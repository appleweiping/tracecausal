"""Tests for the authorized run paths (scripts/_runners.py).

These verify the final-gate fix that the stage authorized branches are **real,
executable run logic** -- NOT blanket ``NotImplementedError`` stubs:

* the pure-CPU analysis stages run end-to-end over on-disk fixtures / frozen
  fixtures and call the actual kernels (binning, r_hat dependent-pair inference,
  the Axis X' sweep, nuisance, and the v5 gate enforcement);
* the v5 gates are enforced *executably* -- G9 routes to ``diagnostic`` when the CI
  lower bound is below m_R, and G9-NOV routes to ``not_novel`` when the simultaneous
  margin CI does not clear 0 (so the acceptance checks bite, they are not pass-through);
* the GPU forward stages bind the model through a provider seam and raise an
  actionable :class:`ForwardProviderUnavailable` (naming the env var) when no
  provider is configured -- a real boundary, not a stub;
* missing on-disk inputs raise an actionable :class:`RunInputError` -- the runner
  never fabricates data.

All pure Python; no model, no GPU, no network. The tests build a minimal
``argparse.Namespace`` and call the runner directly (they never flip the real
``server.authorized`` config -- the guard is exercised separately and stays false).
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import _runners  # noqa: E402


def _args(**kw) -> Namespace:
    base = dict(output=None)
    base.update(kw)
    return Namespace(**base)


# --------------------------------------------------------------------------- #
# Adversarial oracle: fully-implemented, no inputs needed.
# --------------------------------------------------------------------------- #
def test_adversarial_oracle_runs_and_emits_real_sweep(tmp_path):
    out = tmp_path / "ao"
    res = _runners.run_adversarial_oracle(
        _args(output=str(out), regimes=None, xi_grid=None, axis="x_prime"), {}
    )
    assert res["row_count"] == 10  # 2 regimes x 5 xi
    payload = json.loads((out / "axis_x_prime.json").read_text(encoding="utf-8"))
    assert payload["server_authorized"] is False
    # blind regime at xi=1 must collapse R_hat toward 0 (the P5/R4 soundness check).
    blind_top = [
        r for r in payload["sweep"] if r["regime"] == "blind" and r["xi"] == 1.0
    ]
    assert blind_top and abs(blind_top[0]["r_hat_expected"]) <= 1e-6
    assert payload["r4_certification_withdrawn"] is False  # method is sound on fixture


# --------------------------------------------------------------------------- #
# Binning-as-code over on-disk V_sel distances.
# --------------------------------------------------------------------------- #
def test_select_binning_runs_over_disk_distances(tmp_path):
    vsel = tmp_path / "vsel.json"
    vsel.write_text(json.dumps({"distances_to_answer": list(range(12)) * 3}), encoding="utf-8")
    out = tmp_path / "bin"
    res = _runners.run_select_binning(_args(output=str(out), v_sel=str(vsel)), {})
    assert res["k_bin"] >= 1
    payload = json.loads((out / "binning.json").read_text(encoding="utf-8"))
    assert payload["binning"]["delta_pos"] in (1, 2, 4, 8, 16)


def test_select_binning_missing_input_raises_runinputerror(tmp_path):
    out = tmp_path / "bin"
    try:
        _runners.run_select_binning(_args(output=str(out), v_sel=str(tmp_path / "nope.json")), {})
    except _runners.RunInputError:
        return
    raise AssertionError("expected RunInputError for a missing V_sel artifact")


# --------------------------------------------------------------------------- #
# R_hat dependent-pair inference over collected g_ij rows.
# --------------------------------------------------------------------------- #
def _g_ij_rows():
    rows = []
    for s in range(4):
        for t in range(4):
            if s != t:
                rows.append(
                    {
                        "g": 0.1 + 0.01 * (s - t),
                        "mc_var": 4e-4,
                        "source_id": f"s{s}",
                        "target_id": f"t{t}",
                        "g3_class": "date" if (s + t) % 2 == 0 else "entity",
                    }
                )
    return rows


def test_estimate_r_hat_runs_full_inference_stack(tmp_path):
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps({"rows": _g_ij_rows()}), encoding="utf-8")
    out = tmp_path / "rhat"
    res = _runners.run_estimate_r_hat(
        _args(output=str(out), pairs=str(pairs), bootstrap=200, permutations=200, r_int=16), {}
    )
    assert res["row_count"] == len(_g_ij_rows())
    payload = json.loads((out / "rhat.json").read_text(encoding="utf-8"))
    # the full MF-4 stack is present and computed (not DATA_NEEDED placeholders).
    for key in (
        "r_hat",
        "ci_two_way_cluster_bootstrap",
        "class_block_permutation",
        "hajek_projection_var",
        "target_clustered_mc_var",
        "sigma_r",
    ):
        assert key in payload
    ci = payload["ci_two_way_cluster_bootstrap"]
    assert ci["ci_lo"] <= ci["ci_hi"]


# --------------------------------------------------------------------------- #
# Gate enforcement is executable (G9 / G9-NOV / m_R bite).
# --------------------------------------------------------------------------- #
def _gate_inputs(**override):
    gi = {
        "n_val_sel": 400,
        "n_val_inf": 400,
        "m_prime": 10,
        "k_bin": 5,
        "k_op": 20,
        "m_r0": 0.05,
        "kappa_lo_repair": 0.92,
        "r_hat": 0.12,
        "r_hat_ci_lo": 0.07,
        "r_hat_ci_hi": 0.17,
        "perm_p": 1e-4,
        "d_util_repair": 0.01,
        "positivity_excluded_frac": 0.1,
        "class_leakage_ok": True,
        "b4_ci_lo": -0.02,
        "b4_ci_hi": 0.02,
        "r_hat_proposed": 0.12,
        "r_hat_baselines": {"B1": 0.05, "B2": 0.04, "B3": 0.03},
        "positivity_excluded_by_class": {"date": 0.1, "entity": 0.2},
        "g9_nov_margin_ci_low": 0.02,
    }
    gi.update(override)
    return gi


def _run_gates(tmp_path, gi) -> dict:
    p = tmp_path / "gate_inputs.json"
    p.write_text(json.dumps(gi), encoding="utf-8")
    out = tmp_path / "gates"
    _runners.run_eval_gates(_args(output=str(out), detection=str(p)), {})
    return json.loads((out / "gates.json").read_text(encoding="utf-8"))


def test_eval_gates_certifies_a_clean_case(tmp_path):
    d = _run_gates(tmp_path, _gate_inputs())
    assert d["si_path"] == "SI-1"
    # m_R is the Eq. m-R attenuation, NOT the v4 0.05 necessity margin.
    assert abs(d["m_r"] - 0.05 / (2 * 0.92 - 1)) < 1e-9
    assert d["g9_verdict"] == "useful_candidate"
    assert d["g9_novelty_verdict"] == "useful_candidate"


def test_eval_gates_g9_fails_when_ci_below_m_r(tmp_path):
    # CI lower bound 0.04 < m_R (0.0595) -> G9 routes to diagnostic (never invalidated).
    d = _run_gates(tmp_path, _gate_inputs(r_hat_ci_lo=0.04))
    assert d["g9_verdict"] == "diagnostic"


def test_eval_gates_g9_nov_fails_when_margin_ci_not_above_zero(tmp_path):
    d = _run_gates(tmp_path, _gate_inputs(g9_nov_margin_ci_low=-0.01))
    assert d["g9_novelty_verdict"] == "not_novel"


def test_eval_gates_g9_diagnostic_on_utility_breach(tmp_path):
    # repair utility cost above the 0.02 bound -> diagnostic (reframe as abstention).
    d = _run_gates(tmp_path, _gate_inputs(d_util_repair=0.05))
    assert d["g9_verdict"] == "diagnostic"


# --------------------------------------------------------------------------- #
# GPU forward seam: actionable error, not a stub.
# --------------------------------------------------------------------------- #
def test_gpu_stage_without_provider_raises_actionable_error(tmp_path, monkeypatch):
    monkeypatch.delenv("TRACECAUSAL_FORWARD_PROVIDER", raising=False)
    # ensure no provider lingers from another test
    monkeypatch.setattr(_runners, "_PROVIDER", None)
    out = tmp_path / "extract"
    try:
        _runners.run_extract_traces(_args(output=str(out)), {"task": "extract_traces"})
    except _runners.ForwardProviderUnavailable as exc:
        assert "TRACECAUSAL_FORWARD_PROVIDER" in str(exc)
        return
    raise AssertionError("expected ForwardProviderUnavailable when no provider configured")


def test_registered_forward_provider_is_used(tmp_path, monkeypatch):
    calls = {}

    class _StubProvider(_runners.ForwardProvider):
        def __call__(self, *, stage, args, plan, output_dir):
            calls["stage"] = stage
            return {"row_count": 7}

    monkeypatch.setattr(_runners, "_PROVIDER", None)
    _runners.register_forward_provider(_StubProvider())
    try:
        out = tmp_path / "extract"
        res = _runners.run_extract_traces(_args(output=str(out)), {"task": "extract_traces"})
        assert calls["stage"] == "extract_traces"
        assert res["row_count"] == 7
        assert (out / "stage_provenance.json").exists()
    finally:
        _runners.register_forward_provider(None)  # type: ignore[arg-type]
        monkeypatch.setattr(_runners, "_PROVIDER", None)
