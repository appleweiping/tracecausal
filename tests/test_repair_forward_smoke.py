"""No-GPU end-to-end smoke for the v5 repair-transfer headline chain.

Proves the full executable surface flows on tiny SYNTHETIC on-disk fixtures, with
**no model, no GPU, no network**:

    g_ij_rows.json  (per arm: PROPOSED / B1 / B4)
        -> run_estimate_r_hat            -> rhat.json   (R_hat + two-way CI + perm + sigma_R)
        -> run_build_gate_inputs (GLUE)  -> gate_inputs.json (the EXACT schema)
        -> run_eval_gates                -> a G9 / G9-NOV verdict

It also pins:

* the realized R_null is read FROM the g_ij artifact (the r_null_count-vs-r_null fix):
  ``rhat.json`` carries ``r_null_realized`` equal to the artifact's value, not a CLI
  default;
* the G9-NOV margin is computed via ``common_support_pairs`` FIRST (the intersection)
  and then the simultaneous bootstrap, fail-closed on identical pair keys;
* the pure-python claim-span inventory (the documented top-risk assumption) builds a
  deterministic atomic-claim grid with exactly one ``is_target_designated`` span;
* the provider factory + claim-span builder import with NO heavy deps (torch/datasets
  are never imported here).

All pure Python; the GPU provider is exercised only at the import/factory + pure-python
helper level (no model is ever loaded).
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
_SRC = _ROOT / "src"
for _p in (_SCRIPTS, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _runners  # noqa: E402
import repair_forward_provider as rfp  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic fixtures (hand-built; deterministic; no model).
# --------------------------------------------------------------------------- #
def _args(**kw) -> Namespace:
    base = dict(output=None)
    base.update(kw)
    return Namespace(**base)


def _arm_rows(*, base_gain: float, r_null_realized: int) -> dict:
    """Build a tiny g_ij_rows.json payload: 4 sources x 4 targets, in-class pairs.

    ``mc_var`` is the PER-TARGET shared matched-null SE^2 (every pair sharing target t
    carries the same mc_var -- the dependent-pair invariant the cluster bootstrap
    propagates as one shared z_t per target).
    """
    per_target_mc = {f"t{t}": 4e-4 + 1e-5 * t for t in range(4)}
    rows = []
    for s in range(4):
        for t in range(4):
            if s == t:
                continue
            cls = "date" if (s + t) % 2 == 0 else "entity"
            rows.append(
                {
                    "g": base_gain + 0.01 * (s - t),
                    "mc_var": per_target_mc[f"t{t}"],
                    "source_id": f"s{s}",
                    "target_id": f"t{t}",
                    "g3_class": cls,
                }
            )
    return {
        "stage": "repair_transfer_forwards",
        "server_authorized": False,
        "r_null_realized": r_null_realized,
        "r_int": 16,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# 1. The pure-python claim-span inventory (documented top-risk assumption).
# --------------------------------------------------------------------------- #
def test_claim_span_inventory_is_deterministic_with_one_designated():
    spans = rfp.build_target_claim_spans(
        prompt_len=40, answer_index=39, designated_span=(20, 23),
        g3_class="factoid", budget=4, proximity_bin_width=16, prompt_start=8,
    )
    assert spans, "expected a non-empty atomic-claim grid"
    # exactly one designated (oracle) span -> the collapse guard has something to drop
    assert sum(1 for s in spans if s.is_target_designated) == 1
    # all spans carry the cell class and the budget; grid is non-overlapping
    assert all(s.g3_class == "factoid" and s.budget_k == 4 for s in spans)
    starts = sorted(s.span.a for s in spans)
    assert all(b - a >= 4 for a, b in zip(starts, starts[1:]))  # stride == budget


def test_provider_factory_imports_without_heavy_deps():
    prov = rfp.make_provider()
    assert type(prov).__name__ == "RepairForwardProvider"
    # torch / datasets must NOT have been imported merely by building the provider.
    assert "torch" not in sys.modules
    assert "datasets" not in sys.modules


# --------------------------------------------------------------------------- #
# 2. The full chain: g_ij rows -> rhat -> gate_inputs -> eval_gates verdict.
# --------------------------------------------------------------------------- #
def _write(p: Path, payload) -> Path:
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return p


def test_end_to_end_g_ij_to_gate_verdict_cpu_only(tmp_path):
    # --- arm g_ij artifacts (PROPOSED strong, B1 weaker, B4 ~ matched-null) ---
    r_null = 8
    proposed_doc = _arm_rows(base_gain=0.12, r_null_realized=r_null)
    b1_doc = _arm_rows(base_gain=0.04, r_null_realized=r_null)
    b4_doc = _arm_rows(base_gain=0.0, r_null_realized=r_null)
    p_proposed = _write(tmp_path / "g_ij_PROPOSED.json", proposed_doc)
    p_b1 = _write(tmp_path / "g_ij_B1.json", b1_doc)
    p_b4 = _write(tmp_path / "g_ij_B4.json", b4_doc)

    # --- 1) run_estimate_r_hat on PROPOSED -> rhat.json ----------------------
    rhat_out = tmp_path / "rhat"
    res = _runners.run_estimate_r_hat(
        _args(output=str(rhat_out), pairs=str(p_proposed),
              bootstrap=400, permutations=400, r_int=16), {}
    )
    # the realized R_null is read FROM the artifact, NOT a CLI default (the fix).
    assert res["r_null_realized"] == r_null
    rhat = json.loads((rhat_out / "rhat.json").read_text(encoding="utf-8"))
    assert rhat["r_null_realized"] == r_null
    for key in ("r_hat", "ci_two_way_cluster_bootstrap", "class_block_permutation",
                "hajek_projection_var", "target_clustered_mc_var", "sigma_r"):
        assert key in rhat

    # --- nuisance gate inputs (synthetic, on-disk) ---------------------------
    nuisance = {
        "n_val_sel": 400, "n_val_inf": 400, "m_prime": 10, "k_bin": 5,
        "m_r0": 0.05, "kappa_lo_repair": 0.92,
    }
    _write(tmp_path / "nuisance_gate.json", nuisance)
    # operator-freeze artifact carrying the DERIVED k_op (grid cardinality).
    operator_freeze = {"stage": "operator_freeze", "k_op": 20}
    _write(tmp_path / "operator_freeze.json", operator_freeze)

    # --- the gate-input manifest (paths relative to tmp_path) ----------------
    manifest = {
        "rhat_proposed": str(rhat_out / "rhat.json"),
        "arm_rows": {
            "PROPOSED": str(p_proposed),
            "B1": str(p_b1),
            "B4": str(p_b4),
        },
        "nuisance": "nuisance_gate.json",
        "k_op": "operator_freeze.json",
        "b4_ci": [-0.02, 0.02],
        "d_util_repair": 0.01,
        "positivity_excluded_frac": 0.1,
        "positivity_excluded_by_class": {"date": 0.1, "entity": 0.15},
        "class_leakage_ok": True,
    }
    p_manifest = _write(tmp_path / "gate_input_spec.json", manifest)

    # --- 2) run_build_gate_inputs (GLUE) -> gate_inputs.json -----------------
    gi_out = tmp_path / "gateinputs"
    gres = _runners.run_build_gate_inputs(
        _args(output=str(gi_out), gate_input_spec=str(p_manifest), bootstrap=400), {}
    )
    gate_inputs = json.loads((gi_out / "gate_inputs.json").read_text(encoding="utf-8"))
    # the EXACT schema run_eval_gates requires must be present.
    required = (
        "n_val_sel", "n_val_inf", "m_prime", "m_r0", "kappa_lo_repair", "r_hat",
        "r_hat_ci_lo", "r_hat_ci_hi", "perm_p", "d_util_repair",
        "positivity_excluded_frac", "class_leakage_ok", "b4_ci_lo", "b4_ci_hi",
        "r_hat_proposed", "r_hat_baselines",
    )
    for k in required:
        assert k in gate_inputs, f"gate_inputs missing {k}"
    # k_op is the DERIVED grid cardinality read from the operator-freeze artifact.
    assert gate_inputs["k_op"] == 20
    # G9-NOV margin lower CI was computed (common support FIRST, then simultaneous).
    assert "g9_nov_margin_ci_low" in gate_inputs
    assert gate_inputs["g9_nov_margin_ci_low"] is not None
    # PROPOSED out-transfers B1 at the point level (sanity on the synthetic margin).
    assert gate_inputs["r_hat_proposed"] > max(gate_inputs["r_hat_baselines"].values())

    # --- 3) run_eval_gates -> a G9 / G9-NOV verdict --------------------------
    gates_out = tmp_path / "gates"
    eres = _runners.run_eval_gates(
        _args(output=str(gates_out), detection=str(gi_out / "gate_inputs.json")), {}
    )
    gates = json.loads((gates_out / "gates.json").read_text(encoding="utf-8"))
    assert "g9_verdict" in gates and "g9_novelty_verdict" in gates
    # the synthetic case is constructed to certify both headline gates.
    assert gates["g9_verdict"] == "useful_candidate"
    assert gates["g9_novelty_verdict"] == "useful_candidate"
    assert eres["g9_verdict"] == "useful_candidate"


def test_gate_inputs_g9_nov_fails_closed_when_margin_not_above_zero(tmp_path):
    # B1 nearly equals PROPOSED -> the simultaneous margin lower CI should not clear 0,
    # so G9-NOV routes to not_novel (the gate bites; it is not pass-through).
    r_null = 8
    proposed_doc = _arm_rows(base_gain=0.05, r_null_realized=r_null)
    b1_doc = _arm_rows(base_gain=0.05, r_null_realized=r_null)  # ties PROPOSED
    b4_doc = _arm_rows(base_gain=0.0, r_null_realized=r_null)
    p_proposed = _write(tmp_path / "g_ij_PROPOSED.json", proposed_doc)
    p_b1 = _write(tmp_path / "g_ij_B1.json", b1_doc)
    p_b4 = _write(tmp_path / "g_ij_B4.json", b4_doc)

    rhat_out = tmp_path / "rhat"
    _runners.run_estimate_r_hat(
        _args(output=str(rhat_out), pairs=str(p_proposed), bootstrap=400,
              permutations=400, r_int=16), {}
    )
    _write(tmp_path / "nuisance_gate.json", {
        "n_val_sel": 400, "n_val_inf": 400, "m_prime": 10, "k_bin": 5,
        "m_r0": 0.05, "kappa_lo_repair": 0.92,
    })
    _write(tmp_path / "operator_freeze.json", {"k_op": 20})
    manifest = {
        "rhat_proposed": str(rhat_out / "rhat.json"),
        "arm_rows": {"PROPOSED": str(p_proposed), "B1": str(p_b1), "B4": str(p_b4)},
        "nuisance": "nuisance_gate.json",
        "k_op": "operator_freeze.json",
        "b4_ci": [-0.02, 0.02],
        "d_util_repair": 0.01,
        "positivity_excluded_frac": 0.1,
        "class_leakage_ok": True,
    }
    p_manifest = _write(tmp_path / "gate_input_spec.json", manifest)

    gi_out = tmp_path / "gateinputs"
    _runners.run_build_gate_inputs(
        _args(output=str(gi_out), gate_input_spec=str(p_manifest), bootstrap=400), {}
    )
    gates_out = tmp_path / "gates"
    _runners.run_eval_gates(
        _args(output=str(gates_out), detection=str(gi_out / "gate_inputs.json")), {}
    )
    gates = json.loads((gates_out / "gates.json").read_text(encoding="utf-8"))
    assert gates["g9_novelty_verdict"] == "not_novel"


def test_build_gate_inputs_missing_field_raises_runinputerror(tmp_path):
    # a manifest missing a required key must raise RunInputError, never fabricate.
    bad = {"arm_rows": {}}  # missing rhat_proposed, nuisance, k_op, b4_ci, ...
    p = _write(tmp_path / "gate_input_spec.json", bad)
    out = tmp_path / "gi"
    try:
        _runners.run_build_gate_inputs(_args(output=str(out), gate_input_spec=str(p)), {})
    except _runners.RunInputError:
        return
    raise AssertionError("expected RunInputError for an incomplete gate-input manifest")
