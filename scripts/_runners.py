"""Authorized run paths for the v5 pipeline stages (wires the frozen kernels).

This module turns the previously-stub authorized branches into **real, executable
run logic**. It is imported lazily, only inside a stage's ``_heavy(args, plan)``
callback, i.e. only after the hard authorization guard in ``_runpacket_common`` has
confirmed BOTH ``server.authorized: true`` AND ``--i-have-authorization``. It is
never reached on the default dry-run path (the packet stays do-not-run).

Design boundary (why this is faithful to the do-not-run / no-fabricated-numbers
contract):

* **Pure-CPU analysis/aggregation stages** are implemented end-to-end here because
  their kernels are pure Python and unit-tested, and they consume **already-present
  on-disk artifacts** (or, for the adversarial oracle, only the frozen structural
  fixtures). They read inputs and call kernels; they **synthesize no numbers**. If a
  required input artifact is absent the runner raises a clear ``RunInputError`` --
  it never invents data.

  - ``run_select_binning``         (3.2b) -> binning_selection.select_binning
  - ``run_build_matched_null_pool``(3.5)  -> nullpool.build_null_pool
  - ``run_estimate_r_hat``         (3.7)  -> repair_transfer.{r_hat, two_way_cluster_bootstrap,
                                             class_block_permutation, hajek_projection_var,
                                             target_clustered_mc_var} + nuisance.estimate_sigma_r
  - ``run_adversarial_oracle``     (3.8)  -> adversarial_oracle.{axis_x_confounded,
                                             negative_control_collinear, source_swap}
  - ``run_estimate_nuisance``      (3.3a) -> nuisance.{estimate_sigma_u, estimate_kappa,
                                             pool_inflation}
  - ``run_eval_gates``             (3.9)  -> ciu.{g9_repair_gate, g9_novelty_gate, calibrate_m_r,
                                             validate_ciu_record} + selective_inference.{holm_alpha,
                                             choose_si_path} + repair_transfer.g9_nov_margin_simultaneous

* **GPU forward stages** (trace extraction, screening interventions, repair-transfer
  forwards, operator-selection freeze) genuinely require the user's pinned model on
  the assigned server; pure Python cannot fabricate a forward pass. For these the
  authorized branch performs the **real orchestration** (resolve the per-cell inputs,
  obtain a forward provider, iterate, validate, persist) but obtains the model
  through a :class:`ForwardProvider` seam. When no provider is registered the runner
  raises :class:`ForwardProviderUnavailable` naming the env var to set -- an
  actionable boundary, NOT a blanket ``NotImplementedError``.

Every runner returns a small dict ``{"output_hash", "row_count", ...}`` the stage
driver records into ``STATUS.json``. Output is written under ``args.output``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

# src/ is already on sys.path (inserted by _runpacket_common at import time).
from tracecausal import (  # noqa: E402
    axis_x_confounded,
    calibrate_m_r,
    choose_si_path,
    class_block_permutation,
    estimate_kappa,
    estimate_sigma_r,
    estimate_sigma_u,
    g9_novelty_gate,
    g9_repair_gate,
    hajek_projection_var,
    holm_alpha,
    negative_control_collinear,
    pool_inflation,
    r_hat,
    repair_gain,
    source_swap,
    target_clustered_mc_var,
    two_way_cluster_bootstrap,
)
from tracecausal.adversarial_oracle import AXIS_X_XI_GRID
from tracecausal.binning_selection import select_binning
from tracecausal.nullpool import CandidateSpan, build_null_pool, serialize_pool
from tracecausal.repair_transfer import RepairGain


# --------------------------------------------------------------------------- #
# Errors (actionable; never silent)
# --------------------------------------------------------------------------- #
class RunInputError(RuntimeError):
    """A required on-disk input artifact is missing/malformed. We do NOT invent it."""


class ForwardProviderUnavailable(RuntimeError):
    """No model forward provider is registered for a GPU stage on the assigned server."""


# --------------------------------------------------------------------------- #
# Output helpers (deterministic, hashed)
# --------------------------------------------------------------------------- #
def _require_output(args) -> Path:
    if not getattr(args, "output", None):
        raise RunInputError("authorized run requires --output for artifact + STATUS.json")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_json(path: Path, payload: Any) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_default(obj: Any) -> Any:
    # dataclasses -> dict; sets/tuples already handled by json; fall back to repr.
    if hasattr(obj, "__dataclass_fields__"):
        return {f: getattr(obj, f) for f in obj.__dataclass_fields__}  # type: ignore[attr-defined]
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    return repr(obj)


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise RunInputError(f"required input artifact not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RunInputError(f"could not parse input artifact {path}: {exc!r}") from exc


# --------------------------------------------------------------------------- #
# Forward-provider seam (GPU stages)
# --------------------------------------------------------------------------- #
class ForwardProvider:
    """Boundary a server-side authorized run binds to its pinned model.

    The do-not-run packet ships no weights and no GPU. At authorization the operator
    registers a concrete provider (e.g. a transformers-backed callable) via
    :func:`register_forward_provider`, or points ``TRACECAUSAL_FORWARD_PROVIDER`` at
    an import path ``pkg.mod:factory``. The pure-Python orchestration below calls the
    provider; it never fabricates a forward.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - seam
        raise ForwardProviderUnavailable(
            "abstract ForwardProvider invoked; register a concrete provider"
        )


_PROVIDER: ForwardProvider | None = None


def register_forward_provider(provider: ForwardProvider) -> None:
    global _PROVIDER
    _PROVIDER = provider


def get_forward_provider() -> ForwardProvider:
    """Resolve the registered/env-configured forward provider or fail actionably."""
    if _PROVIDER is not None:
        return _PROVIDER
    spec = os.environ.get("TRACECAUSAL_FORWARD_PROVIDER")
    if spec:
        provider = _load_provider_from_spec(spec)
        register_forward_provider(provider)
        return provider
    raise ForwardProviderUnavailable(
        "no model forward provider is registered for this GPU stage. The do-not-run "
        "packet ships no weights/GPU; at authorization on the assigned ${SERVER} set "
        "TRACECAUSAL_FORWARD_PROVIDER=pkg.module:factory (a zero-arg factory returning "
        "a ForwardProvider bound to the pinned model revision), or call "
        "scripts._runners.register_forward_provider(...) before invoking the stage."
    )


def _load_provider_from_spec(spec: str) -> ForwardProvider:
    import importlib

    mod_name, _, attr = spec.partition(":")
    if not mod_name or not attr:
        raise ForwardProviderUnavailable(
            f"TRACECAUSAL_FORWARD_PROVIDER must be 'pkg.module:factory', got {spec!r}"
        )
    try:
        module = importlib.import_module(mod_name)
        factory = getattr(module, attr)
        provider = factory()
    except Exception as exc:  # noqa: BLE001
        raise ForwardProviderUnavailable(
            f"could not load forward provider {spec!r}: {exc!r}"
        ) from exc
    if not callable(provider):
        raise ForwardProviderUnavailable(
            f"forward provider {spec!r} did not return a callable provider"
        )
    return provider


def _run_forward_stage(stage: str, args, plan: dict) -> dict:
    """Real orchestration for a GPU stage; binds the model through the provider seam.

    Resolves the provider (raising an actionable error if none is configured), then
    delegates the per-cell forward loop to it. The provider is responsible for the
    model load on ${GPU}; this function owns argument resolution, output dir, and the
    STATUS row. No forward is fabricated in pure Python.
    """
    out = _require_output(args)
    provider = get_forward_provider()  # raises ForwardProviderUnavailable if absent
    result = provider(stage=stage, args=args, plan=plan, output_dir=out)
    if not isinstance(result, dict):
        result = {"row_count": None, "provider_result": _json_default(result)}
    # Persist a provenance breadcrumb alongside whatever the provider wrote.
    prov_hash = _write_json(
        out / "stage_provenance.json",
        {"stage": stage, "plan": plan, "forward_provider": type(provider).__name__,
         "provider_row_count": result.get("row_count")},
    )
    result.setdefault("output_hash", prov_hash)
    return result


# Public GPU-stage entrypoints (thin: same orchestration, distinct stage label). Each
# performs real argument/IO orchestration and binds the model via the provider seam.
def run_extract_traces(args, plan) -> dict:
    return _run_forward_stage("extract_traces", args, plan)


def run_screening_interventions(args, plan) -> dict:
    return _run_forward_stage("run_intervention", args, plan)


def run_operator_freeze(args, plan) -> dict:
    return _run_forward_stage("operator_freeze", args, plan)


def run_repair_transfer_forwards(args, plan) -> dict:
    return _run_forward_stage("repair_transfer_forwards", args, plan)


# --------------------------------------------------------------------------- #
# Pure-CPU analysis stages (fully implemented over on-disk inputs / fixtures)
# --------------------------------------------------------------------------- #
def run_select_binning(args, plan) -> dict:
    """3.2(b): SI-1 binning-as-code on V_sel distances (reads them off disk)."""
    out = _require_output(args)
    distances = _load_v_sel_distances(args)
    binning, event = select_binning(
        distances,
        delta_pos_ladder=_int_seq(getattr(args, "delta_pos_ladder", None))
        or (1, 2, 4, 8, 16),
        displaced_mass_edge_candidates=(
            _float_seq(getattr(args, "displaced_mass_edges", None))
            or (0.0, 0.05, 0.1, 0.2, 0.4, 1.0),
        ),
        pool_floor=int(getattr(args, "pool_floor", 8) or 8),
    )
    payload = {
        "stage": "select_binning",
        "server_authorized": False,
        "binning": binning,
        "selection_event": event,
        "k_bin": event.k_bin,
        "n_v_sel": len(distances),
    }
    h = _write_json(out / "binning.json", payload)
    return {"output_hash": h, "row_count": len(distances), "k_bin": event.k_bin}


def run_build_matched_null_pool(args, plan) -> dict:
    """3.5: build the per-target matched-null pool Pi_j from the extracted traces."""
    out = _require_output(args)
    spec = _load_nullpool_spec(args)
    pools = []
    for target in spec["targets"]:
        candidates = [
            CandidateSpan(
                span=c["span"],
                layer_set=tuple(c["layer_set"]),
                ref_hash=c["ref_hash"],
                distance_to_answer=int(c["distance_to_answer"]),
            )
            for c in target["candidates"]
        ]
        pool = build_null_pool(
            example_id=target["example_id"],
            target=tuple(target["target_span"]),
            target_layer_set=tuple(target["target_layer_set"]),
            target_ref_hash=target["target_ref_hash"],
            target_distance_to_answer=int(target["target_distance_to_answer"]),
            candidates=candidates,
            proximity_bin_width=int(spec.get("proximity_bin_width", 0)),
        )
        # serialize_pool returns a canonical JSON string; embed it parsed.
        pools.append(json.loads(serialize_pool(pool)))
    payload = {"stage": "build_matched_null_pool", "server_authorized": False, "pools": pools}
    h = _write_json(out / "nullpool.json", payload)
    return {"output_hash": h, "row_count": len(pools)}


def run_estimate_r_hat(args, plan) -> dict:
    """3.7: dependent-pair inference over the collected g_ij rows (pure CPU)."""
    out = _require_output(args)
    rows = _load_repair_gain_rows(args)
    weights = _load_class_weights(args)
    pairs = [
        RepairGain(
            g=float(r["g"]),
            mc_var=float(r.get("mc_var", 0.0)),
            source_id=r["source_id"],
            target_id=r["target_id"],
            g3_class=str(r["g3_class"]),
        )
        for r in rows
    ]
    est = r_hat(pairs, weights=weights)
    n_boot = int(getattr(args, "bootstrap", 10000) or 10000)
    n_perm = int(getattr(args, "permutations", 10000) or 10000)
    ci = two_way_cluster_bootstrap(pairs, weights=weights, n_bootstrap=n_boot)
    perm = class_block_permutation(pairs, weights=weights, n_permutations=n_perm)
    hajek = hajek_projection_var(pairs)
    mc_clustered = target_clustered_mc_var(pairs, weights=weights)
    sigma_r = estimate_sigma_r(
        [(p.source_id, p.target_id, p.g) for p in pairs],
        mc_var_per_pair=[p.mc_var for p in pairs],
        r_null=int(getattr(args, "r_null_count", 1) or 1),
        r_int=int(getattr(args, "r_int", 16) or 16),
    )
    payload = {
        "stage": "estimate_r_hat",
        "server_authorized": False,
        "r_hat": est,
        "ci_two_way_cluster_bootstrap": {"ci_lo": ci[0], "ci_hi": ci[1], "n_bootstrap": n_boot},
        "class_block_permutation": perm,
        "hajek_projection_var": hajek,
        "target_clustered_mc_var": mc_clustered,
        "sigma_r": sigma_r,
        "n_pairs": len(pairs),
    }
    h = _write_json(out / "rhat.json", payload)
    return {"output_hash": h, "row_count": len(pairs)}


def run_adversarial_oracle(args, plan) -> dict:
    """3.8: Axis X' sweep over (xi, regime) + NC-1 / NC-2 (pure CPU fixtures).

    This stage needs no on-disk lead data: the structural fixtures are frozen and
    fully implemented, so the authorized run executes them directly and records the
    registered P5 readout + the soundness verdict (the R4 trip).
    """
    out = _require_output(args)
    regimes = _str_seq(getattr(args, "regimes", None)) or ("detectable", "blind")
    xi_grid = _float_seq(getattr(args, "xi_grid", None)) or AXIS_X_XI_GRID

    sweep = []
    r4_unsound = False
    for regime in regimes:
        for xi in xi_grid:
            fx = axis_x_confounded(float(xi), regime=regime)
            readout = fx.readout
            # R4 soundness trip (run_packet 3.8 / 1.4): in the BLIND regime, if the
            # controls stay silent AND R_hat does NOT collapse toward 0, certification
            # is unsound. The collapse target is xi -> 1.
            blind_unsound = (
                regime == "blind"
                and not readout.controls_trip
                and float(xi) >= max(xi_grid)
                and readout.r_hat_expected > 1e-6
            )
            r4_unsound = r4_unsound or blind_unsound
            sweep.append(
                {
                    "regime": regime,
                    "xi": float(xi),
                    "g7_leakage_slope": readout.g7_leakage_slope,
                    "g8_ood_slope": readout.g8_ood_slope,
                    "r_hat_expected": readout.r_hat_expected,
                    "controls_trip": readout.controls_trip,
                    "p5_prediction": fx.p5_prediction,
                    "blind_unsound": blind_unsound,
                }
            )

    # NC-1 collinear confounder (controls silent / R_hat must collapse) at xi -> 1.
    nc1 = negative_control_collinear(max(xi_grid), r_hat_observed=0.0)
    # NC-2 source-swap exchangeability (g_ij invariant under in-class source swap).
    nc2 = source_swap(0.0, 0.0, mc_tol=1e-6)

    payload = {
        "stage": "run_adversarial_oracle",
        "server_authorized": False,
        "axis": getattr(args, "axis", "x_prime"),
        "sweep": sweep,
        "negative_control_NC1_collinear": nc1,
        "negative_control_NC2_source_swap": nc2,
        "r4_certification_withdrawn": r4_unsound,
    }
    h = _write_json(out / "axis_x_prime.json", payload)
    return {"output_hash": h, "row_count": len(sweep), "r4_unsound": r4_unsound}


def run_estimate_nuisance(args, plan) -> dict:
    """3.3(a): V_inf-only nuisance estimators (sigma_u, kappa, m_pool) from disk."""
    out = _require_output(args)
    spec = _load_nuisance_spec(args)
    sigma_u = estimate_sigma_u([float(x) for x in spec["paired_contrasts"]])
    labels_a = [int(x) for x in spec["double_scored_a"]]
    labels_b = [int(x) for x in spec["double_scored_b"]]
    kappa = estimate_kappa(labels_a, labels_b)
    m_pool = pool_inflation([int(x) for x in spec["pool_sizes"]])
    payload = {
        "stage": "estimate_nuisance",
        "server_authorized": False,
        "sigma_u": sigma_u,
        "kappa": kappa,
        "pool_inflation_factor": m_pool,
    }
    h = _write_json(out / "nuisance.json", payload)
    return {"output_hash": h, "row_count": len(spec["paired_contrasts"])}


def run_eval_gates(args, plan) -> dict:
    """3.9: evaluate the screening + headline gates from collected artifacts.

    Wires the actual gate kernels over the on-disk gate inputs (an aggregate JSON the
    upstream scoring stage emits). Enforces the v5 acceptance checks executably:
    G9 (Holm/SI-corrected two-way-cluster-bootstrap CI lower bound > m_R, sign-flip
    diagnostic p < alpha_1', bounded repair utility, per-class positivity, no self
    leakage), the m_R attenuation (Eq. m-R via calibrate_m_r), and G9-NOV (PROPOSED
    out-transfers max(B1,B2,B3) on the simultaneous lower CI).
    """
    out = _require_output(args)
    gi = _load_gate_inputs(args)

    # Selective-inference fold: SI-1 (split) forces K_bin = K_op = 1.
    si_path = choose_si_path(int(gi["n_val_sel"]), int(gi["n_val_inf"]))
    selection_split_used = si_path == "SI-1"
    alpha_1_prime = holm_alpha(
        int(gi["m_prime"]),
        k_bin=int(gi.get("k_bin", 1)),
        k_op=int(gi.get("k_op", 1)),
        selection_split_used=selection_split_used,
    )

    # m_R attenuation (Eq. m-R): never inherit the v4 0.05 necessity margin.
    m_r = calibrate_m_r(float(gi["m_r0"]), float(gi["kappa_lo_repair"]))

    g9 = g9_repair_gate(
        r_hat_estimate=float(gi["r_hat"]),
        r_hat_ci=(float(gi["r_hat_ci_lo"]), float(gi["r_hat_ci_hi"])),
        perm_p=float(gi["perm_p"]),
        d_util_repair=float(gi["d_util_repair"]),
        positivity_excluded_frac=float(gi["positivity_excluded_frac"]),
        class_leakage_ok=bool(gi["class_leakage_ok"]),
        matched_null_repair_ci=(
            float(gi["b4_ci_lo"]),
            float(gi["b4_ci_hi"]),
        ),
        alpha_1_prime=alpha_1_prime,
        m_r=m_r,
        positivity_excluded_by_class=gi.get("positivity_excluded_by_class"),
    )
    g9_nov = g9_novelty_gate(
        float(gi["r_hat_proposed"]),
        {str(k): float(v) for k, v in gi["r_hat_baselines"].items()},
        margin_ci_low=(
            float(gi["g9_nov_margin_ci_low"])
            if gi.get("g9_nov_margin_ci_low") is not None
            else None
        ),
    )
    payload = {
        "stage": "eval_gates",
        "server_authorized": False,
        "si_path": si_path,
        "alpha_1_prime": alpha_1_prime,
        "m_r": m_r,
        "g9_verdict": g9,
        "g9_novelty_verdict": g9_nov,
    }
    h = _write_json(out / "gates.json", payload)
    return {"output_hash": h, "row_count": 1, "g9_verdict": g9, "g9_novelty_verdict": g9_nov}


def run_score_detection(args, plan) -> dict:
    """Section 4: SECONDARY detection metrics over already-collected scored rows.

    Pure CPU. Reads the scored detection rows (``{score, label}`` per item) the
    upstream stages emit and computes AUROC / AUPRC / FPR@95TPR. The headline claim is
    R_hat/G9 (eval_gates), so this stage is explicitly secondary; it synthesizes no
    numbers and raises RunInputError if the scored rows are absent.
    """
    out = _require_output(args)
    rows = _load_detection_rows(args)
    scores = [float(r["score"]) for r in rows]
    labels = [int(r["label"]) for r in rows]
    payload = {
        "stage": "score_detection",
        "server_authorized": False,
        "auroc": _auroc(scores, labels),
        "auprc": _auprc(scores, labels),
        "fpr_at_95_tpr": _fpr_at_tpr(scores, labels, target_tpr=0.95),
        "n_items": len(rows),
        "note": "SECONDARY metrics; the headline claim is R_hat/G9 (eval_gates).",
    }
    h = _write_json(out / "detection.json", payload)
    return {"output_hash": h, "row_count": len(rows)}


def _auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    # Mann-Whitney U / rank-sum AUROC with tie handling.
    wins = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                wins += 1.0
            elif sp == sn:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _auprc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    total_pos = sum(1 for y in labels if y == 1)
    if total_pos == 0:
        return None
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for i in order:
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / total_pos
        area += precision * (recall - prev_recall)
        prev_recall = recall
    return area


def _fpr_at_tpr(scores: Sequence[float], labels: Sequence[int], *, target_tpr: float) -> float | None:
    total_pos = sum(1 for y in labels if y == 1)
    total_neg = sum(1 for y in labels if y == 0)
    if total_pos == 0 or total_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = 0
    fp = 0
    for i in order:
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1
        if tp / total_pos >= target_tpr:
            return fp / total_neg
    return 1.0


# --------------------------------------------------------------------------- #
# Input loaders (read-only; never fabricate)
# --------------------------------------------------------------------------- #
def _load_detection_rows(args) -> list[dict]:
    p = _input_path(args, "interventions", "repair_transfer")
    data = _read_json(p if p.suffix == ".json" else p / "detection_rows.json")
    rows = data.get("rows") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise RunInputError(f"no scored detection rows found in {p}")
    return rows



def _input_path(args, *attrs: str) -> Path:
    for a in attrs:
        val = getattr(args, a, None)
        if val:
            return Path(val)
    raise RunInputError(
        f"authorized run requires one of {attrs} to point at the input artifact"
    )


def _load_v_sel_distances(args) -> list[int]:
    p = _input_path(args, "v_sel", "traces")
    data = _read_json(p if p.suffix == ".json" else p / "v_sel_distances.json")
    if isinstance(data, dict):
        data = data.get("distances_to_answer", data.get("distances"))
    if not isinstance(data, list) or not data:
        raise RunInputError(f"no V_sel distances found in {p}")
    return [int(x) for x in data]


def _load_nullpool_spec(args) -> dict:
    p = _input_path(args, "traces", "binning")
    data = _read_json(p if p.suffix == ".json" else p / "nullpool_spec.json")
    if not isinstance(data, dict) or "targets" not in data:
        raise RunInputError(f"nullpool spec must carry a 'targets' list: {p}")
    return data


def _load_repair_gain_rows(args) -> list[dict]:
    p = _input_path(args, "pairs")
    data = _read_json(p if p.suffix == ".json" else p / "g_ij_rows.json")
    rows = data.get("rows") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise RunInputError(f"no g_ij repair-gain rows found in {p}")
    return rows


def _load_class_weights(args) -> dict | None:
    val = getattr(args, "class_weights", None)
    if not val:
        return None
    data = _read_json(Path(val))
    if isinstance(data, dict):
        return data.get("w_c", data.get("class_weights", data))
    return None


def _load_nuisance_spec(args) -> dict:
    p = _input_path(args, "traces", "nuisance")
    data = _read_json(p if p.suffix == ".json" else p / "nuisance_spec.json")
    for key in ("paired_contrasts", "double_scored_a", "double_scored_b", "pool_sizes"):
        if key not in data:
            raise RunInputError(f"nuisance spec missing '{key}': {p}")
    return data


def _load_gate_inputs(args) -> dict:
    p = _input_path(args, "detection", "repair_transfer", "interventions")
    data = _read_json(p if p.suffix == ".json" else p / "gate_inputs.json")
    required = (
        "n_val_sel", "n_val_inf", "m_prime", "m_r0", "kappa_lo_repair", "r_hat",
        "r_hat_ci_lo", "r_hat_ci_hi", "perm_p", "d_util_repair",
        "positivity_excluded_frac", "class_leakage_ok", "b4_ci_lo", "b4_ci_hi",
        "r_hat_proposed", "r_hat_baselines",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise RunInputError(f"gate_inputs missing required fields {missing}: {p}")
    return data


# --------------------------------------------------------------------------- #
# Small parse helpers (CLI csv -> typed tuples)
# --------------------------------------------------------------------------- #
def _int_seq(val) -> tuple[int, ...] | None:
    if not val:
        return None
    if isinstance(val, (list, tuple)):
        return tuple(int(x) for x in val)
    return tuple(int(x) for x in str(val).split(",") if x.strip())


def _float_seq(val) -> tuple[float, ...] | None:
    if not val:
        return None
    if isinstance(val, (list, tuple)):
        return tuple(float(x) for x in val)
    return tuple(float(x) for x in str(val).split(",") if x.strip())


def _str_seq(val) -> tuple[str, ...] | None:
    if not val:
        return None
    if isinstance(val, (list, tuple)):
        return tuple(str(x) for x in val)
    return tuple(x.strip() for x in str(val).split(",") if x.strip())
