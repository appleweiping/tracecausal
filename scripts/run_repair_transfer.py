#!/usr/bin/env python
"""run_repair_transfer.py --- v5 repair-transfer entrypoint (THIN, DO-NOT-RUN).

One wrapper covering the four repair-transfer sub-modes in
``reports/run_packet.md``:

  * ``--freeze-operator``           (section 3.3b): OS-1 operator-selection freeze on
                                    V_sel; wraps ``repair_ops.select_operator`` /
                                    ``operator_grid_cardinality`` / ``policy_hash``.
  * ``--build-matched-null-pool``   (section 3.5): per-target matched-null pool Pi_j;
                                    wraps ``tracecausal.nullpool.build_null_pool``.
  * ``--stage repair_transfer``     (section 3.6): the v5 forwards producing g_ij /
                                    RepairGain rows via Variant-C transport
                                    (``repair_ops.transport`` + ``repair_gain``).
  * ``--estimate-r-hat``            (section 3.7): pure-CPU dependent-pair inference;
                                    wraps ``repair_transfer.r_hat`` /
                                    ``two_way_cluster_bootstrap`` /
                                    ``class_block_permutation`` /
                                    ``hajek_projection_var`` /
                                    ``nuisance.estimate_sigma_r``.

The operator-freeze and repair-transfer forwards touch the model/GPU; the pool
build and r_hat estimation are pure CPU. All sub-modes share the same guard:
default = print resolved plan and exit 0.
"""

from __future__ import annotations

import sys

from _runpacket_common import base_parser, run_stage


def _csv(text):
    return tuple(x.strip() for x in text.split(",") if x.strip())


def _csv_floats(text):
    return tuple(float(x) for x in text.split(",") if x.strip())


def build_parser():
    p = base_parser("v5 repair-transfer (do-not-run wrapper)")
    # mode selectors
    p.add_argument("--freeze-operator", action="store_true")
    p.add_argument("--build-matched-null-pool", action="store_true")
    p.add_argument("--estimate-r-hat", action="store_true")
    p.add_argument("--stage", default=None, help="repair_transfer for the g_ij forwards")
    # common
    p.add_argument("--family")
    p.add_argument("--dataset")
    p.add_argument("--model-revision")
    p.add_argument("--split-hash")
    p.add_argument("--seed", type=int)
    p.add_argument("--split", default=None)
    p.add_argument("--traces")
    p.add_argument("--device", default="cpu")
    # operator-freeze (OS-1)
    p.add_argument("--op-grid-ops", default="patch,replay", type=_csv)
    p.add_argument("--op-grid-alpha", default="0.1,0.25,0.5,0.75,1.0", type=_csv_floats)
    p.add_argument("--op-grid-ref-type", default="factual,neutral", type=_csv)
    p.add_argument("--select-objective", default="r_hat_proposed_minus_B4")
    # matched-null pool
    p.add_argument("--binning")
    p.add_argument("--proximity-pool-min", type=int, default=8)
    # repair-transfer forwards (section 3.6)
    p.add_argument("--selector", help="B0..B5 / PROPOSED")
    p.add_argument("--operator-freeze")
    p.add_argument("--nullpool")
    p.add_argument("--transport-variant", default="C")
    p.add_argument("--taxonomy-hash")
    p.add_argument("--r-null", default="DATA_NEEDED_PIN_AT_LOCK",
                   help="matched-null repair draws; pinned at lock to the sigma_MC floor")
    p.add_argument("--r-int", type=int, default=16)
    p.add_argument("--repair-eval-hash")
    p.add_argument("--source-neq-target", action="store_true")
    p.add_argument("--within-class-only", action="store_true")
    # r_hat estimation (section 3.7)
    p.add_argument("--pairs")
    p.add_argument("--class-weights")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--permutations", type=int, default=10000)
    return p


def _resolve_mode(args):
    if args.freeze_operator:
        return "operator_freeze", True, "tracecausal.repair_ops.select_operator"
    if args.build_matched_null_pool:
        return "build_matched_null_pool", False, "tracecausal.nullpool.build_null_pool"
    if args.estimate_r_hat:
        return ("estimate_r_hat", False,
                "tracecausal.repair_transfer.{r_hat,two_way_cluster_bootstrap,"
                "class_block_permutation,hajek_projection_var} + nuisance.estimate_sigma_r")
    if args.stage == "repair_transfer":
        return ("repair_transfer_forwards", True,
                "tracecausal.repair_ops.transport + tracecausal.repair_transfer.repair_gain")
    return "unspecified", False, "(none: pass one of the mode flags)"


def _heavy(args, plan):
    mode = plan["task"]
    if mode in {"operator_freeze", "repair_transfer_forwards"}:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from tracecausal import (  # noqa: F401
            repair_gain, select_operator, transport,
        )
    else:  # pure-CPU aggregation modes
        from tracecausal import (  # noqa: F401
            build_null_pool, class_block_permutation, estimate_sigma_r,
            hajek_projection_var, r_hat, two_way_cluster_bootstrap,
        )
    raise NotImplementedError(
        f"authorized {mode} wires the listed kernels over real artifacts; "
        "the do-not-run packet does not synthesize run inputs"
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    task, uses_gpu, kernel = _resolve_mode(args)
    plan = {
        "task": task,
        "uses_model_or_gpu": uses_gpu,
        "family": args.family,
        "dataset": args.dataset,
        "selector": args.selector,
        "transport_variant": args.transport_variant,
        "r_null": args.r_null,
        "r_int": args.r_int,
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "op_grid": {
            "ops": list(args.op_grid_ops),
            "alpha": list(args.op_grid_alpha),
            "ref_type": list(args.op_grid_ref_type),
            "objective": args.select_objective,
        } if task == "operator_freeze" else None,
        "kernel": kernel,
        "device": args.device,
    }
    return run_stage(task if task != "unspecified" else "run_repair_transfer", args, plan, _heavy)


if __name__ == "__main__":
    sys.exit(main())
