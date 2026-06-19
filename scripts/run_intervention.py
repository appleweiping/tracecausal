#!/usr/bin/env python
"""run_intervention.py --- screening interventions U_hat (THIN, DO-NOT-RUN wrapper).

Matches ``reports/run_packet.md`` section 3.4. Runs the per-detector screening
interventions (mask/patch/replay) that produce ``CIURecord`` necessity rows
(``U_hat``, ``u_deflated``, matched-null pool draws, negative-control deltas) for
the screening gates G1/G2/G5'/G6/G7/G8.

Forwards are involved, so the stage is fully behind the authorization guard.
Default = print resolved plan and exit 0 (no model, no GPU).
"""

from __future__ import annotations

import sys

from _runpacket_common import base_parser, run_stage


def _csv(text):
    return tuple(x.strip() for x in text.split(",") if x.strip())


def build_parser():
    p = base_parser("screening interventions U_hat (do-not-run wrapper)")
    p.add_argument("--stage", default="screening")
    p.add_argument("--method", help="ciu_selector / random_segment / ... / inside_detector")
    p.add_argument("--family")
    p.add_argument("--model-revision")
    p.add_argument("--dataset")
    p.add_argument("--split-hash")
    p.add_argument("--seed", type=int)
    p.add_argument("--traces", help="per-cell-seed trace dir")
    p.add_argument("--binning", help="frozen Binning json for this cell")
    p.add_argument("--operator", default="mask", help="mask / patch / replay")
    p.add_argument("--r-int", type=int, default=16)
    p.add_argument("--negative-controls",
                   default="random_non_causal_segment,shuffled_trace_segment,no_op_intervention",
                   type=_csv)
    p.add_argument("--evaluator-hash")
    p.add_argument("--ciu-scored", action="store_true")
    p.add_argument("--device", default="cpu")
    return p


def _heavy(args, plan):
    # Authorized run: the screening interventions apply mask/patch/replay forwards on
    # ${GPU}, so the model is bound through the ForwardProvider seam; this owns the
    # real orchestration. No forward is fabricated in pure Python; an unconfigured
    # provider raises an actionable ForwardProviderUnavailable. Post-forward CIURecord
    # scoring uses the implemented tracecausal.ciu/interventions kernels via the
    # provider. Reachable only after the §1 authorization flip.
    from _runners import run_screening_interventions as _run  # lazy
    return _run(args, plan)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    plan = {
        "task": "run_intervention",
        "stage": args.stage,
        "method": args.method,
        "family": args.family,
        "model_revision": args.model_revision,
        "dataset": args.dataset,
        "split_hash": args.split_hash,
        "seed": args.seed,
        "operator": args.operator,
        "r_int": args.r_int,
        "negative_controls": list(args.negative_controls),
        "evaluator_hash": args.evaluator_hash,
        "ciu_scored": args.ciu_scored,
        "device": args.device,
        "emits": "CIURecord screening rows (U_hat, u_deflated, matched-null, NC deltas)",
        "kernel": "tracecausal.interventions.{mask,patch,replay} + tracecausal.ciu",
    }
    return run_stage("run_intervention", args, plan, _heavy)


if __name__ == "__main__":
    sys.exit(main())
