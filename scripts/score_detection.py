#!/usr/bin/env python
"""score_detection.py --- detection + repair scoring per cell (THIN, DO-NOT-RUN).

Matches the section 4 summary row for detection scoring in
``reports/run_packet.md`` (queue stage ``score_detection``). PURE-CPU stage over
already-extracted intervention + repair artifacts: computes the SECONDARY
detection metrics (AUROC, AUPRC, FPR@95TPR) and the repair scoring readouts
(repair factuality gain, repair utility delta) per cell. No model, no GPU.

The authorization guard still applies before writing under a run output tree.
Default = print resolved plan and exit 0.
"""

from __future__ import annotations

import sys

from _runpacket_common import base_parser, run_stage


def _csv(text):
    return tuple(x.strip() for x in text.split(",") if x.strip())


def build_parser():
    p = base_parser("detection + repair scoring (do-not-run wrapper)")
    p.add_argument("--family")
    p.add_argument("--dataset")
    p.add_argument("--interventions", help="screening intervention artifacts root")
    p.add_argument("--repair-transfer", help="repair-transfer artifacts root")
    p.add_argument("--metrics",
                   default="auroc,auprc,fpr_at_95_tpr,r_hat,repair_factuality_gain,repair_utility_delta",
                   type=_csv)
    return p


def _heavy(args, plan):
    # Authorized run: pure-CPU SECONDARY scoring over already-collected scored rows
    # (no model, no GPU). Computes AUROC / AUPRC / FPR@95TPR; RunInputError if the
    # rows are absent. The headline claim is R_hat/G9 (eval_gates), not these metrics.
    # Reachable only after the §1 authorization flip.
    from _runners import run_score_detection as _run  # lazy
    return _run(args, plan)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    plan = {
        "task": "score_detection",
        "family": args.family,
        "dataset": args.dataset,
        "interventions": args.interventions,
        "repair_transfer": args.repair_transfer,
        "metrics": list(args.metrics),
        "uses_model_or_gpu": False,
        "note": "detection metrics are SECONDARY; the headline claim is R_hat/G9 (eval_gates.py)",
        "kernel": "tracecausal.metrics + tracecausal.repair_transfer readouts",
    }
    return run_stage("score_detection", args, plan, _heavy)


if __name__ == "__main__":
    sys.exit(main())
