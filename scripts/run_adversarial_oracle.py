#!/usr/bin/env python
"""run_adversarial_oracle.py --- Axis X' oracle sweep (THIN, DO-NOT-RUN wrapper).

Matches ``reports/run_packet.md`` section 3.8. PURE-CPU fixture stage: no lead
data, no model forwards. Wraps
``tracecausal.adversarial_oracle.axis_x_confounded`` (the xi-sweep over the
detectable + blind regimes) plus ``negative_control_collinear`` (NC-1) and
``source_swap`` (NC-2), reproducing the v4 clean oracle at xi=0 and checking the
registered P5 expectation.

Even though no GPU is used, the authorization guard still applies before writing
to a run output tree. Default = print resolved plan and exit 0.
"""

from __future__ import annotations

import sys

from _runpacket_common import base_parser, run_stage


def _csv(text):
    return tuple(x.strip() for x in text.split(",") if x.strip())


def _csv_floats(text):
    return tuple(float(x) for x in text.split(",") if x.strip())


def build_parser():
    p = base_parser("adversarial oracle Axis X' (do-not-run wrapper)")
    p.add_argument("--axis", default="x_prime")
    p.add_argument("--regimes", default="detectable,blind", type=_csv)
    p.add_argument("--xi-grid", default="0.0,0.25,0.5,0.75,1.0", type=_csv_floats)
    p.add_argument("--negative-controls",
                   default="NC-1_collinear_confounder,NC-2_source_swap", type=_csv)
    return p


def _heavy(args, plan):
    # Fully-implemented authorized run: the Axis X' structural fixtures are frozen
    # pure-Python (no lead data, no model, no GPU), so the authorized sweep executes
    # them directly and records the registered P5 readout + the R4 soundness verdict.
    # Reachable only after the §1 authorization flip (server.authorized + the flag).
    from _runners import run_adversarial_oracle as _run  # lazy
    return _run(args, plan)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    plan = {
        "task": "run_adversarial_oracle",
        "axis": args.axis,
        "regimes": list(args.regimes),
        "xi_grid": list(args.xi_grid),
        "negative_controls": list(args.negative_controls),
        "uses_model_or_gpu": False,
        "emits": "adversarial_oracle/axis_x_prime.json (per (xi,regime) P5 readout)",
        "kernel": "tracecausal.adversarial_oracle.{axis_x_confounded,"
                  "negative_control_collinear,source_swap}",
    }
    return run_stage("run_adversarial_oracle", args, plan, _heavy)


if __name__ == "__main__":
    sys.exit(main())
