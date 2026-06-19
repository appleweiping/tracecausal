#!/usr/bin/env python
"""select_binning.py --- SI-1 selection split + binning-as-code (THIN, DO-NOT-RUN).

Matches ``reports/run_packet.md`` section 3.2(b). This is a PURE-CPU,
selection-as-code stage: it wraps
``tracecausal.selective_inference.validate_selection_split`` and
``tracecausal.binning_selection.select_binning`` over the V_sel distances and
records the resulting ``SelectionEvent`` (rungs walked, k_bin) that sizes the
``K_bin`` Holm fold.

No model or GPU is involved at all; but the same authorization guard applies so
that nothing is written to a run output tree until an authorized run is intended.
By default it prints the resolved plan and exits 0.
"""

from __future__ import annotations

import sys

from _runpacket_common import base_parser, run_stage


def _csv_ints(text):
    return tuple(int(x) for x in text.split(",") if x.strip())


def _csv_floats(text):
    return tuple(float(x) for x in text.split(",") if x.strip())


def build_parser():
    p = base_parser("SI-1 selection split + frozen binning (do-not-run wrapper)")
    p.add_argument("--family")
    p.add_argument("--dataset")
    p.add_argument("--v-sel", help="path to the V_sel traces root")
    p.add_argument("--delta-pos-ladder", default="1,2,4,8,16", type=_csv_ints)
    p.add_argument("--displaced-mass-edges", default="0.0,0.05,0.1,0.2,0.4,1.0",
                   type=_csv_floats)
    p.add_argument("--pool-floor", default=8, type=int)
    return p


def _heavy(args, plan):
    # Authorized run: read the actual V_sel distances off disk and call the frozen
    # binning-as-code kernel; emit the Binning + SelectionEvent (k_bin). Pure CPU; it
    # synthesizes no numbers and raises RunInputError if the V_sel artifact is absent.
    # Reachable only after the §1 authorization flip.
    from _runners import run_select_binning as _run  # lazy
    return _run(args, plan)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    plan = {
        "task": "select_binning",
        "family": args.family,
        "dataset": args.dataset,
        "v_sel": args.v_sel,
        "delta_pos_ladder": list(args.delta_pos_ladder),
        "displaced_mass_edges": list(args.displaced_mass_edges),
        "pool_floor": args.pool_floor,
        "emits": "Binning + SelectionEvent (rungs_walked, k_bin)",
        "kernel": "tracecausal.binning_selection.select_binning + "
                  "tracecausal.selective_inference.validate_selection_split",
        "gpu": False,
    }
    return run_stage("select_binning", args, plan, _heavy)


if __name__ == "__main__":
    sys.exit(main())
