#!/usr/bin/env python
"""eval_gates.py --- nuisance estimation, gate evaluation, queue reconcile (THIN).

DO-NOT-RUN wrapper covering three sub-modes from ``reports/run_packet.md``:

  * ``--estimate-nuisance``  (section 3.3a): V_inf-only nuisance estimators
                            (sigma_u, kappa, kappa^repair, m_pool); wraps
                            ``tracecausal.nuisance.{estimate_sigma_u,estimate_kappa,
                            pool_inflation}``.
  * ``--g9 --g9-nov``        (section 3.9): screening G1-G8 + headline G9/G9-NOV
                            with the Holm/SI family; wraps
                            ``tracecausal.ciu.{ciu_gate,g9_repair_gate,g9_novelty_gate,
                            validate_ciu_record}`` and
                            ``tracecausal.selective_inference.{holm_alpha,choose_si_path}``.
  * ``--reconcile-queue``    (section 6.2): pure bookkeeping over STATUS.json files;
                            marks done/failed/partial and emits the remaining
                            pending set in v5 dependency order. This mode is ALWAYS
                            allowed (it loads nothing and only reads checkpoints).

All gate/nuisance numbers are ``DATA_NEEDED`` until an authorized run. Default
(no authorization) = print resolved plan and exit 0; no model, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _runpacket_common import (
    ROOT, base_parser, read_status, run_stage,
)
from _reconcile_queue import load_manifest_jobs, reconcile

DEFAULT_MANIFEST = "experiments/queue_manifest.yaml"


def _csv(text):
    return tuple(x.strip() for x in text.split(",") if x.strip())


def build_parser():
    p = base_parser("nuisance + gates + queue reconcile (do-not-run wrapper)")
    # modes
    p.add_argument("--estimate-nuisance", action="store_true")
    p.add_argument("--reconcile-queue", action="store_true")
    p.add_argument("--g9", action="store_true")
    p.add_argument("--g9-nov", action="store_true")
    p.add_argument("--require-v5", action="store_true")
    # common
    p.add_argument("--family")
    p.add_argument("--dataset")
    p.add_argument("--split", default=None)
    # nuisance
    p.add_argument("--n-val-sigma", type=int, default=200)
    p.add_argument("--n-val-kappa", type=int, default=300)
    p.add_argument("--proximity-pool-min", type=int, default=8)
    p.add_argument("--repair-eval-hash")
    p.add_argument("--traces")
    p.add_argument("--binning")
    # gates
    p.add_argument("--interventions")
    p.add_argument("--repair-transfer")
    p.add_argument("--adversarial-oracle")
    p.add_argument("--detection")
    p.add_argument("--nuisance")
    p.add_argument("--operator-freeze")
    p.add_argument("--g1-on", default="u_deflated")
    p.add_argument("--g7-on", default="leakage_bound_upper_ci")
    p.add_argument("--si-path-rule", default="SI1_if_floors_met_else_SI2")
    p.add_argument("--holm-family",
                   default="g1,g2,g5prime,g6,g7,g8,g9_cells,g9_nov_B1,g9_nov_B2,g9_nov_B3",
                   type=_csv)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--permutations", type=int, default=10000)
    # reconcile
    p.add_argument("--manifest", default=DEFAULT_MANIFEST,
                   help="resumable queue manifest (default: %(default)s)")
    p.add_argument("--out", help="run output root (${OUT}/${RUN_ID}) to scan for STATUS.json")
    p.add_argument("--satisfied-external", type=_csv, default=(),
                   help="comma-separated external precondition tokens to treat as met "
                        "(e.g. server_authorized_true,hashes_pinned,preflight_pass); "
                        "default-deny so a fresh resume reports them as blocking")
    return p


def _reconcile(args) -> int:
    """Rebuild the pending job set from the MANIFEST reconciled against STATUS.json.

    Pure bookkeeping (loads no model/GPU). Parses
    ``experiments/queue_manifest.yaml`` (``--manifest``), reads each job's on-disk
    ``STATUS.json`` checkpoint under ``--out``, and emits the remaining pending set in
    v5 dependency order -- the resumability contract in ``reports/run_packet.md``
    section 6.2. Previously this ignored the manifest and returned an empty pending
    set when ``--out`` was absent; it now reconstructs the queue from the manifest
    even before any output dir exists (every job pending), honouring ``depends_on``.
    """
    out_root = Path(args.out).resolve() if args.out else None

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    jobs, pipeline_order = load_manifest_jobs(manifest_path)

    result = reconcile(
        jobs,
        pipeline_order,
        out_root=out_root,
        status_reader=read_status,
        satisfied_external=getattr(args, "satisfied_external", ()) or (),
    )

    payload = {
        "stage": "eval_gates_reconcile",
        "mode": "RECONCILE_QUEUE",
        "manifest": str(manifest_path),
        "manifest_parsed": bool(jobs),
        "out": str(out_root) if out_root else None,
        "dependency_order": list(pipeline_order),
        "loads_model_or_gpu": False,
        **result,
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _heavy(args, plan):
    # Authorized run: both sub-modes are pure-CPU over on-disk artifacts (no model, no
    # GPU). estimate_nuisance wires the V_inf nuisance estimators; eval_gates ENFORCES
    # the v5 acceptance checks executably -- G9 (Holm/SI-corrected two-way-cluster CI
    # lower bound > m_R, sign-flip diagnostic p < alpha_1', bounded repair utility,
    # per-class positivity, no self leakage), m_R attenuation (Eq. m-R), and G9-NOV
    # (PROPOSED out-transfers max(B1,B2,B3) on the simultaneous lower CI). RunInputError
    # if the upstream artifacts are absent. Reachable only after the §1 flip.
    if plan["task"] == "estimate_nuisance":
        from _runners import run_estimate_nuisance as _run
        return _run(args, plan)
    from _runners import run_eval_gates as _run
    return _run(args, plan)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Reconcile is bookkeeping only and never gated: it loads nothing.
    if args.reconcile_queue:
        return _reconcile(args)

    if args.estimate_nuisance:
        task = "estimate_nuisance"
        plan = {
            "task": task,
            "split": args.split,
            "n_val_sigma": args.n_val_sigma,
            "n_val_kappa": args.n_val_kappa,
            "proximity_pool_min": args.proximity_pool_min,
            "repair_eval_hash": args.repair_eval_hash,
            "uses_model_or_gpu": False,
            "kernel": "tracecausal.nuisance.{estimate_sigma_u,estimate_kappa,pool_inflation}",
            "emits": "nuisance/*.json (sigma_u, kappa, kappa^repair, m_pool) -- all DATA_NEEDED",
        }
    else:
        task = "eval_gates"
        plan = {
            "task": task,
            "family": args.family,
            "dataset": args.dataset,
            "g9": args.g9,
            "g9_nov": args.g9_nov,
            "require_v5": args.require_v5,
            "g1_on": args.g1_on,
            "g7_on": args.g7_on,
            "si_path_rule": args.si_path_rule,
            "holm_family": list(args.holm_family),
            "bootstrap": args.bootstrap,
            "permutations": args.permutations,
            "uses_model_or_gpu": False,
            "kernel": "tracecausal.ciu.{ciu_gate,g9_repair_gate,g9_novelty_gate,"
                      "validate_ciu_record} + selective_inference.{holm_alpha,choose_si_path}",
            "emits": "gates/* verdicts + SI family (m', K_bin, K_op, alpha_1') -- all DATA_NEEDED",
        }
    return run_stage(task, args, plan, _heavy)


if __name__ == "__main__":
    sys.exit(main())
