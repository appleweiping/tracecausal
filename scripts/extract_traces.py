#!/usr/bin/env python
"""extract_traces.py --- v5 trace-extraction entrypoint (THIN, DO-NOT-RUN wrapper).

Matches the run-packet commands in ``reports/run_packet.md`` sections 3.1 / 3.2(a).
Trace extraction is the only stage that loads a model and runs forwards, so it is
fully behind the authorization guard. By default this prints the resolved plan and
exits 0 (no model, no GPU). Heavy work runs ONLY when both
``server.authorized: true`` (in --config) AND ``--i-have-authorization`` are set.

The extracted ``trace_manifest.json`` validates against
``schemas/trace_manifest.schema.json`` via ``tracecausal.validate_trace_manifest``
(imported lazily inside the guarded branch).
"""

from __future__ import annotations

import sys

from _runpacket_common import base_parser, run_stage  # noqa: E402  (pure-python)


def build_parser():
    p = base_parser("v5 trace extraction (do-not-run wrapper)")
    p.add_argument("--family", help="family id, e.g. ar_lead_qwen / ar_lead_llama")
    p.add_argument("--model-revision", help="pinned weight revision (DATA_NEEDED until lock)")
    p.add_argument("--dataset", help="triviaqa / hotpotqa")
    p.add_argument("--split-hash", help="frozen 3-way V_sel/V_inf/test split hash")
    p.add_argument("--prompt-template-hash")
    p.add_argument("--taxonomy-hash", help="frozen G3 class partition hash (A6)")
    p.add_argument("--seed", type=int)
    p.add_argument("--run-tier", default="paper_candidate")
    p.add_argument("--device", default="cpu", help="e.g. cuda:0 (ignored in dry-run)")
    # section 3.2(a) timing-calibration sub-mode
    p.add_argument("--calibrate-cfwd", action="store_true",
                   help="c_fwd timing calibration on V_sel (no analysis)")
    p.add_argument("--split", default=None, help="v_sel / v_inf / test")
    return p


def _heavy(args, plan):
    # Authorized run: the extraction loop genuinely needs the user's pinned model on
    # ${GPU} (pure Python cannot fabricate a forward), so it binds the model through
    # the ForwardProvider seam and owns the real orchestration (argument resolution,
    # output dir, the STATUS row). If no provider is configured the runner raises an
    # actionable ForwardProviderUnavailable (set TRACECAUSAL_FORWARD_PROVIDER), NOT a
    # blanket NotImplementedError. Reachable only after the §1 authorization flip.
    from _runners import run_extract_traces as _run  # lazy
    return _run(args, plan)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    plan = {
        "task": "calibrate_cfwd" if args.calibrate_cfwd else "extract_traces",
        "family": args.family,
        "model_revision": args.model_revision,
        "dataset": args.dataset,
        "split": args.split,
        "split_hash": args.split_hash,
        "prompt_template_hash": args.prompt_template_hash,
        "taxonomy_hash": args.taxonomy_hash,
        "seed": args.seed,
        "run_tier": args.run_tier,
        "device": args.device,
        "emits": "trace_manifest.json (schemas/trace_manifest.schema.json, server_authorized:false const)",
        "kernel": "tracecausal.schemas.validate_trace_manifest",
    }
    return run_stage("extract_traces", args, plan, _heavy)


if __name__ == "__main__":
    sys.exit(main())
