# TraceCausal

TraceCausal is a research-grade project on causal propagation auditing for
LLM hallucinations. It studies whether hallucinations have identifiable
process-level causal sources across autoregressive, reasoning-trace, and
diffusion language models, and whether targeted interventions on those sources
reduce hallucination without damaging answer quality.

This repository is intentionally not a prompt-engineering demo and not a
single-detector benchmark. The defended claim is narrower:

> Hallucinations can be localized to causal process segments whose interventions
> produce measurable factuality gains beyond output-only uncertainty detectors.

## Research Spine

1. Build a common trace schema for autoregressive logits, reasoning steps, and
   diffusion denoising states.
2. Estimate causal contribution of trace segments through counterfactual
   masking, patching, and controlled replay.
3. Compare against output-only and trajectory-only detectors under the same
   datasets, model families, and statistical gates.
4. Validate that the discovered segments are intervention-useful, not merely
   predictive correlates.

## Non-Toy Standard

Paper-facing results require all of the following:

- real hallucination datasets and model traces, not synthetic placeholders;
- at least three baseline families, including current trajectory detectors;
- identical train/valid/test splits and preprocessing across methods;
- 20 random seeds or bootstrap replicates for paper claims;
- paired statistical tests, effect sizes, and 95 percent confidence intervals;
- provenance for model, prompt, dataset, trace extraction, and intervention;
- ARIS research-refine, experiment-plan, experiment-audit, citation-audit, and
  claim-audit gates before submission claims.

Local smoke tests in this repository validate contracts only. They are never
paper evidence.

## Project Layout

```text
.
├── AGENTS.md
├── README.md
├── configs/
├── data/
├── docs/
├── paper/
├── reports/
├── scripts/
├── src/tracecausal/
└── tests/
```

## Current Status

The Stage-1 design is **frozen** and its Phase-2 identification machinery is
**implemented and frozen** (uncommitted): the locked method/theory/analysis plan
is `docs/redesign/REDESIGN_v4.md`; the CIU estimand contract and pure-Python
identification helpers live in `src/tracecausal/` (`ciu.py`, `nuisance.py`,
`oracle_gen.py`, `nullpool.py`) with the passing unit harness
`tests/test_ciu_nulldata.py` (no model, no GPU); the AR-lead freeze is
`configs/experiments/redesign_v4_ar_lead.yaml`; and a Stage-1 paper skeleton with
`DATA_NEEDED` result placeholders (no fabricated numbers) is at `paper/main.tex`
(+ `paper/references.bib`, not yet compiled). See also `docs/active_todo.md`,
`docs/milestones.md`, `docs/paper_claims_status.md`, and
`docs/definition_of_done.md`.

Server experiments are deliberately **not** run from this local setup; every
paper-facing claim remains `pending` and no empirical number exists. The next real
stage is ARIS experiment-plan review of the formal trace extraction and
intervention protocol, followed by server-side trace extraction only after the plan
passes the documented gates and the run is explicitly authorized
(`server.authorized: false` until then).
