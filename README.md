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

The current frozen design is **REDESIGN_v5** (`docs/redesign/REDESIGN_v5.md`,
status `design_frozen_stage1_RR`): the **cross-example repair-transfer
certification** statistic `R_hat` (gate **G9**) with the baseline-conditional
novelty gate **G9-NOV**, valid dependent-pair inference (two-way source×target
cluster bootstrap + class-block-permutation diagnostic + Hájek cross-check), the
selective-inference correction (SI-1 split / SI-2 Bonferroni), and the adversarial
oracle **Axis X′**. v5 is an **additive** layer on the preserved v4 core
(`docs/redesign/REDESIGN_v4.md`): the in-place necessity statistic `U_hat` is
retained as a **screening** filter (no longer the headline), along with the CIU /
matched-null / proper-scoring / G1–G8 machinery.

Both layers are **implemented and frozen** as pure-Python (no model, no GPU). The
kernels live in `src/tracecausal/`: the v4 core (`ciu.py`, `nuisance.py`,
`oracle_gen.py`, `nullpool.py`, `interventions.py`) plus the v5 surfaces
(`repair_transfer.py`, `repair_ops.py`, `binning_selection.py`,
`selective_inference.py`, `adversarial_oracle.py`). The unit harness
`tests/test_ciu_nulldata.py` and the rest of `tests/` are green (no model, no GPU).
The frozen lead plans are `configs/experiments/redesign_v5_ar_lead.yaml` (current)
and `configs/experiments/redesign_v4_ar_lead.yaml` (preserved core); both keep
`server.authorized: false`.

The Stage-1 Registered-Report paper is `paper/main.tex` (+ `paper/references.bib`),
with `DATA_NEEDED` result placeholders (no fabricated numbers); it **compiles**
reproducibly (see `paper/README.md` for the build + TeX-environment note). The
run-later execution packet is `reports/run_packet.md` with the resumable queue
manifest `experiments/queue_manifest.yaml`; the thin CLI entrypoints under
`scripts/` (`extract_traces.py`, `select_binning.py`, `run_intervention.py`,
`run_repair_transfer.py`, `run_adversarial_oracle.py`, `score_detection.py`,
`eval_gates.py`) **default to dry-run** and only execute real work when BOTH
`server.authorized: true` (in `--config`) AND `--i-have-authorization` are set.
See also `docs/active_todo.md`, `docs/milestones.md`,
`docs/paper_claims_status.md`, and `docs/definition_of_done.md`.

Server experiments are deliberately **not** run from this local setup; every
paper-facing claim remains `pending` and no empirical number exists. The next real
stage is the **v5-aware** ARIS experiment-plan re-review (covering G9 / G9-NOV /
Axis X′), followed by server-side trace extraction only after the plan passes the
documented gates and the run is explicitly authorized (`server.authorized: false`
until then).
