# Data And Evaluation Plan

## Data Families

- Open-domain factual QA for output hallucination.
- Multi-hop QA for reasoning trace failure.
- Reasoning-consistency datasets for chain-level hallucination.
- Diffusion-LM trace datasets matching public D-LLM detector protocols.

## Required Manifests

Each dataset must have:

- license and source URL;
- raw and processed hash;
- split definition;
- contamination/leakage check;
- example count and filtering policy.

## Evaluation Tables

1. Detection: AUROC, AUPRC, FPR@95TPR, detector calibration.
2. Localization: segment precision/recall/F1 where labels or perturbation labels
   exist.
3. Intervention: factuality delta, accuracy delta, utility delta.
4. Transfer: held-out model/dataset retention.
5. Efficiency: trace extraction and intervention cost.

## Fairness Policy

All methods use the same examples, prompts, decoding parameters where
applicable, split, and evaluator. If a baseline cannot access traces, it is
listed as output-only rather than silently advantaged or disadvantaged.

