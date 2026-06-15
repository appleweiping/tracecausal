# Pre-Registration

## Scope

This pre-registration covers the first server gate only. It is not a claim that
results exist.

## Frozen Primary Hypothesis

H1: targeted causal trace-segment interventions produce a factuality improvement
at least 0.05 absolute points larger than random segment interventions while
keeping utility drop <= 0.02.

## Primary Outcome

`targeted_delta - random_delta`, measured on matched examples with the same
model, dataset split, prompt, trace extractor, and evaluator.

## Secondary Outcomes

- AUROC/AUPRC for hallucination detection.
- Segment localization F1 where perturbation-derived labels exist.
- Held-out taxonomy retention.
- Trace extraction and intervention cost.

## Analysis Lock

Validation split may tune segment thresholds. Test split must be evaluated once.
Any failed gate downgrades the claim before additional runs.

