# ARIS Experiment-Plan Review

## Baseline Completeness

- Random segment baseline: included.
- Current SOTA-style baseline: diffusion trace detector for D-LLM, reasoning
  consistency detector for reasoning traces, semantic entropy/output signature
  for output-only.
- Simple strong baseline: output entropy and semantic entropy.
- Ablations: predictive-only segment, no intervention score, one-paradigm-only,
  random intervention, no utility gate.
- Baseline count: 6 families, excluding ablations.

## Numeric Decision Gates

| Gate | Metric | Threshold | Failure Action | Verdict |
| --- | --- | --- | --- | --- |
| G1 causal sanity | targeted delta - random delta | >= 0.05 | stop causal claim, redesign intervention | pass in plan |
| G2 utility harm | utility drop | <= 0.02 | downgrade mitigation wording | pass in plan |
| G3 transfer | held-out taxonomy retention | >= 0.80 | narrow to in-domain diagnosis | pass in plan |
| G4 paper evidence | seeds/replicates | >= 20 | label as diagnostic only | pass in plan |

## Compute Feasibility

First server gate is deliberately bounded:

- one model family and one dataset family;
- 20 bootstrap replicates for paper-bound tables after the early gate;
- 30 percent buffer required in the server command packet;
- raw traces stay server-side, only compact artifacts sync back.

## ARIS Scores

| Dimension | Score | Justification |
| --- | ---: | --- |
| Evidence Quality | 8 | Detection and intervention evidence are separated, with paired tests required. |
| Rigor | 8 | Strong baselines, identical splits, and ablations are specified. |
| Gates | 9 | Numeric early kill gates directly test the core causal claim. |
| Feasibility | 8 | The first run is bounded and expands only after a cheap causal gate. |
| Paper Potential | 8 | If intervention-usefulness holds, the story is stronger than detector-only work. |

VERDICT: PROCEED FOR LOCAL DESIGN; SERVER RUN REQUIRES USER APPROVAL
CONFIDENCE: Medium
HARD RULE VIOLATIONS: None

