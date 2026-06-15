# Literature Boundary

## Must-Cite / Must-Beat Work

- TraceDet: D-LLM hallucination detection from denoising traces
  (`https://arxiv.org/abs/2510.01274`).
- Semantic entropy for hallucination detection
  (`https://pmc.ncbi.nlm.nih.gov/articles/PMC11186750/`).
- RACE-style answer/reasoning consistency for hallucination detection
  (`https://arxiv.org/pdf/2506.04832`).
- Output-signature hallucination/data-contamination detectors, represented by
  LOS-style work from the supplied paper bundle.
- Mechanistic and causal tracing work on VLM hallucination, represented by the
  supplied FCCT paper.

## Boundary Statement

TraceCausal does not claim that traces are new. The contribution must be that a
trace segment is useful only if intervening on it changes factuality under a
controlled protocol. Detector-only gains are insufficient.

## Reviewer-Risk Notes

- If the paper only reports AUROC, reviewers will classify it as incremental
  TraceDet/RACE/semantic-entropy work.
- If intervention harms utility, the claim must shift from mitigation to
  diagnosis.
- If only D-LLM traces work, remove cross-paradigm wording.

