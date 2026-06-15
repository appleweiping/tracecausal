# ARIS Research-Refine Audit

## Scores

| Dimension | Score | Justification |
| --- | ---: | --- |
| Novelty | 8 | The claim is not another detector: segments must be intervention-useful under counterfactual replay or patching. |
| Feasibility | 8 | The first formal gate is bounded to one model/dataset pair and expands only after the intervention gate passes. |
| Clarity | 8 | Success is defined by detection, localization, intervention delta, utility drop, and transfer retention. |
| Impact | 8 | A causal audit would matter for reliable deployment and mechanistic understanding. |
| Testability | 9 | The core hypothesis is killed if targeted interventions fail to beat random or detector-only segments. |

Average: 8.2/10.

## Kill Argument

The strongest rejection argument is that TraceCausal may only relabel existing
trajectory signals with causal language. If counterfactual interventions do not
produce stronger factuality gains than random, entropy-selected, or detector-only
segments, then the project is an overclaimed detector rather than a causal
contribution.

## Differentiation

| Closest Approach | Overlap | Differentiation | Strength |
| --- | --- | --- | --- |
| Output-signature hallucination detectors | Use model output distributions. | TraceCausal asks which process segments change factuality under intervention. | Strong |
| Reasoning consistency detectors | Use reasoning traces. | TraceCausal localizes causal bottlenecks and tests interventions, not only consistency. | Medium |
| D-LLM trace detectors | Use denoising trajectories. | TraceCausal treats D-LLM traces as one paradigm in a cross-decoding causal audit. | Medium |

## Verdict

VERDICT: PROCEED
CONFIDENCE: Medium
BLOCKING ISSUE: None for local non-server initialization; server work remains gated by experiment-plan approval.
NEXT ACTION: Draft trace schema and submit the exact first server command for approval.
