# ARIS Research-Refine Audit

## Scores

| Dimension | Score | Justification |
| --- | ---: | --- |
| Novelty | 8 | The causal-intervention requirement differentiates it from detector-only trace work. |
| Feasibility | 6 | Requires trace access across model families and may need substantial server time. |
| Clarity | 8 | Success is measurable through detection, localization, and intervention deltas. |
| Impact | 8 | A causal audit would matter for reliable deployment and mechanistic understanding. |
| Testability | 8 | The core hypothesis can be killed by random interventions matching targeted interventions. |

Average: 7.6/10.

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

VERDICT: PROCEED WITH DESIGN ITERATION
CONFIDENCE: Medium
BLOCKING ISSUE: Formal intervention protocol must be specified before any server run.
NEXT ACTION: Write and review the experiment plan until it passes ARIS hard rules.

