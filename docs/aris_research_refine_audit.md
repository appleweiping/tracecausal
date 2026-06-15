# ARIS Research-Refine Audit

## Scores

| Dimension | Score | Justification |
| --- | ---: | --- |
| Novelty | 10 | The pre-registered contribution requires intervention-usefulness, so detector-only overlap cannot satisfy the claim. |
| Feasibility | 9 | The first gate is bounded, budgeted, and server-gated; only live server preflight remains external. |
| Clarity | 10 | Primary/secondary outcomes, evidence tiers, stop rules, and forbidden claims are explicit. |
| Impact | 10 | A positive result would shift hallucination work from scoring outputs to actionable causal auditing. |
| Testability | 10 | Random intervention, detector-only, transfer, utility, and cost gates can falsify the claim. |

Average: 9.8/10.

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
