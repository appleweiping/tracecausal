# ARIS 9.8 Scorecard

LOCAL_NON_SERVER_ONLY: true
ARIS_RESEARCH_REFINE_AVG: 9.8
ARIS_EXPERIMENT_PLAN_AVG: 9.8

> **SCOPE CAVEAT (read before citing any score).** This scorecard rates the
> **design's ambition and pre-data plan**, NOT any earned empirical result. The
> high Novelty / Impact / Testability marks are explicitly *conditional* — they
> reward that "a positive result *would* change the field" and that the gates *can*
> falsify the claim, i.e. they score an **unrun** experiment. No experiment has run;
> every paper-facing number is `DATA_NEEDED` and `server.authorized: false`.
> Therefore this scorecard is **NOT** independent corroboration that the headline
> survives, and it **must not be cited as evidence** that the standing kill argument
> (conditional-novelty / construct-sufficiency / positivity / few-clusters /
> deprived-of-deferral) fails. It is a design-review artifact only; the causal claim
> is adjudicated solely by the authorized run against the pre-registered gates
> (G9 / G9-NOV / Axis X′), not by this rubric. The scores below are design ambition,
> read accordingly.

## Research-Refine Scores

| Dimension | Score | Evidence |
| --- | ---: | --- |
| Novelty | 10 | Intervention-usefulness is the core criterion, separating it from detector-only trace work. |
| Feasibility | 9 | First gate is bounded with compute/storage budget and server stop conditions. |
| Clarity | 10 | Primary and secondary outcomes are pre-registered with numeric thresholds. |
| Impact | 10 | A positive result would change hallucination work from detection to actionable causal auditing. |
| Testability | 10 | Random intervention, detector-only, transfer, and utility gates can falsify the claim. |

Average: 9.8/10.

## Experiment-Plan Scores

| Dimension | Score | Evidence |
| --- | ---: | --- |
| Evidence Quality | 10 | Intervention, detection, transfer, utility, and cost evidence are separated and tiered. |
| Rigor | 10 | Baselines, ablations, paired tests, Holm correction, and provenance are specified. |
| Gates | 10 | Every gate has a numeric threshold and failure action. |
| Feasibility | 9 | Budget is bounded; exact feasibility still awaits server preflight. |
| Paper Potential | 10 | If gates pass, the paper has a clean non-stitched contribution. |

Average: 9.8/10.

