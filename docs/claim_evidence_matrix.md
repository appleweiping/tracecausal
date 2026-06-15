# Claim Evidence Matrix

| Claim | Status | Required Artifact | Evidence Tier |
| --- | --- | --- | --- |
| TraceCausal defines a common trace schema across AR, reasoning, and D-LLM generation. | pending | `reports/schema_validation/*.json` | diagnostic |
| Causal segment scores improve hallucination detection over output-only baselines. | pending | `reports/main_tables/detection.csv` plus paired tests | paper_result |
| Causal segments are intervention-useful, not merely predictive. | pending | `reports/intervention/intervention_delta.csv` | paper_result |
| Segment taxonomy transfers across datasets and model families. | pending | `reports/transfer/heldout_taxonomy.csv` | official |
| TraceCausal is efficient enough for practical auditing. | pending | `reports/efficiency/cost_latency.json` | official |

No row may be rewritten as completed until the artifact exists and passes ARIS
experiment-audit.

