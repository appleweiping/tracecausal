# Adversarial Review Round 1 Response

## Reviewer Objection 1

TraceCausal may be detector work relabeled as causality.

Response: the primary gate is intervention delta over random segment
intervention, not AUROC. Detector-only improvements cannot satisfy the main
claim.

## Reviewer Objection 2

Cross-paradigm unification may be too broad.

Response: the first gate is deliberately narrow; cross-paradigm claims remain
pending until held-out taxonomy retention reaches 0.80.

## Reviewer Objection 3

Trace extraction may be too expensive.

Response: the compute budget caps first-gate raw traces and requires compact
artifact sync. Cost is a reported metric, not hidden overhead.

## Post-Review Hardening

The adversarial review rejected the earlier 9.8 claim because validators only
checked marker text and the closest-work boundary was too generic. The current
package replaces scorecard-string validation with checks for concrete
provenance and kill gates:

- baseline registry now names Semantic Entropy, RACE, TraceDet, TDGNet,
  SelfCheckGPT, and INSIDE-style comparators with paper URL, implementation
  source, tuning grid, input access, license check, and fairness policy;
- `configs/experiments/first_gate.yaml` fixes negative controls,
  evaluator-leakage gate, causal margin gate, utility gate, and server
  prohibition;
- `schemas/trace_manifest.schema.json` requires `split_hash`,
  `trace_segments`, `candidate_segments`, and `server_authorized: false`;
- `docs/intervention_protocol.md` records invalid-intervention handling and the
  required negative-control IDs.
