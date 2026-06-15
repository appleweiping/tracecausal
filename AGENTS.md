# TraceCausal Agent Operating Contract

This repository is a top-conference research artifact. Treat it as a
publishable system with strict claim discipline, not as a prototype.

## Start Here

Before any nontrivial change, read:

1. `README.md`
2. `docs/research_brief.md`
3. `docs/experiment_protocol.md`
4. `docs/baseline_contract.md`
5. `docs/claim_evidence_matrix.md`
6. `docs/paper_claims_status.md`
7. `docs/research_plan.md`
8. `docs/active_todo.md`
9. `docs/milestones.md`
10. `docs/data_and_evaluation_plan.md`
11. `docs/motivation_ablation_hparam_plan.md`
12. `docs/risks_and_blockers.md`
13. `docs/definition_of_done.md`
14. `docs/aris_research_refine_audit.md`
15. `docs/server_runbook.md`

Use `agentmemory` for live collaboration recall and durable lessons. Follow
`D:\devtools\AGENT-MEMORY-PROTOCOL.md`: project slug is `tracecausal`, concepts
must include exactly one `agent:<who>` tag, and memory is an index to committed
artifacts rather than the only source of truth.

## Hard Rules

- No toy experiments may be used as evidence.
- No paper claim may appear without an artifact path or a `pending` label.
- Do not copy raw source zip contents, agentmemory exports, or external project
  norms into this repository.
- Do not claim SOTA unless all declared baselines are current, fair, and
  audited.
- Do not strengthen the claim from "causal process segments are
  intervention-useful" to "all hallucinations are causally explained."
- Before launching any server experiment, pass ARIS experiment-plan review with
  numeric gates, compute budget, baselines, seeds, and failure actions.

## Collaboration

Complex work must use at least three perspectives when tools are available:

- implementation or harness builder;
- literature/protocol scout;
- hostile reviewer focused on novelty, fairness, and overclaim risk.

Reviewer objections veto paper claims until addressed, downgraded, or recorded
as explicit limitations.

## Evidence Labels

- `paper_result`: >=20 seeds/replicates, significance tests, effect sizes, fair
  comparisons, complete provenance.
- `official`: >=10 seeds/replicates, fair comparison, enough for appendix or
  caveated paper text.
- `diagnostic`: >=5 seeds/replicates, useful internally or in appendix only.
- `pilot`: contract/smoke result only; never paper evidence.
