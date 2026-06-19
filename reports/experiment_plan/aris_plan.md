# TraceCausal ARIS Experiment Plan Packet

Status: design-only local packet. No server command is authorized.

## First Server Gate Draft

Objective: validate whether targeted causal segment intervention beats random
segment intervention on one approved model/dataset pair.

## Preconditions

- ARIS experiment-plan review remains >=8 with no hard-rule violations.
- User approves exact command, model, dataset, output path, and stop condition.
- Server process/GPU/storage preflight is clean.

## Candidate Command Draft

```bash
# Draft only, do not run without approval
python scripts/extract_traces.py \
  --config configs/experiments/redesign_v4_ar_lead.yaml \
  --run-tier diagnostic \
  --output outputs/tracecausal_first_gate
```

## Stop Conditions

- Any baseline cannot be run under the same split/preprocessing policy.
- Targeted intervention margin over random is below 0.05.
- Utility drop exceeds 0.02.
- Trace extraction exceeds the approved storage or wall-clock budget.

