# Server Runbook

No server experiments are authorized yet.

## Before Any Server Run

1. Read `docs/experiment_protocol.md` and update the exact config.
2. Run ARIS experiment-plan review and record the verdict.
3. Estimate GPU hours, storage, and wall time with a 30 percent buffer.
4. Confirm no active server process will be overwritten.
5. Record the command, output directory, log path, and stop condition.

## Server Boundary

Local work prepares code, configs, validators, and paper scaffolding. GPU-scale
trace extraction and model interventions run only after explicit approval.

## Artifact Sync

Copy back only lightweight evidence:

- resolved config;
- environment and git info;
- metrics tables;
- provenance JSON;
- audit reports;
- compact samples for qualitative figures.

Do not commit raw model traces, checkpoints, credentials, or large datasets.

