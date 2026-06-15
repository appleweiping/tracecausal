# Milestones

## M0: Local Initialization

Status: complete.

Evidence:

- GitHub repo: `https://github.com/appleweiping/tracecausal`
- Local validator and tests pass.
- Server runbook blocks unauthorized execution.

## M1: Design Review Packet

Status: pending.

Exit criteria:

- trace schema draft exists;
- dataset/model candidates and resource budget are listed;
- baselines have source candidates and fairness notes;
- ARIS experiment-plan audit reaches proceed or targeted iterate with no fatal
  blocker.

## M2: Early Causal Sanity Gate

Status: blocked by server availability and M1.

Exit criteria:

- targeted interventions beat random segment interventions by the numeric gate;
- utility drop is within threshold;
- artifact package is synced locally.

## M3: Formal Evidence Build

Status: pending.

Exit criteria:

- main detection and intervention tables;
- ablations and transfer analysis;
- statistical audit;
- paper claim audit with no unsupported central claim.

