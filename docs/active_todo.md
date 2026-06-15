# Active TODO

## Current Phase

Phase 0: local non-server initialization and design package.

## Completed

- Project renamed conceptually from temporary `p` to `tracecausal`.
- Private GitHub repository created and pushed.
- Initial README, AGENTS, ARIS audit, protocol, baseline contract, claim matrix,
  server runbook, configs, validator, and tests created.
- Seed idea notes reviewed and abstracted into a causal trace direction.

## Next Local Tasks

1. Draft exact trace schema JSON for AR, reasoning, and D-LLM traces.
2. Convert `docs/experiment_protocol.md` into an ARIS experiment-plan review
   packet with resource estimates.
3. Add citation seed list with verified BibTeX only after citation audit starts.
4. Write small schema validators for trace records before any server extraction.

## Server-Stage TODO

1. Run trace extraction on one approved model/dataset pair.
2. Run early causal sanity gate: targeted segment intervention vs random segment
   intervention.
3. Sync lightweight artifacts back to `reports/` and run experiment audit.

## Next Concrete Command

```powershell
python scripts\validate_project.py; python -m pytest -q
```

## Stop Condition

Do not launch server work until the ARIS experiment-plan audit has no hard-rule
violations and the user approves the exact command.

