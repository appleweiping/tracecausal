# Risks And Blockers

## Active Blockers

| Blocker | Severity | Evidence | Mitigation |
| --- | --- | --- | --- |
| `D:\Research\p` is an empty locked directory. | low | Windows reports the directory is being used by another process. | Treat `D:\Research\tracecausal` as the formal project; delete `p` after the workspace releases it. |
| Server unavailable for GPU/model trace extraction. | high | User instruction says server is temporarily unavailable. | Complete local docs/configs/tests; keep server commands as TODO only. |
| TraceDet-like D-LLM detector novelty risk. | high | Public TraceDet-style work already uses denoising traces for detection. | Center causal intervention, not detection alone. |

## Scientific Risks

- Causal interventions may not beat random interventions.
- Cross-paradigm trace schema may be too abstract.
- Trace extraction may be too expensive for practical auditing.

## Stop Rules

- If intervention-usefulness fails the early gate twice, pivot away from causal
  claims.
- If only one generation paradigm works, narrow the claim and avoid cross-paradigm
  wording.

