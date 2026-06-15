# Intervention Protocol

## Segment Definition

- Autoregressive: contiguous token/logit windows with recorded probability
  signatures and hidden-state references when available.
- Reasoning trace: explicit reasoning-step spans aligned to final answer tokens.
- Diffusion LM: denoising-step subtraces aligned to token positions.

## Intervention Types

- `patch`: replace candidate segment state with a factual-reference or neutral
  state when model internals permit.
- `mask`: suppress segment contribution while keeping prompt and evaluator
  unchanged.
- `replay`: regenerate from a checkpoint before the candidate segment.
- `no_op`: negative control that executes the pipeline without modifying the
  segment.

## Negative Controls

Every intervention table must include random non-causal segments, shuffled
trace segments, and no-op interventions. If targeted gains are matched by
controls, causal wording is forbidden.

Required control IDs are `random_non_causal_segment`,
`shuffled_trace_segment`, and `no_op_intervention`.

## Evaluator Leakage

The intervention trace, evaluator prompt, and answer key must be hashed before
the run. Any evaluator leakage or post-hoc evaluator edits invalidate the run.

## Invalid Intervention Handling

Invalid interventions are logged with reason codes and cannot be silently
dropped. If invalid rate exceeds 5 percent, the run is diagnostic only.
