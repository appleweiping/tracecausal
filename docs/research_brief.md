# Research Brief

## Problem

Recent hallucination work can detect suspicious outputs using next-token
distributions, semantic entropy, reasoning consistency, or diffusion denoising
traces. These methods are often predictive, but they do not establish which
parts of a generation process causally create the hallucination or whether an
intervention on those parts improves factuality.

TraceCausal asks:

> Can hallucination-causing process segments be localized and intervened on
> across autoregressive, reasoning-trace, and diffusion language models?

## Core Hypothesis

Hallucination is not only an output state. In many cases it is a process failure
with identifiable causal bottlenecks: answer-anchored self-reinforcement,
question-evidence detachment, or denoising-state drift. A causal trace audit
should separate predictive signals from intervention-useful sources.

## What Is New

- Cross-paradigm trace schema: logits, reasoning steps, and denoising states are
  represented as comparable process segments.
- Causal segment scoring: counterfactual patching and replay estimate
  intervention effect, not only detector accuracy.
- Dual evidence target: a segment must predict hallucination and improve
  factuality when intervened on.

## Closest Work To Beat

- Output-signature detectors such as LOS-style methods.
- Semantic entropy and guided semantic exploration.
- Reasoning consistency detectors for large reasoning models.
- Diffusion-LM trace detectors such as TraceDet and follow-up hidden-dynamics
  detectors.
- Mechanistic causal tracing for VLM hallucination.

## Kill Arguments

1. The method may only find correlates: if interventions do not improve
   factuality, the project collapses into another detector.
2. Cross-paradigm unification may be too abstract: AR logits, chain-of-thought,
   and diffusion denoising traces may not share a useful causal unit.
3. Baselines are moving fast: a pure D-LLM detector would be incremental unless
   causal intervention is the center of the claim.

## Target Venues

ICLR, NeurIPS, ACL, EMNLP, or NAACL depending on the final balance between
mechanistic analysis and NLP evaluation.

