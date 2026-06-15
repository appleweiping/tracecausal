# Research Plan

## Research Question

Can hallucination-causing process segments be localized and intervened on across
autoregressive, reasoning-trace, and diffusion language models?

## Core Hypotheses

H1. Some hallucinations have trace segments whose causal intervention effect is
larger than random or detector-only selected segments.

H2. A trace segment taxonomy can transfer across at least two dataset families
or model families.

H3. Intervention-useful segments are not always the same as the most predictive
segments, so causal scoring adds value beyond detector accuracy.

## Work Packages

1. Trace schema: define a minimal schema for AR logits, reasoning traces, and
   diffusion denoising states.
2. Counterfactual protocol: specify patching, masking, replay, and abstention
   interventions.
3. Baselines: implement or wrap output entropy, semantic entropy, output
   signatures, reasoning consistency, and diffusion trace baselines.
4. Evaluation harness: detection, localization, intervention, efficiency, and
   statistical tests.
5. Paper evidence: motivation study, main tables, ablations, failure cases,
   claim audit, and citation audit.

## Server Boundary

Local work ends at configs, validators, protocol docs, and lightweight unit
tests. Trace extraction and interventions require server approval because they
need model access, storage, and GPU time.

