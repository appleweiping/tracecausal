# Idea Synthesis

## Source Ideas Read

The seed ideas came from the extracted hallucination and diffusion-LM idea
notes in `D:\devtools\scratch\codex-hallucination-directions-20260615`. They
included:

- hallucination path or trace distributions may differ from factual paths;
- segmentation of generated text may expose where hallucination begins;
- causal reasoning and graph-based diagnosis may be under-benchmarked;
- diffusion LLM hallucination work has trace detectors but lacks causal
  propagation analysis;
- HalluClean-style prompt pipelines and distillation ideas are too close to
  engineering cost reduction unless reframed around a sharper mechanism.

The raw idea files are not copied into this repository.

## Abstractions

The strongest non-stitched abstraction is not "train a smaller hallucination
agent" or "add a graph to a causal reasoning paper." It is:

> hallucination should be audited as a causal process, where a trace segment is
> only scientifically meaningful if changing it changes factuality.

This moves the project away from detector stacking and toward a falsifiable
mechanistic claim.

## Rejected Directions

- HalluClean distillation: useful engineering, weak top-conference novelty.
- Domain-only hallucination detection: likely dataset-paper risk.
- D-LLM trace detection alone: too close to TraceDet-style work.
- Causal DAG plus GNN assistant: likely module stitching unless new causal
  intervention evidence is central.

## Selected Direction

TraceCausal uses the user idea about trace/path distributions and the diffusion
LM causal-analysis gap, but turns it into a stronger contribution:

1. common trace schema across generation paradigms;
2. causal segment scoring through counterfactual replay or patching;
3. intervention-usefulness as the main gate;
4. strict comparison to output-only, reasoning-trace, and diffusion-trace
   detectors.

