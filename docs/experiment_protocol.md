# Experiment Protocol

## Formal Questions

Q1. Do causal trace segment scores detect hallucination better than output-only
and trajectory-only baselines?

Q2. Do high-scoring causal segments produce larger factuality changes under
counterfactual intervention than random or predictive-only segments?

Q3. Does the causal segment taxonomy transfer across datasets, model families,
and decoding paradigms?

## Datasets

Initial formal candidates:

- open-domain QA: Natural Questions style, TriviaQA style, TruthfulQA style;
- multi-hop QA: HotpotQA style;
- hallucination benchmarks: HaluEval style and reasoning-consistency datasets;
- diffusion-LM benchmarks matching current public D-LLM detector papers.

Dataset adapters must record license, split, preprocessing hash, and leakage
checks before any formal run.

## Baselines

Minimum baseline families:

1. random and majority sanity baselines;
2. output-only uncertainty: token probability, entropy, semantic entropy;
3. learned output signature detectors;
4. reasoning consistency detectors;
5. D-LLM trace detectors for diffusion models;
6. causal tracing or intervention baselines where model internals are available.

## Metrics

- detection: AUROC, AUPRC, FPR@95TPR, calibration of detector confidence;
- localization: segment precision/recall/F1 against perturbation-derived labels;
- intervention: factuality delta, answer accuracy delta, fluency/utility delta;
- efficiency: trace extraction cost, intervention latency, memory footprint.

## Statistical Gate

Paper-result gate:

- at least 20 seeds or bootstrap replicates per main comparison;
- paired tests for matched examples;
- Holm correction for multiple comparisons;
- effect size reported for every main improvement;
- 95 percent confidence intervals for all table metrics.

## Early Kill Gates

1. Cheap causal sanity gate: targeted interventions must beat random segment
   interventions by >=5 absolute factuality-delta points on two datasets.
2. Transfer gate: segment taxonomy must retain >=80 percent of in-domain AUROC
   on at least one held-out dataset family.
3. Harm gate: intervention must not reduce answer accuracy or utility by more
   than 2 absolute points unless explicitly framed as abstention.

Failure of any gate triggers an iterate-or-pivot review before more compute.

