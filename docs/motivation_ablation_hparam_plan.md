# Motivation, Ablation, And Hyperparameter Plan

## Motivation Study

Goal: show that predictive trace signals are not enough; intervention effect
must be measured.

Planned figure:

- x-axis: segment detector score rank;
- y-axis: factuality delta after intervention;
- compare causal-selected, predictive-only, entropy-selected, and random
  segments.

## Ablations

- remove causal intervention scoring;
- use predictive detector score only;
- use one trace paradigm only;
- remove transfer taxonomy;
- replace targeted intervention with random segment intervention;
- remove utility/harm gate.

## Hyperparameters

- segment length/window size;
- intervention strength;
- trace feature aggregation;
- replay sample count;
- detector threshold;
- taxonomy clustering threshold.

## Required Curves

- factuality delta vs intervention strength;
- utility drop vs intervention strength;
- AUROC vs segment length;
- cost vs trace resolution.

