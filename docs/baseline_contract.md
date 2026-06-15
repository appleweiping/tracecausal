# Baseline Contract

All baselines must use the same examples, prompts where applicable, model
versions, trace availability, and evaluation code.

## Required Provenance Fields

- dataset name, version, license, split hash;
- model family, exact checkpoint or API model id, access date;
- prompt template hash and decoding parameters;
- trace extraction configuration;
- baseline implementation source, commit, and local modifications;
- prediction file hash and row count;
- metric script version and command;
- seed list and failed-run policy.

## Forbidden Comparisons

- comparing a trace method with extra data against an output baseline without
  stating and ablating the extra data;
- reporting best seed only;
- using different prompts for proposed method and baselines unless prompt
  sensitivity is the experiment;
- calling a wrapper "official" without source commit and provenance;
- importing diffusion-model results into AR-only tables without a separate
  paradigm label.

