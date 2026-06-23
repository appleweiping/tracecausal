# TraceCausal Baselines (verified-repo suite, v6)

> **Status: DESIGN / DOC ONLY — `server.authorized: false`.** This document records
> the **pre-registered** v6 verified-repo baseline suite and the fairness protocol
> under which it will be run. It reports **NO results**: every empirical cell is
> `DATA_NEEDED`. Source of truth for the machine-readable roster is
> `configs/baselines/baseline_registry.yaml`; the design rationale is
> `docs/redesign/REDESIGN_v6.md` (delta) and `docs/redesign/REDESIGN_v5.md`
> (preserved CIU / matched-null / G1-G8 cores). Exact upstream commits and licenses
> are pinned/verified at run authorization (`implementation_commit:
> pending_before_server_run`).

The v6 redesign (ARIS v6) promotes **Qwen2.5-7B-Instruct** (+ **Llama-3.1-8B-Instruct**
secondary) as the matched headline model and **locks a `>=8` OFFICIAL-REPO baseline
suite** (2024-2026, all confirmed real official code) for a top-conference fair
comparison. Our method (CIU: a training-free matched-null intervention-usefulness
certification `Û`/G1, plus the cross-example repair-transfer certificate `R̂`/G9)
must beat these via the pre-registered **G1 (necessity)** + **G9-NOV (novelty)**
gates, with a method-iteration loop if it does not (REDESIGN_v6 §6).

---

## 1. The locked verified-repo baseline suite

Every baseline below is backed by **confirmed real official code** (`official_repo`
URL in the registry, `implementation_source: official_repo`). Each is run on the
**matched model** Qwen2.5-7B-Instruct (the headline), its native signal re-scored to
a segment-level `Û` and run through the **identical repair pipeline** for G9-NOV
(see §2). The `detection-vs-localization` column records which kind of comparator it
is: **DETECTION** baselines are scored by AUROC/AUPRC/FPR@95TPR (the screening / G1 /
detection tables); **LOCALIZATION** comparators produce a per-token/per-component
attribution map (not a detector score) and are the attribution neighbours our
**necessity certification competes with most directly** (scored on the segment-
localization margin: G1 necessity / G6-G7 leakage, **not** AUROC).

| # | Name | Venue | Official repo | Role | Detection vs Localization |
| --- | --- | --- | --- | --- | --- |
| 1 | **ReDeEP** | ICLR 2025 | `github.com/Jeryi-Sun/ReDEeP-ICLR` | `must_beat` | Detection (AUROC) |
| 2 | **RACE** | AAAI 2026 | `github.com/bebr2/RACE` | `must_beat` | Detection (AUROC) |
| 3 | **LapEigvals** | EMNLP 2025 | `github.com/graphml-lab-pwr/lapeigvals` | `must_beat` | Detection (AUROC) |
| 4 | **HaloScope** | NeurIPS 2024 (spotlight) | `github.com/deeplearning-wisc/haloscope` | `must_beat` | Detection (AUROC) |
| 5 | **Lookback-Lens** | EMNLP 2024 | `github.com/voidism/Lookback-Lens` | `must_beat` | Detection (AUROC) |
| 6 | **MIND** | ACL 2024 (Findings) | `github.com/oneal2000/MIND` | `must_beat` | Detection (AUROC) |
| 7 | **INSIDE / EigenScore** | ICLR 2024 | `github.com/D2I-ai/eigenscore` | `must_beat` | Detection (AUROC) |
| 8 | **Semantic Entropy** | Nature 2024 / ICLR 2023 | `github.com/jlko/semantic_uncertainty` | `must_beat` | Detection (AUROC) |
| 9 | **SelfCheckGPT** | EMNLP 2023 | `github.com/potsawee/selfcheckgpt` | `must_beat` | Detection (AUROC) |
| 10 | **Captum Integrated Gradients** | attribution control (IG, 2017; Captum) | `github.com/meta-pytorch/captum` | `control` | **Localization** (token attribution) |
| 11 | **Causal Mediation Analysis** | NeurIPS 2020 (Vig/Belinkov) | `github.com/sebastianGehrmann/CausalMediationAnalysis` | `control` | **Localization** (indirect-effect map) |

**Count: 11 verified-repo baselines = 9 DETECTION (`must_beat`) + 2 LOCALIZATION
(`control`)**, clearing the `>=8` OFFICIAL-REPO contract. (Local sanity baselines
`random_segment` and `output_entropy`, and the diffusion-transfer-only `TDGNet`, are
in the registry but are **not** part of the AR-lead verified-repo headline suite.)

**Cited reference, NOT reproduced — TraceDet (ICLR 2026, arXiv 2510.01274):** see §4.

---

## 2. Fairness protocol

All baselines and PROPOSED are adjudicated under one matched setup. No baseline gets
a different model, prompt, split, or evaluator; a baseline may only differ in its
own native signal and the (declared, hashed) adapter that re-scores it to segment
level.

- **Matched model.** Every baseline and PROPOSED runs on the **same** headline model
  **Qwen2.5-7B-Instruct** (`ar_lead_qwen`, primary confirmatory). The
  **Llama-3.1-8B-Instruct** secondary cells (`ar_lead_llama`) are a budget-gated
  cross-family generalization check, not the primary claim. The 1.5B model is demoted
  to a smoke/contract tier only (REDESIGN_v6 §2).
- **Matched datasets.** **TriviaQA** (factuality) and **HotpotQA** (multi-hop), the
  two frozen lead datasets, with 3-way `V_sel`/`V_inf`/test split hashes (SI-1) pinned
  at authorization. No baseline sees a different split.
- **Matched evaluator + provenance.** Same proper-scored evaluator (`${EVAL_HASH}` for
  `Û`; `${REPAIR_EVAL_HASH}` for the cross-example repair score `Y_j`), same prompt
  template hash, same seed list (`configs/seeds/paper_20.txt`, `>=20`), same leakage
  check. All provenance fields per `docs/baseline_contract.md`.
- **Segment-level re-scoring (`Û`).** Each detector's NATIVE output-/representation-/
  sampling-level signal is re-scored to a **per-segment** `Û` via its declared,
  hashed `segment_adaptation` (registry field, identical-across-baselines adapter
  family). Without an audited adapter a baseline is AUROC-only and **excluded** from
  the `R̂` table (`docs/baseline_contract.md`). This is `ciu_scored: true` in the
  experiment config.
- **Identical repair pipeline (G9-NOV).** For the novelty gate, each detector's
  selected span is fed through the **identical** `repair_ops` pipeline (Variant C
  source-derived repair policy + the frozen anchor/transport map `T`) that PROPOSED
  uses. A **strong** baseline therefore *helps* the baseline beat us — the fairness
  incentive is symmetric, and out-transferring them is the empirical content of the
  novelty claim.
- **Deciding gates.**
  - **G1 (necessity, screening).** PROPOSED's localized segment must show an
    above-matched-null factuality necessity margin `>= 0.05` (on `u_deflated`), the
    margin where the LOCALIZATION comparators (Captum-IG, Causal Mediation) are the
    sharpest competitors.
  - **G9-NOV (baseline-conditional novelty).** `R̂(PROPOSED) − max{R̂(B1), R̂(B2),
    R̂(B3)} > 0` (Holm/SI-corrected simultaneous lower CI), i.e. CIU out-transfers the
    TraceDet-/entropy-/probe-selected localizations through the identical pipeline.
    Fail ⇒ the certification protocol is the contribution, not "causal beats
    correlational" (REDESIGN_v5 §5.2, REDESIGN_v6 §5).

**Detection vs localization, and why it matters for the contract.** The 9 DETECTION
baselines answer an output-/representation-level *predictive* question (is this
generation hallucinated?), so they are compared on AUROC in the screening tables and,
re-scored + repaired, in the `R̂` panel. The 2 LOCALIZATION comparators (Captum-IG,
Causal Mediation) produce an *importance map over tokens/components* — the same kind
of object our necessity certification produces — so they are the comparators our
method competes with **most directly**, scored on the segment-localization margin
(G1 necessity, G6/G7 leakage), not AUROC. Both kinds also enter the G9-NOV repair
panel through the identical pipeline.

---

## 3. Per-baseline "how we run it" notes

Each note states the native signal, the segment-level re-scoring (`Û`), and that the
selected span is fed through the identical repair pipeline for G9-NOV. All run on
Qwen2.5-7B-Instruct / TriviaQA+HotpotQA.

- **ReDeEP (ICLR 2025, detection).** Native: regresses *decoupled external vs
  parametric knowledge* signals from hidden states + attention. Re-score: the
  decoupled-knowledge score is localized to the segment's layers/heads → per-segment
  `Û`. Repair: top-`Û` segment → identical `repair_ops` pipeline.
- **RACE (AAAI 2026, detection).** Native: joint answer-reasoning consistency over
  reasoning traces. Re-score: consistency signal computed per reasoning segment.
  Repair: selected reasoning segment → identical pipeline.
- **LapEigvals (EMNLP 2025, detection).** Native: spectral (Laplacian-eigenvalue)
  features of attention maps. Re-score: eigenvalue features computed over the
  segment-restricted attention submatrix → per-segment `Û`. Repair: identical
  pipeline.
- **HaloScope (NeurIPS 2024 spotlight, detection).** Native: unlabeled-generation
  membership estimation in a hallucination subspace of hidden states. Re-score:
  membership-subspace score localized to segment hidden states. Repair: identical
  pipeline.
- **Lookback-Lens (EMNLP 2024, detection).** Native: attention context-vs-generation
  ratio. Re-score: lookback ratio computed per segment. Repair: identical pipeline.
- **MIND (ACL 2024 Findings, detection).** Native: unsupervised internal-state
  real-time detection. Re-score: internal-state signal over segment-localized hidden
  states. Repair: identical pipeline.
- **INSIDE / EigenScore (ICLR 2024, detection).** Native: internal-state covariance
  eigenscore. Re-score: eigenscore over segment-localized hidden states. Repair:
  identical pipeline.
- **Semantic Entropy (Nature 2024 / ICLR 2023, detection).** Native: meaning-level
  sampling uncertainty. Re-score: per-segment semantic entropy over
  segment-conditioned resamples. Repair: identical pipeline.
- **SelfCheckGPT (EMNLP 2023, detection).** Native: black-box sampling consistency.
  Re-score: NLI consistency over segment-aligned resamples. Repair: identical
  pipeline.
- **Captum Integrated Gradients (attribution control, LOCALIZATION).** Native:
  per-token integrated-gradient attributions. Re-score: token attributions aggregated
  to segment importance. **Scored on the segment-localization margin (G1/G6/G7), not
  AUROC** — this is the closest attribution neighbour our necessity certification
  competes with. Also enters G9-NOV via the identical pipeline.
- **Causal Mediation Analysis (NeurIPS 2020, LOCALIZATION).** Native: indirect-effect
  localization over model components (Vig/Belinkov). Re-score: indirect effect
  localized to segment components. **Scored on the segment-localization margin (G1/
  G6/G7), not AUROC** — the closest *causal*-attribution neighbour. Also enters
  G9-NOV via the identical pipeline.

---

## 4. Why TraceDet is cited, not reproduced

**TraceDet (ICLR 2026, arXiv 2510.01274)** is the sharpest related-work neighbour:
it models a diffusion LLM's denoising as an action trace and, via an
information-bottleneck objective, isolates the sub-trace maximally informative about
hallucination, improving detection AUROC by ~15.2% on average. It is exactly the
"causal beats correlational" rebuttal we must cash in empirically. **It is, however,
CITED and POSITIONED AGAINST — not reproduced as a baseline** — for two concrete
reasons: (1) it has **no public official code** as of the v6 lock, and (2) it is a
**complex diffusion-LLM method** on a paradigm distinct from our frozen AR-LLM lead
(Qwen2.5-7B-Instruct + Llama-3.1-8B-Instruct). The v6 verified-repo contract admits
only confirmed OFFICIAL-REPO methods into the headline suite; a from-scratch
diffusion re-implementation is explicitly barred from that suite (it would be a
faithful-reimplementation we could not audit against upstream, on the wrong
paradigm). We therefore **differentiate** from TraceDet in related work (a matched
null, a selective-inference correction, and a cross-example transfer test it lacks)
rather than reproduce it. TraceDet is absent from both the AR-lead `R̂` table and the
diffusion transfer table as a run detector.

**Distinct and separate:** a **TraceDet-DERIVED AR span-selection adapter** — its IB
sub-trace selection *criterion* re-cast as a training-free AR localization selector —
**does** enter the AR-lead repair panel as **B1**
(`repair_ops.tracedet_ar_span_adapter`). This is **our own code** (no diffusion
weights, no TraceDet implementation): a span selector fed through the identical
`repair_ops` pipeline for G9-NOV fairness. The diffusion **detector** is
cited-not-reproduced; the B1 **adapter** is our re-cast span selector. Keeping these
two separate is what lets us position against TraceDet honestly without claiming to
have reproduced an unreleased diffusion method.

---

## 5. Pointers

- Machine-readable roster + fields: `configs/baselines/baseline_registry.yaml`
- Provenance/fairness rules: `docs/baseline_contract.md`
- v6 delta (headline model promotion, verified-repo contract, fairness protocol,
  method-iteration loop, single-RTX-4090 schedule): `docs/redesign/REDESIGN_v6.md`
- Repair-transfer certification (`R̂`/G9/G9-NOV), matched-null cores, gates:
  `docs/redesign/REDESIGN_v5.md`
- Frozen experiment plan: `configs/experiments/redesign_v5_ar_lead.yaml`
- Run approach (per-baseline + single-GPU schedule): `reports/run_packet.md`
- Paper related-work / baseline panel: `paper/main.tex` (Sec. Related,
  Sec. method:baselines)
