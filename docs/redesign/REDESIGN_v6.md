# REDESIGN v6 — tracecausal (delta over v5)

Status: `design_frozen_stage1_RR` (preserved from v5). **No server run authorized;
`server.authorized: false` is preserved and re-affirmed in every guarded config.**
This document is **additive** over `docs/redesign/REDESIGN_v5.md`: it changes the
**baseline comparison contract, the headline model promotion, and the
method-iteration loop**, and leaves the v5 estimand, gates, inference, and
selective-inference machinery **unchanged**. The frozen estimand is intact:
`Û`/G1/NECESSITY_MARGIN for screening necessity, `R̂`/G9-NOV for cross-example
repair-transfer. **No invented results.** Every empirical slot remains `DATA_NEEDED`;
the only numerics are closed-form margin/variance/feasibility identities ("formula
evaluation, not evidence").

Section numbers in the form "v5 §X" refer to `REDESIGN_v5.md`. This file supersedes
nothing in v5; it **scopes** the baseline panel and the model lead to the ARIS v6
decision and adds the iteration loop and the single-GPU inference-only schedule.

---

## 0. Executive summary (one paragraph)

v5 froze the certification protocol (`R̂`/G9 + G9-NOV, matched-null, selective
inference, adversarial oracle) and an AR-LLM lead. **v6 makes the comparison
top-conference fair and the model headline current**, with three deltas and nothing
else: (i) **promote Qwen2.5-7B-Instruct to the headline matched model** (with
Llama-3.1-8B-Instruct as the secondary cross-family check), **demote 1.5B to a
smoke/contract tier**; (ii) **lock a `>=8` OFFICIAL-REPO verified-repo baseline
suite** (2024-2026, all confirmed real official code), distinguishing DETECTION
comparators (AUROC) from the LOCALIZATION/attribution comparators (Captum-IG, Causal
Mediation) our necessity certification competes with most directly, and **handle
TraceDet as a cited reference, not a reproduced baseline** (no public code + complex
diffusion method); and (iii) add an explicit **method-iteration loop** — if CIU does
not clear G1/G9-NOV against these baselines at 7B, the response is to **diagnose and
improve the METHOD** (never the data, never fabricate), re-run within a capped budget,
else issue an honest negative/limited report. The single-RTX-4090 inference-only
schedule (recompute `c_fwd` at 7B first) is noted. `server.authorized: false`
throughout.

---

## 0.1 Scope label preserved from v5 (G9 = extension; claim-span = proxy; ablation registered)

v6 changes the baseline contract, the model headline, and the iteration loop only; it
**preserves verbatim** three v5 scoping statements that must travel with every place the
estimand/contribution is stated:

- **`R_hat` / G9 / G9-NOV is an ORIGINAL EXTENSION beyond the user's original idea —
  an extension, NOT a direct implementation of it (v5 §3.1, restated §4.2a).** The
  user's original tracecausal idea (`user_ideas/hallucination/ideas of hallucination.txt`,
  idea #2) is the *training-free* certification of **which segment** of the text causes
  the hallucination and the in-place "fix the segmentation"; that is realized **directly**
  by the **necessity** statistic `U_hat` / **G1**. The **cross-example transfer**
  question — "does a repair policy learned by localizing on one example transfer to fix a
  *different* held-out example?" — is the **new, original extension of tracecausal's OWN
  idea**, internal to tracecausal and **not** related to or borrowed from any other
  project. The promotion of `R_hat`/G9 to the v5/v6 headline does **not** relabel it as
  the original idea: it remains the explicitly-labeled extension, with `U_hat`/G1 the
  faithful original-idea line beneath it.
- **The claim-span inventory is a PROXY (v5 §4.2a).** TriviaQA/HotpotQA carry no gold
  atomic-claim segmentation; `build_target_claim_spans` constructs a documented proxy
  (exact rule + assumption in v5 §4.2a). v6 changes nothing about the proxy.
- **The proxy-robustness ablation is registered (v5 §8a).** The headline G9/G9-NOV verdict
  must be invariant to the claim-span construction rule across ≥3 alternative
  segmentations + the default, run through the SAME `g_ij → R_hat → G9/G9-NOV` pipeline
  via the `--claim-span-variant` hook; a sensitivity finding downgrades the claim (R2/R3).
  This ablation is **secondary/diagnostic** (it does not enlarge the v6 Holm family or
  interact with the §5 method-iteration loop) and stays `server.authorized: false` /
  DATA_NEEDED until the authorized run.

---

## 1. What changed from v5 (the delta, and only the delta)

| Surface | v5 state | v6 change | Driver |
| --- | --- | --- | --- |
| **Headline model** | Two AR families pinned (Qwen2.5-7B-Instruct lead_family_1, Llama-3.1-8B-Instruct lead_family_2); 1.5B not the headline | **Qwen2.5-7B-Instruct is the PRIMARY confirmatory headline; Llama-3.1-8B-Instruct SECONDARY cross-family; 1.5B = smoke/contract only** | ARIS v6: current matched-model headline for a top-conf fair comparison |
| **Baseline suite** | Registry lists semantic_entropy, selfcheckgpt, INSIDE, RACE, TraceDet (+ local sanity, diffusion-transfer-only) | **Locked `>=8` OFFICIAL-REPO suite** (ReDeEP, RACE, LapEigvals, HaloScope, Lookback-Lens, MIND, INSIDE, Semantic Entropy, SelfCheckGPT) **+ 2 LOCALIZATION controls** (Captum-IG, Causal Mediation) | ARIS v6: verified-repo fairness contract |
| **Detection vs localization** | implicit (all detectors re-scored) | **Explicit `comparator_class`**: DETECTION (AUROC) vs LOCALIZATION (attribution; G1/G6/G7 margin) | the necessity-cert's most direct competitors are attribution localizers |
| **TraceDet** | `required: true`, faithful-reimpl-required, diffusion detector in transfer table | **`role: cited_reference_not_reproduced`** (no public code + complex diffusion); differentiated-from, not reproduced. B1 AR span adapter (our re-cast) unchanged | no public official code; paradigm mismatch; verified-repo contract |
| **Iteration discipline** | Stage-2 routing R1-R4 (publish/downgrade) | **Add a method-iteration loop BEFORE routing**: G1/G9-NOV miss ⇒ diagnose+improve the METHOD, capped re-run, never data/fabrication | top-conf bar: a near-miss should trigger principled method work, not a premature null |
| **Compute schedule** | v4 feasible point re-checked vs `sigma_R` | **Single-RTX-4090 inference-only note**: recompute `c_fwd` at 7B FIRST; all forwards inference-only (no training) | the assigned hardware tier for v6 |

Everything else in v5 — the `R̂` estimand (Eq. R), the transport map `T` (Variant C),
Identification Lemma 5.1, two-way cluster bootstrap + class-block permutation + Hájek
cross-check, `sigma_R` power model, OS-1 operator freeze, SI-1/SI-2, Axis X′,
gates G1-G8 + G9 + G9-NOV — is **PRESERVED VERBATIM**.

---

## 2. Model promotion: Qwen2.5-7B-Instruct headline, 1.5B demoted to smoke

- **PRIMARY confirmatory (headline).** `ar_lead_qwen` = **Qwen2.5-7B-Instruct**
  (`Qwen/Qwen2.5-7B-Instruct`, revision pinned at run authorization). The headline
  `R̂`/G9/G9-NOV claim and the Holm confirmatory family are adjudicated on the
  pre-registered **2-cell point**: Qwen2.5-7B-Instruct × {TriviaQA, HotpotQA}. This
  matches the registered config `lead_models[0]` and `paper/main.tex` §5.4.
- **SECONDARY cross-family (budget-gated).** `ar_lead_llama` =
  **Llama-3.1-8B-Instruct**. Adds the 4-cell grid (both datasets) to test
  cross-family generalization; **not** the primary claim, gated behind the
  single-GPU budget re-check (§7).
- **1.5B demoted to SMOKE / CONTRACT tier.** Any 1.5B model (e.g. Qwen2.5-1.5B) is
  used **only** for do-not-run contract validation / harness smoke (kernel shape
  checks), **never** as paper evidence. It carries no headline, no confirmatory
  cell, and no baseline comparison. This is a labelling change only; no smoke result
  is ever promoted (`pilot` evidence label per AGENTS.md, never `paper_result`).

All baselines are run on the **same** matched headline model (Qwen2.5-7B-Instruct)
for the primary claim — no baseline gets a different model.

---

## 3. The `>=8` verified-repo baseline contract

The locked suite (all **confirmed real official code**; full roster, repos, venues,
and per-baseline run notes in `BASELINES.md` and
`configs/baselines/baseline_registry.yaml`):

**DETECTION comparators (AUROC), `role: must_beat` — 9:**
ReDeEP (ICLR 2025), RACE (AAAI 2026), LapEigvals (EMNLP 2025), HaloScope (NeurIPS
2024 spotlight), Lookback-Lens (EMNLP 2024), MIND (ACL 2024 Findings), INSIDE /
EigenScore (ICLR 2024), Semantic Entropy (Nature 2024 / ICLR 2023), SelfCheckGPT
(EMNLP 2023).

**LOCALIZATION / attribution comparators, `role: control` — 2:**
Captum Integrated Gradients (attribution control) and Causal Mediation Analysis
(NeurIPS 2020, Vig/Belinkov). These produce a per-token / per-component importance
map (not a detector score), so **our necessity certification competes with them most
directly**; they are scored on the segment-localization margin (G1 necessity / G6-G7
leakage), **not** AUROC.

**=> 9 + 2 = 11 verified-repo baselines, clearing `>=8`.** Local sanity baselines
(`random_segment`, `output_entropy`) and diffusion-transfer-only detectors (TDGNet)
remain in the registry but are **not** part of this AR-lead headline suite.

### 3.1 Merge, not duplicate

The v5 registry already listed `semantic_entropy`, `selfcheckgpt`, `inside_detector`
(INSIDE), `reasoning_consistency_detector` (RACE), and `diffusion_trace_detector`
(TraceDet). v6 **merges** the verified-repo fields (`venue`, `official_repo`,
`implementation_source: official_repo`, `role`, `comparator_class`) into those
existing entries and **adds** the remaining verified-repo baselines (ReDeEP,
LapEigvals, HaloScope, Lookback-Lens, MIND, Captum-IG, Causal Mediation). No baseline
is duplicated.

### 3.2 `implementation_source: official_repo`

Each verified-repo baseline carries `implementation_source: official_repo` with an
`official_repo:` URL and `implementation_commit: pending_before_server_run`. The
exact upstream commit is pinned and the license verified **at run authorization**;
the run is **blocked at preflight** while any required baseline is marked
`pending_before_server_run` (`docs/baseline_contract.md`,
`ciu.baseline_readiness`). No baseline is called "official" without a source commit
and provenance.

### 3.3 Fairness (segment re-scoring + identical repair pipeline)

Each detector's native signal is re-scored to a segment-level `Û` via a declared,
hashed `segment_adaptation` (identical-across-baselines adapter family), and the
selected span is fed through the **identical** `repair_ops` pipeline (Variant C
transport + frozen anchor map `T`) for G9-NOV. A strong baseline therefore *helps*
the baseline — the fairness incentive is symmetric. Without an audited adapter a
baseline is AUROC-only and excluded from the `R̂` table. Full protocol:
`BASELINES.md` §2.

### 3.4 TraceDet handled as a cited reference

**TraceDet (ICLR 2026, arXiv 2510.01274)** is updated to
`role: cited_reference_not_reproduced` /
`implementation_source: cited_reference_not_reproduced`. Reason: **no public official
code** as of the v6 lock, and it is a **complex diffusion-LLM method** on a paradigm
distinct from our AR-LLM lead. The v6 verified-repo contract admits only confirmed
OFFICIAL-REPO methods into the headline suite; a from-scratch diffusion
re-implementation is barred. TraceDet stays as the **sharpest related-work neighbour
we cite + position against** (matched null + selective inference + cross-example
transfer it lacks), **absent from both the AR-lead `R̂` table and the diffusion
transfer table** as a run detector. **Distinct:** the TraceDet-derived **AR
span-selection adapter B1** (its IB selection criterion re-cast as a training-free AR
span selector, `repair_ops.tracedet_ar_span_adapter`) is **our own code** and remains
in the repair panel — no diffusion weights, no TraceDet implementation. Full
rationale: `BASELINES.md` §4.

---

## 4. Fairness protocol (restated; identical to BASELINES.md §2)

Matched model **Qwen2.5-7B-Instruct**; matched datasets **TriviaQA + HotpotQA**
(3-way SI-1 split hashes pinned at authorization); matched proper-scored evaluator
(`${EVAL_HASH}` for `Û`, `${REPAIR_EVAL_HASH}` for the repair score `Y_j`); shared
seeds (`>=20`); **segment-level re-scoring** to `Û` (`ciu_scored: true`) plus the
**identical repair pipeline** for G9-NOV. **Deciding gates: G1 (necessity, screening,
margin `>= 0.05` on `u_deflated`) + G9-NOV (`R̂(PROPOSED) − max{R̂(B1..B3)} > 0`,
Holm/SI-corrected simultaneous lower CI).** DETECTION baselines are AUROC-scored;
the LOCALIZATION controls are scored on the segment-localization margin (G1/G6/G7).

---

## 5. Method-iteration loop (capped; method-only, never data, never fabrication)

The top-conference bar requires that a **near-miss** trigger principled **method**
work, not a premature null. v6 inserts an explicit loop **before** the v5 Stage-2
routing (R1-R4):

> **If CIU does not clear G1 AND G9-NOV against the verified-repo suite at 7B:**
> 1. **DIAGNOSE the method** — read the failing gate's diagnostics (which class /
>    which baseline B1-B3 matched/beat PROPOSED; positivity; matched-null repair B4
>    also-passing; Axis X′ behaviour). Form a **method** hypothesis (e.g. the
>    selector `S*` span boundary, the repair policy `rho` operator/`alpha`/`L_patch`,
>    the anchor rule, the class stratification).
> 2. **IMPROVE the METHOD only** — change the CIU selector / repair operator /
>    transport / stratification. **NEVER** change the data, the splits, the
>    evaluator, the seeds, the baselines, or the gates; **NEVER** fabricate or
>    cherry-pick. Any operator change is re-frozen on `V_sel` (OS-1) and re-paid in
>    the SI correction (`K_op`), so the iteration is multiplicity-honest.
> 3. **RE-RUN, CAPPED** — bounded number of method iterations (pre-registered cap;
>    `method_iteration_cap` below) within the single-GPU inference budget. Each
>    iteration is logged in the reproducibility ledger (what changed, why,
>    re-frozen `V_sel`).
> 4. **ELSE honest report** — if the cap is reached without clearing G1/G9-NOV,
>    issue an **honest negative/limited report** and route per v5 Stage-2 (R2/R3:
>    a G9 null is publishable **only if powered** (`R_power` met) **and**
>    theory-informative; otherwise "inconclusive — request budget"). No overclaim,
>    no quiet baseline weakening, no data tweak.

**Pre-registered cap and guards (DATA_NEEDED where numeric):**

- `method_iteration_cap`: **DATA_NEEDED** (small integer, pinned at authorization;
  e.g. `<= 3`), so the search is bounded and the SI family size is known before lock.
- **Selection-honesty.** Every method iteration re-freezes operator/selector choices
  on `V_sel` only; `V_inf`/test stay sealed. The Holm/SI `K_op` factor accounts for
  the reachable operator grid (v5 §4.7 OS-2). Iterating the method does **not**
  license peeking at the inference split.
- **Forbidden moves (hard).** Never alter data / splits / evaluator / seeds /
  baselines / gate thresholds to pass; never report best-seed; never fabricate; never
  silently drop a failing baseline. (AGENTS.md Hard Rules; `docs/baseline_contract.md`
  Forbidden Comparisons.)
- **Output.** The loop's verdict (cleared at iteration `k`, or honest report at cap)
  is recorded; the paper states the realized iteration count and what method change
  cleared the gate (or that none did).

This loop is the v6 answer to "what if the method does not beat the verified-repo
suite": **improve the method, honestly and capped — never the data.**

---

## 6. Decision routing (v5 R1-R4, with the loop in front)

1. Run the matched comparison at 7B (Qwen2.5-7B-Instruct) against the verified-repo
   suite.
2. If **G1 + G9-NOV clear** ⇒ v5 **R1** (full certification) or **R2** per the v5
   conditions. Done.
3. If they **do not clear** ⇒ enter the **§5 method-iteration loop** (capped).
4. If the loop clears within cap ⇒ R1/R2.
5. If the cap is reached ⇒ v5 **R3** (necessity-only, **only if** powered +
   theory-informative) or "inconclusive — request budget"; or **R4** if Axis X′-blind
   shows unsoundness. Honest report either way.

No new gates; the loop sits between the run and the v5 routing.

---

## 7. Single-RTX-4090 inference-only schedule note

The v6 hardware tier is a **single RTX 4090** (24 GB), **inference-only** (no
training; all baselines and PROPOSED are training-free or use released weights —
CDCR-SFT-style training is out of scope). Scheduling note:

- **Recompute `c_fwd` at 7B FIRST.** The v5 feasible point (`n=850`, `cells=2`,
  `c_fwd <= 4.57e-4` for the 2-cell screening line) was sized at the v4/v5 scale; at
  **Qwen2.5-7B-Instruct** the per-forward cost `c_fwd` **must be re-measured** on the
  `V_sel` split (the §3.2 timing calibration in `reports/run_packet.md`) **before**
  committing cell count. This is a timing measurement, not an experiment, and feeds
  the budget identity `cells * n * forwards_per_example * c_fwd <= budget_gpu_hr`.
- **Inference-only forwards.** Every forward is inference (generation / hidden-state
  read / patch / replay). No gradient steps except Captum-IG's gradient attributions,
  which are inference-time backward passes on a frozen model (no weight update), and
  Causal Mediation's restore-and-read forwards — both fit the inference-only,
  single-GPU envelope.
- **7B fits a single 4090 for inference.** Qwen2.5-7B-Instruct in fp16/bf16 (≈14-15
  GB weights) + activation cache for `patch`/`replay` fits a 24 GB 4090 for
  batch-1/small-batch inference; the OOM degradation ladder
  (`reports/run_packet.md` §6.3: halve batch → microbatch=1 → activation offload)
  applies. `R_int`, `n`, the frozen `R_null`, and seeds are **never** reduced to fit
  memory — only batch/microbatch knobs move.
- **Repair surcharge re-checked.** The v5 G9 forward surcharge
  (`1 + R_null + R_int` per repair-panel selector) is re-checked against the
  re-measured 7B `c_fwd`; if the recomputed total exceeds budget, the v5
  `decision_order` applies (request budget → reduce cells → reduce `R_null`/`R_int`
  to the variance-floored minimum). The SECONDARY Llama-3.1-8B 4-cell grid is gated
  behind this re-check.
- All schedule numbers (`c_fwd` at 7B, `forwards_per_example`, GPU-hours,
  wall-clock) are **DATA_NEEDED**, measured at the authorized run, never fabricated
  here.

---

## 8. DATA_NEEDED slots (v6-specific, in addition to v5)

| Slot | Where | Pinned at |
| --- | --- | --- |
| `c_fwd` re-measured at Qwen2.5-7B-Instruct | run_packet §3.2 timing calibration | run (V_sel timing) |
| `forwards_per_example` (7B repair surcharge `1+R_null+R_int`) | run_packet §5; config `feasible_point` | analysis-lock |
| `method_iteration_cap` (small integer) | this doc §5 | authorization |
| realized method-iteration count + what cleared the gate | reproducibility ledger | post-run |
| per-baseline `Û` segment-adapter hash (verified-repo suite) | baseline_registry `segment_adaptation` + adapter_hash | authorization |
| every `implementation_commit` (official_repo pin) | baseline_registry | authorization |
| every baseline `license` verification | baseline_registry | authorization |
| `R̂(PROPOSED) − max{R̂(B1..B3)}` (G9-NOV) | paper Tab. g9nov | post-run |
| AUROC/AUPRC/FPR@95TPR per DETECTION baseline | paper detection table | post-run |
| segment-localization margin per LOCALIZATION control | paper G1/G6/G7 tables | post-run |

No number above is fabricated in this document.

---

## 9. Preserved from v5 (unchanged under v6)

- The `R̂` estimand (Eq. R), transport map `T` (Variant C), Identification Lemma 5.1,
  assumptions A5-A9.
- Inference: two-way cluster bootstrap + class-block sign-flip diagnostic + Hájek
  two-projection cross-check; nested matched-null MC; `sigma_R` power model
  (Eq. R-VAR / R-POWER), `nuisance.r_power_repair`.
- OS-1 operator freeze on `V_sel`; SI-1/SI-2 selective inference; Holm family.
- Adversarial oracle Axis X′ (detectable + blind), NC-1/NC-2, P5/P6.
- Gates G1-G8 (screening) + G9 (headline) + G9-NOV (novelty), fail-closed, never
  `invalidated`.
- The frozen `Û`/G1/NECESSITY_MARGIN screening estimand and the AR-LLM lead with
  diffusion/reasoning demoted to a transfer study.
- `server.authorized: false`, additive/uncommitted discipline, DATA_NEEDED results,
  build-now/run-later, zero fabricated numbers.

---

## 10. Run discipline

```
server:
  authorized: false
  reason: >-
    v6 is a baseline-contract + model-promotion + iteration-loop DELTA over the v5
    design_frozen_stage1_RR design. No run, no model load, no GPU, no training. ARIS
    v6 design review (verified-repo suite + Qwen2.5-7B headline + G9/G9-NOV/Axis X')
    must be on file and >=8 before the §1 authorization flip in reports/run_packet.md.
```

**Phase-B gating rule (preserved).** Status `design_frozen_stage1_RR`, not
`stage1_ready`. The v5 modules are implemented as do-not-run pure-Python under a green
unit harness; v6 adds **no new module** (it is a config + doc + contract delta). The
Stage-1-ready label is earned only when the harness stays green AND the authorized 7B
run produces the pre-registered evidence. `server.authorized: false`.
