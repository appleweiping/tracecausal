# REDESIGN v2 — tracecausal

Status: `design_only`. No server run authorized. `server.authorized: false` is
preserved. This document is **additive**: it does not modify, delete, or
supersede any existing file (including `REDESIGN_v1.md`, which remains on disk as
the prior iteration). It reuses the existing governance (`AGENTS.md`,
`docs/experiment_protocol.md`, `docs/intervention_protocol.md`,
`docs/baseline_contract.md`, `docs/pre_registration.md`), the existing code
contracts (`src/tracecausal/metrics.py::passes_intervention_gate`,
`src/tracecausal/schemas.py`), the `configs/baselines/baseline_registry.yaml`
registry, and the `configs/seeds/paper_20.txt` seed manifest. All numeric gates
below are *identical* to the ones already pre-registered (margin `0.05`, utility
drop `0.02`, transfer retention `0.80`, seeds `>= 20` as a floor — v2 adds a
*power-derived* seed/replicate requirement on top, never below, that floor).
This redesign does not weaken any pre-registered gate.

Naming note (unchanged from v1). The deliverable brief inherits a generic clause
requiring a "propriety PROOF for riskcal". `tracecausal` has no module named
`riskcal`. The load-bearing object the clause maps onto here is the
**causal-usefulness estimator** `\hat U` (the targeted-minus-random factuality
contrast that gates every causal claim, implemented today as
`passes_intervention_gate`). **v2 changes what we prove about it.** v1 claimed a
*propriety / non-gameability theorem* asserting that pure detectors provably
score `0`. The adversarial review (GPT-5.5, 3/10) correctly held that this
*assumes away* the main threat: real detector covariates (entropy, token
position, claim-bearing-ness, suffix instability) plausibly **correlate** with
intervention sensitivity, so "detectors score 0" was stipulated, not proved.
v2 therefore replaces the theorem with an **Identification Lemma** under
explicit, empirically **testable** assumptions, and reframes the detector-floor
as an empirical **prediction** tested by the random-segment control — not a
guarantee. The "propriety PROOF" clause is honoured at the only level the design
can actually support: a proof of *unbiasedness of the matched-control contrast
for the causal estimand under stated, falsifiable conditions* (Lemma 2.5),
plus a stress test that can reject those conditions.

---

## Changelog vs v1 (each GPT-5.5 fix → what changed here)

This table is the spine of v2. Every blocking issue in
`D:/Research/_aris_orchestration/reviews/tracecausal/redesign_v1_gpt55.md` is
mapped to a concrete change.

| GPT-5.5 blocking issue (v1) | Correction in v2 | Where |
| --- | --- | --- |
| **B1.** Theorem 2.5 *assumes away* the threat; "pure detectors score 0" is stipulated by exchangeability, not proved. Detector covariates (entropy, position, claim-bearing-ness, suffix instability) plausibly correlate with intervention sensitivity. | **Theorem 2.5 deleted.** Replaced by **Identification Lemma 2.5** that proves only *unbiasedness of the matched contrast for the causal estimand* under explicit assumptions (A1′–A4), and explicitly lists the empirically testable conditions. The detector-floor is downgraded from a theorem to a **falsifiable empirical prediction** (§3) tested by the random-segment control. We no longer claim detectors score 0; we claim it is a prediction that can lose. | §2.5, §3 |
| **B2.** "certifies" / "proper score" / "cannot be gamed" overclaim; design-only repo with all evidence rows pending. | All such words removed. "would be consistent with" replaces "certifies"; "matched-control contrast" replaces "proper score"; the non-gameability theorem is gone. New **Claim-Calibration table (§2.8)** maps every phrase to its allowed strength and the evidence that would license it. | §2.5, §2.6, §2.8, §3 |
| **B3.** Code cannot enforce the design: `passes_intervention_gate` is scalar-only; `TraceSegment` lacks edit budget, null-pool def, reference-run hash, matched-control provenance, evaluator hash → CIU is prose, not a reproducible estimand. | **§5 now specifies concrete schema/code CONTRACTS** that persist all of these as required fields, with a `validate_ciu_record` contract and a `CIURecord` dataclass. `passes_intervention_gate` is *wrapped* (not modified) by a gate that refuses to run unless the provenance fields are present and hash-checked. CIU becomes a **reproducible estimand** defined by persisted records. | §5 |
| **B4.** Reference-run patch/replay may measure *information injection*, not causal localization. | New **reference-construction ablation suite (§2.7, §4)**: `gold` / `placebo` / `neutral` / `wrong-but-plausible` / `no-gold`. A causal-localization claim requires the gold-reference effect to **exceed** the wrong-but-plausible and placebo effects by a pre-registered margin; otherwise the result is labelled *information injection*, not localization. | §2.7, §4 |
| **B5.** Detector-floor failure was only an "audit note"; baseline fairness unresolved (pending `implementation_commit`/`license`; output-only baselines can't natively select internal trace segments). | Detector-floor failure is now a **hard kill** (G5) that **invalidates the causal estimator**, not an audit note. Baseline fairness resolved via: (i) a `baseline_readiness` gate that blocks the run while any required baseline has `pending` commit/license; (ii) an **audited segment adapter** for output-only baselines, declared as part of the baseline and provenance-tracked. | §4 (G5), §5 |
| **B6.** Power story weak: 2 sizes × 2 datasets × 3 operators × granularities under Holm; 20 seeds insufficient. | New **§4.5 power analysis**: minimum-detectable-effect (MDE) under Holm at the realized family size, evaluator-noise bound feeding the variance, and a **replicate budget derived from MDE** (the `>= 20` floor is kept but the binding requirement is the power-derived count). Family size is shrunk by pre-registering operator/granularity as nuisance dimensions analysed at fixed primary settings, reducing the Holm family. | §4.5 |
| **B6′.** AR-LLM frozen as lead; reasoning/diffusion as transfer. | Unchanged from v1 (this was a *strength* the review credited). Re-stated and tightened. | §2.0, §2.9 |

Items the review credited as **strengths** and that v2 **preserves**: AR-led
single main claim; diffusion demoted to a transfer study that may fail without
killing the paper; the matched-budget control idea; the kill-gate structure with
numeric thresholds; the additive, server-gated discipline.

---

## 1. Defect being fixed

### 1.1 The audited problem, stated precisely

Two layers of defect remain in scope. The *first* layer (v1's D1–D3) is the
original scaffold weakness. The *second* layer (v2's D4–D6) is what the
adversarial review exposed in v1's own fix.

**D1 — Scope creep / underpowered design.** The pre-registration
(`docs/pre_registration.md`) fixes a *single* primary outcome
`targeted_delta - random_delta`, but the formal config implies it must hold
across a `paradigm x dataset x baseline x operator x granularity` grid. Under
Holm correction the per-test `alpha` collapses and the paired-bootstrap CI
half-width `~ z_{1-alpha/2} * sigma / sqrt(n)` widens, so `20` replicates cannot
power the advertised grid. (Carried from v1; now *quantified* in §4.5.)

**D2 — Diffusion novelty is thin and dominates risk.** A D-LLM denoising-trace
*detector* is "too close to TraceDet-style work" (arXiv:2510.01274) and TDGNet
(arXiv:2602.08048). If diffusion is co-equal in the main claim, a reviewer can
kill the paper on the diffusion section alone. (Carried from v1; fix preserved:
demote to transfer.)

**D3 — "Just another detector" risk under-specified at the estimand level.** The
gate `passes_intervention_gate(targeted_delta, random_delta, utility_drop)` is
defined on two caller-supplied scalars with no formal statement of the sampling
law of `targeted_delta`/`random_delta`, what makes the control *matched*, or why
a *predictive* (non-causal) segment cannot pass. (Carried from v1.)

**D4 — (NEW, from review B1) The v1 "non-gameability" proof was circular.**
v1's Theorem 2.5 defined a detector as a selector *independent of the true causal
effect `tau` given matching covariates*, then concluded detectors score `0`. But
real detectors choose segments by entropy, token position, claim-bearing-ness,
and suffix instability — covariates that are **plausibly correlated with `tau`**
(high-entropy or answer-adjacent spans may genuinely be more intervention-
sensitive). So the independence premise is empirically *false in general*, and
the conclusion was an artifact of the assumption. The defect is **overclaim by
unfalsifiable definition.**

**D5 — (NEW, from review B3) The estimand was prose, not a persisted object.**
Nothing in `schemas.py` or `metrics.py` persists the edit budget, the matched
null-pool definition `Pi`, the reference-run hash, the matched-control
provenance, or the evaluator hash. Two runs could both "pass `G1`" while
estimating *different* quantities. CIU was not reproducible.

**D6 — (NEW, from review B4/B5/B6) Three unguarded confounds.** (a) Reference-run
patch/replay can inject information rather than localize a cause; (b) output-only
baselines cannot natively select internal segments, so the comparison is unfair
unless an audited adapter is declared; (c) the power story is asserted, not
computed.

### 1.2 Formal symptom of D3/D4

Let `S` be a candidate segment, `Y in {0,1}` final-answer factuality, `do(I)` an
intervention on `S`. The scaffold scores `Delta(S) = E[Y | do(I_S)] - E[Y |
no_op]`. A pure detector selects `S` to maximize `Corr(feature(S), 1 - Y)`. The
v1 redesign tried to *prove* this cannot pass; v2 instead makes it a
**measurable contrast with a stated null** and an **empirical test** of whether
detector-selected segments clear it. Nothing is asserted to be impossible; the
design is built so that, if detectors do clear the gate, the *causal*
interpretation is invalidated rather than the result celebrated (§4 G5).

---

## 2. New method — Causal Intervention-Usefulness (CIU)

### 2.0 One-paragraph statement

We freeze **autoregressive LLM generation** as the lead paradigm and define a
single contribution: a segment `S` of a generation trace is *intervention-useful*
iff a **reference intervention** on `S` moves the answer's factuality more than a
**budget-matched random-segment intervention**, and does so **without paying it
back in utility**, **and** the gain is **larger under a correct (gold) reference
than under a wrong-but-plausible or placebo reference**. We name the estimator
the **Causal Intervention-Usefulness (CIU)** statistic `\hat U`. We prove that
the matched-control contrast is an **unbiased estimator of a causal contrast
under explicit, testable assumptions** (Identification Lemma 2.5) — *not* that
detectors are provably ungameable. Whether detectors clear the gate is an
**empirical prediction** (§3) the experiment can falsify. Reasoning-trace and
diffusion-LM become a **transfer study**: we test whether the *same* CIU
operator, ported unchanged, still selects intervention-useful segments — a
falsifiable generalization claim, not a parallel main result.

### 2.1 Trace, segments, granularity (reuses `schemas.py`)

A generation produces a token sequence `x_{1:T}` with hidden states
`h_{1:T}^{(l)}` at layers `l` and next-token distributions `p_t`. We reuse the
existing `TraceStep` (a scored `text_span`) and `TraceSegment` (a set of
`step_ids` + `selector` + `intervention`) dataclasses, and **extend** them
additively (§5) with the provenance fields the estimand requires.

**Granularity (lead = autoregressive).** A *segment* is a contiguous window of
`w` decoding positions, `S = [a, b]` with `b - a + 1 = w`, plus its layerwise
residual-stream slice `H_S = { h_t^{(l)} : t in [a,b], l in L_patch }`. To
control the multiple-comparison family (§4.5), **granularity and operator are
pre-registered as nuisance dimensions analysed at a single primary setting**, and
swept only as secondary robustness curves:

- **primary granularity:** claim span `S` = the minimal token span an
  atomic-claim extractor attributes a check-worthy proposition to (semantic);
- **secondary (robustness only):** token window `w in {1, 4, 8}`;
- **transfer only:** reasoning-step window (one chain-of-thought step).

Trace extraction (deterministic, hashable, server-side only):
`(1)` temperature-fixed decode under frozen `prompt_template_hash`;
`(2)` record `x_{1:T}, p_{1:T}, h^{(L_patch)}`; `(3)` atomic-claim segmentation;
`(4)` emit a `TraceManifest` with `split_hash` and `server_authorized=false`.
This is the `schemas.trace_manifest.schema.json` contract.

### 2.2 Intervention operators (reuses `intervention_protocol.md` IDs)

For a segment `S` we define three operators, paired with the existing control IDs
`random_non_causal_segment`, `shuffled_trace_segment`, `no_op_intervention`.

1. **Counterfactual masking `M_S`** (`mask`). Renormalise attention from
   positions `> b` to positions in `[a,b]` to zero, forcing downstream tokens to
   be produced as if `S` carried no evidence. No external reference; isolates
   *necessity*. (Reference-free — immune to the information-injection confound of
   D6a; this is why `mask` is the **primary operator** in v2.)

2. **Activation patching with a reference `P_S^{ref}`** (`patch`). Given a paired
   *reference run* on the same question, copy its residual stream into `H_S`:
   `h_t^{(l)} <- (1-alpha) h_t^{(l)} + alpha * h_t^{(l),ref}`, `t in [a,b]`,
   `l in L_patch`, `alpha in (0,1]`. Isolates *sufficiency* of a reference state.
   **The reference type is an experimental factor** (§2.7), not fixed to gold.

3. **Replay-from-checkpoint `R_S`** (`replay`). Roll back to the decoder state
   before position `a`, re-decode `[a,b]` under a reference policy, then free-run
   the suffix. Isolates *trajectory* correction. Reference type is again a factor.

Each operator preserves an explicit **edit budget** `c(I)` (number of altered
positions / patched coordinates), persisted per record (§5). The random control
must spend the *same* budget (§2.3). Invalid interventions (NaN logits, decode
failure) are logged with reason codes per `intervention_protocol.md`; `> 5%`
invalid → diagnostic only, and the invalid count is a **persisted field** of the
CIU record so the estimand denominator is reproducible.

### 2.3 The matched random-segment control (the identification design for D3)

Let `I_S^{theta}` be operator `theta in {mask, patch, replay}` on segment `S`
with budget `c(I_S^{theta}) = k`. Define the **matched random control** `\tilde S`
drawn from the *budget-matched, position-stratified* null:

```
\tilde S ~ Pi(S) :=  Uniform{ S' :  c(I_{S'}^{theta}) = k,
                                    len(S') = len(S),
                                    layer-set L_patch identical,
                                    S' disjoint from S,
                                    reference run identical (same ref_hash),
                                    same example x_i }
```

`\tilde S` is matched on operator, edit budget, length, layers, reference, and
example; only *which positions are edited* varies. This is the
`random_non_causal_segment` control made precise and **persisted** (the realized
`Pi` is serialized by its `null_pool_hash`, §5, so the estimand is reproducible —
fixing D5). `shuffled_trace_segment` additionally permutes within-segment
ordering; `no_op_intervention` sets `alpha=0` / `k=0`.

**What the matching can and cannot buy (honest statement, fixing D4).** Matching
removes *budget/length/layer/reference/example* as explanations of any contrast.
It does **not** remove the possibility that the selector's covariates (entropy,
position, claim-bearing-ness) are themselves correlated with the local causal
effect `tau`. That residual correlation is exactly the empirically open question;
the design *measures* it via the reference ablations (§2.7) and the detector-
floor test (§3, §4 G5) rather than *assuming* it away.

### 2.4 Per-example causal contrast and the CIU estimator

Let `Y in {0,1}` be evaluator factuality and `\tilde Y` the post-intervention
answer. Under operator `theta` and reference type `rho`:

```
delta_i(S; rho)        =  Y_i(do(I_S^{theta,rho}))                      - Y_i(no_op)
delta_i(\tilde S; rho) =  E_{\tilde S ~ Pi(S)}[ Y_i(do(I_{\tilde S}^{theta,rho})) ] - Y_i(no_op)
```

Per-example **causal-usefulness contrast**:

```
u_i(S; rho) = delta_i(S; rho) - delta_i(\tilde S; rho).
```

**CIU estimator** over `n` matched examples (paired bootstrap per
`statistical_analysis_plan.md`):

```
\hat U(selector; rho)  =  (1/n) sum_{i=1}^{n} u_i(S^*(x_i); rho),
```

where `S^*(x_i)` is the selector's segment on example `i`. The **utility guard**
uses the same controls on a fluency/accuracy score `Q in [0,1]`:

```
\hat D_util = (1/n) sum_i [ Q_i(no_op) - Q_i(do(I_{S^*}^{theta,rho})) ].
```

**Gate (unchanged thresholds, wraps `passes_intervention_gate`).** A causal
*usefulness* claim is licensed for the lead paradigm iff, at the primary operator
`theta=mask` and gold reference (or reference-free for `mask`):

```
passes_intervention_gate(targeted_delta = mean_i delta_i(S^*),
                         random_delta   = mean_i delta_i(\tilde S),
                         utility_drop   = \hat D_util,
                         min_margin     = 0.05,
                         max_utility_drop = 0.02)  ==  True
```

i.e. `\hat U >= 0.05` **and** `\hat D_util <= 0.02`, with the paired-bootstrap
95% CI of `\hat U` excluding `0.05` after Holm. No change to the gate function;
v2 supplies the *sampling law* (`Pi`), the *estimand* (`u_i`), the *reference
factor* (`rho`), and the *persisted provenance* (§5) that make the inputs an
identified, reproducible estimand.

### 2.5 Identification of the CIU contrast (the required proof, corrected)

> **This section replaces v1's Theorem 2.5 and Corollary 2.6.** We do *not* prove
> non-gameability. We prove a strictly weaker, defensible statement: the matched
> control yields an **unbiased estimate of a causal contrast under explicit,
> testable assumptions**, and we list the conditions and how each is tested.

**Estimand.** Define the **per-position average causal effect** of editing `S`,
relative to no-op, as
`tau(S) = E_i[ Y_i(do(I_S^{theta,rho})) - Y_i(no_op) ]`, and the **null-pool
mean** `bar tau(Pi) = E_{S' ~ Pi}[ tau(S') ]`. The CIU target is the contrast
`U(selector) = E_i[ tau(S^*(x_i)) ] - bar tau(Pi)`.

**Assumptions (each tagged TESTABLE / STRUCTURAL).**
- **(A1′, budget/covariate matching — STRUCTURAL, enforced by `Pi`).** `S` and
  `\tilde S` share operator, budget `k`, length, layers, reference run, and
  example. Enforced by construction (§2.3) and *checked* by the
  `null_pool_hash` contract (§5).
- **(A2, shared no-op — STRUCTURAL).** `Y_i(no_op)` is the same un-intervened run
  for `S` and `\tilde S`, so it cancels in `u_i`. Enforced by re-using one no-op
  record per example (`noop_run_hash`, §5).
- **(A3, evaluator pre-commitment — STRUCTURAL).** `Y, Q` come from an evaluator
  whose prompt and key are hashed before the run (`evaluator_hash`, §5;
  `intervention_protocol.md` leakage clause). `Y_i(·)` is a fixed measurable
  function of the intervened output.
- **(A4, control-pool unbiasedness for the null — TESTABLE).** The matched random
  draw is an unbiased estimate of the null-pool mean:
  `E[ delta_i(\tilde S) ] = bar tau(Pi)`. This holds by construction **iff** the
  *act of editing a random matched segment* carries no systematic effect beyond
  position identity. It can fail (positional bias near the answer; budget that is
  systematically easier to "spend" on some regions). **Test:** the placebo /
  shuffled / no-op controls in §2.7 and the detector-floor test in §3 directly
  probe A4; G5 (§4) makes A4 failure a hard invalidation.

**Lemma 2.5 (unbiasedness of the matched contrast).** Under (A1′), (A2), (A3),
(A4),
```
E[ \hat U(selector) ] = E_i[ tau(S^*(x_i)) ] - bar tau(Pi) = U(selector).
```
*Proof.* By (A2) the shared no-op cancels inside `u_i`, so
`E[u_i] = E[delta_i(S^*)] - E[delta_i(\tilde S)]`. By (A3) both terms are
expectations of a fixed measurable factuality function. By (A4)
`E[delta_i(\tilde S)] = bar tau(Pi)`. By definition `E[delta_i(S^*)] =
tau(S^*(x_i))` (averaged over `i`). Hence `E[\hat U] = E_i[tau(S^*(x_i))] -
bar tau(Pi)`. ∎

**What Lemma 2.5 does and does not give us.**
- It gives: `\hat U` is an **unbiased estimator of a genuine causal contrast**
  (selected-segment effect minus budget-matched null effect) — *provided A4
  holds*. A positive, CI-clearing `\hat U` therefore *would be consistent with*
  the selected segments carrying above-null causal effect.
- It does **not** give: that a detector-like selector must score `0`. Whether a
  detector's covariates correlate with `tau` (making `E_i[tau(S^*_det)] >
  bar tau`) is an **empirical** matter. v2 makes no theorem about it.

**Corollary 2.6 (what passing the gate would be consistent with — softened).**
Passing `G1` under the matched control of §2.3 **would be consistent with**
`Var_{S' ~ Pi}(tau) > 0` and the selector concentrating on the high-`tau` tail —
i.e. that editing the selected segments restores factuality beyond the matched
null. This is **stronger than an AUROC/detection statement only if A4 holds and
the detector-floor prediction (§3) is borne out**; if a pure detector also clears
the gate, the design treats A4 as suspect (G5) and withholds causal wording. The
utility guard `\hat D_util <= 0.02` would additionally be consistent with the
gain not being bought by answer degradation. (No "certifies"; no "proper score";
no "cannot be gamed".)

### 2.6 Algorithm box

```
Algorithm CIU-AR  (lead paradigm: autoregressive; design only, no run)
Input: frozen model f, split D (split_hash), evaluator E (evaluator_hash),
       operator theta in {mask*, patch, replay}, reference type rho, budget k,
       layers L_patch, selector pi_sel (method under test),
       replicate count R_rep >= R_power (see 4.5; floor 20).
Output: CIURecord{ \hat U, CI, \hat D_util, provenance hashes, invalid_count },
        gate verdict in {useful_candidate | diagnostic | invalidated}.

assert server.authorized == false                      # never run locally
for each seed/replicate, each example x_i in D_test:
  1. trace_i        <- extract_trace(f, x_i)            # TraceManifest, hashed
  2. Y0, Q0         <- E(no_op run)                     # noop_run_hash
  3. S*             <- pi_sel(trace_i)                  # method's segment
  4. Yt, Qt         <- E( f under do(I_{S*}^{theta,rho}) )   # targeted
  5. for r in 1..R:                                     # matched random control
       S~_r         <- sample Pi(S*)   # budget/len/layer/ref-matched, disjoint
       Yr_r         <- E( f under do(I_{S~_r}^{theta,rho}) )
     delta_rand_i   <- mean_r (Yr_r - Y0)               # estimates bar tau(Pi)
  6. delta_tgt_i    <- Yt - Y0 ;  u_i <- delta_tgt_i - delta_rand_i
     dutil_i        <- Q0 - Qt
collect {u_i}, {delta_tgt_i}, {delta_rand_i}, {dutil_i}, invalid_count
\hat U, CI         <- paired_bootstrap(u_i, B>=R_power, Holm-corrected)
record             <- CIURecord(... + null_pool_hash, ref_hash, evaluator_hash,
                                 noop_run_hash, edit_budget=k, invalid_count)
gate               <- passes_intervention_gate(mean(delta_tgt_i),
                          mean(delta_rand_i), mean(dutil_i),
                          min_margin=0.05, max_utility_drop=0.02)
verdict            <- ciu_gate(record, gate)   # 4: invalidated if G5/A4 fails
return record, verdict
```

### 2.7 Reference-construction ablations (fixing D6a / review B4)

Patch/replay can inject information rather than localize a cause. v2 makes the
**reference type `rho` a first-class experimental factor** with five levels, all
sharing budget, length, layers, example, and null pool:

| `rho` | Reference run definition | Purpose |
| --- | --- | --- |
| `gold` | reference run whose answer is evaluator-confirmed factual | the intended causal reference |
| `neutral` | no-evidence / prompt-only reference (no answer content) | controls for "any state change" |
| `placebo` | reference resampled from the *same* model on the *same* question with no factuality conditioning | controls for benign re-draw |
| `wrong_plausible` | reference run whose answer is fluent but evaluator-confirmed *non-factual* | controls for *information injection* — a wrong reference that still "stabilizes decoding" |
| `no_gold` | mask operator only, **no reference at all** (reference-free necessity) | upper-bound on reference-free signal |

**Pre-registered localization criterion (new).** A *causal-localization* claim
(as opposed to a mere *information-injection* effect) is licensed only if
```
\hat U(gold) - \hat U(wrong_plausible) >= 0.03   (Holm-corrected CI > 0)
AND \hat U(gold) - \hat U(placebo) >= 0.03.
```
If `\hat U(wrong_plausible)` is statistically indistinguishable from
`\hat U(gold)`, the patch/replay effect is reported as **information injection,
not localization**, and the causal-localization wording is dropped (the mask /
`no_gold` reference-free result still stands for *necessity*). This directly
answers B4: the gold-reference effect must *beat* a wrong-but-plausible reference,
not merely beat a budget-matched random position.

### 2.8 Claim-calibration table (fixing review B2 overclaim)

Every load-bearing phrase, its allowed strength, and the evidence that would
license it. Until that evidence exists (all rows currently `pending` per
`docs/claim_evidence_matrix.md`), only the design-level wording is used.

| Phrase | Status in v2 | Allowed only when |
| --- | --- | --- |
| "proper score" / "propriety" | **removed** | never (replaced by "unbiased matched contrast") |
| "certifies" | **removed** → "would be consistent with" | n/a |
| "cannot be gamed" / "non-gameable" | **removed** | only after detector-floor prediction (§3) is empirically borne out across datasets |
| "pure detectors score 0" | **removed** → empirical prediction (§3) | reported as observed, never as theorem |
| "causal localization" | gated | §2.7 localization criterion passes |
| "intervention-useful" | gated | G1 + G2 pass with power (§4.5) |
| "transfers across paradigms" | gated | G3 passes incl. diffusion |

### 2.9 Lead-vs-transfer split (preserved from v1; fixes D1 + D2)

- **Lead (main result):** autoregressive LLM only, **frozen as the lead
  paradigm**. One model family, the open-domain + multi-hop datasets, primary
  operator `mask`, full reference ablations on `patch`/`replay`. The `\hat U`
  gate must pass here for the paper to exist.
- **Transfer study (secondary):** port the *identical* CIU operator and gate to
  (i) reasoning-trace LRM and (ii) diffusion LM, gated on
  `heldout_taxonomy_retention >= 0.80`. This demotes TraceDet-adjacent diffusion
  work to "does our causal operator transfer?". Diffusion can **fail** without
  killing the paper.

---

## 3. Why this is NOT stitching — and the falsifiable prediction

**Single crisp scientific contribution.** A trace segment is scientifically
meaningful **iff intervening on it restores factuality beyond a budget-matched
random edit, by more than a wrong-but-plausible reference would** — and we give
the estimator `\hat U` plus the matched null `Pi` and reference factor `rho` that
make this an *identified, reproducible, falsifiable* quantity (Lemma 2.5), not a
stack of detector + graph + mitigation prompt. The novelty is the **estimand and
its identification design** (matched null + reference ablations + persisted
provenance), not any individual operator. No prior hallucination-tracing work
supplies a control-matched, reference-ablated intervention estimand with persisted
provenance and a pre-registered detector-floor kill.

**Sharper than the neighbours.**
- vs **TraceDet** (arXiv:2510.01274) / **TDGNet** (arXiv:2602.08048):
  detection-only on diffusion traces; no intervention, no causal estimand. CIU
  demotes diffusion to a transfer test and adds the intervention axis.
- vs **RACE** (arXiv:2506.04832): reasoning-consistency *detector*; predictive,
  not interventional. `delta_i(S)` is what RACE cannot produce.
- vs **semantic entropy** (Nature 2024) / **SelfCheckGPT** / **INSIDE**:
  output/sampling/internal-state *uncertainty*; CIU asks the orthogonal
  interventional question.

**Falsifiable prediction (pre-registered; the detector-floor as PREDICTION, not
theorem — fixing B1/B5).** *If* hallucination in autoregressive LLMs has
localizable causal segments **and assumption A4 holds**, *then*:
1. the CIU selector achieves `\hat U >= 0.05` (Holm-corrected paired-bootstrap CI
   lower bound `> 0.05`) with `\hat D_util <= 0.02` on `>= 2` lead datasets; and
2. **every output-only / sampling / detector baseline (scored on `\hat U` through
   the audited segment adapter of §5) has `\hat U` whose CI overlaps `0`** — this
   is the **detector-floor prediction**, tested by the random-segment control,
   *expected* but **not guaranteed**.

**This prediction can lose, and losing is informative.** If detector baselines
*also* clear `0.05`, we do **not** celebrate; per G5 (§4) the causal estimator is
**invalidated** (A4 is suspect: the matched null is not capturing the right
counterfactual, or detector covariates genuinely track `tau`), the project
downgrades to a detector-comparison diagnosis, and no causal wording is used. The
random-segment control is thus simultaneously the estimator and its own falsifier.

---

## 4. Experiment design to confirm/kill (design only — NO runs)

**Datasets (lead uses the first two; reused from `data_and_evaluation_plan.md`).**
open-domain factual QA (TruthfulQA-style + TriviaQA-style); multi-hop QA
(HotpotQA-style) for trajectory-correction; hallucination benchmark
(HaluEval-style); diffusion-LM trace dataset (transfer only). Each needs license,
raw+processed hash, split, leakage check.

**Base models (pending user approval; AR frozen as lead).**
- Lead AR: one open-weights instruct family at **two sizes** (e.g. 7–9B and
  13–14B class) to test CIU is not a single-checkpoint artifact.
- Transfer reasoning: one open large-reasoning model.
- Transfer diffusion: one open diffusion-LM matching TraceDet's public protocol.
All checkpoints, prompt-template hashes, decoding params recorded per
`baseline_contract.md`.

**Baselines (reuse `baseline_registry.yaml`).** `random_segment`,
`output_entropy`, `semantic_entropy` (Nature 2024), `output_signature_detector`,
`reasoning_consistency_detector` (RACE), `selfcheckgpt`, `inside_detector`,
`diffusion_trace_detector` (TraceDet), `tdgnet_trace_detector`. **Every baseline
is also scored on `\hat U`**, not only AUROC — this is the matched-setup test
that probes the detector-floor prediction.

**Baseline fairness — audited segment adapter (fixing B5).** Output-only
baselines (entropy, semantic entropy, SelfCheckGPT) do **not** natively select
internal trace segments. To score them on `\hat U` we must wrap each with a
**declared segment adapter** that maps the baseline's output-level signal to a
segment selection (e.g. highest-entropy claim span). That adapter:
- is declared a **part of the baseline** in `baseline_registry.yaml`
  (`segment_adapter:` block), with its own provenance and `adapter_hash`;
- is held **identical across all output-only baselines** so the comparison tests
  the *signal*, not the adapter;
- is itself audited (its selection policy is logged), because a too-clever adapter
  could smuggle causal selection into a "detector".
Without an audited adapter, output-only baselines are reported AUROC-only and are
**not** placed in the `\hat U` table (no silent advantage/disadvantage, per the
existing fairness policy).

**Baseline-readiness gate (fixing B5).** The run is **blocked** while any required
baseline still has `implementation_commit: pending_before_server_run` or
`license: verify_before_run` in `baseline_registry.yaml`. This is enforced by a
`baseline_readiness` check (§5) at preflight, not left as a note.

**Metrics.** Detection: AUROC, AUPRC, FPR@95TPR. Localization: segment P/R/F1 vs
perturbation-derived labels. Intervention: `factuality_delta`,
`answer_accuracy_delta`, `utility_delta`, and the headline
`\hat U = targeted_delta - random_delta` (per `rho`). Efficiency: trace-extract
cost, intervention latency.

**Matched-budget protocol.** Same examples, prompts, decoding, split, evaluator
across methods. For `\hat U`, targeted and random arms additionally share
operator `theta`, budget `k`, length, layers `L_patch`, reference run `rho`
(§2.3). All methods consume `configs/seeds/paper_20.txt` (floor) extended to the
power-derived replicate count (§4.5).

**The kill-gate (pre-registered; thresholds unchanged; G5 is new/promoted).**
- **G1 causal usefulness (existential):** `\hat U >= 0.05` with Holm-corrected
  paired-bootstrap CI lower bound `> 0.05`, on `>= 2` lead datasets, at primary
  operator `mask`. *Fail ⇒ stop causal claim; project becomes a detector-
  comparison diagnosis.*
- **G2 utility harm:** `\hat D_util <= 0.02`. *Fail ⇒ reframe as abstention.*
- **G3 transfer:** held-out taxonomy retention `>= 0.80`. *Fail ⇒ drop
  cross-paradigm wording, keep AR-only.*
- **G4 paper evidence:** replicates `>= max(20, R_power)` (§4.5), paired tests,
  effect sizes, 95% CI. *Fail ⇒ `diagnostic` label only.*
- **G5 detector-floor = HARD invalidation of the causal estimator (NEW; was a
  v1 audit note).** If **any** non-causal baseline (scored via the audited
  adapter) has a `\hat U` CI whose lower bound `> 0` after Holm, then assumption
  A4 is rejected: the matched null does not isolate causal identity. *Action:
  the causal estimator is **invalidated**; `G1` may **not** be reported as a
  causal result regardless of CIU's own value; the run is relabelled a
  detector-comparison diagnosis.* (This is the teeth the review demanded.)
- **G6 localization vs injection (NEW, from §2.7):** for `patch`/`replay`,
  `\hat U(gold) - \hat U(wrong_plausible) >= 0.03` and
  `\hat U(gold) - \hat U(placebo) >= 0.03` (Holm CI `> 0`). *Fail ⇒ report as
  information injection, not localization; mask/`no_gold` necessity result
  unaffected.*

**Ablations.** remove intervention scoring (predictive-only); one operator only;
one paradigm only; replace targeted with random (positive control — must collapse
`\hat U` to ~`0`); remove utility gate; the **reference suite** of §2.7; sweep
`w`, `alpha`, `k`, `L_patch`, replay sample count (robustness curves only).

No run is scheduled. `server.authorized: false` remains. Server execution
requires the existing ARIS experiment-plan approval + explicit user command per
`docs/server_runbook.md`.

### 4.5 Power, minimum-detectable-effect, and evaluator-noise bounds (fixing B6)

The lead design has a multiple-comparison family and a binary, evaluator-scored
outcome. v2 makes the replicate budget a **derived** quantity, not the asserted
`20`.

**Family-size control first.** We shrink the Holm family by pre-registering
operator and granularity as nuisance dimensions analysed at a single primary
setting (mask, claim-span). The **primary family** is then:
`F = {2 sizes} x {2 datasets} x {G1, G2} = 8` confirmatory tests (transfer G3 and
G5 are evaluated in their own families). Holm at family-wise `alpha = 0.05` over
`m = 8` gives a worst-case per-test `alpha_1 = 0.05/8 ≈ 0.00625`.

**Evaluator-noise bound feeds the variance.** `Y in {0,1}` is produced by a
hashed evaluator with finite agreement rate `kappa` against human/gold labels.
Per-example label noise inflates `Var(u_i)`; we bound the *usable* effect by
`|U|_eff <= |U| * (2*kappa - 1)` (attenuation under symmetric label noise), and
require the evaluator's measured `kappa` (on a held-out audited subset) to be
reported as a **persisted field** (`evaluator_kappa`) that lower-bounds CIU
precision. We will not claim `\hat U` precision finer than the evaluator's own
agreement interval (stated limitation, §6).

**MDE / replicate count (formula evaluation, not evidence).** For a paired binary
contrast with per-example contrast SD `sigma_u`, the paired-bootstrap CI
half-width is approximately `h ≈ z_{1-alpha_1/2} * sigma_u / sqrt(n)`. To make
the gate (`CI lower bound > 0.05`) detectable at a true effect `U_true = 0.08`
(margin `0.05` + a `0.03` buffer matching the localization margin) we need
`h <= U_true - 0.05 = 0.03`, i.e.
```
n  >=  ( z_{1-alpha_1/2} * sigma_u / 0.03 )^2.
```
*Formula evaluation, not evidence:* with the Holm per-test `alpha_1 ≈ 0.00625`
(`z ≈ 2.74`) and a conservative bounded-contrast SD `sigma_u ≈ 0.35`
(the contrast `u_i = delta_tgt - delta_rand` lies in `[-1,1]`; `0.35` is a
plausible upper-mid estimate to be **re-estimated from the validation split
before the test split is unlocked**), this gives
`n >= (2.74 * 0.35 / 0.03)^2 ≈ (31.97)^2 ≈ 1022` matched examples per cell. This
arithmetic is a *placeholder pending the validation-split variance estimate*; it
already shows the binding constraint is the **per-cell example count (order 10^3),
not the seed count** — the `20` figure governs only the bootstrap/seed dimension,
not statistical power. `R_power` is therefore set from this formula at lock time;
`G4` requires `>= max(20, R_power)` replicates and the realized per-cell `n` to
meet the MDE computed from the *validation-split* `sigma_u`. If the budget cannot
fund the MDE-required `n`, the lead shrinks further (one size or one dataset)
rather than reporting an underpowered grid.

---

## 5. What changes in code vs the current scaffold

All changes are **additive and uncommitted**. Nothing existing is modified or
deleted by this document. The table below is the *proposed* future diff, to be
implemented only after design approval. **The point of this section (fixing B3)
is that the identification design becomes an enforced CONTRACT, not prose:** CIU
is defined by a persisted `CIURecord` whose required provenance fields make the
estimand reproducible, and a gate that refuses to license a causal reading unless
those fields are present, hash-consistent, and pass G5/G6.

**Files this redesign ADDS now:**
- `docs/redesign/REDESIGN_v2.md` (this file). *Only file written.*

**Files a future additive implementation would ADD (not done here):**
- `src/tracecausal/ciu.py` — `causal_usefulness(deltas_targeted, deltas_random)`,
  `matched_random_null(...)` sampler spec, `paired_bootstrap_U` with Holm, and
  `ciu_gate(record, scalar_gate)` that returns
  `{useful_candidate | diagnostic | invalidated}` and **invalidates** on G5/G6
  failure. Wraps, does not replace, `metrics.passes_intervention_gate`.
- `src/tracecausal/operators.py` — pure-Python *interfaces/contracts* for `mask`,
  `patch`, `replay`, `no_op` with budget accounting `c(I)`; no model calls.
- `configs/experiments/redesign_v2_ar_lead.yaml` — lead AR config: 1 family x 2
  sizes, 2 datasets, primary operator `mask`, reference suite, `gates` block
  copied verbatim (`0.05/0.02/0.80`) plus `detector_floor_invalidation: true`
  (G5) and `localization_margin: 0.03` (G6), `server.authorized: false`,
  `seeds.paper_minimum: 20`, `replicates.power_derived: true`.
- `configs/experiments/redesign_v2_transfer.yaml` — reasoning + diffusion
  transfer config, gated on `heldout_taxonomy_retention: 0.80`.
- `tests/test_ciu_estimator.py` — asserts the *positive control* numerically (a
  random-targeted selector yields `\hat U ≈ 0`); asserts the gate **invalidates**
  when a synthetic detector baseline clears `0` (G5 path); asserts budget/length/
  layer/reference mismatch **raises**; asserts missing provenance fields
  **raise**. (No claim that a detector *must* score 0 — only that the gate
  *invalidates the causal reading* when one clears the floor.)
- `tests/test_ciu_contract.py` — round-trips a `CIURecord` and checks
  `validate_ciu_record` rejects records missing any required hash.

**Schema/code CONTRACTS the implementation would ADD (the enforced estimand):**
A new `CIURecord` (additive dataclass) persists the estimand's provenance so it
is reproducible:
```
CIURecord(
  selector_id, operator, reference_type,        # which estimand
  edit_budget: int,                             # k, must match control
  null_pool_hash: str,                          # serialized Pi (A1')
  noop_run_hash: str,                           # shared no-op (A2)
  evaluator_hash: str, evaluator_kappa: float,  # A3 + noise bound (4.5)
  ref_hash: str,                                # reference run identity (2.7)
  matched_control_provenance: tuple[str, ...],  # the R drawn \tilde S ids
  invalid_count: int, n_examples: int,          # estimand denominator
  u_hat: float, ci_low: float, ci_high: float,
  d_util: float, server_authorized: bool = False,
)
```
with `validate_ciu_record(record) -> list[str]` returning violations if any hash
is empty, if `edit_budget` differs between targeted and control provenance, if
`server_authorized` is True, or if `n_examples` is below the MDE-required count.

**Files a future implementation would EXTEND (additively, back-compatible):**
- `src/tracecausal/schemas.py` — `TraceSegment` gains optional
  `edit_budget: int | None = None`, `reference_hash: str | None = None`,
  `null_pool_hash: str | None = None` (defaults preserve back-compat).
- `src/tracecausal/metrics.py` — **unchanged signature**;
  `passes_intervention_gate` is *wrapped* by `ciu_gate`, never edited.
- `configs/baselines/baseline_registry.yaml` — add `ciu_scored: true` and an
  optional `segment_adapter:` block (with `adapter_hash`) per output-only
  baseline; the audited adapter is identical across them. No baseline removed.
- `src/tracecausal/contracts.py` — add a `baseline_readiness(registry)` check
  that flags any required baseline with `pending_before_server_run` /
  `verify_before_run`, used at preflight (blocks the run, fixing B5).

**Contract invariants kept:** `validate_manifest` still requires
`server.authorized == false`, `>= 3` baselines, `>= 20` paper seeds (floor);
`passes_intervention_gate` signature and thresholds unchanged; all `required_docs`
paths in `contracts.py` continue to exist (this file is additional).

No `git commit`, no `git push`, no run. (Repo currently has no `.git`; nothing to
commit regardless.)

---

## 6. Open risks + honest limitations

- **(A4) is now the load-bearing, TESTABLE assumption — and its failure is a hard
  kill, not a footnote.** If masking/patching a random matched segment carries a
  systematic effect beyond position identity (positional bias near the answer,
  budget easier to spend on some regions), `bar tau(Pi)` is biased and a detector
  could leak a positive `\hat U`. v2 does not assume this away (the v1 error);
  G5 (§4) **invalidates the causal estimator** if any detector baseline clears
  the floor. The detector-floor is a *prediction we can lose*, not a theorem.
- **Reference-run dependence / information injection.** `patch`/`replay` can
  inject information rather than localize a cause. The reference suite (§2.7) and
  G6 quantify this: a gold reference must beat a wrong-but-plausible one by a
  pre-registered margin, else we report *information injection, not
  localization*. The reference-free `mask` operator is the primary, injection-
  immune result.
- **Baseline-adapter risk.** Scoring output-only baselines on `\hat U` requires a
  segment adapter that is itself a modelling choice. We hold it identical across
  baselines and audit it; a too-clever adapter would *help* detectors clear the
  floor, which under G5 invalidates *our* causal claim — so the incentive is to
  keep the adapter neutral, and we report adapter provenance (`adapter_hash`).
- **Evaluator is the ground truth of `Y`, and bounds `\hat U` precision.** We
  report `evaluator_kappa` and attenuate the usable effect by `(2*kappa - 1)`
  (§4.5); we will not claim precision finer than the evaluator's agreement
  interval.
- **Power may exceed budget.** The MDE analysis (§4.5) suggests per-cell example
  counts of order `10^3`; if the budget cannot fund this, the lead shrinks (one
  size or one dataset) rather than reporting an underpowered grid. The `20`
  figure governs only the seed/bootstrap dimension, not statistical power.
- **Diffusion transfer may fail.** Acceptable by design (G3); the cross-paradigm
  story is conditional. We will not write "across all generation paradigms"
  unless G3 passes for diffusion.
- **Replay is expensive.** Roll-back + re-decode multiplies cost; CIU may be an
  *audit-time* tool, framed as offline causal auditing, not a live decoder.
- **Scope of the claim.** Per `AGENTS.md`, we never strengthen "causal process
  segments are intervention-useful" to "all hallucinations are causally
  explained." A CI-clearing `\hat U` *would be consistent with* the existence and
  concentration of intervention-useful segments, not coverage of every
  hallucination.
- **No empirical evidence exists.** Everything above is design + theory; the only
  numeric work is the §4.5 MDE arithmetic, explicitly labelled *formula
  evaluation, not evidence*. Every claim-evidence row in
  `docs/claim_evidence_matrix.md` and `docs/paper_claims_status.md` remains
  `pending`; this document changes no status label and authorizes no run.
