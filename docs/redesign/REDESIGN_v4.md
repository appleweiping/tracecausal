# REDESIGN v4 — tracecausal

Status: `design_frozen_phase2_implemented`. **No server run authorized;
`server.authorized: false` is preserved.** This is a **Stage-1 Registered Report
grade design** whose method, theory, and pre-registered analysis plan are now
**locked**, and whose Phase-2 identification machinery is **implemented and frozen
in `src/` with a passing pure-Python unit harness** (no model, no GPU, no run). No
experiment, training, model load, or GPU/heavy-CPU job is run on the server or this
local PC. The only numerics are closed-form / quadrature checks, each labelled
**"formula evaluation, not evidence."** No number here is empirical; the deferred
GPU run remains the sole source of empirical numbers and is unauthorized.

**Closure state (what is now DONE, vs. the deferred run).**
- *Method + analysis plan:* **frozen** (this document; `configs/experiments/redesign_v4_ar_lead.yaml`).
- *Theory:* **discharged where provable** — the propriety/curvature identity
  (§2.11, Theorem in `paper/main.tex`), the assumption-free residual-estimand
  tautology (Prop. 2.5a), and the identification lemma (Lemma 2.5) are proved;
  A4★ is an explicit *testable assumption* with a positive probe, not a theorem,
  by design.
- *Phase-2 code (the §5 plan):* **implemented + frozen, uncommitted** —
  `src/tracecausal/ciu.py` (`CIURecord`, `validate_ciu_record`, `ciu_gate`,
  `leakage_slope_regression`, `ood_deflation`), `src/tracecausal/nuisance.py`
  (`estimate_sigma_u`, `estimate_kappa`, `r_power`, `u_target`, `pool_inflation`,
  the deterministic κ-fallback), `src/tracecausal/oracle_gen.py` (the graded
  Axis P/M/D fixtures), `src/tracecausal/nullpool.py` (proximity-stratified `Pi_i`
  + `null_pool_hash`), with the consolidated harness `tests/test_ciu_nulldata.py`
  **passing** (pure-Python, no GPU). The `configs/experiments/redesign_v4_ar_lead.yaml`
  freeze pins the AR lead (Qwen2.5-7B-Instruct + Llama-3.1-8B-Instruct) and lead
  datasets (TriviaQA + HotpotQA).
- *Paper:* a Stage-1 skeleton is written at `paper/main.tex` (+ `paper/references.bib`);
  every results table/figure is a `DATA_NEEDED` placeholder (no fabricated numbers),
  not yet compiled.
- *Empirical results:* **none** — every claim-evidence row remains `pending`;
  `server.authorized: false`; no `git commit`.

This document is **additive**. It ADDs `docs/redesign/REDESIGN_v4.md` only. It
does not modify or delete `REDESIGN_v1.md`, `REDESIGN_v2.md`, `REDESIGN_v3.md`,
any `src/`, `configs/`, `docs/`, or schema file. It **extends** v3's public
contracts and design objects additively; it does not change the meaning of any
existing public API. It reuses the existing governance (`AGENTS.md`,
`docs/experiment_protocol.md`, `docs/intervention_protocol.md`,
`docs/baseline_contract.md`, `docs/pre_registration.md`,
`docs/statistical_analysis_plan.md`, `docs/compute_budget.md`), the code
contracts (`src/tracecausal/metrics.py::passes_intervention_gate`,
`src/tracecausal/schemas.py`, `src/tracecausal/contracts.py`), the
`configs/baselines/baseline_registry.yaml` registry, and the
`configs/seeds/paper_20.txt` seed manifest. No pre-registered gate is weakened:
margin `0.05`, utility drop `0.02`, transfer retention `0.80`, seeds `>= 20` as a
**floor**. v4 *tightens* the identification probes, *pre-registers* the nuisance
re-estimation that v3 left as a forward promise, *corrects a derivative-labelling
slip* in §2.11, and *closes* the budget/power inequality with an exhibited
feasible point. It relaxes nothing.

**Carry-forward.** Everything substantive in v3 — the AR-led single claim;
diffusion demoted to transfer; the per-example matched null `Pi_i`; Prop. 2.5a
(assumption-free tautology) vs Assumption A4★ (the one substantive causal claim);
the deletion of v2's invalid G5 and its replacement by the detector-beating
novelty gate G5′; the operational `mask` spec; the proper-scoring evaluator
mechanism; gates G1–G8; the CIU `Phase-2` contract; additive/server-gated
discipline — is **preserved unchanged unless explicitly revised below**. v4
sharpens five specific surfaces that the GPT-5.5 review (7/10, "major revision;
the registered-report prose is ahead of the executable preregistration") and the
ML-research finalization pass identified as still soft: the positive-ID probe was
*present but easy* (only the clean oracle), the nuisance re-estimation
(`sigma_u`, `kappa`) was *promised but not pre-specified as an estimator + CI +
freeze rule*, §2.11's stationarity *prose mislabelled the second-order condition*,
mask OOD handling was *a binary SHAM pass/fail* rather than a dose-response, and
the power/budget reconciliation *named an inequality but exhibited no satisfying
point*.

---

## Changelog vs v3

Five required fixes, each mapped to a concrete change and its section. Verified-
correct v3 parts are preserved (carry-forward note above).

| v3 → v4 fix | What changes | Where |
| --- | --- | --- |
| **(a) Oracle/G7 probe was binary-easy.** v3's §2.10(i) oracle only exhibited the *clean* regime (`tau=1` planted span, `tau=0` everywhere, `bar tau_i(Pi)=0` exactly) where the estimator trivially passes. A reviewer cannot see whether `\hat U` *degrades correctly* as identification erodes. | **Replaced by a GRADED oracle family (§2.10G).** Three controlled stress axes — **partial-leakage** (inert spans carry a tunable `tau_inert = beta*Delta_pos` up to and past the `0.03` margin), **multi-cause** (the planted effect is split across `m_c` spans so a single-span selector under-recovers), and **distractor-span** (a high-detector-signal but causally inert span sweeps the false-positive axis). The probe pre-registers the **expected monotone degradation curve** of `\hat U` and the **planted crossing point** at `beta*Delta_pos = 0.03`, so G7 demonstrates correct degradation, not just the easy pass. G7 is restated to gate on the **upper CI** of the leakage bound (see fix (c)). | §2.10G, §4 G7 |
| **(b) `sigma_u` / `kappa` re-estimation was promised, not specified.** v3 said both are "re-estimated from the validation split" but gave no estimator, CI, minimum validation `n`, freeze rule, or interaction with the proximity-stratified `Pi` sampler (which shrinks the per-example pool and can inflate the random-arm variance). | **§4.6 pre-registers both as leakage-safe procedures:** named estimators (paired-contrast variance for `sigma_u`; Cohen/Fleiss agreement for `kappa`), their CIs (bootstrap for `sigma_u`, Wilson/Fleiss-CI for `kappa`), a **minimum validation `n_val`** with a stated rule, a **freeze-at-lock** rule (validation-only; test split stays sealed; no re-estimation after unlock), and an explicit **pool-shrinkage inflation factor** `(1 + 1/\bar m_pool)` that the proximity stratifier induces, propagated into `R_power`. | §4.6 |
| **(c) §2.11 stationarity wording mislabelled the second-order condition.** The propriety algebra is correct, but v3 wrote "since `G''=2>0` this stationary point is the **unique maximum**." `G''>0` makes the *generator* convex (which is what makes the rule proper); it is **not** the second-order condition for the *expected score* `E[S]` having a maximum. The correct statement: first-order condition `d/dp E[S] = 2(q-p) = 0`; **strict concavity of `E[S]`** with curvature `d^2/dp^2 E[S] = -G''(p) = -2 < 0`. Only the derivative *labels* were wrong. | **§2.11 reworded** (FOC `2(q-p)=0`; SOC `-G''=-2<0` ⇒ strict concavity ⇒ unique max). The proper-scoring/answer-adjacent regression yielding `beta` is **pre-registered with a CI**, and **G7 is gated on the UPPER CI** of the leakage bound `beta*Delta_pos`. | §2.11, §2.10G, §4 G7 |
| **(d) Mask OOD handling was a binary SHAM check.** v3's G8 only asked "is SHAM-MASK `\hat U` CI overlapping 0?" — a single pass/fail that cannot tell a small genuine effect from a small operator artifact, nor detect a *dose-dependent* OOD footprint. | **§2.12 adds a `displaced_mass`-stratified analysis:** on **inert spans only**, calibrate the operator's own footprint `\hat U_inert` as a function of `displaced_mass` (the post-mask softmax mass that would have gone to `[a,b]`), binned. A **monotone `\hat U_inert` vs `displaced_mass` slope with CI** is the OOD dose-response; G8 is extended to gate on **both** the SHAM null **and** a bounded calibrated slope, and the real-span `\hat U` is **deflated** by the calibrated operator footprint at matched `displaced_mass`. | §2.12, §4 G8 |
| **(e) Power vs compute named an inequality but exhibited no satisfying point.** v3 left `c_fwd` and the affordable `n` "filled at lock," so a reviewer could not see that the MDE + attenuation + budget system is *jointly satisfiable at all*. | **§4.7 pre-registers a `c_fwd` ceiling** `c_fwd <= c_fwd_max` and **exhibits at least one concrete `(n, cells, kappa)` point** that satisfies the MDE inequality, the `(2kappa-1)` attenuation requirement, and the GPU-hour budget simultaneously (formula evaluation, not evidence). The locked config still records the measured `c_fwd`; v4 additionally pins the *existence* of a feasible operating point and the decision order if the measured `c_fwd` exceeds the ceiling. | §4.7 |

**Net effect.** v3 already fixed the score-capping logical flaw (deleting v2's
invalid G5). The remaining 7/10 gap was "prose ahead of executable
preregistration" plus four softness surfaces. v4 hardens exactly those: the
positive-ID probe now *demonstrates graded degradation*, the nuisance estimators
are *fully pre-specified and leakage-safe*, the §2.11 SOC is *correctly labelled*,
mask OOD is a *calibrated dose-response not a binary check*, and the power/budget
system is *exhibited as jointly satisfiable*. No claim is strengthened beyond what
the design supports; no gate is loosened.

---

## 1. Context — what v4 inherits and what it sharpens

### 1.1 Inherited unchanged from v3 (not reopened)

AR generation frozen as the lead paradigm; the CIU estimator `\hat U` and its
per-example matched null `Pi_i(S)`; Prop. 2.5a (assumption-free) split from
Assumption A4★ (the single substantive causal assumption); deletion of v2's
invalid G5 and its replacement by the detector-beating novelty gate G5′; the
operational `mask`/`patch`/`replay`/`no_op` operators with `keep_absolute`
position policy and `displaced_mass` accounting; the strictly-proper
Bregman/Savage evaluator-elicitation rule (closing the gamed-label loophole, =
A3); the reference-construction ablations (G6); gates G1–G8; the four-sampling-
level statistical separation (examples `i`, seeds `s`, intervention-repeats
`R_int`, bootstrap `B_boot`); the CIU Phase-2 contract (`CIURecord`,
`validate_ciu_record`, `ciu_gate`, `baseline_readiness`); and the
additive/uncommitted/server-gated discipline. The claim-calibration table (v3
§2.8) is carried forward verbatim with the two row-edits in §2.13.

### 1.2 The five residual softness surfaces v4 fixes

- **R1 (fix a).** The positive identification probe existed but only in its easy
  regime; it could not exhibit *graceful degradation*, the property a top-venue
  reviewer demands of any estimator claimed to be identified.
- **R2 (fix b).** The two nuisance parameters that set the power budget
  (`sigma_u`) and the attenuation correction (`kappa`) were re-estimated "from
  validation" with no estimator, no CI, no minimum `n_val`, no freeze rule, and
  no accounting for the variance inflation the proximity stratifier induces.
- **R3 (fix c).** §2.11's second-order reasoning was mislabelled: it invoked the
  generator's convexity `G''>0` where the expected-score concavity `-G''<0` is
  the actual second-order condition. The conclusion (`p=q` is the unique max) is
  correct; the justification text was wrong, which a careful reviewer flags.
- **R4 (fix d).** Mask OOD was a binary SHAM pass/fail; it could not separate a
  small genuine necessity signal from a small operator artifact, nor expose a
  dose-dependent OOD footprint.
- **R5 (fix e).** The power/compute reconciliation pre-committed an inequality and
  a decision order but exhibited no point proving the system is satisfiable.

---

## 2. Method deltas (v4)

The method is v3's CIU, unchanged in its core. v4 adds/edits the subsections
below. All other §2 subsections of v3 (2.0–2.9, 2.13's antecedents) stand.

### 2.10G Graded positive-identification probe family (fixing a)

> **Replaces v3 §2.10(i)'s single clean oracle with a graded family.** v3 §2.10(ii)
> (the analytic `beta*Delta_pos` bound) is retained and upgraded in §2.11 / G7 to
> gate on the bound's UPPER CI. All oracle runs are Phase-2 **unit fixtures** on a
> small frozen templated generator; they produce **no paper numbers** and use
> synthetic/oracle labels, never server runs.

The probe must show the estimator **degrades correctly**, not merely passes the
easy case. We pre-register three controlled stress axes, each a one-parameter
sweep around the clean oracle, with a **pre-registered expected `\hat U` curve**
and a **planted crossing point** at the localization margin `0.03`.

**Clean baseline (the v3 oracle, retained as the sweep origin).** Templated
generator `f_oracle`; a designated claim span `S_designated` is, by construction,
the sole determinant of factuality; all other spans are causally inert *given the
prompt*; `tau_i(S_designated)=1`, `tau_i(S')=0` for inert `S'`, so
`bar tau_i(Pi)=0` exactly and A4★ holds by construction. Expected:
`\hat U(oracle_selector)\to 1`, `\hat U(random_selector)\to 0`,
`\hat U(detector_on_inert_feature)\to 0`.

**Axis P (partial-leakage).** Inject a controlled positional-leakage effect into
the inert spans: an inert span at proximity gap `Delta_pos` from the answer is
given a planted effect `tau_inert(Delta_pos) = beta * Delta_pos`, sweeping
`beta*Delta_pos \in \{0, 0.01, 0.02, 0.03, 0.04, 0.06\}` (i.e. up to and *past*
the `0.03` margin). Because `bar tau_i(Pi)` now picks up the leakage term, the
matched-null arm is biased upward and `\hat U` for a *correct* selector is
deflated by exactly `beta*Delta_pos` (the §2.11 bound, here installed by
construction). **Pre-registered expected curve:**
`\hat U_clean(oracle) = 1 - tau_inert` to first order; the **crossing of the
`0.03` gate floor** occurs when the residual leakage the proximity stratifier
fails to absorb reaches `0.03`. This axis demonstrates the estimator *degrades
linearly and crosses the gate at the predicted point*, and is the empirical
companion to the analytic bound of §2.11.

**Axis M (multi-cause).** Split the planted causal effect across `m_c \in
\{1,2,3,4\}` co-equal spans, each carrying `tau = 1/m_c`. A single-span selector
recovers only `\hat U \approx 1/m_c` (under-recovery); a top-`m_c` selector
recovers `\to 1`. **Pre-registered expected curve:** single-span
`\hat U \approx 1/m_c` (monotone decreasing in `m_c`), top-`k` selector flat near
`1`. This demonstrates the estimator reports *honest under-recovery* when the
true causal structure is distributed, rather than spuriously saturating.

**Axis D (distractor-span).** Add a span that is *causally inert*
(`tau = 0`) but carries a **high detector signal** (planted high entropy /
high claim-bearing-ness / high RACE-style consistency feature). A naive
detector selects it; its `\hat U \to 0` (it is inert), while the causal selector
ignores it. Sweep the distractor's detector-signal strength
`d \in \{0.5,1,2,4\}\times` the true span's. **Pre-registered expectation:**
detector-on-distractor `\hat U` stays at `0` regardless of `d`; the proposed
selector's `\hat U` stays at `1`; G5′'s selector-minus-detector difference stays
`\geq 0.03`. This is the *false-positive* axis: it shows `\hat U` is not fooled by
detector-salient-but-inert spans, the exact failure mode the novelty gate must
survive.

**What the graded family buys.** Each axis has a **monotone, pre-registered
`\hat U` response** with a **planted crossing point**, so G7 (below) checks not
only that the estimator passes the clean case but that it *tracks the planted
ground truth as identification erodes* — partial leakage (degrade with `beta`),
distributed cause (under-recover with `m_c`), and detector-salient distractors
(stay at zero). A flat or non-monotone response on any axis is itself a G7
failure: it would mean `\hat U` is insensitive to the very structure A4★ governs.

### 2.11 Proper-scoring proof — corrected stationarity wording (fixing c)

> **Labelled: closed-form check, not experimental evidence.** Verified by stdlib
> arithmetic in-session (exact polynomial identity over sampled `p\in[0,1]`,
> `y\in\{0,1\}`; grid-argmax for the stationary point). The propriety **algebra is
> unchanged from v3 and correct**; v4 only **fixes the second-order-condition
> label**, which v3 stated as `G''=2>0` (the generator's convexity) where the
> expected-score concavity `-G''=-2<0` is the actual SOC.

**Claim 1 (exact score identity — unchanged from v3, correct).** For
`G_B(p)=1-p+p^2`, `G_B'(p)=2p-1`, the Savage/Bregman binary score
`S(p,y)=G_B(p)+G_B'(p)(y-p)` equals `1-(p-y)^2`. *Proof.*
`S = (1-p+p^2) + (2p-1)(y-p) = 1 - p^2 + (2p-1)y`. At `y=0`: `S=1-p^2=1-(p-0)^2`.
At `y=1`: `S=1-p^2+2p-1=2p-p^2=1-(p-1)^2`. Since `y\in\{0,1\}`,
`S(p,y)=1-(p-y)^2` identically. ∎

**Claim 2 (strict propriety — CORRECTED derivative labels).** Under
`Y\sim Bernoulli(q)`, the expected score is
```
E[S(p,Y)] = G_B(p) + G_B'(p)(q - p).
```
**First-order condition (stationarity).** Differentiating, the
`G_B(p)` and `-G_B'(p)\,p` terms' derivatives combine so that
```
d/dp E[S]  =  G_B''(p) (q - p)  =  2 (q - p),
```
which is `0` **iff `p = q`** (the unique stationary point). This is the FOC
`2(q-p)=0`, unchanged and correct.

**Second-order condition (CORRECTED).** The curvature of the *expected score* is
```
d^2/dp^2 E[S]  =  G_B'''(p)(q-p) - G_B''(p)  =  0 - 2  =  -G_B''(p)  =  -2  <  0,
```
using `G_B'''\equiv 0`. So **`E[S]` is strictly concave** in `p` with curvature
`-G_B'' = -2 < 0`, and the stationary point `p=q` is therefore the **unique
maximum**. The role of `G_B''=2>0` is to make the **generator** strictly convex —
that is what makes the scoring rule *strictly proper* (Savage/Bregman) — **not**
the second-order condition for the maximizer of `E[S]`. v3's text invoked the
generator's convexity in the maximizer's place; the conclusion (`p=q` unique max)
was right, the justification label was wrong, now corrected. ∎
(Grid check: `argmax_p E[S] = q` recovered at `q\in\{0.2,0.5,0.8\}`; numerically
estimated curvature `\approx -2` at all three.)

**Role (unchanged).** Claim 2 is the mechanism that makes **A3** (evaluator
pre-commitment) incentive-compatible: the elicited factuality probability is the
evaluator's true belief, closing the gamed-label loophole. It says nothing about
detectors and makes no causal claim about `\hat U`.

**Answer-adjacent leakage regression yielding `beta` (pre-registered, with CI).**
The leakage slope `beta` of §2.10/§2.11's bound `|E[\hat U]-U_true| <=
beta*Delta_pos` is **estimated, not assumed**. On the **answer-adjacent control**
(§2.2: masking the span immediately preceding the answer, high positional
leverage, typically not the evidence) plus inert spans at varying proximity, we
fit the pre-registered regression
```
delta_inert_i  =  beta * proximity_i  +  gamma  +  eps_i,
```
where `delta_inert_i` is the masked-inert-span factuality change and `proximity_i`
the (negative) distance-to-answer. We report `\hat beta` with a **paired-bootstrap
95% CI** `[beta_lo, beta_hi]` (validation split only; frozen at lock, §4.6). The
A4★ bias bound used by G7 is then the **upper-CI** product
`B_UCI := beta_hi * Delta_pos`, where `Delta_pos` is the proximity-bin width of
the stratified `Pi_i` (so `Delta_pos = bin_width`). Gating on `beta_hi` (not
`\hat beta`) makes G7 conservative: A4★ is declared positively supported only if
even the worst-case leakage slope keeps the bound under the margin.

### 2.12 Mask OOD as a `displaced_mass`-stratified dose-response (fixing d)

> **Extends v3 §2.2/G8 beyond the binary SHAM check.** Carries forward the SHAM-
> MASK null (do-nothing renormalisation + answer-disjoint non-check-worthy span)
> and *adds* a calibrated dose-response on inert spans, using the already-persisted
> `displaced_mass` field. No new operator; `displaced_mass` is the v3 field.

A binary "is SHAM-MASK `\hat U` CI overlapping 0?" cannot distinguish a small
genuine necessity effect from a small operator artifact, nor reveal whether the
operator's OOD footprint **grows with how much attention mass it displaces**. v4
calibrates the operator's own footprint as a function of `displaced_mass`.

**Inert-span calibration set.** Take spans the atomic-claim extractor labels
*non-check-worthy and answer-disjoint* (provably-irrelevant; the same population
SHAM-MASK draws from). For each, apply the *identical* `mask` mechanics and record
`(displaced_mass, \hat U_inert)`. Because these spans are causally inert, **any
non-zero `\hat U_inert` is operator artifact by construction** (OOD footprint of
the partition-function/renorm change), not necessity.

**Stratified analysis.** Bin `displaced_mass` into `\{[0,0.05),[0.05,0.1),
[0.1,0.2),[0.2,0.4),[0.4,1]\}`. Per bin, estimate `\hat U_inert` with a paired-
bootstrap CI. Fit the pre-registered calibration slope
```
\hat U_inert  =  s_OOD * displaced_mass  +  c_OOD  +  noise,
```
reporting `\hat s_OOD` and its 95% CI. `s_OOD` is the **dose-response of the OOD
artifact**: how much spurious `\hat U` the operator manufactures per unit
displaced mass.

**Deflation of real-span `\hat U` (pre-registered).** For each real (candidate)
span at observed `displaced_mass = d^*`, the **calibrated operator footprint**
`\widehat{art}(d^*) = \hat s_OOD\,d^* + \hat c_OOD` is subtracted:
```
\hat U_deflated(S^*)  =  \hat U(S^*)  -  \widehat{art}(displaced_mass(S^*)),
```
with the footprint's CI propagated into `\hat U_deflated`'s CI (added in
quadrature). The gate G1 is evaluated on `\hat U_deflated`, so a necessity signal
must clear `0.05` **after** removing the matched-dose operator artifact. This
converts the binary SHAM check into a quantitative correction: a real effect that
survives deflation at its own `displaced_mass` is not an OOD artifact.

**Near-vacuous-mask guard (carried).** `displaced_mass \approx 0` still means `S`
was barely attended to and the mask is near-vacuous; such examples are flagged
(v3 sanity field) and reported separately, since both `\hat U` and the deflation
are uninformative there.

### 2.13 Claim-calibration table — two row edits

The v3 §2.8 table is carried forward verbatim except:
- the row *"intervention-useful" gated by G1+G2+G7+G8 with power* now reads
  **gated by G1(on `\hat U_deflated`)+G2+G7(on the bound's upper CI)+G8(SHAM null
  AND bounded calibrated `s_OOD`) with power**;
- a new row: *"identification degrades gracefully"* — **allowed only when the
  §2.10G graded family matches its pre-registered `\hat U` curves on all three
  axes (G7)**; never stated from the clean oracle alone.

---

## 3. Contribution + falsifiable predictions (v4 deltas)

The single scientific contribution is unchanged from v3 (the *estimand and its
identification design* plus the *detector-beating novelty gate*). v4 sharpens the
falsifiable predictions tied to the five fixes; v3's P1/P2/P3 stand and are
extended:

- **P1 (existential), extended.** The CIU selector achieves
  `\hat U_deflated >= 0.05` (Holm CI lower bound `>0.05`, on the *deflated*
  estimator of §2.12) with `\hat D_util <= 0.02` on `>=2` lead datasets at
  operator `mask`. *Now stated on the OOD-deflated estimator.*
- **P2 (novelty), unchanged.** The proposed selector's `\hat U` exceeds the best
  adapted detector's by `>=0.03` (Holm CI `>0`); a detector matching/beating it is
  a publishable novelty downgrade, **not** a causal-identification failure.
- **P3 (operator honesty), extended.** SHAM-MASK has `\hat U` CI overlapping `0`
  **and** the calibrated OOD slope `s_OOD` is bounded (CI excludes a slope large
  enough to manufacture `0.05` over the observed `displaced_mass` range); the
  answer-adjacent regression yields `beta_hi` with `beta_hi*Delta_pos < 0.03`.
- **P4 (graded identification, NEW).** On the §2.10G family, `\hat U` follows its
  pre-registered curves: linear degradation with `beta*Delta_pos` crossing the
  `0.03` floor at the planted point (Axis P); `\approx 1/m_c` under-recovery for a
  single-span selector (Axis M); flat-at-zero for detector-on-distractor across
  all distractor strengths (Axis D). *A flat or non-monotone response on any axis
  fails G7.*

Each prediction can lose and is routed to the right conclusion: P1 fails ⇒ kills
the causal claim; P2 fails ⇒ demotes novelty (not causality); P3 fails ⇒ operator
artifact; P4 fails ⇒ A4★ not positively supported / estimator not behaving as an
identified contrast (causal wording withheld pending wider stratification).

---

## 4. Pre-registered analysis plan deltas

Gates G1–G6 keep v3's thresholds. v4 restates G7 and G8 per fixes (a)/(c)/(d) and
adds §4.6 (nuisance re-estimation) and §4.7 (power/budget feasibility point). The
Holm confirmatory family is unchanged in membership (G1, G2, G5′, G6, G7, G8 +
per-baseline best-detector `\hat U` contrasts across `{2 sizes} x {2 datasets}`);
v4 notes G1 is now evaluated on `\hat U_deflated` and G7 on the **upper-CI** bound.

### 4.1 Restated gates G7 and G8

- **G7 positive A4★ support (RESTATED, fixes a + c).** Two conjuncts, both
  required:
  1. **Graded-oracle recovery (§2.10G).** On the clean oracle, planted selector
     `\hat U \to 1`, random/inert-detector `\to 0`. On the graded family, `\hat U`
     **matches its pre-registered curve on all three axes** (Axis P linear
     degradation with the crossing at `0.03`; Axis M `\approx 1/m_c`; Axis D
     flat-at-zero). Non-monotone / off-curve response ⇒ fail.
  2. **Upper-CI leakage bound (§2.11).** The reported bias bound uses the
     **upper CI** of the answer-adjacent leakage slope: `B_UCI = beta_hi *
     Delta_pos < 0.03` (the localization margin). *Fail ⇒ A4★ not positively
     supported; causal wording withheld pending wider proximity-stratification
     (smaller `Delta_pos` = finer bins) until `B_UCI < 0.03`.*
- **G8 operator-artifact control (RESTATED, fix d).** Two conjuncts, both required:
  1. **SHAM null (carried).** SHAM-MASK and answer-adjacent controls have
     `\hat U` CI overlapping `0`.
  2. **Bounded calibrated OOD slope (§2.12).** The `displaced_mass`-stratified
     calibration slope `s_OOD` has a CI that **excludes** a slope large enough to
     manufacture `\hat U_inert >= 0.05` over the observed `displaced_mass` range;
     and G1 is evaluated on the **deflated** `\hat U_deflated`. *Fail ⇒ the `mask`
     necessity signal is an OOD/positional artifact; report as such, do not claim
     necessity.*

### 4.6 Leakage-safe re-estimation of `sigma_u` and `kappa` (fixing b)

Both nuisance parameters are estimated **only on the validation split**, with a
named estimator, a CI, a minimum `n_val`, and a **freeze-at-lock** rule; the test
split stays sealed and neither parameter is re-estimated after unlock. This is the
analysis-lock discipline of `pre_registration.md`, made specific.

**(B1) `sigma_u` — per-example paired-contrast standard deviation.**
- *Estimator.* On the validation split, compute the paired per-example contrasts
  `u_i = delta_tgt_i - delta_rand_i` (`delta_rand_i` MC-averaged over `R_int`
  draws), then `\hat sigma_u = sample_sd(\{u_i\})`. This is the **realized paired**
  sd, which the v3 §4.5(E2) bound `sigma_u <= 0.707` caps from above; the paired
  design typically yields far less.
- *CI.* Bootstrap the validation `\{u_i\}` (`>=10,000` resamples) for a 95% CI
  `[sigma_lo, sigma_hi]`. **Power uses `sigma_hi`** (the upper CI), so the MDE
  `n` is conservative against an underestimated variance.
- *Minimum validation `n_val`.* `n_val >= 200` paired examples per cell **and**
  `n_val` large enough that the bootstrap CI half-width on `sigma_u` is
  `<= 0.05*\hat sigma_u` (a 5% relative-precision rule); if not met, enlarge the
  validation draw before locking. `n_val` examples are **disjoint** from the test
  `n`.
- *Freeze rule.* `\hat sigma_u`, its CI, and the derived
  `R_power := \lceil (z_{1-alpha_1/2} \cdot sigma_hi / 0.03)^2 \cdot
  (1+1/\bar m_pool) \rceil` are written into the locked config at analysis-lock
  and **never re-estimated** on test data.

**(B2) `kappa` — evaluator agreement for attenuation.**
- *Estimator.* On a validation subset double-scored by the hashed evaluator and a
  held-out reference labeler, `\hat kappa` = Cohen's `kappa` (2 raters) or Fleiss'
  `kappa` (`>2`), on the binary factuality label.
- *CI.* Bootstrap (or Fleiss closed-form) 95% CI `[kappa_lo, kappa_hi]`. The
  attenuation correction uses the **lower CI** `kappa_lo`, so the target
  `U_target = 0.05/(2*kappa_lo - 1)` is conservative against an overestimated
  agreement.
- *Minimum validation `n_val`.* `>= 300` double-scored items per cell, or enough
  that the `kappa` CI half-width is `<= 0.03`; else enlarge before lock.
- *Freeze rule.* `\hat kappa`, its CI, and `U_target` are locked; the `kappa<0.90`
  fallback (widen margin to the attenuation-adjusted value, **or** aggregate to
  claim-level factuality proportion to raise effective `kappa`) is selected **at
  lock from the validation `kappa_lo`**, removing v3's post-hoc discretion: the
  rule is *"if `kappa_lo < 0.90`, use claim-level aggregation; re-measure
  `kappa_lo` on the aggregated outcome; if still `< 0.90`, widen the margin to
  `0.05/(2*kappa_lo-1)`."* The branch is deterministic given validation data.

**(B3) Interaction with the proximity-stratified `Pi` sampler.** The proximity
stratifier (v3 §2.10(ii): require `\tilde S` to match `S^*` on a distance-to-answer
bin) **shrinks the per-example null pool** `Pi_i` from its full size to the in-bin
size `m_pool(i)`. A smaller pool raises the variance of the random-arm MC estimate
`delta_rand_i`: with a finite pool of size `m_pool` sampled `R_int` times, the
random-arm contributes an extra finite-pool factor. We pre-register the
**inflation factor**
```
infl  =  1 + 1/\bar m_pool,        \bar m_pool = mean_i m_pool(i)  (validation),
```
applied to `R_power` (B1's freeze formula). If any cell's `\bar m_pool` is below a
floor `m_pool_min = 8` (the stratifier shrank the pool too far), the pre-registered
response is to **coarsen the proximity bin one step** (larger `Delta_pos`, hence a
slightly larger `B_UCI` in G7) **until** `\bar m_pool >= 8`, accepting the
G7-bound/variance trade-off explicitly rather than silently. This couples fix (b)
to fix (a)/(c): the proximity-bin width is the shared knob between the leakage
bound (G7) and the pool-shrinkage variance (power), and v4 pre-registers how it is
set when the two pull against each other.

### 4.7 Power/compute feasibility — exhibited point (fixing e)

v3 §4.5(E5) stated the inequality `4 * n * 18 * c_fwd <= 14` GPU-hr and a decision
order but exhibited no satisfying `(n, cells, kappa, c_fwd)`. v4 pre-registers a
**`c_fwd` ceiling** and **exhibits one concrete feasible point** (formula
evaluation, not evidence).

**`c_fwd` ceiling.** Pre-register `c_fwd <= c_fwd_max := 4.57e-4` GPU-hr per
intervention-forward (`\approx 1.65` s/forward) — the **budget-binding ceiling at
the pre-registered feasible operating point** (2 cells, `n = 850`, 18
forwards/example), rounded **down** from the exact `14/(2*850*18) = 4.5752e-4` so
the ceiling stays within budget: `2 * 850 * 18 * 4.57e-4 = 13.98 <= 14` GPU-hr.
(Rounding *up* to `4.58e-4` would give `14.01`, tipping over the line.) (An earlier draft
mislabelled `5.4e-3` as the ceiling; that figure was a forward-timing scale note,
**not** a feasibility ceiling — `2*850*18*5.4e-3 = 165.2` GPU-hr, far over the
14-GPU-hr line — and is corrected here and in `redesign_v4_ar_lead.yaml`.) The
locked config records the **measured** `c_fwd` from the validation-split timing
calibration (a timing measurement, **not** an experiment); if measured
`c_fwd > c_fwd_max`, the decision order of v3 §4.5(E5) applies (request a budget
increase line; else reduce `n` per the decision order). The cell count is set by
the cell-count thresholds below: 4 cells is affordable only if the measured `c_fwd`
reaches the tighter 4-cell ceiling `2.29e-4`.

**Exhibited feasible point (all three constraints simultaneously).** Take the
moderate-noise, moderate-variance operating cell:
- **`kappa = 0.92`** ⇒ attenuation `2*kappa-1 = 0.84` ⇒ `U_target =
  0.05/0.84 = 0.0595 <= 0.08` (the MDE design effect `U_true=0.08` clears it).
  *(Attenuation requirement satisfied.)*
- **`sigma_u = 0.30`** (within the §4.5(E2) bound `0.707`; the paired design's
  plausible regime, locked from validation per B1), family `m = 8`
  (`alpha_1 = 0.00625`, `z = 2.734`), proximity inflation `infl = 1.125`
  (`\bar m_pool = 8`). MDE: `n >= (z * sigma_hi / 0.03)^2 * infl`. With
  `sigma_hi = 0.30`: `(2.734*0.30/0.03)^2 = 27.34^2 = 747.5`, times `1.125`
  `= 841`. Round to **`n = 850` examples/cell**. *(MDE inequality satisfied at
  `U_true=0.08`, CI half-width `<=0.03`.)*
- **Budget:** `cells = 4` (`{2 sizes} x {2 datasets}`), `(1+1+R_int) = 18`
  forwards/example at `R_int=16`. At the non-binding `5.4e-3` forward-timing scale
  figure the total intervention GPU-hr would be
  `= 4 * 850 * 18 * 5.4e-3 = 61200 * 5.4e-3 = 330.5` ... which **exceeds** the
  14-GPU-hr intervention+baseline line by `>20x` — which is exactly why `5.4e-3` is
  **not** the feasibility ceiling. The feasible point is reached by the
  pre-registered decision order, and `c_fwd_max` is set to the budget-binding value:

  **Feasible exhibited operating point.** Drop to **`cells = 2`** (1 size x 2
  datasets, per v3's decision order) **and** set the `c_fwd` ceiling to the
  budget-binding value: solving `2 * n * 18 * c_fwd <= 14` at `n = 850` gives
  `c_fwd <= 14/(2*850*18) = 14/30600 = 4.5752e-4` GPU-hr/forward, locked
  **rounded down** to `c_fwd_max = 4.57e-4` (`\approx 1.65` s/forward) so the
  ceiling stays within budget. So the **jointly satisfiable point** is
  ```
  (n = 850 examples/cell,  cells = 2,  kappa = 0.92,  c_fwd <= 4.57e-4 GPU-hr/fwd),
  ```
  which satisfies: MDE (`n=850 >= 841`), attenuation (`U_target=0.0595 <= 0.08`),
  **and** budget (`2*850*18*4.57e-4 = 13.98 <= 14` GPU-hr at the locked, rounded-
  down ceiling). This is an *existence proof* that the MDE + attenuation + budget
  system has a solution; it pins the `c_fwd` the calibration must beat and the cell
  count it implies.

  **Cell-count thresholds (corrected).** The affordable `c_fwd` ceiling depends on
  the cell count, because the budget binds on `cells * n * 18 * c_fwd <= 14`. At
  `n = 850`, `18` forwards/example (exact value, then the locked rounded-DOWN
  ceiling that stays within budget):
  - **2 cells** affordable iff `c_fwd <= 14/(2*850*18) = 14/30600 = 4.5752e-4`
    GPU-hr/fwd; **locked ceiling `c_fwd_max = 4.57e-4`** (`13.98 <= 14`).
  - **4 cells** affordable iff `c_fwd <= 14/(4*850*18) = 14/61200 = 2.2876e-4`
    GPU-hr/fwd (half the 2-cell ceiling); **locked ceiling `2.28e-4`**
    (`13.95 <= 14`). Rounding *up* (`4.58e-4` / `2.29e-4`) would tip over the line.

  So the earlier `c_fwd_max = 5.4e-3` ceiling is *not* a feasibility ceiling for
  the 4-cell design — it gives `4*850*18*5.4e-3 = 330.5` GPU-hr, far over budget.
  The **pre-registered feasible operating point is 2 cells**
  `(n = 850, cells = 2, kappa = 0.92, c_fwd <= 4.57e-4 GPU-hr/fwd)`. The 4-cell
  full-`n` design becomes affordable **only if the measured `c_fwd <= 2.28e-4`**
  (not `4.57e-4`); if measured `c_fwd` is in `(2.28e-4, 4.57e-4]`, the 2-cell point
  holds; if above `4.57e-4`, a budget-increase line is requested (or `n` is reduced
  per the v3 §4.5(E5) decision order). The locked config records the measured
  `c_fwd`, the selected `(n, cells)`, and which branch of the decision order fired.

(Quadrature/closed-form note: the §4.6 `R_power` inflation, the §4.7 arithmetic,
the §2.10G expected curves, and the §2.11 identity/curvature were checked
numerically in-session and are labelled **formula evaluation, not evidence**.)

---

## 5. Phase-2 code — IMPLEMENTED + FROZEN (additive, uncommitted)

All v3 Phase-2 additions stand. v4 **extends them additively** — no existing
public API meaning changes; new optional fields and new pure-Python helpers only.
**The Phase-2 plan below is now implemented and frozen in `src/` and exercised by a
passing pure-Python unit harness** (no model, no GPU, no run). It remains
**uncommitted** and authorizes nothing: `server.authorized: false`, no `git
commit`, no `git push`, no experiment. The list below now reads as the *as-built*
manifest rather than a forward promise.

**Files this redesign ADDED (now on disk, frozen, uncommitted):**
`docs/redesign/REDESIGN_v4.md` (this file); `src/tracecausal/ciu.py`,
`src/tracecausal/nuisance.py`, `src/tracecausal/oracle_gen.py`,
`src/tracecausal/nullpool.py`, `src/tracecausal/interventions.py`,
`src/tracecausal/_numerics.py`; `configs/experiments/redesign_v4_ar_lead.yaml`;
`tests/test_ciu_nulldata.py`; and the Stage-1 paper skeleton
`paper/main.tex` + `paper/references.bib`.

**Phase-2 additions/extensions — AS BUILT (implemented + frozen):**
- `src/tracecausal/ciu.py` — add `graded_oracle_family(axis, param) ->
  fixtures` (pure-Python templated-generator fixtures for Axes P/M/D, no model
  calls); `ood_deflation(u_hat, displaced_mass, slope, intercept, slope_ci) ->
  (u_deflated, ci)`; `leakage_slope_regression(delta_inert, proximity) ->
  (beta_hat, beta_lo, beta_hi)`. `ciu_gate` is **extended** (additive optional
  args `u_deflated`, `beta_hi`, `s_ood_ci`, `graded_curve_pass`) — its existing
  return contract `{useful_candidate | diagnostic | not_novel}` is unchanged; it
  still **never** returns `invalidated` for a detector with positive `\hat U`.
- `src/tracecausal/nuisance.py` (implemented) — `estimate_sigma_u(u_i) ->
  (sigma_hat, sigma_lo, sigma_hi)`; `estimate_kappa(labels_a, labels_b) ->
  (kappa_hat, kappa_lo, kappa_hi)`; `r_power(sigma_hi, z, margin, infl) -> int`;
  `pool_inflation(m_pool_per_example) -> float`; plus the deterministic κ-fallback
  (`claim_level_aggregate`, `apply_kappa_fallback`) and `u_target`. All
  validation-split-only; callers pass frozen arrays; no test-split access path
  exists in the API.
- `src/tracecausal/ciu.py::CIURecord` (the dataclass lives in `ciu.py`, not a
  separate `ciu_record.py`) — `CIURecord` carries the optional, defaulted fields
  (back-compatible): `u_deflated: float | None = None`,
  `ood_slope: float | None = None`, `ood_slope_ci: tuple[float,float] | None =
  None`, `beta_hi: float | None = None`, `proximity_bin_width: float | None =
  None`, `m_pool_mean: float | None = None`. `validate_ciu_record` gains checks
  (only fire when the new fields are populated): G7 requires
  `beta_hi*proximity_bin_width < localization_margin`; G8 requires the OOD slope
  CI bounded; G1 evaluated on `u_deflated` when present. Existing checks
  unchanged.
- `configs/experiments/redesign_v4_ar_lead.yaml` (frozen on disk) — copies the v3 config and
  adds `graded_oracle: {axes: [partial_leakage, multi_cause, distractor],
  crossing_margin: 0.03}`, `ood_deflation: true`, `displaced_mass_bins: [...]`,
  `leakage_slope_ci: required`, `sigma_u_source: validation_bootstrap_upper_ci`,
  `kappa_source: validation_lower_ci`, `n_val_min: {sigma_u: 200, kappa: 300}`,
  `proximity_pool_min: 8`, `c_fwd_max: 4.57e-4` (the budget-binding 2-cell
  ceiling, `floor(14/(2*850*18), 3sf)`; the legacy `5.4e-3` figure is retained only as a non-binding scale note),
  `feasible_point: {n: 850, cells: 2, kappa: 0.92}`,
  `server.authorized: false`. All v3 markers retained.
- The v4 graded-oracle, OOD-deflation, and nuisance-estimator tests are
  **consolidated into `tests/test_ciu_nulldata.py`** (single pure-Python module,
  no GPU) rather than three separate files, and the suite **passes**. Specifically:
  - *Graded oracle* (`test_axis_p_degrades_linearly_and_crosses_floor`,
    `test_axis_m_underrecovers_to_one_over_mc`,
    `test_axis_d_detector_on_distractor_stays_zero`) — on synthetic fixtures,
    asserts Axis P `\hat U` degrades linearly and crosses `0.03` at the planted
    point; Axis M yields `\approx 1/m_c`; Axis D stays at `0` across distractor
    strengths.
  - *OOD deflation* (`test_ood_deflation_subtracts_calibrated_footprint`,
    `test_sham_mask_empty_set_is_vacuous_and_yields_zero_effect`) — asserts a
    real-span `\hat U` is correctly deflated by the calibrated footprint and the
    SHAM/do-nothing path yields `\hat U \approx 0`.
  - *Nuisance estimators* (`test_sigma_and_kappa_estimators_return_point_and_ci`,
    `test_r_power_reproduces_exhibited_feasible_point`,
    `test_pool_inflation_matches_one_plus_one_over_mean`) — asserts `sigma_u`/`kappa`
    estimators return point + CI, `r_power` uses the upper `sigma` CI and the
    inflation factor, and the validation-only API exposes no test-split path.

**Contract invariants kept:** `validate_manifest` still requires
`server.authorized == false`, `>=3` baselines, `>=20` paper seeds (floor);
`passes_intervention_gate` signature/thresholds **unchanged** and still only
*wrapped* by `ciu_gate`; all `required_docs` paths continue to exist; the new
`CIURecord` fields are optional with back-compatible defaults so v3 records still
validate. No `git commit`, no `git push`, no run.

---

## 6. Honest limitations (v4 deltas)

Carries v3 §6 limitations; adds/updates:
- **Graded oracle is synthetic.** §2.10G demonstrates correct *degradation* on a
  templated generator where ground truth is planted; it shows the estimator
  behaves as an identified contrast under controlled erosion, **not** that real
  AR traces have the same `tau` structure. It is a correctness fixture (no paper
  numbers), as labelled.
- **`beta` upper-CI gating can be conservative.** G7 gates on `beta_hi*Delta_pos`;
  a wide leakage-regression CI (small `n_val`) can fail G7 even when `\hat beta`
  is small. The pre-registered response is finer proximity bins (smaller
  `Delta_pos`) and/or larger `n_val`, traded against the §4.6(B3) pool-shrinkage
  variance — an explicit, pre-registered trade, not a hidden knob.
- **OOD deflation assumes the inert-span footprint transfers.** §2.12 calibrates
  `s_OOD` on inert spans and subtracts it from real spans at matched
  `displaced_mass`; this assumes the operator's OOD footprint depends on the
  displaced mass, not on span content. The `displaced_mass`-binned analysis tests
  the mass-dependence; residual content-dependence is an accepted limitation,
  reported.
- **Feasibility point is one exhibited solution, not the locked design.** §4.7's
  `(n=850, cells=2, kappa=0.92, c_fwd<=4.57e-4)` proves the system is satisfiable;
  the *actual* `(n, cells, c_fwd)` are set at lock from the measured `c_fwd`. If
  the measured `c_fwd` exceeds the ceiling and no budget increase is granted, the
  2-cell point is the floor; below it, the 4-cell design is preferred.
- **No empirical evidence exists.** Everything here is design + theory; the only
  numerics are the §2.10G / §2.11 / §4.6 / §4.7 closed-form / quadrature checks,
  explicitly **"formula evaluation, not evidence."** Every claim-evidence row in
  `docs/claim_evidence_matrix.md` and `docs/paper_claims_status.md` remains
  `pending`; this document changes no status label and authorizes no run.
  `server.authorized: false`.
