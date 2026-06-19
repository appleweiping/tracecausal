# REDESIGN v5 — tracecausal

Status: `design_frozen_stage1_RR` (proposal-to-design promotion; **the v5 / Phase-B
modules ARE implemented as do-not-run pure-Python, exercised by a green unit harness
of 113 passing tests — no model load, no GPU, no server run**). **No server run
authorized; `server.authorized: false` is preserved and re-affirmed.** This is a **Stage-1 Registered Report grade design** that
re-opens the v4 *central contribution* under a harder top-venue bar, names the
single weakness that capped v4 below 9, and locks a sharpened v5 contribution at
**equation level**. It addresses every must-fix item from the GPT-5.5 adversarial
review (7/10, `reviews/tracecausal/redesign_v5_gpt55.md`). **No experiment, no
model load, no GPU/heavy-CPU job, ZERO fabricated numbers.** The only numerics
below are closed-form arithmetic, each labelled **"formula evaluation, not
evidence."** Every empirical result slot remains `DATA_NEEDED`.

This document is **additive** at design level. The Phase-B modules whose frozen
spec it records in §9 **are now implemented** under `src/tracecausal/` as do-not-run
pure-Python and are exercised by the green unit harness (113 passing tests, no model,
no GPU); this document itself does not re-derive that code. Section
numbers in the form "v4 §X" refer to `REDESIGN_v4.md`; "MF-n" refers to GPT-5.5
must-fix item n (§2 of the review). This file supersedes the working note
`_v5_proposal.md`, which it absorbs and corrects.

---

## 0. Executive summary (one paragraph)

v4 builds an *honest* matched-null causal estimand (CIU, `U_hat`) and proves its
internal machinery leakage-safe, but its central claim is certified by **internal
consistency** — `U_hat` is unbiased for a contrast the method itself defines, and
every kill-gate G1–G8 checks that the *estimator* is well-behaved, never that the
localized segment is *useful for anything a detector cannot already do*. v5 closes
that gap with one principle — **localization certified by intervention usefulness
earned out-of-sample under an uncontrolled reference** — operationalized as a
**cross-example repair-transfer** certification statistic `R_hat` (headline) sitting
on top of `U_hat` (now demoted to a *screening* statistic). The patching/repair
machinery (`interventions.patch`/`.replay`), idle in v4, becomes the star. v5 does
**not** stop at the relabel the reviewer flagged as lateral: it (i) replaces the
indefensible "iff" thesis with a **named estimand under five explicit assumptions
plus an identification lemma** (MF-1); (ii) specifies a **frozen source→target
transport protocol** and pins exactly which reference state crosses examples
(MF-2); (iii) adds **detector-to-repair baselines run through the identical repair
pipeline** so the differentiator is empirical, not rhetorical (MF-3); (iv) replaces
the invalid paired bootstrap with a **two-way (source × target) cluster /
class-block-permutation estimator with nested matched-null Monte-Carlo variance**
(MF-4); (v) derives a **new `sigma_R` power model** rather than reusing `sigma_u`
(MF-5); (vi) **freezes operator/hyperparameter selection on `V_sel`** (MF-6);
(vii) **strengthens Axis X** with a proven control-failure regime and direct
confounding negative controls (MF-7); (viii) **implements all v5
modules as do-not-run pure-Python under a green 113-test unit harness** (no model,
no GPU), with the Stage-1 readiness label still gated on that harness staying green
(MF-8); (ix) **narrows the
novelty claim** to "first matched-null, selective-inference-correct, cross-example
repair-transfer *certification*" and makes it **conditional on the baselines
losing** (MF-9); and (x) **hardens Stage-2 routing** so a broad G9 failure is only
publishable if the null is itself powered and theory-informative (MF-10). The CIU
estimand, the Prop 2.5a (tautology) / A4★ (testable assumption) split, the
proper-scoring theorem, the hard kill-gates G1–G8 (no silent bypass), and the
AR-LLM lead (diffusion/reasoning demoted to a transfer study) are **preserved**.

---

## 1. The capping weakness of v4 (what 7/10 diagnosed, restated precisely)

### 1.1 The single capping issue

> **v4's central claim is certified by internal consistency, not external
> usefulness.** Lemma 2.5 proves `E[U_hat] = E_i[tau_i(S*)] − E_i[bar tau_i(Pi)]`,
> which is true *by construction of `Pi` and the matched arm*. Calling that
> contrast "intervention usefulness" is unearned: the design measures an effect; it
> never shows the effect *does* something a detector cannot. A 9-tier paper needs
> "we can do X that prior work cannot, here is the falsifiable test that X holds."
> v4's X is "we can *measure* an above-pool causal contrast honestly" — honest
> measurement of a self-defined quantity reads as **infrastructure, not
> discovery**. TraceDet already detects from traces with a concrete downstream win
> (+15.2% AUROC avg; Table 1, `TRACEDET.pdf`); v4's reply "ours is causal not
> correlational" is only persuasive if causality buys a capability detection lacks,
> and v4 never cashes that in.

### 1.2 Two secondary soundness gaps v4 leaves open

- **Selective-inference gap.** v4 *adapts* the proximity-bin width `Delta_pos`
  data-adaptively (coarsen until `bar m_pool >= 8`, v4 §4.6 B3) and the
  `displaced_mass` bin edges (v4 §2.12), then runs confirmatory G1/G7/G8 tests on
  the *same* data scale. The Holm correction (v4 §4.2) covers the *family of gates*
  but not the *data-adaptive selection of the binning that defines those gates' test
  statistics*. Post-selection inference without conditioning on the selection event
  is anti-conservative.
- **Prose-ahead-of-code on the selection loop.** The *estimators* exist
  (`ciu.ood_deflation`, `nuisance.pool_inflation`), but the data-adaptive
  *selection procedure* — the loop that picks `Delta_pos`/bin edges and the event
  being conditioned on — is not a frozen, unit-tested function.

### 1.3 What the GPT-5.5 review additionally established (and v5 must honor)

The review (7/10) confirmed v5's *direction* is right and G9 + selective-inference
are real improvements over v4, but found the proposal **still 7/10** because:
(a) the novelty gap vs FCCT/IRI and CDCR-SFT is **overstated**; (b) the "a detector
cannot exhibit `R_hat`" claim is **false** (a detector can select a span and feed it
to the same repair adapter); (c) the cross-example estimand is **under-defined**
(no transport map, positivity, SUTVA); (d) the **paired bootstrap is statistically
invalid** under pair reuse; (e) reusing `sigma_u` for G9 power is **unjustified**;
(f) the "iff" thesis is **causally indefensible**; (g) Axis X controls **need not**
detect hidden confounding; (h) at review time the modules were **unimplemented**
(now resolved: implemented as do-not-run pure-Python, green 113-test harness — see
§9/§11); (i) the v5/v4
delta is **partly lateral relabeling**; (j) Stage-2 routing **over-sells** a broad
G9 null. v5 (this document) resolves each — §2 below maps every item.

---

## 2. Must-fix → fix map (audit table; every GPT-5.5 item addressed)

| MF | GPT-5.5 must-fix | v5 resolution | Where |
| --- | --- | --- | --- |
| 1 | Replace "iff" thesis with a formal estimand theorem (transport, consistency, positivity, within-class exchangeability, reference validity, matched-null repair validity) | Thesis reworded to a **certified-claim** (not "iff"); estimand `R` named with **assumptions A5–A9** + **Identification Lemma 5.1** | §3.1, §4.1, §4.4 |
| 2 | Explicit alignment/repair protocol; specify *which* reference state crosses examples | **Transport protocol T**: target receives a **source-derived repair *policy*** (Variant C), not the source reference state; Variants A/B named and rejected with reasons | §4.2 |
| 3 | Detector-to-repair baselines through identical `repair_ops` | **G9 baseline panel B0–B5** (TraceDet-, entropy-, probe-, random-same-class-, self-, oracle-selected), all through identical `repair_ops`; the differentiator is the *margin over B1–B3* | §4.3, §5.2 |
| 4 | Valid dependent-pair estimator (U-statistic / multiway cluster bootstrap / class-block permutation) with nested matched-null MC uncertainty | `R_hat` is a **two-sample U-statistic**; inference by **two-way (source,target) cluster bootstrap** + **class-block permutation** null; nested matched-null MC variance added in quadrature | §4.5 |
| 5 | New G9 power model; estimate `sigma_R` separately (pair reuse, class imbalance, matched-null MC, repair-op variance) | **`sigma_R` decomposition** (Eq. R-VAR) and **`R_power`** derived from the U-statistic design-effect, **not** `sigma_u` | §4.6 |
| 6 | Freeze operator selection (`patch`/`replay`, `rho`/`alpha`, span budget, target mapping, taxonomy) on `V_sel` or pre-register | **Operator-selection freeze (OS-1)** on `V_sel`; the choice is a **selection event** folded into the SI correction (`K_op` factor) | §4.7, §6.2 |
| 7 | Strengthen Axis X; prove/simulate control-failure regimes; add direct-confounding negative controls | **Axis X′**: a *proven* regime where G7/G8 do **not** trip (collinear confounder), plus **direct negative controls NC-1/NC-2** and a revised P5 that can genuinely fail | §7 |
| 8 | Implement v5 modules before claiming Stage-1 readiness | **Phase-B gating rule**: no Stage-1-ready label until all §9 modules implemented + harness green; this doc is `design_frozen`, *not* `stage1_ready` | Status line, §9, §11 |
| 9 | Narrow novelty to "matched-null, SI-correct, cross-example repair-transfer certification"; conditional on baselines losing | Net-novelty sentence rewritten and made **conditional on B1–B3 losing G9** | §3.4, §5.2 |
| 10 | Harsher Stage-2 routing; a broad G9 null is not "still a complete paper" by default | **Routing R1–R4**: necessity-only downgrade is publishable **only if** the G9 null is *powered* (`R_power` met) **and** theory-informative (a pre-registered mechanism prediction) | §8 |

---

## 3. The sharpened v5 central contribution

### 3.1 The thesis (corrected — a certified claim, not an "iff"; MF-1, MF-9)

> **A trace segment is *repair-transfer-certified* for a hallucination class when a
> training-free, source-derived localized repair policy, applied to held-out
> in-class targets, recovers factuality beyond a matched-null repair, at a bounded
> edit budget and bounded utility cost — under the registered operator, transport
> map, reference construction, class partition, and matched-null assumptions (A5–A9,
> §4.4).** Certification is **sufficient** evidence of causal usefulness, not a
> definition of it: repair-transfer is *neither necessary nor sufficient* for
> causal usefulness in general (a segment may be causally necessary in-place yet
> not single-span-repairable; a generic repair may transfer for non-causal
> reasons), so the word "iff" is **dropped** and the claim is scoped to the
> registered design.

Operationally: v4's `U_hat` (above-pool factuality change under `mask`, on the
*same* example) is the **screening** statistic; the new **certification** statistic
`R_hat` (above-matched-null repair-transfer, *across* examples) is the **headline**.

### 3.2 The unifying principle (so it is not a bag of tricks)

> **Counterfactual usefulness must be earned out-of-sample, under a reference the
> method does not control.**

Three surfaces are *instances* of this one principle, not additive tricks:
- **Certification** (`R_hat`, G9) earns usefulness out-of-sample **across examples**
  (held-out targets).
- **Selective-inference correction** (§6) earns gate verdicts out-of-sample **across
  the data-adaptive binning and the operator selection** (both are selection events
  that must be paid for).
- **Adversarial oracle** (Axis X′, §7) earns A4★/A9 support out-of-distribution **in
  the reference** (a regime where the reference is misspecified and the controls
  must catch it — and a regime where, provably, they cannot).

Remove any one and the principle is incompletely tested. One sentence states the
whole paper: *causal localization is only as good as the repair it licenses on data
and references you did not get to choose.*

### 3.3 Why `R_hat` is not just a second `U_hat`

`U_hat` measures **necessity in-place** (mask `S*` in `x_i`, read the factuality drop
in `x_i`): within-example, ablation-style. `R_hat` measures **sufficiency across
examples** (a source-derived repair *policy* applied to held-out `x_j`, read the
factuality *gain* in `x_j`): cross-example, repair-style, at a budget, against a
matched-null repair baseline. The two are decorrelated in a scientifically
interesting way (a necessary-in-place span need not yield a transferable repair),
which is exactly why G9 is a *new* falsifiable test and not a restatement of G1.

### 3.4 Net novelty statement (narrowed and conditional; MF-9)

> *Conditional on the detector-to-repair baselines (B1–B3, §4.3) failing G9, we are
> the first to **certify** hallucination localization by a **matched-null,
> selective-inference-correct, cross-example repair-transfer** estimand — a
> training-free, multiplicity-correct edit-transfer test — distinguishing it from
> detection-based tracing (TraceDet, an IB sub-trace optimized for AUROC), from
> fixed-grid recovery-rate tracing (FCCT/IRI), and from SFT-based causal mitigation
> (CDCR-SFT). If a detector-selected span fed through the identical repair pipeline
> matches our margin, the contribution downgrades to "the matched-null + SI
> certification protocol," not "causal-vs-correlational localization."*

This is deliberately weaker than `_v5_proposal.md`'s "first to certify by
repair-transfer," which the review flagged as inflated.

---

## 4. The certification statistic `R_hat` at equation level

### 4.1 Objects and notation (carried from v4 unless noted)

- `x_i` — an example (prompt + AR generation trace). `S*(x_i) = [a_i, b_i]` — the
  selector's localized claim span on `x_i` (the *source* localization).
- `class(x_i) in C` — the **frozen G3 taxonomy bucket** (same atomic-claim /
  error-type partition v4 already pins; hash persisted, §10). No new researcher
  freedom (MF-6 for taxonomy).
- `Y_j(·) in [0,1]` — the **proper-scored** evaluator factuality on target `x_j`
  (Theorem `thm:propriety`, preserved; SOC corrected in v4 §2.11). `no_op` is the
  unedited target run.
- `Pi_j` — the per-example **matched-null pool** on the *target* `x_j` (proximity-
  and budget-stratified; `nullpool.py`, preserved): the set of in-budget spans on
  `x_j` matched to the transported span in position bin and length.

### 4.2 The source→target transport protocol T (MF-2 — the crux the review demanded)

The review's decisive objection: "no formal transport map from source span to
target trace." v5 pins it. **Three variants are named; v5 freezes Variant C** and
records the rejected alternatives so the estimand is unambiguous.

- **Variant A (reference-state transport, REJECTED).** Inject the *source's*
  reference residual state `h^{ref}(x_i)` at mapped target positions. *Rejected:*
  cross-example residual states are not comparable (different prompt, hidden
  geometry), positivity fails, and the edit is not a well-defined `patch` on `x_j`.
- **Variant B (target-own reference at mapped positions, REJECTED as the headline).**
  Patch the *target's own* `h^{ref}(x_j)` at positions mapped from `S*(x_i)`.
  *Rejected as headline* because it reduces to a within-target repair whose only
  cross-example content is the *position map* — too weak to certify transfer; **kept
  as an ablation** (it isolates the position-map contribution).
- **Variant C (source-derived repair *policy* transport, FROZEN HEADLINE).** The
  source localization induces a **repair policy** `rho := (op, alpha, L_patch,
  budget_k, ref_type, anchor_rule)` — *not* a state. The policy is **applied on the
  target's own run**: the target's claim span is identified by the **anchor rule**
  (the frozen alignment map `T`, below), and the target's *own* reference state is
  patched/replayed there under the *source-derived* policy `rho`. Cross-example
  content is the **policy + anchor**, which is what "the localization licenses a
  reusable repair" actually means. This is the estimand `R` below.

**The anchor/alignment map `T` (frozen).** `T(S*(x_i), x_j)` returns the target span
`[a_j, b_j]` by: (i) restricting to atomic claim spans of the *same G3 class* on
`x_j` (taxonomy-matched); (ii) selecting the in-class span whose **proximity bin to
the answer matches** the source span's proximity bin (the same `Delta_pos` grid the
matched null uses); (iii) matching the **edit budget** `k` (length/coordinate count)
exactly; (iv) if multiple candidates remain, the **first by position** (deterministic
tie-break). If no in-class, budget-matched, proximity-matched span exists on `x_j`,
the **positivity guard fails** and the pair `(i,j)` is **excluded** (recorded; this
is the A7 positivity event, §4.4).

### 4.3 The G9 baseline panel (MF-3 — the detector contrast made empirical)

The review correctly: a detector *can* select a span and feed it to the same
repair adapter, so "a detector cannot exhibit `R_hat`" is false. v5 removes that
claim and instead makes the differentiator an **empirical margin** over detector-
selected localizations run through the **identical** `repair_ops` pipeline:

- **B0** — `no_op` floor (target unedited).
- **B1** — **TraceDet-selected** source span (the IB sub-trace localized to a span),
  same `repair_ops`, same transport `T`.
- **B2** — **entropy/uncertainty-selected** source span (LN-Entropy / perplexity
  peak), same pipeline.
- **B3** — **probe-selected** source span (latent linear probe), same pipeline.
- **B4** — **random same-class source** span (the matched-null repair baseline of
  Eq. g-ij, below; this is the *control inside* `R_hat`).
- **B5** — **oracle-selected** source span (planted ground truth, oracle fixtures
  only; upper bound).
- **PROPOSED** — the CIU selector's `S*`.

The headline differentiator is **`R_hat`(PROPOSED) − max(R_hat(B1), R_hat(B2),
R_hat(B3))**, with its own Holm-corrected CI. If that margin's lower CI does not
clear `0`, the novelty downgrades per §3.4 (MF-9). B4 is the within-`g` control;
B5 is the ceiling.

### 4.4 The estimand and its assumptions (MF-1)

Define the per-pair **repair gain** (Variant C transport, matched-null-controlled):

```
g_{ij} = [ Y_j( do( phi_rho^{T(S*(x_i), x_j)} ) ) − Y_j( no_op ) ]            (localized repair on target)
       − E_{ S~Pi_j }[ Y_j( do( phi_rho^{S} ) ) − Y_j( no_op ) ]              (matched-null repair on target)
                                                                               (Eq. g-ij)
```

where `phi_rho^{[a,b]}` is the **frozen repair policy** `rho` applied at target span
`[a,b]` (a `patch` at `alpha` over `L_patch`, or a `replay` of `[a,b]` under
`ref_type`). The **repair-transfer estimand** is the within-class, source≠target
mean:

```
R(selector; rho) = E_{ (i,j): class(i)=class(j), i != j, positivity(i,j) }[ g_{ij} ].   (Eq. R)
```

**Assumptions under which `R` is an identified causal estimand (named so the
reviewer can attack each):**

- **A5 (intervention consistency / SUTVA for the repair).** Applying `phi_rho` at
  `[a_j,b_j]` on `x_j` yields the same `Y_j` whether the policy was derived from
  `x_i` or any other in-class source; no cross-target interference (each target is
  edited and scored independently). *Testable:* swap the source within class; `g_{ij}`
  must be invariant up to MC noise (a registered diagnostic, §7 NC-2).
- **A6 (within-class exchangeability).** Targets are exchangeable within a class
  given the matched-null stratification, so `E[g_{ij}]` does not depend on *which*
  source in the class induced `rho` beyond the policy `rho` itself. *Tested by* the
  source-swap diagnostic (A5/NC-2) and by class-block permutation (§4.5).
- **A7 (positivity).** For a non-vanishing fraction of in-class target pairs, an
  in-class, proximity-matched, budget-matched target span exists (`T` is defined).
  Pairs failing positivity are excluded and the **excluded fraction is reported**;
  G9 requires excluded fraction `< 0.5` per class (else the class is under-powered
  and routed to "insufficient positivity," not a null).
- **A8 (reference validity).** The reference run used to build `rho` and to patch the
  target is a valid counterfactual reference (the factual reference state is the
  state the model would occupy had it conditioned on the evidence). This is the
  cross-example analogue of v4's A4★; it is the assumption **Axis X′ attacks** (§7).
- **A9 (matched-null repair validity).** `Pi_j` is the correct counterfactual repair
  control: editing *any* matched in-budget target span the same way is the right "what
  if we had repaired a generic span" baseline. This is the repair-operator analogue of
  v4's per-example matched null; the `g_{ij}` contrast subtracts it.

**Identification Lemma 5.1 (assumption-free part + assumption-loaded part; mirrors
the v4 Prop 2.5a / A4★ split, preserved as scaffold).**

- *Prop 5.1a (tautology, assumption-free).* By construction of `Pi_j` and the paired
  `no_op` subtraction,
  ```
  E[g_{ij}] = ( target-repair effect of the transported policy )
            − ( mean target-repair effect of a matched-null policy ),
  ```
  which holds **definitionally** for the registered `Pi_j`, exactly as v4's Lemma 2.5
  holds for `U_hat`. No causal claim yet.
- *A8+A9 (testable assumptions).* Under A5–A9, `R(selector; rho)` equals the
  **average controlled repair-transfer effect** attributable to the *localized policy*
  rather than to generic repair or to confounding. **`R > 0` is then interpreted as
  causal usefulness only under A5–A9**, never definitionally — closing the review's
  "operational score, not an identified estimand" objection by stating precisely what
  must hold and how each piece is tested (A7 by the positivity report; A5/A6 by the
  source-swap diagnostic and class-block permutation; A8 by Axis X′; A9 by the
  matched-null repair control and SHAM).

### 4.5 Valid inference: U-statistic, two-way cluster bootstrap, class-block permutation (MF-4)

The review is right that a naive paired bootstrap over `(i,j)` is invalid because
examples recur in many pairs (each `x_k` is a source for many targets and a target
for many sources): the pairs are **dependent**. `R_hat` is therefore treated as a
**two-sample / within-class U-statistic**, not a mean of i.i.d. paired draws.

- **Estimator.** Per class `c`, `R_hat_c = mean over ordered in-class pairs (i,j),
  i!=j, positivity ok, of g_{ij}`. Overall `R_hat = sum_c w_c R_hat_c` with `w_c`
  the pre-registered class weights (inverse-variance or equal; **frozen on `V_sel`**,
  MF-6).
- **Variance (primary): two-way cluster bootstrap.** Resample **source clusters and
  target clusters independently** (multiway / pigeonhole cluster bootstrap over the
  two example-index margins), recompute `R_hat` each replicate (`>= 10,000`
  replicates). This accounts for the fact that a single example contaminates many
  pairs through *both* margins. The CI is the bootstrap percentile interval; G9 gates
  on its **Holm-corrected lower bound**.
- **Variance (cross-check): U-statistic Hájek projection (BOTH projections).** Report
  the closed-form Hoeffding/Hájek variance for the **ordered/asymmetric** kernel
  `Var(R_hat_c) ≈ zeta_10 / n_source + zeta_01 / n_target`, where `zeta_10` is the
  variance of the **source** projection `E[g_{ij} | i]` and `zeta_01` the variance of
  the **target-margin** projection `E[g_{ij} | j]`. This includes BOTH projections so
  the analytic floor is not understated; the symmetric shorthand `(4/n_c) zeta_1`
  collapsed the two margins and is no longer used (findings 4, 10). It is the analytic
  sanity bound the cluster bootstrap must not undershoot (matches
  `repair_transfer.hajek_projection_var` and `nuisance.estimate_sigma_r`).
- **Null (corroborating diagnostic, NOT the confirmatory test): class-block sign-flip.**
  Under H0 (`R = 0`), flip an independent `+-1` sign on each (class, source) block of
  the matched-null-centred `g_{ij}` and compare the observed `R_hat` to the sign-flip
  distribution. This **probes source-block sign-symmetry / A6 within-class
  exchangeability** and **corroborates** the confirmatory two-way cluster-bootstrap CI;
  it is **NOT exact for `R = 0`**. Matched-null centring + A6 give *mean-zero +
  within-class exchangeability*, which are strictly weaker than the *distributional
  sign-symmetry* a `+-1` flip would require (a skewed mean-zero block is exchangeable
  yet not sign-symmetric), so the confirmatory significance burden rests on the
  cluster-bootstrap CI lower bound, with this diagnostic required to agree. (The
  earlier "exact-by-construction under A6" claim was an over-claim; demoted per G9-FIX.)
- **Nested matched-null Monte-Carlo uncertainty.** The matched-null term in Eq. g-ij
  is an expectation estimated by `R_null` draws from `Pi_j`; its MC variance
  `sigma_MC^2 / R_null` is **added in quadrature** into each `g_{ij}`'s contribution
  and propagated through the cluster bootstrap (so the reported CI includes both the
  pair-dependence and the inner sampling noise). The repair-op stochasticity (e.g.
  `replay` decoding noise) is a third variance component, also propagated (§4.6).

### 4.6 The G9 power model — `sigma_R`, derived not borrowed (MF-5)

v4's `sigma_u` is the SD of a **within-example paired contrast**; it is the wrong
scale for a **cross-example U-statistic**. v5 derives `sigma_R` from first
principles:

```
Var(R_hat) ≈ zeta_10 / n_source + zeta_01 / n_target   +   (1 / N_pair) * ( sigma_MC^2 / R_null + sigma_op^2 / R_int )   (Eq. R-VAR)
```

- The kernel `g_{ij}` is **ordered/asymmetric** (source role != target role), so the
  first-order projection has **two** parts: `zeta_10` — variance of the **source**
  projection `E[g_{ij} | i]`, divided by the source cluster count `n_source`; and
  `zeta_01` — variance of the **target-margin** projection `E[g_{ij} | j]`, divided by
  `n_target`. Both projections are retained so the asymmetric-kernel variance is not
  understated; the symmetric shorthand `(4 / n_eff) * zeta_1` collapsed the two margins
  and is **no longer used** (findings 4, 10). `n_eff = min(n_source, n_target)` is kept
  for design-effect bookkeeping only. This matches `repair_transfer.hajek_projection_var`
  and `nuisance.estimate_sigma_r`.
- `sigma_MC^2 / R_null` — matched-null repair MC variance (nested, §4.5).
- `sigma_op^2 / R_int` — repair-operator stochasticity (e.g. `replay`), averaged over
  `R_int` repeats.
- **Class imbalance** enters through `n_source`/`n_target` and the class weights `w_c`
  (an effective design-effect factor `D_eff >= 1`, pre-registered, that inflates `n`).

Then

```
R_power := ceil( ( z_{1 - alpha_1' / 2} * sigma_R_hi / m_R )^2 * D_eff )      (Eq. R-POWER)
```

with `sigma_R_hi` the **upper CI** of `sigma_R` estimated on `V_sel` (conservative),
`m_R` the repair margin (§4.8), `alpha_1'` the Holm/SI-corrected per-test level
(§6.3). **`R_power` is computed in `nuisance.r_power_repair(...)`, a NEW function —
v4's `r_power` is NOT reused for G9.** No number is fabricated here; `zeta_10`,
`zeta_01`, `sigma_MC`, `sigma_op`, `D_eff`, `n_source`, `n_target` are all
`DATA_NEEDED`, estimated on `V_sel` at lock; Eq. R-VAR/R-POWER are the frozen
identities the locked config will evaluate.

### 4.7 Operator-selection freeze (MF-6)

Every degree of freedom in the repair policy `rho` is a selection layer that must be
paid for or pre-registered:

- **OS-1 (freeze on `V_sel`).** `op in {patch, replay}`, `alpha in
  PATCH_RHO_LEVELS`, `L_patch` (layer set), span/budget `k`, `ref_type in {factual,
  neutral}`, the anchor tie-break, and the class weights `w_c` are **selected on the
  selection split `V_sel` only**, by a frozen rule (maximize `R_hat`(PROPOSED) −
  B4 on `V_sel`, ties broken by smallest budget then `patch` over `replay`). Frozen
  into the locked config; never re-touched on `V_inf`/test.
- **OS-2 (selection event recorded).** The set of policies the freeze rule *could*
  have chosen (the discrete grid `PATCH_RHO_LEVELS × {patch,replay} × L_patch_grid ×
  ref_types`) has bounded cardinality `K_op`, recorded before data. If the SI-2
  fallback (§6.2) is used, the confirmatory level pays a Bonferroni factor `K_op` in
  addition to `K_bin`. Under SI-1 (selection split), `K_op = 1` for the inference
  family because the choice was made on `V_sel`.

### 4.8 The repair margin `m_R` (formula evaluation, not evidence)

`m_R` is **not** silently inherited from the necessity margin. It is set to a
**design repair margin `m_R0`** (a meaningful factuality recovery on the target,
`DATA_NEEDED`, pinned on `V_sel`), attenuation-adjusted by the **same `kappa`
machinery** (preserved `nuisance.u_target`) but with the cross-example evaluator
agreement `kappa^{repair}` re-estimated on target labels:

```
m_R := m_R0 / (2 * kappa_lo^{repair} − 1).      (Eq. m-R)
```

Whether `m_R0` numerically equals v4's `0.05` is a **`V_sel` calibration decision**,
not an assumption; the locked config records the realized `m_R0`, `kappa_lo^{repair}`,
and `m_R`. (Formula evaluation only; no value asserted here.)

---

## 5. The falsifiable kill-gate G9 (and the baseline-conditional novelty gate)

### 5.1 G9 — repair-transfer certification (headline falsifiable gate)

> **G9 (repair-transfer usefulness).** On `>= 2` lead datasets, the cross-example
> repair-transfer estimator `R_hat` (Eq. R) clears the repair margin with valid
> dependent-pair inference: the **confirmatory Holm/SI-corrected
> two-way-cluster-bootstrap CI lower bound `> m_R`**, corroborated by the
> **class-block sign-flip diagnostic `p < alpha_1'`** (G9-FIX: a diagnostic, not the
> confirmatory test), with **bounded
> utility cost** `D_util^{repair} <= 0.02`, **positivity** (excluded pair fraction
> `< 0.5` per class, A7), and **no class/self leakage** (source ≠ target; an example
> is never repaired by a policy localized on itself; B4/B5 separated). *Fail ⇒ route
> per §8 (necessity-only downgrade is conditional, not automatic).*

The gate **never** returns `invalidated` (it reuses the v4 `ciu_gate` fail-closed,
no-silent-bypass template). Its loss routes:
- `U_hat` (G1) passes, `R_hat` (G9) fails ⇒ **necessity without usefulness**
  (downgrade *only if* §8 R3 powered-null condition holds).
- `R_hat` passes but the `g_{ij}` matched-null term (B4) also passes (CI overlaps 0
  because *any* matched repair transfers) ⇒ **repair is generic, not localized** ⇒
  **diagnostic**, never `invalidated`.
- `R_hat` passes with `D_util^{repair} > 0.02` ⇒ **trades factuality for utility** ⇒
  reframe as abstention (mirrors G2).

### 5.2 G9-NOV — baseline-conditional novelty gate (MF-3, MF-9)

> **G9-NOV.** `R_hat(PROPOSED) − max_b{ R_hat(B1), R_hat(B2), R_hat(B3) } > 0`
> (Holm/SI-corrected lower CI), i.e. the CIU localization out-transfers
> TraceDet-/entropy-/probe-selected localizations through the **identical**
> `repair_ops` pipeline. *Fail ⇒ the certification protocol is the contribution, not
> "causal beats correlational"; the abstract/§3.4 novelty downgrades accordingly.*
> This is the gate that makes the detector contrast **empirical, not rhetorical.**

### 5.3 Why G9/G9-NOV are genuinely falsifiable

`R_hat` can be `<= 0` even when `U_hat` is large and clean (necessity ≠ transferable
sufficiency). B1–B3 can match or beat PROPOSED (a detector-selected span may license
an equally transferable repair). Both are real, common, publishable outcomes — and
each is a clean falsification of a *distinct* claim (usefulness; causal-vs-
correlational), routed honestly by §8.

---

## 6. Selective-inference correction (re-derive G7/G8/G1 + fold in operator selection)

The principle: **every data-adaptive choice is a selection event that must be paid
for.** v5 corrects the inference for *two* selection layers v4 left uncorrected
(binning) and v5 adds (operator selection).

### 6.1 SI-1 — selection split (primary)

Partition validation into a **selection split `V_sel`** and an **inference split
`V_inf`**, both disjoint from test. ALL data-adaptive choices — proximity-bin width
`Delta_pos`, `displaced_mass` bin edges, the repair policy `rho` (OS-1), class
weights `w_c`, and `m_R0` — are **frozen using `V_sel` only**. Nuisance estimates
(`sigma_u`, `sigma_R`, `kappa`, `kappa^{repair}`, `s_OOD`, `beta`) and ALL gate
statistics are computed on `V_inf`/test with every selection **held fixed**. Because
the selections are independent of the inference data, the Holm family on
`V_inf`/test is valid **without** post-selection adjustment: `K_bin = K_op = 1` for
the inference family.

### 6.2 SI-2 — selection-conditioned Bonferroni (fallback when `V_sel` too small)

If splitting costs too much power (the `n_val` floors of v4 §4.6 bind on both
splits), v5 pre-registers a **multiplicity penalty over the reachable selection
grid**: the binning ladder (finite, ordered `Delta_pos in {w_1 > w_2 > ...}`) and the
`displaced_mass` candidate edge set give `K_bin`; the operator grid (§4.7 OS-2) gives
`K_op`; both are **bounded and enumerated before data**. The confirmatory level is

```
alpha_1' = 0.05 / ( m' * K_bin * K_op ),      (Eq. SI-ALPHA)
```

folded into the Holm family. Conservative but selection-honest. The **deterministic
choice rule** (SI-1 iff the `n_val` floors are met on both splits, else SI-2) is
frozen, so the power cost is known before lock, not discovered after.

### 6.3 Family size and feasibility re-check (closed-form, not evidence)

G9 + G9-NOV add contrasts to the Holm family: `m' = m + (#G9 cells) + (#G9-NOV
baseline contrasts)`. Under SI-1 (`K_bin = K_op = 1`), `m'` grows by the per-cell
G9 contrasts plus the B1–B3 baseline margins only; `z = z_{1 - alpha_1'/2}` rises a
small amount. **`R_power` is recomputed from Eq. R-POWER with `sigma_R` (NOT
`sigma_u`)**; the v4 feasible point `(n=850, cells=2, kappa=0.92, c_fwd<=4.57e-4)` is
**re-checked, not assumed**, against the *new* forward count (a localized-repair
forward + `R_null` matched-null-repair forwards + `R_int` repair-op repeats per
target). The forward surcharge is persisted as `forwards_per_example`. If the
recomputed `R_power` or budget exceeds the v4 point, the **pre-registered decision
order** (request budget line → reduce cells → reduce `R_null`/`R_int` to the
variance-floored minimum) applies and is recorded. **No number here is fabricated;
this is the frozen budget identity re-evaluated at the new family size and the new
variance model.** The locked config records realized `m'`, `K_bin`, `K_op`, `z`,
`sigma_R_hi`, `forwards_per_example`, and the re-checked feasible point.

### 6.4 The selection procedure is code, not prose (closes the prose-ahead-of-code gap)

The deliverable is a **frozen, unit-tested `binning_selection.select_binning(V_sel)
-> (binning, selection_event)`** and **`repair_ops`/operator-freeze on `V_sel`**,
each emitting the chosen value **and** the selection event (the ladder walked / the
operator grid). The SI correction is then computable from the recorded event, not
narrated.

---

## 7. Adversarial oracle Axis X′ (genuine falsification risk, strengthened) (MF-7)

### 7.1 The problem v4's oracle cannot solve, and why the v5 proposal's Axis X was insufficient

v4's graded oracle plants `tau` and pre-registers the recovered curve; A4★ holds by
construction on every axis, so it can only show graceful degradation of a *correctly
specified* estimator — never that the method **detects its own misspecification**.
The `_v5_proposal.md` Axis X asserted "the controls (G7/G8) trip before the headline
under confounding," but the review correctly objected: **the G7/G8 controls need not
detect hidden confounding** — there exist confounders collinear with the leakage and
SHAM controls that the controls are *blind* to. An oracle whose only failure mode is
"controls always catch it" is still designed to pass. v5 fixes this by including a
**provable control-failure regime** and **direct negative controls**.

### 7.2 Axis X′ — hidden-confounder / misspecified-reference oracle (revised)

> **Axis X′.** The templated generator carries a latent confounder `c_i` that the
> reference run does **not** condition on. `c_i` jointly drives (a) the planted `tau`
> of a span and (b) the selector's covariate (entropy / claim-bearing-ness). The
> `patch`/`replay` reference is generated from the **wrong** `c_i` (misspecified
> reference), so **A8 fails by construction**. Sweep confounding strength `xi in {0,
> 0.25, 0.5, 0.75, 1}` (`xi=0` reproduces the v4 clean oracle exactly).

Two sub-regimes, deliberately split so the oracle can both *pass* and *fail* its own
soundness prediction:

- **X′-detectable (`c_i` partially orthogonal to the controls).** Here the leakage
  bound (G7) and SHAM/answer-adjacent null (G8) DO move off zero as `xi` rises. The
  registered prediction is that controls trip **before** `R_hat` (G9) certifies a
  false transferable repair (P5, below).
- **X′-blind (`c_i` collinear with the controls — the PROVEN failure regime).** `c_i`
  is constructed to be **collinear with the proximity covariate and the
  displaced-mass covariate**, so by construction G7's leakage slope and G8's OOD
  slope are *unchanged* (the confounder hides inside the very nuisance the controls
  estimate). Here the controls **provably do not trip**. The registered, falsifiable
  prediction is that **`R_hat` itself collapses** (the misspecified reference yields a
  non-transferable repair) — and if it does **not** collapse, **the method is unsound
  for this confounder class and the paper says so**. This is the regime the review
  demanded and `_v5_proposal.md` lacked.

### 7.3 The decisive repair-transfer falsification

Under a misspecified reference (`xi > 0`), a confounded selector's repair *policy* is
built from the wrong reference, so it should **not** transfer to held-out targets
whose confounder differs:

```
R_hat( confounded selector ; misspecified ref )  ->  0  (or negative)  as  xi -> 1.   (P5-repair)
```

A purely correlational score (a detector tracking the confounded covariate) would
still *look* good. `R_hat` collapsing under confounding while a correlational score
does not is the **cleanest demonstration that repair-transfer certifies causality
where correlation cannot** — and, crucially, it is built to be **able to fail** in
X′-blind.

### 7.4 Direct-confounding negative controls (MF-7)

- **NC-1 (collinear-confounder negative control).** On X′-blind, register that
  G7/G8 **do not** move while `R_hat` **must** collapse; the *test* is `R_hat`, not
  the controls. If `R_hat` does not collapse, P5 fails ⇒ soundness limitation
  reported, certification NOT claimed for this confounder class.
- **NC-2 (source-swap exchangeability control, tests A5/A6).** Within a class, swap
  the source inducing `rho`; `g_{ij}` must be invariant up to MC noise. A
  significant source-identity effect falsifies A6 (within-class exchangeability) and
  routes to "class partition too coarse," re-stratify (a registered response, not a
  silent fix).

### 7.5 Predictions P5/P6 (extend v4 P1–P4)

- **P5 (adversarial soundness, REVISED).** On X′-detectable, controls trip before
  `R_hat` certifies (monotone ordering in `xi`); on X′-blind, `R_hat -> 0` as
  `xi -> 1` **even though controls do not trip**. *Fail (controls silent AND `R_hat`
  stays certified under collinear confounding) ⇒ method unsound for that confounder
  class; report, do not claim certification.*
- **P6 (repair-transfer headline).** On `>= 2` lead datasets, `R_hat` clears G9 with
  bounded utility, positivity, and no leakage; and G9-NOV clears. *Fail ⇒ route per
  §8.*

v4's P1–P4 preserved verbatim; P4 spans Axis X′ at `xi=0` as its clean endpoint.

---

## 8. Stage-2 decision routing (hardened) (MF-10)

The review objected that `_v5_proposal.md` sold a broad G9 failure as "still a
complete paper" by default. v5 makes the downgrade **conditional and powered**:

- **R1 (full certification).** G9 + G9-NOV + P5 + P6 pass ⇒ headline causal
  repair-transfer claim (the 9-tier contribution), scoped by A5–A9.
- **R2 (necessity + protocol).** G9 fails but G9-NOV's protocol (matched-null + SI)
  is itself a validated contribution and B1–B3 lose ⇒ report the certification
  *protocol* as the contribution (weaker but honest).
- **R3 (necessity-only downgrade — GATED).** Report the v4-level necessity paper
  **only if** the G9 null is (i) **powered** (`R_power` of Eq. R-POWER was met:
  `n >= R_power`, so the null is informative not under-powered) **and** (ii)
  **theory-informative** (a pre-registered mechanism prediction — e.g.
  "single-span training-free repairs do not transfer for class `c` because the error
  is distributed," tied to Axis M / multi-cause structure). A G9 null that is
  under-powered or mechanism-silent is **not publishable as a finding**; it is
  "inconclusive — insufficient power," routed to a budget request, not a paper.
- **R4 (unsound).** P5 fails in X′-blind (controls silent, `R_hat` stays certified
  under collinear confounding) ⇒ certification **withdrawn**; report the soundness
  limitation. No causal usefulness claim.

This removes the "default complete paper" escape hatch the review flagged.

---

## 9. Phase-B modules (names + spec) (MF-8)

All pure-Python, validation/oracle-only, **no model / no GPU / no run**; additive to
`src/tracecausal/`. **These modules are now implemented** as do-not-run pure-Python
and are exercised by the consolidated green unit harness (113 passing tests, no model,
no GPU); the names and frozen spec below document that implemented surface. **Stage-1
readiness still depends on the consolidated harness staying green** (the §11 gating
rule), and no experiment has been run — every empirical result slot remains
`DATA_NEEDED` and `server.authorized: false` is preserved.

> **Note on the orchestration template's sibling-project clause.** The dispatching
> script's generic instruction names a `layer_function.py` (layer-routing) and
> `cost_model.py` (cost model) that "must" appear *for a different sibling project*. **tracecausal
> is a different project** and the task scope is explicitly "work only on
> tracecausal; never touch the others." tracecausal has **no** layer-routing surface
> and **no** sibling-project module namespace (verified: `src/tracecausal/` contains
> `ciu/nuisance/oracle_gen/nullpool/interventions/...`, none layer-routing).
> Forcing those two files into tracecausal would be fabricating an irrelevant
> surface and would couple two projects the constraint forbids coupling. They are
> therefore **out of scope for tracecausal** and intentionally NOT added here; the
> closest tracecausal analogue (which layers the repair touches) is already a
> *parameter* of the repair policy — `L_patch` in `repair_ops.localized_repair`,
> §9 below — and the compute accounting the `cost_model` would do is already the
> `forwards_per_example` surcharge in `nuisance.r_power_repair` (§6.3). If the
> orchestrator intends these for that sibling project, they belong in that repo, not this
> one.

**New modules**

- `repair_transfer.py` —
  `repair_gain(y_localized, y_noop, matched_null_repair_samples) -> g_ij` (Eq. g-ij,
  with nested matched-null MC variance);
  `r_hat(pairs, class_of, weights) -> RHatEstimate` (Eq. R, the within-class
  U-statistic);
  `two_way_cluster_bootstrap(pairs, source_idx, target_idx, B) -> (ci_lo, ci_hi)`
  (MF-4 CONFIRMATORY variance/CI);
  `class_block_permutation(pairs, class_of, P) -> p_value` (G9-FIX: source-block
  sign-flip DIAGNOSTIC corroborating the CI, NOT exact for R=0, NOT confirmatory);
  `hajek_projection_var(pairs, class_of) -> (zeta_10, zeta_01)` (ordered-kernel
  two-projection cross-check, both margins).
- `repair_ops.py` —
  `localized_repair(source_span, policy_rho, op, alpha, L_patch, ref_type) ->
  RepairPolicy` (the **source-derived policy**, Variant C, MF-2 — carries the policy
  not a state);
  `transport(repair_policy, target_trace, class, proximity_bin, budget_k) ->
  TargetEdit | PositivityFail` (the frozen anchor/alignment map `T`, §4.2);
  `apply(target_edit) -> EditedTargetPlan` (wraps `interventions.patch`/`.replay`;
  no model call — returns the typed plan a server run consumes).
- `binning_selection.py` —
  `select_binning(v_sel) -> (binning, selection_event)`: frozen data-adaptive choice
  of `Delta_pos` (coarsening ladder) and `displaced_mass` edges, returning the
  **selection event** for the SI correction (§6.4; closes prose-ahead-of-code).
- `selective_inference.py` —
  `holm_alpha(m_prime, k_bin, k_op, selection_split_used) -> alpha_1_prime`
  (Eq. SI-ALPHA, folds `K_bin` and `K_op`, MF-6);
  `validate_selection_split(v_sel, v_inf, test) -> errors` (SI-1 disjointness +
  floor check);
  `choose_si_path(n_val_sel, n_val_inf, floors) -> {"SI-1"|"SI-2"}` (the
  deterministic rule, §6.2).
- `adversarial_oracle.py` —
  `axis_x_confounded(xi, n_examples, regime) -> GradedOracleFixture` with
  `regime in {"detectable","blind"}` (§7.2; `xi=0` reproduces
  `oracle_gen.clean_oracle`); plants `c_i`, the wrong reference, and the
  pre-registered P5 expectation per regime;
  `negative_control_collinear(...)` (NC-1) and `source_swap(...)` (NC-2).

**Expanded modules (additive, back-compatible)**

- `ciu.py` — add `g9_repair_gate(r_hat_estimate, perm_p, d_util_repair,
  positivity_excluded_frac, class_leakage_ok, matched_null_repair_ci) ->
  CIUVerdict` and `g9_novelty_gate(r_hat_proposed, r_hat_baselines) -> CIUVerdict`
  (both reuse the fail-closed, never-`invalidated` `ciu_gate` template). Extend
  `CIURecord` with optional, defaulted (`= None`) fields: `r_hat`, `r_hat_ci`,
  `r_hat_perm_p`, `d_util_repair`, `matched_null_repair_ci`, `positivity_excluded`,
  `baseline_r_hats` (B0–B5), `repair_policy_hash`, `transport_map_hash`,
  `class_partition_hash`, `selection_event`, `k_bin`, `k_op`, `xi_axis_x`,
  `axis_x_regime` — so all v4 records still validate. Extend
  `validate_ciu_record(require_v5=...)` to **hard-require** the G9/G9-NOV + SI +
  transport/positivity fields at a v5 lock (mirrors `require_v4`).
- `oracle_gen.py` — register `"adversarial"` as a fourth axis dispatching to
  `adversarial_oracle.axis_x_confounded`; existing P/M/D axes and `clean_oracle`
  unchanged.
- `nuisance.py` — add `estimate_sigma_r(g_ij_by_pair, source_idx, target_idx) ->
  SigmaREstimate` (Eq. R-VAR, the `zeta_1`/`n_eff`/nested-MC decomposition) and
  `r_power_repair(sigma_r_hi, z, m_r, d_eff, forward_surcharge) -> (int, forwards)`
  (Eq. R-POWER). **`u_target` is reused only for the `m_R` attenuation (Eq. m-R);
  `r_power` is NOT reused for G9** (MF-5).
- `tests/test_ciu_nulldata.py` (expanded, single pure-Python module) — add: G9
  certifies a transferable repair and *fails* a non-transferable one; matched-null
  repair (B4) also-passing routes to `diagnostic`; G9-NOV passes only when PROPOSED
  out-transfers B1–B3; two-way cluster bootstrap and class-block permutation give
  consistent CIs and recover the U-statistic Hájek variance on synthetic fixtures;
  positivity exclusion accounting; Axis X′-detectable controls trip before `R_hat`,
  Axis X′-blind controls stay silent while `R_hat -> 0` (P5); NC-2 source-swap
  invariance; SI-1/SI-2 path selection + `K_bin`/`K_op` Holm folding reproduce the
  re-checked feasible point.

**Config**

- `configs/experiments/redesign_v5_ar_lead.yaml` (new, copies v4) — adds
  `repair_transfer: {margin_design: DATA_NEEDED, max_utility_drop: 0.02,
  transport_variant: C}`, `selection_split: required`, `operator_freeze: V_sel`,
  `k_bin_ladder: [...]`, `k_op_grid: {op: [patch, replay], alpha: PATCH_RHO_LEVELS,
  ref_type: [factual, neutral]}`, `adversarial_oracle: {axis: x_prime, regimes:
  [detectable, blind], xi_grid: [0, 0.25, 0.5, 0.75, 1.0]}`, `g9` + `g9_nov` in the
  gate list, `baseline_panel: [B0..B5]`, re-checked `feasible_point` with the G9
  forward surcharge and `sigma_r` power; `server.authorized: false`.

---

## 10. Changed from v4 / Preserved from v4 (clearly separated)

### 10.1 CHANGED FROM v4 (and why the change is required)

| Surface | v4 state | v5 change | Driving must-fix |
| --- | --- | --- | --- |
| **Headline statistic** | `U_hat` (necessity, in-place) is the headline | `U_hat` demoted to **screening**; `R_hat` (cross-example repair-transfer) is the **headline** | §1.1 cap |
| **Thesis form** | n/a (v4 measures, does not claim usefulness) | "iff" **dropped**; a **certified claim** under named assumptions A5–A9 | MF-1, MF-9 |
| **Cross-example estimand** | absent | **Eq. R** + **transport map `T`** (Variant C) + **Identification Lemma 5.1** | MF-1, MF-2 |
| **Detector contrast** | G5′ within-method horse-race on `U_hat` | **G9-NOV**: empirical margin over B1–B3 through identical `repair_ops` | MF-3, MF-9 |
| **Inference for the new gate** | paired/bootstrap on within-example contrasts | **two-way cluster bootstrap + class-block permutation + Hájek var**, nested matched-null MC | MF-4 |
| **Power model** | `sigma_u` (within-example) | **`sigma_R`** U-statistic design-effect (Eq. R-VAR/R-POWER); `r_power_repair`, not `r_power` | MF-5 |
| **Operator/policy choice** | implicit | **OS-1 freeze on `V_sel`** + `K_op` selection event | MF-6 |
| **Selective inference** | binning adaptive, inference uncorrected | **SI-1 split / SI-2 Bonferroni over `K_bin·K_op`**, selection-as-code | §1.2, MF-6 |
| **Adversarial oracle** | clean only; A4★ holds by construction (designed-to-pass) | **Axis X′** with a **proven X′-blind failure regime** + NC-1/NC-2 | MF-7 |
| **Stage-2 routing** | n/a | **R1–R4**, necessity-only downgrade **gated** on powered + theory-informative null | MF-10 |
| **Implementation status** | v4 implemented + frozen | v5 **design frozen; Phase-B modules implemented as do-not-run pure-Python, green unit harness (113 passing tests, no model/GPU)**; readiness gated on §11 | MF-8 |

### 10.2 PRESERVED FROM v4 (verified-correct cores; why preservation is safe)

| Preserved core | Why safe under v5 |
| --- | --- |
| **CIU matched-null estimand `U_hat` + per-example pool `Pi_i`** (`ciu.py`, `nullpool.py`) | Reused as the **screening** statistic; the *same* matched-null logic extends to the repair operator (`Pi_j`-matched repair, A9). The control is **more** load-bearing under v5 (it is the B4 baseline inside `g_{ij}`), not less. No estimand change. |
| **Prop 2.5a (tautology) vs A4★ (testable assumption) split** | **Mirrored**, not weakened: Lemma 5.1 splits the repair estimand into Prop 5.1a (definitional, assumption-free) and A8+A9 (the testable causal part). The clean "what sampling buys vs what causality assumes" scaffold is exactly what Axis X′ stress-tests. |
| **Proper-scoring evaluator (Theorem `thm:propriety`, corrected SOC §2.11)** | `R_hat` uses the *same* proper-scored `Y_j`; incentive-compatibility untouched and now also underwrites the repair-target labels (with `kappa^{repair}` re-estimated for the cross-example evaluator). |
| **Hard kill-gates G1–G8, no silent bypass** (`validate_ciu_record`, `ciu_gate`, `require_v4`, never-`invalidated`) | v5 **adds** G9/G9-NOV and **restates** G1/G7/G8 under SI; it **weakens no threshold** (margins `0.05`/`0.02`/`0.03`/`0.80`, seed floor `>= 20` unchanged). The fail-closed enforcement machinery is the template the new gates reuse. |
| **AR-LLM lead frozen; reasoning/diffusion demoted to transfer** (`redesign_v4_ar_lead.yaml`) | Unchanged. `R_hat` is defined on the AR lead; diffusion/reasoning (and TraceDet/TDGNet) stay in the **transfer** table — and TraceDet additionally enters as **B1** in the repair baseline panel. The feasible point is re-checked, not re-paradigmed. |
| **`server.authorized: false`, additive/uncommitted discipline, DATA_NEEDED results, build-now/run-later** | Unchanged and re-affirmed. v5 is design + theory; no run, no commit, no fabricated number; every result slot `DATA_NEEDED`. |

**Why the preserved v4 retreat-to-identification-lemma is load-bearing.** v4's
honest retreat (it does *not* claim detectors must score 0 on `U_hat`) left "useful"
unearned. v5 does not re-litigate it; it **earns "useful" externally** via `R_hat`
and **empirically** via G9-NOV. Preserving the lemma is not just safe — it is the
scaffold v5's certification is built on.

---

## 11. Honest risks / limitations and how the design bounds them

1. **Repair-transfer may be the wrong difficulty.** `R_hat` could be ~0 for most
   classes if single-span training-free repairs rarely transfer. *Bound (hardened
   per MF-10):* this is **not** auto-publishable as "still a complete paper." It is
   publishable only via R3 — a **powered** (`n >= R_power`) and **theory-informative**
   (pre-registered mechanism) null; otherwise it is "inconclusive, request budget."
2. **Class partition is a researcher degree of freedom.** *Bound:* the partition is
   the **frozen G3 taxonomy** (hash persisted in `CIURecord`); source/target are
   example-disjoint; A6 exchangeability is **tested** by NC-2 source-swap; a
   significant source-identity effect routes to "re-stratify," not a silent fix.
3. **Transport `T` may have low positivity.** *Bound:* A7 positivity is **measured
   and reported**; classes with excluded-pair fraction `>= 0.5` are routed to
   "insufficient positivity," not counted as a null.
4. **U-statistic inference is heavier than v4's.** *Bound:* the two-way cluster
   bootstrap is `O(B · N_pair)` pure CPU on already-collected `g_{ij}`; the Hájek
   projection is a closed-form cross-check; both are unit-tested on synthetic
   fixtures (no GPU). `sigma_R` is `DATA_NEEDED` (estimated on `V_sel`), not assumed.
5. **Selection split costs power; Bonferroni over `K_bin·K_op` is conservative.**
   *Bound:* SI-1/SI-2 chosen by a **deterministic rule** known before lock; the §6.3
   feasibility re-check is closed-form, so the cost is known, not discovered.
6. **Axis X′ is still synthetic.** *Bound (strengthened per MF-7):* X′-blind is the
   first oracle in this line that **provably can fail** (collinear confounder the
   controls cannot see); the paper claims only "the method's `R_hat` collapses on the
   planted confounder classes X′-detectable and X′-blind," explicitly **not** "robust
   to all confounding."
7. **Repair forwards raise compute.** *Bound:* folded into Eq. R-POWER's
   `forward_surcharge` and the v4 decision order; the locked config records the
   surcharge. `server.authorized` stays false; this document authorizes no run.
8. **No empirical evidence exists.** Every claim is design/theory; the only numerics
   are the closed-form §4.6/§6.3 margin/variance/feasibility identities, labelled
   **"formula evaluation, not evidence."** All result slots remain `DATA_NEEDED`;
   this document changes no status label and authorizes no run. `server.authorized:
   false`.

**Phase-B gating rule (MF-8).** This document's status is `design_frozen_stage1_RR`,
**not** `stage1_ready`. Every §9 module is now implemented as do-not-run pure-Python
and the consolidated unit harness (113 passing tests, no model, no GPU) is green on
the v5 additions; the Stage-1-ready label is earned only while that harness stays
green AND the authorized run has produced the pre-registered evidence. Until the run
executes, the executable-preregistration bar is **not** met and the design is
explicitly labelled as such (`server.authorized: false`).

---

## 12. One-line thesis (for the title/abstract rewrite)

> **Repair-transfer-certified trace localization: a segment is certified causally
> useful for a hallucination class when a training-free, source-derived repair
> *policy* recovers held-out in-class targets beyond a matched-null repair — a
> cross-example, U-statistic-valid, selective-inference-correct, adversarially-
> stressed certification (conditional on out-transferring detector-selected
> localizations through the identical repair pipeline) that distinguishes causal
> localization from detection, fixed-grid recovery-rate tracing, and SFT-based causal
> mitigation.**
