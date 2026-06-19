# REDESIGN v5 proposal — tracecausal (HARDER-BAR pass)

Status: `proposal_only`. **No server run authorized; `server.authorized` stays
false.** This is a Stage-1 Registered Report proposal: it re-opens the *core
design* under a harder top-venue bar, names the single weakness that caps v4
below 9, and proposes a sharpened v5 contribution. **No experiment, no model
load, no GPU/heavy-CPU job, ZERO fabricated numbers.** The only numerics below
are closed-form arithmetic, each labelled **"formula evaluation, not evidence."**
Every empirical slot remains `DATA_NEEDED`. This file is **additive**: it ADDs
`docs/redesign/_v5_proposal.md` and modifies nothing in `src/`, `configs/`,
`paper/`, or any other doc. The Phase-B module list below is a *plan*, not an
edit.

This document is written to be read against `REDESIGN_v4.md` and `paper/main.tex`;
section numbers in the form "v4 §X" refer to those.

---

## 0. One-paragraph executive summary

v4 builds an *honest* matched-null causal estimand (CIU, `U_hat`) and proves its
internal machinery is leakage-safe, but its **central claim is certified only by
internal consistency** — `U_hat` is unbiased for a *contrast it itself defines*,
and every kill-gate checks that the estimator is *well-behaved*, never that the
localized segment is *good for anything downstream*. That is the gap a hostile
reviewer exploits: "you have built a beautiful, self-referential estimand; show
me it *does* something a detector cannot." v5 closes it with a single principle —
**localization certified by intervention usefulness, not correlation**: a segment
is causally useful **iff a training-free, localized repair applied to it
generalizes** (fixes held-out instances of the same hallucination at a budget a
matched-null repair cannot match). This makes the **patching/repair machinery the
star** (the real differentiator vs TraceDet, a detector), upgrades the graded
oracle from a *designed-to-pass* fixture to one carrying **genuine falsification
risk** (an adversarial oracle axis with a misspecified reference run / hidden
confounder), and closes the **selective-inference gap** (multiplicity over
data-adaptive bin selection) that v4 silently leaves open. The CIU estimand,
the Prop 2.5a/A4★ split, the proper-scoring theorem, and the hard kill-gates
G1–G8 are **preserved**; v5 *adds an outer certification layer and a falsifiable
repair-transfer gate*, and *re-derives* only the two surfaces that fail the harder
bar (selective inference; oracle falsification risk).

---

## 1. The biggest weakness of v4's central claim

### 1.1 Statement (the single capping issue)

> **v4's central claim is certified by *internal consistency*, not by *external
> usefulness*. The estimand `U_hat` is provably unbiased for a contrast that the
> method itself defines (selected-segment effect minus matched-pool effect), and
> every kill-gate G1–G8 verifies that this estimator is *honest and well-behaved*
> — but nothing in v4 demonstrates that a CIU-localized segment is *useful for any
> downstream task a detector could not already serve*. The contribution, as
> framed, is "a self-consistent measurement of a quantity we invented," which is a
> 7-tier contribution, not a 9-tier one.**

This is the weakness that the GPT-5.5 review's "prose ahead of executable
preregistration" was a *symptom* of: v4 responded by hardening the machinery
(graded oracle, OOD deflation, leakage CI, feasibility point) — all genuinely
good — but hardening the *measurement* does not answer the *"so what"*. Three
concrete consequences a hostile reviewer will name:

1. **The estimand is closed under its own definition.** Lemma 2.5 (`paper/main.tex`
   `lem:unbiased`) proves `E[U_hat] = E_i[tau_i(S*)] − E_i[bar tau_i(Pi)]`. This is
   true *by construction of `Pi` and the matched arm*. A reviewer asks: what
   licenses calling `tau_i(S*) − bar tau_i(Pi)` **"causal usefulness"** rather than
   "the above-pool effect size of whatever the selector happened to pick"? v4's
   own honesty — the deliberate retreat from a non-gameability theorem to an
   identification lemma — *removes* the load-bearing claim and replaces it with a
   tautology plus a testable assumption. After that retreat, **the word "useful"
   in "intervention-useful" is unearned**: the design measures an effect, it never
   shows the effect is *useful*.

2. **The novelty gate G5′ is necessary but not sufficient for impact.** "Our
   selector's `U_hat` beats the best adapted detector's by ≥0.03" establishes that
   the selector finds *higher-above-pool-effect* segments than a detector adapted
   to the same scoring. But `U_hat` is *our own metric*; beating detectors *on our
   own metric* is close to circular unless the metric is independently anchored to
   something a practitioner wants. A reviewer: "you defined the ruler, then showed
   you win on your ruler." Without an external anchor, G5′ is a within-method
   horse-race.

3. **The graded oracle is built to pass.** v4 §2.10G plants the ground-truth `tau`
   structure (`oracle_gen.py::clean_oracle`, `_axis_partial_leakage`,
   `_axis_multi_cause`, `_axis_distractor`) and pre-registers the *exact curve the
   estimator should trace*. Every axis is a sweep whose answer is known and whose
   estimator is constructed to recover it (`expected_u_hat = tau_selected −
   bar_tau_pool`). This demonstrates *correct degradation under controlled erosion*
   — valuable — but it **cannot fail in a way that falsifies the method**, because
   the fixture and the estimator share the same generative assumption (A4★ holds by
   construction). A correctness fixture that is *designed to pass* is not a
   falsification risk; a reviewer reads it as a unit test promoted to a scientific
   claim.

### 1.2 Two secondary soundness gaps that compound the cap

- **Selective-inference gap (multiplicity over data-adaptive bin selection).** v4
  *adapts* two analysis choices to the data and then runs confirmatory tests on
  the *same* data scale: (i) the proximity-bin width `Delta_pos` is **coarsened
  data-adaptively** until `bar m_pool >= 8` (v4 §4.6 B3; `nuisance.POOL_MIN`,
  `nullpool.PROXIMITY_POOL_MIN`), and the *same* `Delta_pos` then sets the G7
  leakage bound `B_UCI = beta_hi * Delta_pos`; (ii) `displaced_mass` bins
  (`displaced_mass_bins`) define the OOD-deflation regression whose slope feeds the
  deflated `U_hat` that G1 gates on. The Holm correction (v4 §4.2) covers the
  *family of gates*, but **not the data-adaptive selection of the binning that
  defines those gates' test statistics**. This is textbook selective inference
  (post-selection inference / the "garden of forking paths"): a confirmatory `p`/CI
  computed after a data-driven model-selection step, without conditioning on the
  selection event, is **anti-conservative**. v4 mentions the bound/variance trade
  as "explicit, pre-registered" but never *corrects the inference* for the
  adaptivity. A statistics-literate reviewer flags this immediately.

- **Prose-ahead-of-code on the binning algorithms.** The OOD-deflation binning and
  the proximity-bin coarsening are described in prose (v4 §2.12, §4.6 B3) and the
  *estimators* exist (`ciu.ood_deflation`, `nuisance.pool_inflation`), but the
  **data-adaptive selection procedure itself** — the loop that picks `Delta_pos`,
  the deflation-bin edges, and (critically) the *event being conditioned on* — is
  not a frozen, unit-tested function. The exact failure the prior review named
  ("registered-report prose ahead of the executable preregistration") survives in
  these two procedures. The code freezes the *consequences* of a selection; it does
  not freeze the *selection*.

### 1.3 Why this caps the score below 9 (reviewer voice)

A 9-tier paper at NeurIPS/ICML/ICLR needs a claim of the form *"we can do X that
prior work cannot, and here is the falsifiable test that X holds."* v4's X is "we
can *measure* an above-pool causal contrast honestly." Honest measurement of a
self-defined quantity, however rigorous, reads as **infrastructure, not
discovery**. TraceDet (the base paper) already *detects* hallucinations from
traces and reports a concrete downstream win (+15.2% AUROC). v4's reply — "but our
quantity is causal, not correlational" — is **only persuasive if causality buys
something detection does not**, and v4 never cashes that in. The cap is: *the
differentiator vs the base paper is asserted (causal > correlational) but never
operationalized into a capability the base paper lacks.*

---

## 2. The sharpened v5 central contribution

### 2.1 The thesis (one crisp sentence)

> **A trace segment is causally useful for hallucination iff a training-free,
> localized *repair* applied to it *transfers* — i.e. fixes held-out instances of
> the same hallucination class at an edit budget that a matched-null repair
> provably cannot — and we certify localization by this *repair-transfer*
> usefulness, not by any correlational or self-defined effect-size contrast.**

Operationally: v4's `U_hat` (above-pool factuality *change* under `mask`) becomes
the **screening** statistic, and a new **certification** statistic `R_hat`
(above-null *repair-transfer*) becomes the **headline**. The patching/repair
machinery — `patch`/`replay` in `interventions.py`, currently demoted to
"other operators" — is promoted to **the star**: it is what *certifies* the
localization, because a repair is the one thing a detector categorically cannot
produce.

### 2.2 The certification statistic `R_hat` (equation-level)

Let `S*(x_i)` be the selector's localized segment on a *source* example `x_i`. A
**localized repair** `phi(S*)` is a training-free edit derived **only from
`S*` and a reference run** (e.g. a `patch` at mixing weight `alpha` toward the
factual reference state on `[a,b]`, or a `replay` re-decode of `[a,b]` under the
reference policy — both already specified, `interventions.patch` / `.replay`). The
repair is then **applied to a held-out target example `x_j` of the same
hallucination class** (same atomic-claim type / same error taxonomy bucket; the
G3 taxonomy already partitions these), at the **matched edit budget** `k`.

Define the per-instance **repair gain** as the factuality recovery the localized
repair produces on the target, minus the recovery a **matched-null repair**
(the *same* operator at the *same* budget on a segment drawn from the per-example
matched pool `Pi_j`, §2.3) produces:

```
g_{ij} = [ Y_j( do(phi^{rho}_{S*}) ) − Y_j(no_op) ]               (localized repair)
       − E_{tilde S ~ Pi_j}[ Y_j( do(phi^{rho}_{tilde S}) ) − Y_j(no_op) ].   (matched-null repair)
```

`Y_j` is the proper-scored evaluator factuality (Theorem `thm:propriety`,
preserved). The **repair-transfer estimand** is the cross-example mean

```
R(selector; rho) = E_{(i,j): class(i)=class(j), i != j}[ g_{ij} ],
                                                                        (R-estimand)
```

and `R_hat` is its paired-bootstrap estimator over matched source/target pairs.
The headline kill-gate (G9, §3) requires `R_hat`'s lower CI to clear a
pre-registered repair margin `m_R` **with a strictly bounded utility cost**.

**Why this is not just a second `U_hat`.** `U_hat` measures *necessity on the same
example you localized on* (mask `S*` in `x_i`, see factuality drop in `x_i`):
within-example, ablation-style, and — crucially — **a detector's covariates can
also produce a positive `U_hat`** (v4's own admission, the reason G5′ retreated to
a novelty gate). `R_hat` measures *whether the thing you localized is a reusable
repair across examples*: cross-example, **sufficiency-style** (you *fix*, not
*break*), at a budget. A detector outputs a score; it does not output a
**transferable edit**. So `R_hat > 0` is a capability **no detector can exhibit by
construction**, which is exactly the differentiator the cap demanded.

### 2.3 The single unifying principle (so it is not a bag of tricks)

All three HARDER-BAR moves are *instances of one principle*:

> **Counterfactual usefulness must be earned out-of-sample, under a reference the
> method does not control.**

- The **certification** (`R_hat`, G9) earns usefulness *out-of-sample across
  examples* (held-out target `x_j`).
- The **selective-inference correction** (§3.2) earns the gate verdicts
  *out-of-sample across the data-adaptive binning* (the bin choice is a selection
  event that must be paid for, via a held-out selection split or a
  selection-conditioned correction).
- The **adversarial oracle** (§4) earns A4★ support *out-of-distribution in the
  reference*: the oracle now includes a regime where the reference run is
  **misspecified** (a hidden confounder the method is *not* told about), so the
  graded family can **genuinely fail** rather than recover a planted answer.

One sentence states the whole paper: *causal localization is only as good as the
repair it licenses on data and references you did not get to choose.* That is the
spine; the modules hang off it.

### 2.4 Novelty vs the base paper and named prior art (specific, no hand-waving)

**vs TraceDet (Chang et al., ICLR 2026 — the base paper).** TraceDet is a
**detector**: it formulates the D-LLM denoising trajectory as an action trace,
applies an information-bottleneck objective `min −I(Y;A_sub) + beta I(A;A_sub)`
(their Eq. 3) to extract a maximally-informative sub-trace `A_sub = g(A)`, and
trains a classifier `f(A_sub)` reporting AUROC (their Table 1, +15.2% avg). It is
**correlational by design** — `I(Y;A_sub)` is mutual information between a sub-trace
and the *label*, with no intervention, no counterfactual, no edit, no reference
run. v5's `R_hat` is the quantity TraceDet's framework **cannot express**:
TraceDet's `A_sub` is "the part of the trace most *predictive* of hallucination";
v5 certifies "the segment whose *localized repair transfers*." A reviewer who
likes TraceDet will see that v5 answers the question TraceDet's own conclusion
defers ("future work: mitigation"). TraceDet enters v5 only as a segment-adapted
detector baseline in the diffusion *transfer* table (preserved from v4), and its
IB sub-trace is, at best, a strong *screening* selector our `R_hat` certification
must out-transfer.

**vs FCCT / IRI (Li et al., AAAI-26 — "Causal Tracing of Object Representations
in LVLMs").** This is the closest *interventional* neighbour and must be named
explicitly. FCCT does genuine causal tracing: Gaussian-noise corruption + clean
activation restoration, scored by **Recovery Rate** `RR = (P_patched −
P_corrupted)/(P_clean − P_corrupted)` (their Eq. 2) over a **fixed grid** of
components × layers × seven token categories; IRI is their training-free
inference-time repair. Three sharp differences, each a v5 contribution:
   - **Unit and grid.** FCCT traces *model components* (MHSA/MLP/hidden-state) on a
     *pre-enumerated* token taxonomy; v5 localizes a *data-adaptive claim span in
     the generation trace*, where *which span* is the discovery. FCCT has no
     multiplicity problem because its grid is fixed a priori; v5's grid is
     data-selected, which is *why* v5 must (and does, §3.2) solve the
     selective-inference problem FCCT never faces.
   - **No matched-null / no above-pool contrast.** `RR` compares patched-vs-corrupted
     for a *single* component; it has **no per-example matched-null pool**, so it
     cannot separate "this component matters" from "editing *any* matched component
     this much matters." v5's `Pi`-matched contrast (preserved) is precisely this
     control, and v5 extends it to the *repair* operator (the `Pi_j`-matched repair
     baseline in `g_{ij}`).
   - **Certification by transfer.** FCCT/IRI validate the repair (IRI) by *aggregate
     downstream AUROC/accuracy on the same distribution*; they never test that a
     repair **localized on one instance transfers to a held-out instance of the
     same error class at a matched budget**. `R_hat` (G9) is exactly that test.
     v5's relationship to FCCT is the relationship of a *certified, multiplicity-
     correct, matched-null repair-transfer estimand* to an *uncorrected fixed-grid
     recovery-rate score*.

**vs CDCR-SFT (Li et al., AAAI-26 — "Mitigating Hallucinations via Causal
Reasoning").** CDCR-SFT is **supervised fine-tuning** on causal-DAG construction
(their CausalDR dataset, 25,368 samples). It is the *opposite* design point:
training-heavy, dataset-building, operating on *variable-level DAGs* over a
reasoning trace. v5 is **training-free, localization+repair**, operating on
*token spans* of an AR generation. The user's original annotation floated "GNN
over the causal graph / structured reward instead of SFT"; v5's principled answer
is that CDCR-SFT already occupies the SFT corner, so v5 differentiates by being
the **training-free, intervention-certified** corner — the `R_hat` repair *is* the
"structured-reward-without-SFT" idea cashed out as a falsifiable estimand rather
than a loss function.

**vs classic causal mediation / ROME / activation patching (Vig 2020; Meng 2022;
Geiger 2021).** These localize *factual associations or circuits* by patching and
reading a recovery, on curated probe sets. v5 borrows the intervention-as-evidence
stance (preserved citation stance from `paper/main.tex` `sec:related`) but adds
the three things none of them have together: a **per-example matched-null pool**,
a **selective-inference-correct data-adaptive localization**, and a
**cross-example repair-transfer certification**. The novelty is the *certification
principle and its falsifiable gate*, not any single operator.

**Net novelty statement (for the abstract).** *We are the first to certify
hallucination localization by training-free repair-transfer — fixing held-out
instances of the same error class with a localized, matched-null-controlled,
multiplicity-correct edit — turning the counterfactual patching machinery from a
measurement into a falsifiable capability that detection-based tracing (TraceDet)
and fixed-grid recovery-rate tracing (FCCT) cannot provide.*

---

## 3. The falsifiable test / kill-gate that makes v5 NON-stitched

The principle of §2.3 yields **one new headline gate (G9)** plus a **selective-
inference correction (G7/G8/G1 restatement)**. These are not additive tricks; they
are the *same* "earn-it-out-of-sample" requirement applied to three surfaces.

### 3.1 G9 — Repair-transfer certification (the headline falsifiable gate)

> **G9 (repair-transfer usefulness).** On `>=2` lead datasets, the cross-example
> repair-transfer estimator `R_hat` (§2.2) clears a pre-registered repair margin:
> Holm-corrected paired-bootstrap CI lower bound `> m_R`, with **bounded utility
> cost** `D_util^{repair} <= 0.02` and **no class leakage** (source and target are
> disjoint examples of the same error class; an example is never repaired by a
> segment localized on itself). *Fail ⇒ the localization does not license a
> transferable repair: the causal claim downgrades to "necessity-only" (the v4
> claim) and the paper reports `U_hat` necessity without the usefulness headline.*

**Why this is genuinely falsifiable (can lose, routes honestly).** `R_hat` can be
≤0 even when `U_hat` is large and clean: a segment can be *necessary in-place*
(masking it breaks `x_i`) yet its repair may *not transfer* (the localized edit is
example-idiosyncratic, or the hallucination class is not repairable by a single-
span training-free edit). That is a real, common, scientifically interesting
outcome — and it is a **falsification of the usefulness thesis**, cleanly
separated from a falsification of the necessity claim (G1). The gate has three
distinct loss routes:
- `U_hat` passes, `R_hat` fails ⇒ **necessity without usefulness** (honest
  downgrade to the v4 claim; still publishable, weaker).
- `R_hat` passes only with the matched-null repair *also* passing (i.e. `g_{ij}`
  CI overlaps 0 because *any* matched repair transfers) ⇒ **repair is generic, not
  localized** ⇒ the localization is not what licenses the repair (a novelty/
  identification distinction, routed to "diagnostic," never "invalidated").
- `R_hat` passes with utility cost > 0.02 ⇒ **repair trades factuality for
  utility** ⇒ reframe as abstention (mirrors G2).

**Pre-registered margin `m_R` (formula evaluation, not evidence).** `m_R` is set
by the *attenuation-adjusted* repair effect under the *same* `kappa`-attenuation
machinery as G1 (preserved `nuisance.u_target`): `m_R := m_R0 / (2*kappa_lo − 1)`
with `m_R0` the design repair margin. To keep the power budget feasible without
re-opening §4.7, `m_R0` is pinned at the **necessity margin** `0.05`
(`NECESSITY_MARGIN`) so G9 reuses the exact MDE/`R_power` arithmetic v4 already
exhibited (`n=850`, `sigma_hi=0.30`, `z` from the Holm family — now with the
family enlarged by the G9 tests, see §3.3). No new fabricated number; the
feasibility re-check is a closed-form re-evaluation of v4 §4.7 at the new family
size `m` (§3.3).

### 3.2 Selective-inference correction (re-derive G7/G8/G1 under data-adaptive binning)

The principle says the **bin selection is a selection event that must be paid
for**. v5 pre-registers a **selection split** discipline (the clean, top-venue-
defensible fix) plus a fallback **selection-conditioned correction**:

- **(SI-1) Selection split (primary).** Partition the validation data into a
  **selection split** `V_sel` and an **inference split** `V_inf`, disjoint from
  test. The data-adaptive choices — proximity-bin width `Delta_pos` (the
  coarsening loop of v4 §4.6 B3) and the `displaced_mass` bin edges (v4 §2.12) —
  are frozen **using `V_sel` only**. The nuisance estimates (`sigma_u`, `kappa`,
  `s_OOD`, `beta`) and all gate statistics are then computed on `V_inf` / test
  with the binning **held fixed**. Because the binning is independent of the
  inference data, the Holm family on `V_inf`/test is valid without
  post-selection adjustment. This is the **same out-of-sample principle** as G9,
  applied to the analysis pipeline.

- **(SI-2) Selection-conditioned bound (fallback, when `V_sel` is too small).** If
  splitting costs too much power (the `n_val` floors of v4 §4.6 bind), v5 pre-
  registers a **multiplicity penalty over the data-adaptive bin grid**: the
  coarsening loop visits an ordered, finite, pre-enumerated ladder of bin widths
  `Delta_pos in {w_1 > w_2 > ...}` (already discrete: the "coarsen one step"
  ladder), and the `displaced_mass` edges are chosen from a pre-enumerated finite
  candidate set. The number of *reachable* binnings `K_bin` is therefore
  **bounded and known before data**. v5 corrects the G7/G8/G1 confirmatory tests by
  a **Bonferroni-over-binnings** factor `K_bin` *folded into the Holm family*:
  `alpha_1 = 0.05 / (m * K_bin)`. This is conservative but **selection-honest** —
  the gates pay for every binning the adaptive loop could have reached, not just
  the one it did.

Either way, the **deliverable is a frozen, unit-tested selection procedure**
(§5, `binning_selection.py`) that emits *both* the chosen binning *and the
selection event* (the ladder it walked / the candidate set it chose from), so the
correction is computable and the "prose-ahead-of-code" gap is closed: the
selection is code, not prose.

### 3.3 Family-size and feasibility re-check (closed-form, not evidence)

Adding G9 (one repair-transfer contrast per lead cell) and, under SI-2, the
`K_bin` factor enlarges the Holm family from v4's `m` to `m' = m + (#G9 contrasts)`
and, under SI-2, multiplies by `K_bin`. The MDE arithmetic of v4 §4.7 is
re-evaluated closed-form: with the SI-1 selection-split path (preferred), `K_bin =
1` and `m'` grows by the per-cell G9 contrasts only, nudging `z` up by a small
amount; the §4.7 feasible point `(n=850, cells=2, kappa=0.92, c_fwd<=4.57e-4)` is
**re-checked, not assumed**, and if `m'`'s larger `z` pushes `R_power` above 850
the pre-registered response is the *existing* v4 decision order (request budget /
reduce cells), now also recording the G9 forward-count surcharge (a repair forward
+ a matched-null repair forward per target, persisted as `forwards_per_example`
+= the repair arms). **No number here is fabricated; this is the same closed-form
budget identity v4 already froze, re-evaluated at the new family size.** The
locked config records the realized `m'`, `K_bin`, `z`, and the re-checked
feasible point.

### 3.4 Why the whole thing is one principle, not a stitch

G9, SI-1/SI-2, and the adversarial oracle (§4) are **three applications of "earn
counterfactual usefulness out-of-sample under an uncontrolled reference."** Remove
any one and the principle is incompletely tested (usefulness untested across
examples / across binnings / across reference misspecification). They share one
falsifiable spine, one estimand family (matched-null contrasts), and one honesty
discipline (out-of-sample). That is the difference between a unifying contribution
and a bag of tricks.

---

## 4. The adversarial oracle axis (genuine falsification risk)

### 4.1 The problem v4's oracle cannot solve

v4's graded oracle (§2.10G; `oracle_gen.py`) plants `tau` and pre-registers the
recovered curve; **A4★ holds by construction on every axis**, so the oracle can
only show *graceful degradation of a correctly-specified estimator*. It cannot
show the estimator **detecting its own misspecification**, which is the property a
reviewer actually wants: *does the method know when its reference/counterfactual is
wrong?*

### 4.2 Axis X — adversarial / misspecified-reference oracle (new)

> **Axis X (hidden-confounder / misspecified reference).** The templated generator
> is extended with a **latent confounder** `c_i` that the *reference run does not
> condition on*: `c_i` jointly drives (a) the planted factuality `tau` of a span
> **and** (b) the selector's covariate (entropy / claim-bearing-ness). The
> reference state used for `patch`/`replay` is generated from the **wrong** `c_i`
> (a misspecified reference), so A4★ **fails by construction**: the matched-null
> pool is *not* the correct counterfactual, and a confounded selector achieves a
> positive `U_hat` *that is pure confounding*.

The sweep parameter is the **confounding strength** `xi in {0, 0.25, 0.5, 0.75,
1}` (0 = the v4 clean regime, recovering current behaviour; 1 = full
confounding). The oracle is **designed to make the method fail** at high `xi`,
and the pre-registered claim is **not** "the estimator recovers the truth" but:

> **The method's *own controls* must catch the misspecification.** As `xi`
> increases, the SHAM-MASK / answer-adjacent null (G8.i) and the leakage bound
> (G7.ii) must **trip** (their CIs must move off zero / the bound must exceed
> 0.03) **before** `R_hat` (G9) reports a false transferable repair. The
> pre-registered, falsifiable prediction (P5, §below) is a **monotone "controls
> trip before the headline fires" ordering** in `xi`.

This carries **genuine falsification risk**: if the method's controls *do not*
trip before `R_hat` fires under confounding, **the method is unsound and the paper
says so** — that is a real way for v5 to lose at Stage-2, which is exactly what a
registered report is supposed to expose. The clean oracle becomes the `xi=0`
endpoint of Axis X, so v4's preserved oracle is a *special case*, not deleted.

### 4.3 The repair-transfer falsification under Axis X (the sharpest test)

The decisive adversarial test combines G9 with Axis X: under a **misspecified
reference** (`xi>0`), a confounded selector's localized "repair" is built from the
**wrong** reference state, so it **should not transfer** to held-out targets whose
confounder differs. Pre-registered prediction:

```
R_hat(confounded selector; misspecified ref)  ->  0  (or negative)  as xi -> 1,
                                                                        (P5-repair)
```

while a detector that merely tracks the confounded covariate would *look* good on
any correlational metric. **`R_hat` collapsing under confounding while a
correlational score does not is the single cleanest demonstration that repair-
transfer certifies causality where correlation cannot** — it is the empirical
heart of the paper, and it is *built to be able to fail*.

### 4.4 New falsifiable predictions (extend v4's P1–P4)

- **P5 (adversarial soundness, NEW).** On Axis X, the controls trip monotonically
  before the headline: there exists `xi* < 1` such that for `xi >= xi*` G7/G8
  fail (controls catch misspecification) while for `xi < xi*` `R_hat` is
  certified; and `R_hat(confounded; misspecified ref) -> 0` as `xi -> 1`
  (P5-repair). *Fail (controls do not trip before `R_hat` fires) ⇒ the method is
  not robust to reference misspecification; report as a soundness limitation, do
  not claim certification.*
- **P6 (repair-transfer headline, NEW).** On `>=2` lead datasets, `R_hat` clears
  G9 with bounded utility and no class leakage. *Fail ⇒ necessity-only downgrade.*

v4's P1–P4 are preserved verbatim (P1 deflated necessity, P2 novelty margin, P3
operator honesty, P4 graded identification); P4 now spans Axis X at `xi=0` as its
clean endpoint.

---

## 5. Exactly what changes from v4 and what is preserved

### 5.1 PRESERVED (verified-correct; preservation is safe, with the reason)

| Preserved core | Why preservation is safe under v5 |
| --- | --- |
| **CIU matched-null estimand `U_hat` + per-example pool `Pi_i`** (`ciu.py`, `nullpool.py`, `paper/main.tex` eq. `pool`/`Uhat`) | v5 *reuses* it as the **screening** statistic and *extends* the same matched-null logic to the repair operator (`Pi_j`-matched repair in `g_{ij}`). Nothing about the estimand changes; it gains an outer certification layer. The matched-null control is *more* needed under v5 (it is the repair baseline), not less. |
| **Prop 2.5a (assumption-free tautology) vs A4★ (testable assumption) split** (`paper/main.tex` `prop:tautology`, `ass:a4`) | v5's `R_hat` does not weaken the split; the repair-transfer estimand has its **own** A4★-analogue (the `Pi_j`-matched repair is the correct counterfactual repair), which is tested by the same SHAM/answer-adjacent controls *plus* the new Axis X. The clean separation of "what sampling buys" from "what causality assumes" is exactly the scaffold the adversarial oracle stress-tests. |
| **Proper-scoring evaluator (Theorem `thm:propriety`, corrected SOC)** | `R_hat` uses the *same* proper-scored `Y_j`; the incentive-compatibility argument is untouched and now also underwrites the repair-target labels. |
| **Hard kill-gates G1–G8, no silent bypass** (`ciu.validate_ciu_record`, `ciu_gate`, `require_v4`, `require_controls`) | v5 **adds** G9 and *restates* G7/G8/G1 under selective-inference correction; it does **not** weaken any threshold (margins `0.05`/`0.02`/`0.03`/`0.80` and the seed floor `>=20` are unchanged). The fail-closed enforcement machinery (`require_v4`, fail-closed provenance, no `invalidated` verdict) is the template G9 reuses. |
| **AR-LLM lead frozen; reasoning/diffusion demoted to transfer** (`configs/.../redesign_v4_ar_lead.yaml`) | Unchanged. `R_hat` is defined on the AR lead; the diffusion/reasoning transfer study (and TraceDet/TDGNet baselines) stay in the transfer table. The feasible point is re-checked, not re-paradigmed. |
| **`server.authorized: false`, additive/uncommitted discipline, DATA_NEEDED results** | Unchanged and re-affirmed. v5 is a proposal; no run, no commit, no fabricated number. |

**The retreat-to-identification-lemma is preserved and is *why* v5 works.** v4's
honest retreat (it does *not* claim detectors must score 0) left the "useful"
unearned (§1.1). v5 does not re-litigate that retreat — it **earns "useful"
externally** via `R_hat`, which is the right repair for the right gap. So
preservation of the lemma is not just safe; it is load-bearing for v5's framing.

### 5.2 RE-DERIVED (only the surfaces that fail the harder bar)

| Surface | v4 state | v5 re-derivation |
| --- | --- | --- |
| **Selective inference over data-adaptive binning** | Trade named "explicit/pre-registered" but inference **not corrected** for adaptivity (§1.2) | **SI-1 selection split** (primary) + **SI-2 selection-conditioned Bonferroni-over-`K_bin`** (fallback), with the selection procedure frozen as code (§3.2, §5). |
| **Oracle falsification risk** | Designed-to-pass; A4★ holds by construction on every axis (§1.1.3) | **Axis X adversarial oracle** (hidden confounder / misspecified reference) with the "controls trip before headline" prediction P5 (§4). Clean oracle = `xi=0` endpoint. |
| **Binning algorithms prose-ahead-of-code** | Estimators exist; the *selection loop* is prose (§1.2) | Frozen `binning_selection.py` emitting the chosen binning **and** the selection event (§5). |
| **The "useful" in intervention-useful** | Asserted via a self-defined contrast | Re-grounded on `R_hat` repair-transfer (§2), the new headline. |

### 5.3 ADDED (the new contribution surface)

The repair-transfer certification layer (`R_hat`, G9), the adversarial oracle
(Axis X), and the selective-inference correction. Everything else is v4.

---

## 6. New / expanded Phase-B modules (names + 1-line spec each)

All pure-Python, validation/oracle-only, **no model / no GPU / no run**; additive
to `src/tracecausal/`. Specs only — this proposal writes none of them.

**New modules**

- `repair_transfer.py` — `repair_gain(y_localized, y_noop, y_matched_null_repair) -> g_ij`
  and `r_hat(pairs) -> (R_hat, ci_lo, ci_hi)`: the cross-example repair-transfer
  estimator (§2.2) with paired bootstrap over class-matched source/target pairs.
- `repair_ops.py` — `localized_repair(span, reference_state, rho, op) -> RepairPlan`:
  wraps `interventions.patch` / `.replay` into a *transferable* repair object
  (carries the source span + reference + budget) applicable to a held-out target.
- `binning_selection.py` — `select_binning(v_sel) -> (binning, selection_event)`:
  the frozen data-adaptive selection of `Delta_pos` (coarsening ladder) and
  `displaced_mass` edges, returning the **selection event** (ladder walked /
  candidate set) for the SI correction (§3.2) — closes the prose-ahead-of-code gap.
- `selective_inference.py` — `holm_alpha(m, k_bin, selection_split) -> alpha_1` and
  `validate_selection_split(v_sel, v_inf, test) -> errors`: the SI-1 split-validity
  check + SI-2 Bonferroni-over-`K_bin` folding into the Holm family (§3.2, §3.3).
- `adversarial_oracle.py` — `axis_x_confounded(xi, n_examples) -> GradedOracleFixture`:
  the hidden-confounder / misspecified-reference oracle (§4.2), with planted
  confounder `c_i`, wrong reference, and the pre-registered "controls trip before
  `R_hat`" expectation; `xi=0` reproduces `oracle_gen.clean_oracle`.

**Expanded modules (additive, back-compatible)**

- `ciu.py` — add `g9_repair_gate(r_hat_ci, d_util_repair, class_leakage_ok,
  matched_null_repair_ci) -> CIUVerdict` (the G9 headline gate, reusing the
  `ciu_gate` fail-closed / never-`invalidated` template); extend `CIURecord` with
  optional `r_hat`, `r_hat_ci`, `d_util_repair`, `matched_null_repair_ci`,
  `selection_event`, `k_bin`, `xi_axis_x` (all defaulted `None`, v4 records still
  validate); extend `validate_ciu_record(require_v5=...)` to hard-require the G9 +
  SI fields at a v5 lock.
- `oracle_gen.py` — register `"adversarial"` as a fourth axis dispatching to
  `adversarial_oracle.axis_x_confounded`; the existing P/M/D axes and
  `clean_oracle` are unchanged.
- `nuisance.py` — add `r_power_with_repair(sigma_hi, z, margin, infl,
  repair_forward_surcharge) -> int` (the §3.3 family-size + forward-count
  re-check); `u_target` reused verbatim for `m_R` attenuation.
- `configs/experiments/redesign_v5_ar_lead.yaml` (new, copies v4) — adds
  `repair_transfer: {margin: 0.05, max_utility_drop: 0.02}`, `selection_split:
  required`, `k_bin_ladder: [...]`, `adversarial_oracle: {axis: x, xi_grid: [0,
  0.25, 0.5, 0.75, 1.0]}`, G9 in the gate list, re-checked `feasible_point` with
  the G9 forward surcharge; `server.authorized: false`.
- `tests/test_ciu_nulldata.py` (expanded, single pure-Python module) — add: G9
  certifies a transferable repair and *fails* a non-transferable one;
  matched-null-repair-also-passes routes to diagnostic; Axis X controls trip
  monotonically before `R_hat` as `xi` rises (P5); `R_hat -> 0` under misspecified
  reference (P5-repair); SI selection-split validity + `K_bin` Holm folding
  reproduce the re-checked feasible point.

---

## 7. Honest risks / limitations and how the design bounds them

1. **Repair-transfer may be the wrong difficulty.** `R_hat` could be ~0 for *most*
   hallucination classes simply because single-span training-free repairs rarely
   transfer — making the headline gate fail broadly. *Bound:* this is **routed,
   not fatal** (G9 fail ⇒ honest necessity-only downgrade to the v4 claim, still a
   complete paper). The registered-report framing makes a broad G9 failure a
   *reportable scientific finding* ("localization rarely licenses transferable
   repair"), not a rejection-worthy null. The pre-registered `m_R` and the
   class-disjoint design keep the test fair.

2. **Class definition for source/target matching is a researcher degree of
   freedom.** Choosing the "same hallucination class" partition could be gamed to
   inflate `R_hat`. *Bound:* the class partition is the **frozen G3 taxonomy**
   (no new freedom), source/target are **example-disjoint** (no self-repair), and
   the matched-null repair baseline `Pi_j` absorbs any "generic repair transfers"
   confound (the `g_{ij}` contrast, §2.2). The class partition hash is persisted in
   `CIURecord` so it cannot be re-chosen post hoc.

3. **Selection-split costs power; Bonferroni-over-`K_bin` is conservative.** SI-1
   spends validation data; SI-2 over-penalizes. *Bound:* the two are
   **pre-registered with a deterministic choice rule** (SI-1 when the `n_val`
   floors of v4 §4.6 are met on both splits, else SI-2), and the §3.3 feasibility
   re-check is closed-form, so the power cost is *known before lock*, not
   discovered after. `K_bin` is bounded because the coarsening ladder and
   `displaced_mass` candidate edges are finite and pre-enumerated.

4. **Adversarial oracle is still synthetic.** Axis X plants a confounder; it shows
   the *controls* can catch *a* misspecification, not *every* real-world
   confounder. *Bound:* same honest caveat as v4's synthetic-oracle limitation
   (preserved), now *strengthened* — Axis X is the first oracle in this line that
   can **fail**, so it is a genuine (if synthetic) soundness probe rather than a
   tautological pass. The paper claims only "the controls trip before the headline
   on the planted confounder," explicitly not "robust to all confounding."

5. **Repair forwards raise the compute budget.** `g_{ij}` needs a localized-repair
   forward + a matched-null-repair forward per target pair. *Bound:* folded into
   the §3.3 closed-form re-check via `forwards_per_example` surcharge and the
   existing v4 decision order (reduce cells / request budget); the locked config
   records the surcharge. No GPU run is implied by this proposal;
   `server.authorized` stays false.

6. **No empirical evidence exists.** Every claim here is design/theory; the only
   numerics are the closed-form §3.1/§3.3 margin/feasibility re-evaluations,
   labelled **"formula evaluation, not evidence."** All result slots remain
   `DATA_NEEDED`; this proposal authorizes no run and changes no status label.
   `server.authorized: false`.

---

## 8. One-line thesis (for the title/abstract rewrite)

> **Intervention-useful trace segments, certified by repair-transfer: a segment is
> causally useful iff its training-free localized repair fixes held-out instances
> of the same hallucination class beyond a matched-null repair — a falsifiable,
> selective-inference-correct, adversarially-stressed certification that
> counterfactual *usefulness* (not correlation) is what distinguishes causal
> localization from detection.**
