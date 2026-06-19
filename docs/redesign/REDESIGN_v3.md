# REDESIGN v3 — tracecausal

Status: `design_only`. **No server run authorized; `server.authorized: false` is
preserved.** This document is a **Stage-1 Registered Report grade design**:
method + theory + a *pre-registered, pre-data analysis plan*. No experiment is
run on the GPU server or locally; no result is fabricated. The only numerics
here are closed-form / quadrature checks, each labelled **"formula evaluation,
not evidence."**

This document is **additive**. It ADDs `docs/redesign/REDESIGN_v3.md` only. It
does not modify or delete `REDESIGN_v1.md`, `REDESIGN_v2.md`, any `src/`,
`configs/`, `docs/`, or schema file. It reuses the existing governance
(`AGENTS.md`, `docs/experiment_protocol.md`, `docs/intervention_protocol.md`,
`docs/baseline_contract.md`, `docs/pre_registration.md`,
`docs/statistical_analysis_plan.md`, `docs/compute_budget.md`), the code
contracts (`src/tracecausal/metrics.py::passes_intervention_gate`,
`src/tracecausal/schemas.py`, `src/tracecausal/contracts.py`), the
`configs/baselines/baseline_registry.yaml` registry, and the
`configs/seeds/paper_20.txt` seed manifest. No pre-registered gate is weakened:
margin `0.05`, utility drop `0.02`, transfer retention `0.80`, seeds `>= 20` as
a **floor**. v3 *adds* power-derived requirements on top of that floor and
*tightens* the causal interpretation; it relaxes nothing.

**Naming note (carried, unchanged in substance from v1/v2).** The deliverable
brief inherits a generic clause requiring a "propriety PROOF for riskcal".
`tracecausal` has no module named `riskcal`. The load-bearing object the clause
maps onto here is the **causal-usefulness estimator** `\hat U`. v1 over-claimed a
non-gameability theorem; v2 correctly retreated to an *unbiased-matched-contrast*
Identification Lemma; **v3 keeps that retreat but (a) de-circularizes the lemma
with per-example notation, (b) removes the logically invalid G5, and (c) adds a
genuinely propriety-flavoured mechanism check** for the `riskcal` analogue, namely
the **strictly-proper Bregman/Savage scoring rule** `G_B(p)=1-p+p^2 ⇒
S(p,y)=1-(p-y)^2`, whose stationary point is `p=q` — verified below as *formula
evaluation, not evidence* and used only to justify that the evaluator's
factuality probability `Y` is elicited by a strictly proper rule, so the
evaluator has no incentive to misreport. This is the only level of "propriety
proof" the design honestly supports; it is **not** a claim that detectors score 0.

---

## Changelog vs v2

The reviewer (GPT-5.5 6/10; Opus 6/10) gave one blocking *logical* defect, one
identification defect, and five required fixes. Every one is mapped below to a
concrete change with its section. Verified-correct v2 parts (AR-led single claim;
diffusion demoted to transfer; matched-budget control; reference ablations;
numeric kill-gates; additive/server-gated discipline; the *unbiased-matched-
contrast* retreat from v1's theorem) are **preserved**.

| Review item (v2) | Fix in v3 | Where |
| --- | --- | --- |
| **A — Central logical flaw: G5 makes "any detector with `\hat U>0` invalidates A4".** A detector can legitimately select high-`tau` segments while the matched null stays unbiased; that is a *novelty* problem, not a *causal-identification* problem. | **G5 deleted and replaced.** (i) The new **novelty gate G5′** requires the proposed causal selector to **beat the best ADAPTED detector** on `\hat U` by a pre-registered margin `0.03`. (ii) A detector with `\hat U>0` is **no longer** treated as a causal-identification failure; it is treated as evidence that detector covariates track `tau` (expected, not fatal). A4 is now probed *only* by constructions that actually bear on the null's unbiasedness (B). | §2.5, §4 G5′, §4 G7 |
| **B — A4 only falsifiable, never positively supportable.** | New **§2.10 positive identification probe**: (i) a **synthetic/oracle construction** with *known* ground-truth causal segments where `\hat U` must recover them and the null is unbiased *by construction*; (ii) an **analytic sensitivity bound** on the matched-null bias `bar tau(Pi)` (the positional-leakage bound) so A4 is *supportable within a stated interval*, not merely rejectable. | §2.10, §4 G7 |
| **C — Lemma 2.5 conflates a tautology with the substantive claim; A4 circular.** | **Lemma rewritten with per-example `tau_i(S)` and `Pi_i(S)`.** The **tautology** ("a uniform draw from `Pi_i` estimates the pool mean `bar tau_i(Pi)`") is split out as **Prop. 2.5a** (assumption-free). The **substantive causal claim** ("`Pi_i` is the correct counterfactual pool") is isolated as **Assumption A4★** with its own test. A4 is no longer used to prove its own premise. | §2.5 |
| **D — `mask` operator under-specified; attention-zeroing ≠ "removing evidence"; no SHAM/OOD control; no answer-adjacent control.** | **§2.2 rewritten operationally** (exact KV/attention/logit behaviour, position-id and RMS-renorm handling). Adds a **SHAM-MASK control** (mask a provably-irrelevant span / do-nothing renormalisation) and an **answer-adjacent-span control**, so a "necessity" signal cannot be an OOD or positional artifact. | §2.2, §2.3, §4 G8 |
| **E — Statistical plan conflates examples/seeds/intervention-repeats/bootstrap; Holm family omits G2/G5/best-baseline; `sigma_u=0.35` placeholder; binary-Y attenuation; ~10³/cell vs budget unreconciled.** | **§4.5 rewritten.** Four sampling levels separated with named symbols; Holm family redefined to include G1,G2,G5′,G6,G7 + best-baseline contrasts; `sigma_u` placeholder replaced by a **bounded-Bernoulli-contrast variance formula** with a pre-registered validation-split re-estimation rule; the `(2κ−1)` attenuation band is computed and the gate target is **raised to keep 0.05/0.03 detectable at κ≥0.90**, with a widen-or-aggregate fallback below that; the `~10³/cell` count is **reconciled against `docs/compute_budget.md`** with an explicit GPU-hour line and a feasible cell count (the "lead shrinks" escape hatch is replaced by a concrete budget table). | §4.5 |
| **F — CIU contract not implemented; run-readiness over-claimed.** | **§5 specifies the full Phase-2 contract** (`CIURecord`, `validate_ciu_record`, `ciu_gate`, `baseline_readiness`, null-pool sampler, adapter/null-pool/evaluator hashes) **plus a runnable null-data unit harness** proving (i) a random selector ⇒ `\hat U ≈ 0` and (ii) the **revised** novelty gate G5′ fires. AR-LLM stays the frozen lead; reasoning/diffusion stay a transfer study. | §5 |
| Proper-scoring mechanism for the `riskcal` analogue (reviewer verified the algebra but noted it is "not present in v2"). | **Now present**: §2.0a states and uses the strictly-proper rule `G_B(p)=1-p+p^2 ⇒ S(p,y)=1-(p-y)^2`; §2.11 gives the proof and the `p=q` stationarity, both **formula evaluation, not evidence**. | §2.0a, §2.11 |

**Net effect on the score-limiting issue.** The reviewer wrote: *"the design
still has a central logical flaw: G5 treats detector baselines with positive
`\hat U` as invalidating A4 … that would weaken novelty, not invalidate
causality. This is the main reason I cannot score above 6."* v3 removes exactly
that inference (fix A) and supplies the missing positive support for A4 (fix B).

---

## 1. Defect being fixed (context)

### 1.1 What was right in v2 and is kept

v2 already (correctly) froze AR generation as the lead, demoted diffusion to a
falsifiable transfer test, defined a budget/length/layer/reference/example-matched
random control `Pi`, introduced reference-construction ablations against
information injection, and retreated from v1's circular non-gameability theorem to
an *unbiasedness-of-the-matched-contrast* lemma. None of this is reopened.

### 1.2 The four residual defects v3 must fix

**D7 — Invalid kill-gate inference (review item A; the score-capping flaw).**
v2 G5 said: *if any non-causal baseline has `\hat U` CI lower bound `>0`, then
A4 is rejected and the causal estimator is invalidated.* This is a non-sequitur.
Let `tau_i(S)` be the true per-example effect of editing `S`. A detector that
selects segments correlated with `tau_i` will have
`E_i[tau_i(S^*_{det})] > bar tau_i(Pi)`, i.e. positive `\hat U`, **while the
matched random draw `\tilde S ~ Pi_i` remains an unbiased estimate of the
null-pool mean** (Prop. 2.5a, which is assumption-free). So a detector's positive
`\hat U` is fully consistent with A4 holding. It threatens **novelty** (the
detector is also "causally useful"), not **identification**. Conflating the two
both (a) mislabels a novelty result as a causality failure and (b) hands an
adversary a trivial way to nuke the paper by tuning one detector slightly above 0.

**D8 — A4 was falsifiable but never positively supportable (review item B).**
v2 could *reject* A4 (via controls) but offered no construction under which A4 is
*shown to hold*. A pre-registered design that can only ever reject its key
assumption is asymmetric: a null result is uninformative about whether the method
*works*, only about whether the assumption *failed*. v3 adds (i) an oracle/
synthetic generator with known causal segments and a provably-unbiased null, and
(ii) a closed-form positional-leakage bound on `bar tau(Pi)`.

**D9 — Lemma 2.5 conflated a tautology with the causal claim (review item C).**
v2's A4 read "the matched random draw is an unbiased estimate of the null-pool
mean … holds by construction **iff** the act of editing a random matched segment
carries no systematic effect beyond position identity." The first half is a
tautology of uniform sampling; the second half is the entire substantive
counterfactual claim. Bundling them lets the proof appear to *establish* the
counterfactual when it only re-states sampling. v3 splits them.

**D10 — `mask` semantics + statistics under-specified (review items D, E).** v2
said `mask` "renormalise attention … to zero, forcing downstream tokens to be
produced as if `S` carried no evidence." Attention-zeroing is not equivalent to
"`S` carried no evidence": it changes the partition function, position indexing,
and can put the residual stream off-distribution (OOD), so an apparent
"necessity" signal may be an OOD artifact. And the statistical plan conflated four
distinct sampling levels. Both are fixed (§2.2, §4.5).

---

## 2. New method — Causal Intervention-Usefulness (CIU), v3

### 2.0 One-paragraph statement

We freeze **autoregressive LLM generation** as the lead paradigm. A trace segment
`S` is *intervention-useful* iff editing `S` moves answer factuality more than a
**budget/length/layer/reference/example-matched random edit drawn per-example from
`Pi_i(S)`**, *without* paying it back in utility, *and* the proposed causal
selector **beats the best adapted detector baseline on `\hat U` by a pre-registered
margin** (the novelty requirement; this is the corrected role of what used to be
G5). We name the estimator the **Causal Intervention-Usefulness (CIU)** statistic
`\hat U`. We prove (Lemma 2.5) that `\hat U` is an **unbiased estimator of a causal
contrast** under explicit per-example assumptions, separating the *tautology* that
a uniform draw estimates the pool mean (Prop. 2.5a, assumption-free) from the
*substantive* claim that the matched pool is the correct counterfactual (A4★,
testable + positively probed in §2.10). We do **not** claim detectors must score
0; whether they do is an empirical, novelty-bearing question. Reasoning-trace and
diffusion-LM remain a **transfer study** with the identical operator.

### 2.0a The proper-scoring (`riskcal` analogue) mechanism, stated up front

`Y_i \in \{0,1\}` is a factuality label produced by a hashed evaluator. To make
the evaluator's *probabilistic* report `p` incentive-compatible (no benefit to
hedging or misreporting), the evaluator is elicited under the **strictly proper
Bregman/Savage binary score**
```
S_G(p, y) = G(p) + G'(p) (y - p),     with generator  G_B(p) = 1 - p + p^2.
```
Then (proof and stationarity in §2.11; both **formula evaluation, not evidence**)
```
S(p, y) = 1 - (p - y)^2,     argmax_p E_{Y~Bernoulli(q)}[S(p,Y)] = q,  uniquely (G'' = 2 > 0).
```
This is the load-bearing role the brief's "propriety PROOF for riskcal" maps onto:
a **strictly proper scoring rule guarantees the evaluator's elicited factuality
probability is its true belief**, which is exactly assumption **A3** (evaluator
pre-commitment) made mechanistically honest. It says nothing about detectors and
makes no causal claim; it only closes the loophole that `Y` could be a gamed label.

### 2.1 Trace, segments, granularity (reuses `schemas.py`)

A generation produces tokens `x_{1:T}` with hidden states `h_{1:T}^{(l)}` and
next-token distributions `p_t`. We reuse `TraceStep` and `TraceSegment` and
*extend* them additively (§5) with provenance fields.

- **primary granularity (lead):** *claim span* `S` = the minimal token span an
  atomic-claim extractor attributes a check-worthy proposition to;
- **secondary (robustness only):** token window `w in {1,4,8}` (nuisance);
- **transfer only:** one reasoning-step / one denoising-subtrace.

Granularity and operator are **pre-registered nuisance dimensions** analysed at a
single primary setting (`mask`, claim-span) to control the Holm family (§4.5).

### 2.2 Intervention operators — operational `mask` spec (fixing D / review D)

Let the segment be token positions `S = [a,b]`, layers `L_patch`. Define the
"downstream" positions as `t > b` (the suffix that produces the answer).

**Primary operator — `mask` (counterfactual evidence ablation), exact behaviour.**
For every downstream query position `t > b` and every head `h` at layers
`l \in L_attn` (the attention layers in scope), we **set the pre-softmax attention
logits from `t` to every key position `j \in [a,b]` to `-inf`** *before* the
softmax, then **renormalise the softmax over the remaining keys** `j \notin [a,b]`.
Concretely, for attention weights `\alpha_{t,j} = softmax_j(q_t·k_j/\sqrt{d})`:
```
mask:  for j in [a,b]:  logit_{t,j} <- -inf ;   alpha_{t,·} <- softmax over j \notin [a,b].
```
This is **KV-masking**, not state corruption: the keys/values at `[a,b]` are
*excluded from the attention mixture* for downstream queries, so the suffix is
produced as if positions `[a,b]` were absent from the context, *without* editing
the residual stream, RoPE/position ids of surviving tokens, or the logit head.
We **do not** shift position ids of downstream tokens (positions remain their
original absolute indices; only the *attendable set* shrinks), because re-indexing
would itself be an intervention; this is recorded as `mask_position_policy:
keep_absolute`. The edit budget `c(mask) = b - a + 1` (number of masked key
positions). We log the post-mask softmax mass that *would have* gone to `[a,b]`
(`displaced_mass`) as a persisted field, because a near-zero displaced mass means
`S` was never attended to and the mask is near-vacuous (a built-in sanity field).

> **Why this is not "removing evidence" by fiat.** Masking changes the partition
> function and can move the residual stream OOD even when `[a,b]` was causally
> irrelevant. v3 therefore does **not** interpret any `mask` effect as causal on
> its own; it is interpreted **only relative to the matched controls below**, and
> a "necessity" reading additionally requires passing the SHAM-MASK and
> answer-adjacent controls (G8). This is the precise answer to the reviewer's
> "attention zeroing ≠ removing evidence."

**`patch` (sufficiency, reference-state injection).** For `t \in [a,b]`,
`l \in L_patch`: `h_t^{(l)} <- (1-\alpha) h_t^{(l)} + \alpha h_t^{(l),ref}`,
`\alpha \in (0,1]`. Reference type `\rho` is a factor (§2.7). Budget
`c(patch) = |[a,b]| \cdot |L_patch|` patched coordinates.

**`replay` (trajectory).** Roll back to the decoder state before `a`, re-decode
`[a,b]` under a reference policy, free-run the suffix. Budget = re-decoded
positions. Reference type is a factor.

**`no_op`** sets `\alpha=0` / `k=0` (negative control).

**New controls demanded by review D (persisted, scored on `\hat U`):**
- **SHAM-MASK (OOD control).** Apply the *identical* `mask` mechanics to a span
  that is **provably irrelevant** to the answer — operationalised as (i) a
  *do-nothing renormalisation* that masks the **empty set** but still re-runs the
  softmax-renorm code path (isolates the numerical/OOD footprint of the operator
  itself), and (ii) a span the atomic-claim extractor labels *non-check-worthy and
  answer-disjoint* (e.g. boilerplate/punctuation runs). **If SHAM-MASK produces a
  positive `\hat U`, the `mask` "necessity" signal is an OOD/operator artifact**
  and is reported as such (G8). SHAM-MASK is *not* the same as the random control
  `Pi` (which is budget-matched but still potentially causal); SHAM-MASK targets
  *provably-irrelevant* spans to isolate the operator's own footprint.
- **Answer-adjacent-span control.** Mask the span immediately preceding the answer
  tokens (high positional leverage, often *not* the evidence). This separates
  *necessity of evidence* from *positional/recency* effects; it feeds the A4★
  positional-leakage probe (§2.10) and G8.

Invalid interventions (NaN logits, decode failure, empty attendable set) are
logged with reason codes per `intervention_protocol.md`; `>5%` invalid ⇒
diagnostic only; `invalid_count` is persisted (§5).

### 2.3 The matched random-segment control, per example (the identification design)

For operator `\theta` on `S` with budget `c(I_S^\theta)=k`, define the
**per-example** matched null pool:
```
Pi_i(S) := Uniform{ S' in example x_i :  c(I_{S'}^\theta) = k,
                                         len(S') = len(S),
                                         layer-set L_patch identical,
                                         S' disjoint from S,
                                         reference run identical (same ref_hash),
                                         same example x_i }
```
`\tilde S_i ~ Pi_i(S)` is matched on operator, budget, length, layers, reference,
and example; only *which positions are edited* varies. The realized `Pi_i` is
serialized by `null_pool_hash` (§5), so the estimand is reproducible. The pool is
**per-example** (subscript `i`) — this is the notational change review item C
required; the contrast and lemma below are all defined example-by-example and then
averaged.

**Honest statement of what matching buys (fixing D9).** Matching removes
budget/length/layer/reference/example as explanations of any contrast. It does
**not** remove the possibility that the selector's covariates (entropy, position,
claim-bearing-ness) correlate with the local causal effect `tau_i`. That residual
correlation is **the empirically interesting quantity, not a bug**: a causal
selector *should* concentrate on high-`tau_i` segments. v3 measures it (it is the
numerator of the novelty gate G5′), rather than treating it as an identification
failure.

### 2.4 Per-example causal contrast and the CIU estimator

Let `Y_i \in \{0,1\}` (evaluator factuality, elicited under the proper rule of
§2.0a) and `Q_i \in [0,1]` (utility). Under operator `\theta`, reference `\rho`:
```
delta_i(S; \rho)        =  Y_i(do(I_S^{\theta,\rho}))                              - Y_i(no_op)
delta_i(\tilde S; \rho) =  E_{\tilde S ~ Pi_i(S)}[ Y_i(do(I_{\tilde S}^{\theta,\rho})) ] - Y_i(no_op)
u_i(S; \rho)            =  delta_i(S; \rho) - delta_i(\tilde S; \rho).
```
**CIU estimator** over `n` matched examples (paired bootstrap):
`\hat U(selector; \rho) = (1/n) \sum_i u_i(S^*(x_i); \rho)`. Utility guard:
`\hat D_util = (1/n) \sum_i [Q_i(no_op) - Q_i(do(I_{S^*}^{\theta,\rho}))]`.

**Gate (unchanged thresholds; wraps `passes_intervention_gate`).** At primary
operator `mask` (reference-free):
```
passes_intervention_gate(targeted_delta = mean_i delta_i(S^*),
                         random_delta   = mean_i delta_i(\tilde S),
                         utility_drop   = \hat D_util,
                         min_margin     = 0.05,
                         max_utility_drop = 0.02) == True
```
i.e. `\hat U >= 0.05` and `\hat D_util <= 0.02`, with the Holm-corrected
paired-bootstrap 95% CI lower bound of `\hat U` exceeding `0.05`. No change to the
gate function; v3 supplies the per-example sampling law `Pi_i`, the estimand
`u_i`, the reference factor `\rho`, the proper-scoring evaluator (§2.0a), and the
persisted provenance (§5).

### 2.5 Identification of the CIU contrast — de-circularized (fixing C / review C)

> **Replaces v2 §2.5.** The proof now separates an assumption-free tautology from
> the one substantive causal assumption, using per-example `tau_i` and `Pi_i`.

**Per-example estimands.** For example `i`:
- selected-segment effect `tau_i(S^*) = E[ Y_i(do(I_{S^*}^{\theta,\rho})) - Y_i(no_op) ]`
  (expectation over intervention/decoding randomness at fixed `x_i`);
- per-example null-pool mean
  `bar tau_i(Pi) = E_{S' ~ Pi_i(S^*)}[ tau_i(S') ]`.

The CIU target is `U(selector) = E_i[ tau_i(S^*) - bar tau_i(Pi) ]`.

**Proposition 2.5a (the TAUTOLOGY — assumption-free).** For each `i`, if
`\tilde S_i` is drawn uniformly from `Pi_i(S^*)` and `R` such draws are averaged,
then `E_{draws}[ delta_i(\tilde S; \rho) ] = bar tau_i(Pi)` exactly, with Monte
Carlo error `O(1/\sqrt{R})`. *Proof.* Uniform sampling: the expectation of a
uniform draw of `tau_i(S')` over `S' \in Pi_i` is the pool mean, by definition of
the mean. No causal or distributional assumption is used. ∎
**This is all uniform sampling gives.** It does **not** say `bar tau_i(Pi)` is the
right counterfactual baseline — only that we estimate *that pool's mean* without
bias. (This is the half of v2's A4 that was tautological.)

**Assumption A4★ (the SUBSTANTIVE causal claim — TESTABLE, the only causal
assumption).** *The matched pool `Pi_i` is the correct counterfactual for the
selected segment*, i.e. editing a budget-matched random segment carries **no
systematic effect on factuality beyond the causal effect of position identity that
we intend to net out**:
```
A4★ :   bar tau_i(Pi)  is the no-op-adjusted effect attributable to "an edit of
        this budget/length/layers somewhere unrelated in example i",
        with no positional-leakage term correlated with the answer.
```
A4★ can **fail** (positional bias near the answer; budget systematically easier to
spend in some regions). It is (i) *bounded analytically* by the positional-leakage
sensitivity bound (§2.10), and (ii) *tested* by the SHAM-MASK and answer-adjacent
controls (G8) and the oracle construction (§2.10). **Crucially, A4★ failure is no
longer inferred from a detector having positive `\hat U`** (that inference, v2's
G5, is deleted — fix A).

**Lemma 2.5 (unbiasedness of the matched contrast).** Under (A1′ matching, A2
shared no-op, A3 proper-scored pre-committed evaluator, A4★),
```
E[ \hat U(selector) ] = E_i[ tau_i(S^*) ] - E_i[ bar tau_i(Pi) ] = U(selector).
```
*Proof.* By A2 the shared no-op cancels in each `u_i`. By A3 (and the proper
rule, §2.0a) both terms are expectations of a fixed, incentive-compatible
factuality function. By **Prop. 2.5a** (assumption-free) the random arm estimates
`bar tau_i(Pi)` without bias. By definition the targeted arm estimates
`tau_i(S^*)`. Averaging over `i` gives the claim. A4★ enters **only** in
interpreting `bar tau_i(Pi)` as the *correct counterfactual*; it is not used to
establish Prop. 2.5a. ∎

**What Lemma 2.5 gives / does not give.**
- *Gives:* `\hat U` is an unbiased estimator of the contrast "selected-segment
  effect minus matched-pool effect." A positive, CI-clearing `\hat U` *is
  consistent with* the selected segments carrying above-pool causal effect.
- *Does not give:* that detectors must score 0. If a detector's covariates track
  `tau_i`, it will also have positive `\hat U` **with A4★ intact**. That is a
  novelty question (G5′), not an identification failure.

**Corollary 2.6 (softened, unchanged in spirit).** Passing G1 under the matched
control *is consistent with* `Var_{S' ~ Pi_i}(tau_i) > 0` and the selector
concentrating on the high-`tau_i` tail. The utility guard additionally is
consistent with the gain not being bought by degradation. No "certifies"; no
"proper score for causal identity"; no "cannot be gamed."

### 2.6 Algorithm box (lead paradigm; design only, no run)

```
Algorithm CIU-AR  (lead = autoregressive; design only)
Input: frozen f, split D (split_hash), proper-scored evaluator E (evaluator_hash),
       operator theta in {mask*, patch, replay}, reference rho, budget k,
       layers L_patch, selector pi_sel, replicate floor 20, n >= R_power (§4.5).
Output: CIURecord{ \hat U, CI, \hat D_util, provenance hashes, invalid_count,
                   displaced_mass }, verdict in {useful_candidate | diagnostic | not_novel}.

assert server.authorized == false
for each seed s, each example x_i in D_test:
  1. trace_i      <- extract_trace(f, x_i)                 # TraceManifest, hashed
  2. Y0,Q0        <- E(no_op)                              # noop_run_hash (A2)
  3. S*           <- pi_sel(trace_i)
  4. Yt,Qt        <- E( f under do(I_{S*}^{theta,rho}) )   # targeted
  5. for r in 1..R_int:                                    # intervention-repeats
       S~_r       <- sample Pi_i(S*)                       # per-example matched null
       Yr_r       <- E( f under do(I_{S~_r}^{theta,rho}) )
     delta_rand_i <- mean_r (Yr_r - Y0)                    # Prop 2.5a: unbiased for bar tau_i(Pi)
  6. delta_tgt_i  <- Yt - Y0 ;  u_i <- delta_tgt_i - delta_rand_i ;  dutil_i <- Q0 - Qt
collect {u_i},{delta_tgt_i},{delta_rand_i},{dutil_i}, invalid_count, displaced_mass
\hat U, CI       <- paired_bootstrap_over_examples(u_i, B_boot, Holm-corrected)   # §4.5 levels
record           <- CIURecord(...+null_pool_hash, ref_hash, evaluator_hash, evaluator_kappa,
                              noop_run_hash, edit_budget=k, invalid_count, displaced_mass)
# Gates: G1 (scalar gate) AND G5' (beat best adapted detector by >=0.03) AND G7/G8 controls
verdict          <- ciu_gate(record, scalar_gate, best_detector_U, sham_U, oracle_pass)
return record, verdict
```

Note the four sampling levels are now distinct loops/symbols: **examples** `i`,
**seeds** `s`, **intervention-repeats** `R_int` (the random-control draws), and
**bootstrap resamples** `B_boot` (over examples). §4.5 fixes which one each
quantity is computed over.

### 2.7 Reference-construction ablations (preserved from v2)

Reference type `\rho \in {gold, neutral, placebo, wrong_plausible, no_gold}`, all
sharing budget/length/layers/example/null-pool. **Localization criterion (G6):**
`\hat U(gold) - \hat U(wrong_plausible) >= 0.03` and
`\hat U(gold) - \hat U(placebo) >= 0.03` (Holm CI `>0`); else the patch/replay
effect is *information injection, not localization*, and the `mask`/`no_gold`
reference-free necessity result still stands. (Unchanged from v2.)

### 2.8 Claim-calibration table (preserved + updated)

| Phrase | Status in v3 | Allowed only when |
| --- | --- | --- |
| "proper score" / "propriety" | applies **only** to the evaluator-elicitation rule (§2.0a/§2.11), never to non-gameability of selectors | always, for the evaluator rule; never re-extended to detectors |
| "certifies" | removed → "is consistent with" | n/a |
| "cannot be gamed" / "non-gameable" | removed | never (not claimed) |
| "pure detectors score 0" | removed → empirical, novelty-bearing | reported as observed, never as theorem |
| "more causally useful than detectors" | gated by **G5′** | proposed selector beats best adapted detector on `\hat U` by `>=0.03` |
| "causal localization" | gated by G6 | §2.7 criterion passes |
| "intervention-useful" | gated by G1+G2+G7+G8 with power | all pass |
| "transfers across paradigms" | gated by G3 | incl. diffusion |

### 2.9 Lead-vs-transfer split (preserved from v2)

- **Lead:** autoregressive LLM only, frozen; two sizes; open-domain + multi-hop;
  primary operator `mask`; full reference ablations on `patch`/`replay`. The
  `\hat U` gate (G1) **and** the novelty gate (G5′) must pass for the paper.
- **Transfer (secondary):** port the *identical* CIU operator to reasoning-LRM and
  diffusion-LM, gated on `heldout_taxonomy_retention >= 0.80` (G3). Diffusion may
  fail without killing the paper.

### 2.10 Positive identification probe for A4★ (fixing B / review B)

A4★ must be **supportable**, not merely rejectable. Two pre-registered devices:

**(i) Oracle / synthetic construction with known causal segments.** Build a
controlled generator `f_oracle` (a small frozen model run on a templated dataset)
in which a *designated* claim span is, **by construction**, the sole determinant
of factuality: the answer token is a deterministic function of the content at the
designated span, and all other spans are causally inert *given* the prompt. On
this generator:
- the **true causal effect is known**: `tau_i(S_designated) = 1`,
  `tau_i(S') = 0` for inert `S'`, so `bar tau_i(Pi) = 0` *exactly* and A4★ holds
  **by construction** (no positional leakage because inert spans are answer-disjoint);
- a correct CIU selector must recover `\hat U(oracle_selector) \to 1` and
  `\hat U(random_selector) \to 0` and `\hat U(detector_on_inert_feature) \to 0`.
This is a **positive** test: it demonstrates a regime where A4★ provably holds and
`\hat U` recovers the planted ground truth. (Built in Phase-2 as a *unit harness*,
§5; it uses synthetic/oracle labels, not server runs, and produces **no paper
numbers** — it is a correctness fixture, labelled as such.)

**(ii) Analytic sensitivity bound on matched-null bias `bar tau(Pi)`
(formula evaluation, not evidence).** Decompose the matched-null effect into the
intended position-identity term and an unwanted **positional-leakage** term
`L_pos` (effect of editing *near the answer* regardless of content). Let `\beta`
be the maximum per-position leakage slope (factuality change per unit positional
proximity to the answer) and `\Delta_pos` the maximum proximity gap the pool `Pi`
allows between `S^*` and `\tilde S` (zero if `Pi` is **proximity-stratified**).
Then the worst-case A4★ bias is bounded:
```
| E[\hat U] - U_true |  <=  \beta * \Delta_pos.
```
**Mitigation, pre-registered:** make `Pi_i` *proximity-stratified* — require
`\tilde S` to match `S^*` on a discretized distance-to-answer bin — driving
`\Delta_pos \to (bin width)` and the bias to `\beta * (bin width)`. The
answer-adjacent control (§2.2) estimates `\beta`; the bound is then a *reported
interval* on residual A4★ bias, so A4★ is **supportable within a stated bound**
rather than only falsifiable. (This adds `proximity_bin` to the matching keys of
`Pi_i`; back-compatible additive field, §5.)

### 2.11 Proper-scoring proof for the `riskcal` analogue (formula evaluation, not evidence)

> **Labelled: closed-form check, not experimental evidence.** Verified by stdlib
> arithmetic in this session (exact polynomial identity over sampled `p∈[0,1]`,
> `y∈{0,1}`, and grid-argmax for stationarity).

**Claim 1 (exact score identity).** For `G_B(p)=1-p+p^2`, `G_B'(p)=2p-1`, the
Savage/Bregman binary score `S(p,y)=G_B(p)+G_B'(p)(y-p)` equals `1-(p-y)^2`.
*Proof.* `S = (1-p+p^2) + (2p-1)(y-p) = 1 - p^2 + (2p-1)y`. At `y=0`:
`S = 1-p^2 = 1-(p-0)^2`. At `y=1`: `S = 1-p^2+2p-1 = 2p-p^2 = 1-(p-1)^2`. Since
`y\in\{0,1\}`, `S(p,y)=1-(p-y)^2` for both, hence identically. ∎

**Claim 2 (strict propriety / stationary point at `p=q`).** Under `Y\sim
Bernoulli(q)`, `E[S(p,Y)] = G_B(p) + G_B'(p)(q-p)`. Then
`d/dp E[S] = G_B''(p)\,(q-p) = 2(q-p)`, which is `0` iff `p=q`, and since
`G_B''=2>0` this stationary point is the **unique maximum**. So an evaluator
maximizing expected score reports its true belief `p=q`. ∎
(Grid check: `argmax_p E[S] = q` recovered at `q\in\{0.2,0.5,0.8\}`.)

**Role.** Claim 2 is exactly the mechanism that makes **A3** (evaluator
pre-commitment) incentive-compatible: the factuality probability fed into `Y` is
the evaluator's true belief, closing the "gamed label" loophole. It is **not** a
statement about detectors or about `\hat U`'s non-gameability.

---

## 3. Contribution + falsifiable predictions

**Single crisp scientific contribution.** A trace segment is scientifically
meaningful **iff intervening on it restores factuality beyond a per-example
budget-matched random edit, by more than a wrong-but-plausible reference would,
and by more than the best adapted detector achieves** — and we give the estimator
`\hat U`, the per-example matched null `Pi_i`, the reference factor `\rho`, the
proper-scored evaluator, and the persisted provenance that make this an
*identified, reproducible, falsifiable* quantity (Lemma 2.5 + Prop. 2.5a +
A4★-bound). The novelty is the **estimand and its identification design**, plus
the **detector-beating novelty gate**, not any individual operator.

**Sharper than the neighbours** (labels sanity-checked against primary pages):
- vs **TraceDet** (arXiv:2510.01274) / **TDGNet** (arXiv:2602.08048): detection-
  only on diffusion traces; no intervention estimand. Demoted to transfer.
- vs **RACE** (arXiv:2506.04832): reasoning-consistency *detector*; predictive.
  `delta_i(S)` is what RACE cannot produce — **but** if RACE-selected segments are
  causally useful, that is captured by G5′ (RACE becomes a strong *adapted
  detector* the proposed selector must beat), not dismissed.
- vs **semantic entropy** (Nature 2024) / **SelfCheckGPT** (arXiv:2303.08896) /
  **INSIDE** (arXiv:2402.03744): output/sampling/internal uncertainty; orthogonal
  interventional question.

**Falsifiable predictions (pre-registered, pre-data).**
- **P1 (existential).** If hallucination in AR-LLMs has localizable causal
  segments **and A4★ holds within the §2.10 bound**, the CIU selector achieves
  `\hat U >= 0.05` (Holm CI lower bound `>0.05`) with `\hat D_util <= 0.02` on
  `>=2` lead datasets, at primary operator `mask`.
- **P2 (novelty, the corrected detector claim).** The proposed causal selector's
  `\hat U` **exceeds the best adapted detector's `\hat U` by `>=0.03`** (Holm CI
  `>0`). *If a detector matches or beats it, the result is "detectors are also
  causally useful" — a real, publishable finding that demotes our novelty, **not**
  a causal-identification failure.* (This is the central fix.)
- **P3 (operator honesty).** SHAM-MASK and answer-adjacent controls have `\hat U`
  CI overlapping `0`; the oracle construction recovers `\hat U \to 1` for the
  planted selector and `\to 0` for random/inert-detector selectors. *If SHAM-MASK
  is positive, the `mask` necessity signal is an OOD artifact (G8).*

**These predictions can lose, and losing is informative and correctly diagnosed:**
P1 failing kills the causal claim; P2 failing demotes novelty (not causality); P3
failing flags an operator artifact. v3's gates route each failure to the right
conclusion — the specific defect the reviewer flagged.

---

## 4. Pre-registered analysis plan + kill-gates + power

**Datasets (lead = first two; from `data_and_evaluation_plan.md`).** open-domain
factual QA (TruthfulQA-style + TriviaQA-style); multi-hop QA (HotpotQA-style);
hallucination benchmark (HaluEval-style); diffusion-LM trace set (transfer only).
Each needs license, raw+processed hash, split, leakage check.

**Base models (pre-declared; AR frozen as lead).** Lead AR: one open-weights
instruct family at two sizes (7–9B and 13–14B class). Transfer reasoning: one open
LRM. Transfer diffusion: one open diffusion-LM matching TraceDet's public
protocol. Checkpoints/prompt-hashes/decoding params recorded per
`baseline_contract.md`. **Exact model/dataset ids are locked in the experiment
config before the test split is unlocked** (analysis-lock, `pre_registration.md`).

**Baselines (reuse `baseline_registry.yaml`).** `random_segment`,
`output_entropy`, `semantic_entropy`, `output_signature_detector`,
`reasoning_consistency_detector` (RACE), `selfcheckgpt`, `inside_detector`,
`diffusion_trace_detector` (TraceDet), `tdgnet_trace_detector`. **Every baseline
is scored on `\hat U`** via the audited segment adapter (below) — this is what
feeds the novelty gate G5′.

**Baseline fairness — audited segment adapter.** Output-only baselines do not
natively select internal segments; each is wrapped with a **declared, hashed,
identical-across-baselines** `segment_adapter` (e.g. highest-signal claim span),
declared part of the baseline with `adapter_hash`. Because under the corrected G5′
a strong adapter *helps the detector beat us* (the opposite of v2's incentive
inversion), the fairness risk is symmetric and we report the adapter so reviewers
can audit it. Without an audited adapter a baseline is AUROC-only and excluded
from the `\hat U` table.

**Baseline-readiness gate.** The run is **blocked** while any required baseline
has `implementation_commit: pending_before_server_run` or
`license: verify_before_run` (currently true for 7 of 9 baselines in
`baseline_registry.yaml`). Enforced by `baseline_readiness` at preflight (§5).

**The kill-gates (pre-registered; G1–G4, G6 unchanged thresholds; G5 replaced;
G7/G8 new).**
- **G1 causal usefulness (existential).** `\hat U >= 0.05`, Holm CI lower bound
  `>0.05`, `>=2` lead datasets, operator `mask`. *Fail ⇒ stop causal claim;
  project becomes detector-comparison diagnosis.*
- **G2 utility harm.** `\hat D_util <= 0.02`. *Fail ⇒ reframe as abstention.*
- **G3 transfer.** held-out taxonomy retention `>= 0.80`. *Fail ⇒ AR-only.*
- **G4 paper evidence.** `>= max(20, R_power)` replicates (§4.5), paired tests,
  effect sizes, 95% CI. *Fail ⇒ diagnostic label only.*
- **G5′ novelty (REPLACES v2 G5 — the central fix).** The proposed causal
  selector's `\hat U` **exceeds the best adapted detector's `\hat U` by a
  pre-registered margin `0.03`** (Holm CI of the *difference* `>0`). *Fail ⇒ the
  causal claim G1 may still stand, but the contribution is **downgraded to
  "detectors are also causally useful; our selector is not distinguishably better"**
  — a novelty downgrade, **not** a causal invalidation.* **A detector with
  positive `\hat U` is NOT treated as A4★ failure.**
- **G6 localization vs injection.** `\hat U(gold)-\hat U(wrong_plausible) >= 0.03`
  and `\hat U(gold)-\hat U(placebo) >= 0.03` (Holm CI `>0`). *Fail ⇒ information
  injection, not localization; mask/no_gold necessity unaffected.*
- **G7 positive A4★ support (NEW).** On the oracle construction (§2.10), the
  planted selector achieves `\hat U \to 1` and random/inert-detector selectors
  `\to 0`, **and** the reported §2.10(ii) bias bound `\beta*\Delta_pos` is below
  the localization margin `0.03`. *Fail ⇒ A4★ is not positively supported; causal
  wording is withheld pending a wider proximity-stratification.* (This is the
  positive identification teeth replacing v2's only-rejectable A4.)
- **G8 operator-artifact control (NEW).** SHAM-MASK and answer-adjacent controls
  have `\hat U` CI overlapping `0`. *Fail ⇒ the `mask` necessity signal is an
  OOD/positional artifact; report as such, do not claim necessity.*

**Holm family (corrected — review item E).** The confirmatory family includes,
at the locked primary settings, **G1, G2, G5′ (selector-vs-best-detector
difference), G6, G7, G8, and the per-baseline best-detector `\hat U` contrasts**,
across `{2 sizes} x {2 datasets}`. Operator and granularity are nuisance
dimensions at a single primary setting (not multiplied into the family). Transfer
G3 is its own family. We report the exact family size `m` and per-test
`alpha_1 = 0.05/m` in the locked config; the power computation below uses
`m \in [8,12]` (the realized confirmatory count) and is recomputed at lock.

**Ablations.** remove intervention scoring (predictive-only); one operator only;
one paradigm only; replace targeted with random (positive control — must collapse
`\hat U` to ~0); remove utility gate; the reference suite (§2.7); SHAM-MASK /
answer-adjacent / oracle (§2.2, §2.10); sweep `w,\alpha,k,L_patch,R_int`
(robustness curves only).

No run is scheduled. `server.authorized: false`. Server execution requires ARIS
experiment-plan approval + explicit user command per `docs/server_runbook.md`.

### 4.5 Statistical plan: four sampling levels, MDE, attenuation, budget reconciliation

**(E1) Four distinct sampling levels (un-conflated; review item E).**
| Symbol | Level | What it varies | Used for |
| --- | --- | --- | --- |
| `i = 1..n` | **examples** | the test items | the unit of the paired bootstrap; `\hat U` is a mean over `i` |
| `s = 1..S_seed` | **seeds** | decode/init RNG (floor 20) | stability of `\hat U` across runs; reported as seed-variance, **not** as the bootstrap |
| `r = 1..R_int` | **intervention-repeats** | the random-control draws `\tilde S ~ Pi_i` | Monte-Carlo estimate of `bar tau_i(Pi)` (Prop. 2.5a; `O(1/\sqrt{R_int})`) |
| `b = 1..B_boot` | **bootstrap resamples** | resampling examples with replacement | the 95% CI / Holm test on `\hat U` |
These were conflated in v2 ("`B>=R_power`", "20 seeds or bootstrap replicates").
v3 separates them: **`n`** drives power (below), **`S_seed>=20`** is the seed
floor, **`R_int`** (default `>=16`) controls null-arm MC error, **`B_boot`**
(default `>=10,000`) controls CI resolution. `G4` requires `S_seed>=max(20,?)` and
`n>=R_power`.

**(E2) Variance of the contrast — replacing the `sigma_u=0.35` placeholder.** Each
`u_i = delta_tgt_i - delta_rand_i` is a difference of bounded means of Bernoulli
factuality outcomes. With per-arm factuality probabilities `p_t` (targeted) and
`p_r` (random, MC-averaged over `R_int` draws), the per-example contrast variance
is bounded by the **Bernoulli-difference formula**
```
Var(u_i)  <=  p_t(1-p_t) + p_r(1-p_r)/R_int  +  2|Cov|,
```
maximized (worst case, `p=1/2`, `R_int` large, paired positive covariance reducing
it) at `sigma_u^2 <= 1/4 + 1/4 = 1/2`, i.e. `sigma_u <= 0.707` as a hard upper
bound, but the **paired** design (shared no-op, same example) makes the realized
`sigma_u` much smaller. v3 does **not** assert a number; it pre-registers the
estimator: `sigma_u` is **re-estimated from the validation split** before the test
split is unlocked, and `R_power` is set from that estimate at lock time. We report
both the hard bound (`0.707`) and the validation estimate; the §4.5 arithmetic
below is **formula evaluation, not evidence**, shown for three `sigma_u` values to
bracket the budget.

**(E3) MDE / per-cell example count (formula evaluation, not evidence).** The
paired-bootstrap CI half-width is `h \approx z_{1-alpha_1/2}\,sigma_u/\sqrt{n}`. To
keep the gate (`CI lower bound > 0.05`) detectable at a true effect
`U_true = 0.08` (margin `0.05` + `0.03` buffer), need `h <= 0.03`, i.e.
`n >= (z_{1-alpha_1/2}\,sigma_u/0.03)^2`. Computed at the corrected family size:

| family `m` | `alpha_1=0.05/m` | `z` | `sigma_u=0.30` | `sigma_u=0.35` | `sigma_u=0.40` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.00625 | 2.734 | **n>=748** | n>=1018 | n>=1329 |
| 10 | 0.00500 | 2.807 | n>=788 | n>=1072 | n>=1401 |
| 12 | 0.00417 | 2.865 | n>=821 | n>=1117 | n>=1460 |

So the binding constraint is **per-cell example count `n` of order 7.5×10²–1.5×10³**,
*not* the seed count (the `20` floor governs only `S_seed`). At the *validation*
`sigma_u` (re-estimated at lock), `R_power := \lceil (z\,sigma_u/0.03)^2 \rceil`.

**(E4) Binary-Y attenuation — keeping 0.05/0.03 detectable (review item E).**
Under symmetric label noise at evaluator agreement `\kappa`, the observed contrast
attenuates: `U_obs = (2\kappa-1)\,U_true`. So the gate `U_obs >= 0.05` requires
`U_true >= 0.05/(2\kappa-1)`. Computed (formula evaluation, not evidence):

| `\kappa` | `2\kappa-1` | `U_true` for 0.05 gate | `U_true` for 0.03 margin |
| ---: | ---: | ---: | ---: |
| 0.95 | 0.90 | 0.0556 | 0.0333 |
| 0.90 | 0.80 | 0.0625 | 0.0375 |
| 0.85 | 0.70 | 0.0714 | 0.0429 |
| 0.80 | 0.60 | 0.0833 | 0.0500 |
| 0.75 | 0.50 | 0.1000 | 0.0600 |

**Pre-registered rule.** We **raise the design target** to
`U_target = 0.05/(2\kappa-1)` using the *measured validation* `\kappa`, so the
0.05 gate stays detectable. At `\kappa >= 0.90` this is `<= 0.0625` (achievable;
this is why the MDE uses `U_true=0.08`). **If `\kappa < 0.90`**, the band makes the
0.05 gate require `U_true>0.0625`; we then **either** widen the margin to the
attenuation-adjusted value **or** aggregate the binary outcome to a less-noisy
score (claim-level factuality proportion per example, which raises effective
`\kappa`), pre-registered as the fallback. We will not claim `\hat U` precision
finer than the evaluator's agreement interval (limitation §6).

**(E5) Budget reconciliation against `docs/compute_budget.md` (replaces the
"lead shrinks" escape hatch with a feasible cell count).** The first-gate budget
allots **22 GPU-hours** (8 trace-extraction + 6 interventions + 8 baselines, +30%
buffer) and 300 GB trace storage. We translate the `n`-per-cell requirement into a
GPU-hour line and pick a **feasible cell count**:

- Cells in the confirmatory grid: `{2 sizes} x {2 datasets} = 4` lead cells
  (operator/granularity fixed at primary).
- Per-cell cost driver: for each example, `1` no-op + `1` targeted + `R_int`
  random-control interventions + per-baseline scoring. With `R_int = 16` and
  `\approx 9` baselines scored on `\hat U`, the dominant term is the
  `(1 + 1 + R_int) \approx 18` forward interventions per example for the proposed
  method plus baseline forwards.
- Let `c_fwd` be the GPU-hours per intervention-forward (a hardware constant fixed
  at lock from a micro-benchmark on the validation split — **not** an experiment,
  a timing calibration). Then per-cell GPU-hours `\approx n \cdot 18 \cdot c_fwd`,
  and total lead `\approx 4 \cdot n \cdot 18 \cdot c_fwd`.
- **Feasibility constraint:** `4 \cdot n \cdot 18 \cdot c_fwd <= 14` GPU-hr
  (the 6+8 intervention+baseline lines), i.e. `n <= 14 / (72\,c_fwd)`. The locked
  config records `c_fwd` and solves for the **largest `n` the budget funds**; if
  that `n` is below the MDE-required `n` (E3), the **pre-registered response is to
  request a budget increase line in `compute_budget.md` to fund the MDE `n`** (the
  feasible action), and only if that is denied do we drop to `1` size **or** `1`
  dataset (2 cells), which **doubles the affordable `n` per cell**. This is a
  concrete, budgeted cell count, not an open-ended "shrinks further." The number
  `c_fwd` and the resulting `n` are filled at lock from the timing calibration;
  the design pre-commits the *inequality and the decision order*, which is what a
  Stage-1 registered report requires pre-data.

(Quadrature/closed-form note: the `n` table (E3), the attenuation band (E4), and
the riskcal identity (§2.11) were checked numerically in-session and are labelled
**formula evaluation, not evidence**.)

---

## 5. Code plan for Phase-2 (CIU contract + runnable null-data harness)

All changes are **additive and uncommitted**; nothing existing is modified or
deleted by this document. **The point (review item F): the identification design
becomes an enforced CONTRACT with a runnable null-data unit harness proving the
estimator behaves and the revised novelty gate fires — before any run-readiness is
claimed.** AR-LLM stays the frozen lead; reasoning/diffusion stay transfer.

**Files this redesign ADDS now:** `docs/redesign/REDESIGN_v3.md` (this file only).

**Files a future additive implementation would ADD (Phase-2, not done here):**
- `src/tracecausal/ciu.py` — `causal_usefulness(deltas_targeted, deltas_random)`;
  `matched_random_null(...)` **null-pool sampler** (per-example `Pi_i` with the
  matching keys incl. `proximity_bin`); `paired_bootstrap_U(u_i, B_boot, holm=...)`
  over **examples**; `ciu_gate(record, scalar_gate, best_detector_U, sham_U,
  oracle_pass) -> {useful_candidate | diagnostic | not_novel}`. **`ciu_gate`
  encodes the corrected logic:** a detector with `\hat U>0` returns `not_novel`
  (G5′) only if it beats the proposed selector within margin — it **never** returns
  `invalidated` for that reason. Wraps, does not replace,
  `metrics.passes_intervention_gate`.
- `src/tracecausal/operators.py` — pure-Python interfaces/contracts for `mask`
  (with the exact KV/attention/renorm policy of §2.2, `keep_absolute` positions,
  `displaced_mass` accounting), `patch`, `replay`, `no_op`, **`sham_mask`**, and
  budget accounting `c(I)`; no model calls.
- `src/tracecausal/ciu_record.py` — the `CIURecord` dataclass + `validate_ciu_record`.
- `configs/experiments/redesign_v3_ar_lead.yaml` — 2 sizes, 2 datasets, operator
  `mask`, reference suite, `gates` block copied verbatim (`0.05/0.02/0.80`) plus
  `novelty_margin: 0.03` (G5′), `localization_margin: 0.03` (G6),
  `oracle_recovery: true` (G7), `sham_mask_null: true` (G8),
  `server.authorized: false`, `seeds.paper_minimum: 20`,
  `replicates.power_derived: true`, `r_int: 16`, `b_boot: 10000`,
  `proximity_stratified_null: true`, `c_fwd_gpu_hours: pending_timing_calibration`.
- `configs/experiments/redesign_v3_transfer.yaml` — reasoning + diffusion, gated
  on `heldout_taxonomy_retention: 0.80`.

**Runnable null-data unit harness (the key Phase-2 deliverable, no GPU):**
- `tests/test_ciu_estimator.py` — on **synthetic null data** (random labels, no
  planted effect): asserts a **random selector ⇒ `\hat U \approx 0`** within MC
  tolerance `O(1/\sqrt{R_int})`; asserts the **positive control** (planted
  `Var(tau)>0` oracle of §2.10) gives `\hat U \to 1` for the planted selector;
  asserts budget/length/layer/reference/proximity-bin mismatch in `Pi_i`
  **raises**; asserts missing provenance hashes **raise**.
- `tests/test_ciu_novelty_gate.py` — **proves the revised novelty gate fires**:
  constructs a synthetic detector whose selected segments have `\hat U` above the
  proposed selector and asserts `ciu_gate(...) == "not_novel"` (G5′) **and** that
  the record is **not** marked `invalidated` (the v2 bug is regression-tested away);
  constructs the reverse (proposed beats detector by `>=0.03`) and asserts
  `useful_candidate`.
- `tests/test_ciu_contract.py` — round-trips a `CIURecord`, checks
  `validate_ciu_record` rejects empty hashes, `server_authorized=True`,
  `edit_budget` mismatch between targeted and control provenance, and
  `n_examples` below the MDE-required count.
- `tests/test_sham_mask.py` — asserts SHAM-MASK on the do-nothing renormalisation
  path yields `\hat U \approx 0` on null data (G8 wiring).

**The enforced estimand — `CIURecord` (additive dataclass):**
```
CIURecord(
  selector_id, operator, reference_type,         # which estimand
  edit_budget: int,                              # k, must match control
  null_pool_hash: str,                           # serialized per-example Pi_i (A1')
  noop_run_hash: str,                            # shared no-op (A2)
  evaluator_hash: str, evaluator_kappa: float,   # A3 (proper-scored) + attenuation (4.5)
  ref_hash: str,                                 # reference identity (2.7)
  adapter_hash: str | None,                      # baseline segment adapter (G5')
  proximity_bin: int | None,                     # A4* proximity stratification (2.10)
  displaced_mass: float,                         # mask sanity (2.2)
  matched_control_provenance: tuple[str, ...],   # the R_int drawn \tilde S ids
  invalid_count: int, n_examples: int,           # estimand denominator
  r_int: int, b_boot: int, s_seed: int,          # the four sampling levels (4.5)
  u_hat: float, ci_low: float, ci_high: float,
  d_util: float, best_detector_u: float | None,  # for G5'
  server_authorized: bool = False,
)
```
`validate_ciu_record(record) -> list[str]` returns violations if any hash is
empty, `server_authorized` is True, `edit_budget` differs between targeted and
control provenance, `s_seed < 20`, or `n_examples` is below the MDE-required count.

**Files a future implementation would EXTEND (additive, back-compatible):**
- `src/tracecausal/schemas.py` — `TraceSegment` gains optional `edit_budget`,
  `reference_hash`, `null_pool_hash`, `proximity_bin` (defaults preserve
  back-compat; the four shipped fields stay).
- `src/tracecausal/metrics.py` — **unchanged signature**;
  `passes_intervention_gate` is *wrapped* by `ciu_gate`, never edited.
- `configs/baselines/baseline_registry.yaml` — add `ciu_scored: true` and an
  optional identical `segment_adapter:` block (`adapter_hash`) per output-only
  baseline. No baseline removed.
- `src/tracecausal/contracts.py` — add `baseline_readiness(registry)` flagging any
  baseline with `pending_before_server_run` / `verify_before_run`, used at
  preflight (blocks the run while 7/9 baselines are pending).

**Contract invariants kept:** `validate_manifest` still requires
`server.authorized == false`, `>=3` baselines, `>=20` paper seeds (floor);
`passes_intervention_gate` signature/thresholds unchanged; all `required_docs`
paths continue to exist. No `git commit`, no `git push`, no run.

---

## 6. Honest limitations

- **A4★ is the one substantive causal assumption.** v3 makes it *both* falsifiable
  (G8, controls) *and* positively supportable (oracle construction + the
  `\beta*\Delta_pos` bias bound, §2.10 / G7), and — the central correction —
  **stops inferring A4★ failure from a detector's positive `\hat U`** (v2's invalid
  G5). The residual risk is that the §2.10(ii) bound is loose if `\beta` is large
  and proximity bins are coarse; we report the bound and widen stratification if it
  exceeds the `0.03` margin.
- **Novelty vs causality are now correctly separated.** If detectors are *also*
  causally useful (G5′ fails), the paper's contribution shrinks to the estimand +
  identification design and an honest "detectors also pass," which is a weaker but
  *true* result — not a hidden failure. This is a real acceptance risk we accept.
- **`mask` may still be partly OOD.** KV-masking changes the partition function;
  SHAM-MASK (G8) bounds how much of any signal is operator artifact, but cannot
  prove zero OOD effect. The `displaced_mass` field flags near-vacuous masks.
- **Reference-run information injection.** `patch`/`replay` can inject information;
  G6 + the reference suite quantify it; the reference-free `mask` is the primary,
  injection-immune result.
- **Adapter risk is now symmetric, not adversarial.** Under G5′ a strong adapter
  helps detectors *beat* us, so there is no incentive to weaken adapters; we still
  audit and hash them.
- **Evaluator bounds `\hat U` precision.** The proper-scoring rule (§2.0a/§2.11)
  makes the elicited probability incentive-compatible but does not make `\kappa=1`;
  we report `evaluator_kappa`, attenuate by `(2\kappa-1)`, and raise the target or
  aggregate the outcome below `\kappa=0.90` (§4.5 E4).
- **Power may exceed the first-gate budget.** The MDE is `n` of order 7.5×10²–1.5×10³
  per cell (E3) against a 22-GPU-hr first-gate budget (E5); the pre-registered
  response is a budgeted inequality with a request-increase-then-shrink-cells
  decision order, not an open-ended escape hatch.
- **Diffusion transfer may fail** (G3) — acceptable by design; no "across all
  paradigms" claim unless G3 passes.
- **Replay is expensive** — CIU is framed as offline causal *auditing*, not a live
  decoder.
- **Scope.** A CI-clearing `\hat U` *is consistent with* the existence and
  concentration of intervention-useful segments, never "all hallucinations are
  causally explained" (`AGENTS.md`).
- **No empirical evidence exists.** Everything here is design + theory; the only
  numerics are the §2.11 / §4.5 closed-form / quadrature checks, explicitly
  **"formula evaluation, not evidence."** Every claim-evidence row in
  `docs/claim_evidence_matrix.md` and `docs/paper_claims_status.md` remains
  `pending`; this document changes no status label and authorizes no run.
  `server.authorized: false`.
