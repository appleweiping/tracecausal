# REDESIGN v1 — tracecausal

Status: `design_only`. No server run authorized. `server.authorized: false` is
preserved. This document is **additive**: it does not modify or supersede any
existing file. It reuses the existing governance (`AGENTS.md`,
`docs/experiment_protocol.md`, `docs/intervention_protocol.md`,
`docs/baseline_contract.md`, `docs/pre_registration.md`), the existing code
contracts (`src/tracecausal/metrics.py::passes_intervention_gate`,
`src/tracecausal/schemas.py`), the `configs/baselines/baseline_registry.yaml`
registry, and the `configs/seeds/paper_20.txt` seed manifest. All numeric gates
below are *identical* to the ones already pre-registered (margin `0.05`, utility
drop `0.02`, transfer retention `0.80`, seeds `>= 20`); this redesign does not
weaken any gate, it only sharpens the estimator they are applied to.

Naming note. The deliverable brief inherits a generic clause requiring a
proper-scoring propriety obligation. `tracecausal` has no proper-scoring-reward
module. The
load-bearing object that the clause maps onto here is the **causal-usefulness
estimator** `\hat U` (the targeted-minus-random factuality contrast that gates
every causal claim, implemented today as `passes_intervention_gate`). Section 2.5
gives that estimator a formal **propriety property** (Definition 2.4) and a
**proof** (Theorem 2.5) that it is non-positive for non-causal segments and
strictly positive only for genuinely causal ones, i.e. the estimator cannot be
gamed by a pure detector. That is the analogue of a proper-scoring construction:
the quantity is *maximized in expectation only by the truth-tracking object*.

---

## 1. Defect being fixed

### 1.1 The audited problem, stated precisely

The current scaffold commits to **one method evaluated simultaneously across
three generation paradigms** (autoregressive LLM, reasoning-trace LRM,
diffusion LM) `x` 8 baselines (`baseline_registry.yaml`) `x` 4 dataset families
(`formal_tracecausal.yaml`). Three coupled defects follow.

**D1 — Scope creep / underpowered design.** The pre-registration
(`docs/pre_registration.md`) fixes a *single* primary outcome
`targeted_delta - random_delta` but the formal config asks it to hold across the
full `3 x 8 x 4` grid. With Holm correction over `m` paradigm `x` dataset `x`
baseline cells, the per-cell significance threshold is `alpha / m`. For the main
intervention claim the effective `m` is on the order of `3 (paradigms) x 4
(datasets) x >=2 (control contrasts) = 24+` family-wise tests. Holding the
family-wise error at `0.05` forces per-test `alpha ~ 0.002`, and the paired
bootstrap CI half-width scales as `~ z_{1-alpha/2} \cdot sigma / sqrt(n)`. The
budget in `configs/compute/first_gate_budget.yaml` funds *one* model family and
*one* dataset family for 20 replicates — it cannot power the advertised grid.
The contribution is therefore spread too thin to clear its own gate.

**D2 — Diffusion novelty is thin and dominates risk.** `literature_boundary.md`
and `risks_and_blockers.md` already flag that a D-LLM denoising-trace *detector*
is "too close to TraceDet-style work" (arXiv:2510.01274) and TDGNet
(arXiv:2602.08048). If diffusion is co-equal in the main claim, a reviewer can
kill the whole paper on the diffusion section alone, even if the autoregressive
result is strong. Risk is concentrated in the *weakest, most-crowded* paradigm.

**D3 — Existential "just another detector" risk is under-specified.** The kill
gate exists (`G1`, margin `>= 0.05`) but the object it gates,
`passes_intervention_gate(targeted_delta, random_delta, utility_drop)`, is
currently defined on **two scalars supplied by the caller** with no formal
statement of (a) what probability law `targeted_delta` and `random_delta` are
drawn from, (b) what makes the random control *matched*, or (c) why a positive
contrast cannot be produced by a segment that is merely *predictive* of
hallucination rather than *causal*. Without that, a detector that selects
high-entropy segments could pass `G1` by selecting segments that are easy to
perturb, not segments that cause hallucination. The gate is falsifiable in name
but not yet *identified*.

### 1.2 Formal symptom of D3

Let `S` be a candidate segment, `Y in {0,1}` the factuality of the final answer
(1 = factual), and `do(I)` an intervention operator on `S`. The scaffold scores
`Delta(S) = E[Y | do(I_S)] - E[Y | no_op]`. A pure detector selects `S` to
maximize `Corr(feature(S), 1 - Y)`. Nothing in the current definition prevents

```
Delta(S_detector) > 0  AND  Delta(S_detector) - Delta(S_random) >= 0.05
```

from holding for a `S_detector` chosen by correlation alone, because perturbing
any segment that *co-occurs* with hallucination can mechanically nudge `Y` (e.g.
by destabilising decoding). The redesign must make the estimator *reward causal
sufficiency/necessity*, not co-occurrence. That is the core fix in Section 2.

---

## 2. New method — Causal Intervention-Usefulness (CIU)

### 2.0 One-paragraph statement

We freeze **autoregressive LLM generation** as the lead paradigm and define a
single contribution: a segment `S` of a generation trace *matters* iff a
**factuality-preserving counterfactual reference intervention** on `S` moves the
answer's factuality more than a **matched random-segment intervention with the
same edit budget**, and does so **without paying it back in utility**. We name
the estimator the **Causal Intervention-Usefulness (CIU)** statistic `\hat U`,
prove it is *propriety-sound* (positive in expectation only for causal
segments), and gate every causal claim on it. Reasoning-trace and diffusion-LM
become a **transfer study**: we test whether the *same* CIU operator, ported
unchanged, still selects intervention-useful segments — a falsifiable
generalization claim, not a parallel main result.

### 2.1 Trace, segments, granularity (reuses `schemas.py`)

A generation produces a token sequence `x_{1:T}` with hidden states
`h_{1:T}^{(l)}` at layers `l` and next-token distributions `p_t`. We reuse the
existing `TraceStep` (a scored `text_span`) and `TraceSegment` (a set of
`step_ids` + `selector` + `intervention`) dataclasses unchanged.

**Granularity (lead = autoregressive).** A *segment* is a contiguous window of
`w` decoding positions, `S = [a, b]` with `b - a + 1 = w`, plus its layerwise
residual-stream slice `H_S = { h_t^{(l)} : t in [a,b], l in L_patch }`. Three
granularities are pre-registered and swept as the segment-length hyperparameter
already listed in `motivation_ablation_hparam_plan.md`:

- **token window** `w in {1, 4, 8}` (fine);
- **claim span** `S = `the minimal token span that an atomic-claim extractor
  attributes a check-worthy proposition to (semantic);
- **reasoning-step window** (transfer paradigm only): one explicit
  chain-of-thought step, aligned to answer tokens via the
  `reasoning_trace` selector.

Trace extraction procedure (deterministic, hashable, server-side only):
`(1)` greedy/temperature-fixed decode under the frozen `prompt_template_hash`;
`(2)` record `x_{1:T}, p_{1:T}, h^{(L_patch)}`; `(3)` atomic-claim segmentation;
`(4)` emit a `TraceManifest` with `split_hash` and `server_authorized=false`.
This is exactly the `schemas.trace_manifest.schema.json` contract.

### 2.2 Intervention operators (reuses `intervention_protocol.md` IDs)

For a segment `S` we define three operators, each paired with the existing
control IDs `random_non_causal_segment`, `shuffled_trace_segment`,
`no_op_intervention`.

1. **Counterfactual masking `M_S`** (`mask`). Suppress the segment's causal
   contribution by replacing its key/value contribution with the model's
   *prompt-only* prior: attention from positions `> b` to positions in `[a,b]`
   is renormalised to zero, forcing downstream tokens to be produced as if `S`
   carried no evidence. No external reference needed; isolates *necessity*.

2. **Activation patching with a factual/neutral reference `P_S^{ref}`**
   (`patch`). Given a paired *reference run* on the same question whose answer is
   known-factual (or a neutral no-evidence run), copy its residual stream into
   `H_S`:
   `h_t^{(l)} <- (1-alpha) h_t^{(l)} + alpha \, h_t^{(l),ref)}, t in [a,b],
   l in L_patch`, with patch strength `alpha in (0,1]`. Isolates *sufficiency*
   of a factual state at `S`.

3. **Replay-from-checkpoint `R_S`** (`replay`). Roll back to the decoder state
   immediately before position `a`, then re-decode `[a,b]` under a
   factuality-promoting reference policy (reference-run teacher forcing or
   constrained decoding), then free-run the suffix. Isolates *trajectory*
   correction while holding the prefix fixed.

Each operator must preserve the **edit budget** `c(I)` (number of altered token
positions / patched coordinates), because the random control must spend the
*same* budget (Section 2.3). Invalid interventions (NaN logits, decode failure)
are logged with reason codes per `intervention_protocol.md`; `> 5%` invalid ->
diagnostic only.

### 2.3 The matched random-segment control (the identification fix for D3)

Let `I_S^{theta}` be operator `theta in {mask, patch, replay}` applied to
segment `S` with budget `c(I_S^{theta}) = k`. Define the **matched random
control** `\tilde S` as a segment drawn from the *budget-matched, position-
stratified* null:

```
\tilde S ~ Pi(S) :=  Uniform{ S' :  c(I_{S'}^{theta}) = k,
                                    len(S') = len(S),
                                    layer-set L_patch identical,
                                    S' disjoint from S,
                                    reference run identical }
```

Crucially `\tilde S` is matched on **everything that is not causal identity**:
same operator, same edit budget, same length, same layers, same reference, same
example. The only thing that varies is *which* positions are edited. This is the
`random_non_causal_segment` control made precise. `shuffled_trace_segment`
additionally permutes the within-segment ordering (controls for *content* vs
*position*); `no_op_intervention` sets `alpha=0` / `k=0` (controls for pipeline
noise).

### 2.4 Per-example causal contrast and the CIU estimator

Let `Y in {0,1}` be evaluator factuality and `\tilde Y` the answer
post-intervention. Define the **per-example targeted effect** and **matched
random effect** under operator `theta`:

```
delta_i(S)        =  Y_i(do(I_{S}^{theta}))        -  Y_i(no_op)
delta_i(\tilde S) =  E_{\tilde S ~ Pi(S)}[ Y_i(do(I_{\tilde S}^{theta})) ] - Y_i(no_op)
```

The **per-example causal-usefulness contrast** is

```
u_i(S) = delta_i(S) - delta_i(\tilde S).
```

The **CIU estimator** over `n` matched examples (paired bootstrap as in
`statistical_analysis_plan.md`) is

```
\hat U(S-selector)  =  (1/n) sum_{i=1}^{n} u_i(S^*(x_i)),
```

where `S^*(x_i)` is the segment chosen by the method's selector on example `i`.
The **utility guard** uses the same controls on a fluency/accuracy score
`Q in [0,1]` (answer-accuracy + fluency, from `metrics`/evaluator):

```
\hat D_util = (1/n) sum_i [ Q_i(no_op) - Q_i(do(I_{S^*}^{theta})) ]   (utility drop).
```

**Gate (unchanged thresholds, reuses `passes_intervention_gate`).** A causal
claim is licensed iff

```
passes_intervention_gate(targeted_delta = mean_i delta_i(S^*),
                         random_delta   = mean_i delta_i(\tilde S),
                         utility_drop   = \hat D_util,
                         min_margin     = 0.05,
                         max_utility_drop = 0.02)  ==  True
```

i.e. `\hat U >= 0.05` **and** `\hat D_util <= 0.02`, with the paired-bootstrap
95% CI of `\hat U` excluding `0.05` after Holm correction. No code change is
required to the gate function; the redesign supplies the *sampling law* (`Pi`)
and the *estimand* (`u_i`) that make its inputs identified.

### 2.5 Propriety of the CIU estimator (the required proof)

We now show CIU behaves like a **proper score for causal identity**: it is
non-positive in expectation for any selector that is not causally truth-tracking,
and strictly positive only for selectors that pick segments with genuine causal
effect. This is what stops the method collapsing into "another detector".

**Setup / assumptions.**
- (A1, *budget-matched exchangeability*) Conditional on the matching set
  (operator `theta`, budget `k`, length, layers `L_patch`, reference run,
  example `x_i`), the chosen segment `S` and a control draw `\tilde S ~ Pi(S)`
  are exchangeable in everything except their *position identity*. Formally the
  potential outcomes `{Y_i(do(I_{S'}^{theta}))}` for `S'` in the matched pool
  depend on `S'` only through its causal effect, not through the matching
  covariates (those are held fixed by construction in Section 2.3).
- (A2, *no-op baseline shared*) `Y_i(no_op)` is identical across `S` and
  `\tilde S` (same un-intervened run), so it cancels in `u_i`.
- (A3, *evaluator pre-commitment*) `Y, Q` come from an evaluator whose prompt
  and key are hashed before the run (`intervention_protocol.md` leakage clause);
  hence `Y_i(\cdot)` is a fixed measurable function of the intervened output, not
  adaptively chosen.

Define the **average causal effect of editing positions `S`** as
`tau(S) = E_i[ Y_i(do(I_S^{theta})) - Y_i(no_op) ]` and the **null pool mean**
`bar tau = E_{S' ~ Pi}[ tau(S') ]`. By construction `delta_i(\tilde S)`
estimates `bar tau` and `delta_i(S)` estimates `tau(S)`, both relative to the
*same* `no_op` (A2).

**Definition 2.4 (causal selector).** A selector is *causal* if it picks
`S^* = argmax_{S' in pool} tau(S')` (the segment with the largest genuine
factuality-restoring effect). It is *non-causal / detector-like* if its choice
is conditionally independent of `tau` given the matching covariates, i.e.
`S^* perp tau | matching`.

**Theorem 2.5 (propriety / non-gameability).** Under (A1)-(A3):
1. **Unbiasedness vs the null.** `E[\hat U(selector)] = E_i[ tau(S^*(x_i)) ] -
   bar tau`. (The matched random control is an unbiased estimate of the null
   pool mean.)
2. **Detector floor.** For any *non-causal* selector,
   `E[\hat U] = E[tau(S^*)] - bar tau = 0`, because `S^* perp tau | matching`
   makes `E[tau(S^*)] = E_{S'~Pi}[tau(S')] = bar tau`. A pure detector therefore
   has CIU expectation exactly `0`, so it **cannot** clear the `>= 0.05` gate
   except by sampling noise (excluded by the paired-bootstrap CI + Holm).
3. **Strict positivity only under causal effect.** For the causal selector of
   Definition 2.4, `E[\hat U] = E[max_{S'} tau(S')] - bar tau >= 0`, with strict
   inequality iff `Var_{S' ~ Pi}(tau(S')) > 0`, i.e. iff some segments genuinely
   carry more factuality-restoring causal effect than the budget-matched average.

*Proof.* (1) Linearity of expectation and (A2): the shared `no_op` term cancels
inside `u_i`, leaving `E[u_i] = tau(S^*(x_i)) - E_{\tilde S~Pi}[tau(\tilde S)]
= tau(S^*(x_i)) - bar tau`; average over `i`. (2) Non-causal ⇒ `S^* perp tau |
matching`; the tower rule gives `E[tau(S^*)] = E[E[tau(S^*)|matching]] =
E[E_{S'~Pi}[tau(S')|matching]] = bar tau`, so the contrast is `0`. (3) The
maximum over the pool is `>= ` the pool mean, with equality iff `tau` is constant
on the pool (zero variance); a non-degenerate causal structure has
`Var(tau) > 0`, giving strict positivity. ∎

**Corollary 2.6 (what the gate certifies).** Passing `G1` with the matched
control of Section 2.3 certifies `Var_{S'~Pi}(tau) > 0` *and* that the selector
concentrates on the high-`tau` tail — i.e. the method has found segments whose
*editing causally restores factuality beyond the matched null*. This is strictly
stronger than any AUROC/detection claim, and is false for a pure detector by
Theorem 2.5(2). The utility guard `\hat D_util <= 0.02` additionally certifies
the gain is not bought by degrading the answer (so it is mitigation, not
sabotage). This is the precise sense in which CIU is "proper": the gated
quantity is maximized in expectation only by the truth-tracking (causal)
selector, never by a correlational one.

### 2.6 Algorithm box

```
Algorithm CIU-AR  (lead paradigm: autoregressive; design only, no run)
Input: frozen model f, dataset split D with split_hash, evaluator E (hashed),
       operator theta in {mask, patch, replay}, budget k, layers L_patch,
       segment selector pi_sel (the method under test), seeds 0..S-1 (>=20).
Output: \hat U, CI, \hat D_util, gate verdict (paper_result | diagnostic).

assert server.authorized == false                      # never run locally
for each seed s, each example x_i in D_test:
  1. trace_i        <- extract_trace(f, x_i)            # TraceManifest, hashed
  2. Y0, Q0         <- E(no_op run)                     # no_op control
  3. S*             <- pi_sel(trace_i)                  # method's segment
  4. Yt, Qt         <- E( f under do(I_{S*}^theta) )    # targeted
  5. for r in 1..R:                                     # matched random control
       S~_r         <- sample Pi(S*)   # budget/len/layer/ref-matched, disjoint
       Yr_r         <- E( f under do(I_{S~_r}^theta) )
     delta_rand_i   <- mean_r (Yr_r - Y0)
  6. delta_tgt_i    <- Yt - Y0
     u_i            <- delta_tgt_i - delta_rand_i
     dutil_i        <- Q0 - Qt
collect {u_i}, {delta_tgt_i}, {delta_rand_i}, {dutil_i}
\hat U, CI         <- paired_bootstrap(u_i, B>=20, Holm-corrected)
verdict            <- passes_intervention_gate(mean(delta_tgt_i),
                          mean(delta_rand_i), mean(dutil_i),
                          min_margin=0.05, max_utility_drop=0.02)
return \hat U, CI, mean(dutil_i), verdict
```

### 2.7 Lead-vs-transfer split (fixes D1 + D2)

- **Lead (main result):** autoregressive LLM only. One model family, the
  open-domain + reasoning-restoration datasets, all three operators, full
  ablations. This is where the `\hat U` gate must pass for the paper to exist.
- **Transfer study (secondary section):** *port the identical CIU operator and
  gate* to (i) reasoning-trace LRM (segment = reasoning step) and (ii)
  diffusion LM (segment = denoising-step subtrace). The claim is *generalization
  of the operator*, gated by the `heldout_taxonomy_retention >= 0.80`
  pre-registered transfer gate — **not** a new SOTA detector. This demotes the
  TraceDet-adjacent diffusion work to "does our causal operator transfer?",
  which is novel *because TraceDet is detection-only and has no intervention
  estimator to compare against*. Diffusion can now *fail* without killing the
  paper.

---

## 3. Why this is NOT stitching

**Single crisp scientific contribution.** A trace segment is scientifically
meaningful **iff intervening on it restores factuality beyond a budget-matched
random edit** — and we give the estimator `\hat U` that makes this an
*identified, properly-scored, falsifiable* quantity (Theorem 2.5), not a stack
of a detector + a graph + a mitigation prompt. The novelty is the **estimand and
its propriety**, not any individual operator (masking/patching/replay are known
mechanistic tools; semantic entropy / TraceDet / RACE are known detectors). No
prior hallucination-tracing work supplies a control-matched intervention
estimator with a non-gameability guarantee that *pure detectors provably score
0* on.

**Sharper than the neighbours.**
- vs **TraceDet** (arXiv:2510.01274) and **TDGNet** (arXiv:2602.08048):
  detection-only on diffusion traces; no intervention, no causal estimand. CIU
  *demotes* diffusion to a transfer test and adds the missing causal axis.
- vs **RACE** (arXiv:2506.04832): reasoning-consistency *detector*; predictive,
  not interventional. CIU's `delta_i(S)` is exactly what RACE cannot produce.
- vs **semantic entropy** (Nature 2024) / **SelfCheckGPT** / **INSIDE**:
  output/sampling/internal-state *uncertainty*; Theorem 2.5(2) shows any such
  selector has CIU expectation `0`.

**Falsifiable prediction (pre-registered, single number).** *If* hallucination
in autoregressive LLMs has localizable causal segments, *then* the CIU selector
will achieve `\hat U >= 0.05` (paired-bootstrap CI lower bound `> 0.05` after
Holm) with `\hat D_util <= 0.02` on `>= 2` datasets; *and* every output-only /
sampling / detector baseline in `baseline_registry.yaml` will have `\hat U`
indistinguishable from `0`. If the detector baselines also clear `0.05`, the
causal claim is **falsified** and the project downgrades to diagnosis per the
existing stop rule. This is a prediction that can lose.

---

## 4. Experiment design to confirm/kill (design only — NO runs)

**Datasets (4 families, reused from `data_and_evaluation_plan.md`; lead uses the
first two).** open-domain factual QA (TruthfulQA-style + TriviaQA-style);
multi-hop QA (HotpotQA-style) for trajectory-correction; hallucination benchmark
(HaluEval-style); diffusion-LM trace dataset (transfer only). Each needs
license, raw+processed hash, split definition, leakage check (existing manifest
requirements).

**Base models w/ sizes (pending user approval; design names only).**
- Lead AR: one open-weights instruct family at **two sizes** (e.g. 7-9B and
  13-14B class) to test that CIU is not a single-checkpoint artifact.
- Transfer reasoning: one open large-reasoning model (step-structured traces).
- Transfer diffusion: one open diffusion-LM matching TraceDet's public protocol.
All checkpoints, prompt-template hashes, decoding params recorded per
`baseline_contract.md`.

**Baselines (reuse `baseline_registry.yaml`, incl. the base/closest papers).**
`random_segment` (sanity), `output_entropy`, `semantic_entropy` (Nature 2024),
`output_signature_detector` (LOS-style), `reasoning_consistency_detector`
(RACE), `selfcheckgpt`, `inside_detector`, `diffusion_trace_detector`
(TraceDet, base paper for the transfer section), `tdgnet_trace_detector`.
**Every baseline is also scored on `\hat U`**, not only AUROC — this is the
matched-setup test that exposes the detector floor (Theorem 2.5(2)).

**Metrics (reuse `experiment_protocol.md`).** Detection: AUROC, AUPRC,
FPR@95TPR. Localization: segment P/R/F1 vs perturbation-derived labels.
Intervention: `factuality_delta`, `answer_accuracy_delta`, `utility_delta`, and
the headline `\hat U = targeted_delta - random_delta`. Efficiency: trace-extract
cost, intervention latency.

**Matched-budget protocol.** Same examples, prompts, decoding params, split,
evaluator across all methods (existing fairness policy). For `\hat U`, the
*targeted* and *random* arms additionally share operator `theta`, edit budget
`k`, segment length, layer set `L_patch`, and reference run (Section 2.3). All
methods consume the identical `configs/seeds/paper_20.txt` seed list.

**The kill-gate (pre-registered, unchanged thresholds).**
- **G1 causal sanity (existential):** `\hat U >= 0.05` with Holm-corrected
  paired-bootstrap CI lower bound `> 0.05`, on `>= 2` lead datasets. *Fail ⇒
  stop causal claim, project becomes a detector-comparison diagnosis.*
- **G2 utility harm:** `\hat D_util <= 0.02`. *Fail ⇒ reframe as abstention, not
  mitigation.*
- **G3 transfer:** held-out taxonomy retention `>= 0.80`. *Fail ⇒ drop
  cross-paradigm wording, keep AR-only.*
- **G4 paper evidence:** `>= 20` seeds/replicates, paired tests, effect sizes,
  95% CI. *Fail ⇒ `diagnostic` label only.*
- **Detector-floor check (new, derived from Theorem 2.5):** all non-causal
  baselines must have `\hat U` CI overlapping `0`. *If a detector baseline clears
  `0.05`, the identification (A1) is suspect ⇒ audit the matching set `Pi`
  before any causal wording.*

**Ablations (reuse `motivation_ablation_hparam_plan.md`).** remove intervention
scoring (predictive-only); one operator only; one paradigm only; replace
targeted with random (must collapse `\hat U` to `0` — a positive control for the
estimator); remove utility gate; sweep `w`, `alpha`, `k`, `L_patch`, replay
sample count.

No run is scheduled. `server.authorized: false` remains. Server execution
requires the existing ARIS experiment-plan approval + explicit user command per
`docs/server_runbook.md`.

---

## 5. What changes in code vs the current scaffold

All changes are **additive and uncommitted**. Nothing existing is modified or
deleted by this document. The table below is the *proposed* future diff, to be
implemented only after design approval.

**Files this redesign ADDS now:**
- `docs/redesign/REDESIGN_v1.md` (this file). *Only file written.*

**Files a future additive implementation would ADD (not done here):**
- `src/tracecausal/ciu.py` — `causal_usefulness(deltas_targeted,
  deltas_random)`, `matched_random_null(...)` sampler spec, `paired_bootstrap_U`
  with Holm. Wraps, does not replace, `metrics.passes_intervention_gate`.
- `src/tracecausal/operators.py` — pure-Python *interfaces/contracts* for
  `mask`, `patch`, `replay`, `no_op` with budget accounting `c(I)`; no model
  calls, contract-level only (smoke-testable like existing
  `validate_trace_manifest`).
- `configs/experiments/redesign_v1_ar_lead.yaml` — lead AR config: 1 model
  family `x` 2 sizes, 2 datasets, operators, `gates` block copied verbatim
  (`0.05/0.02/0.80`), `server.authorized: false`, `seeds.paper_minimum: 20`.
- `configs/experiments/redesign_v1_transfer.yaml` — reasoning + diffusion
  transfer config, gated on `heldout_taxonomy_retention: 0.80`.
- `tests/test_ciu_estimator.py` — asserts Theorem 2.5(2) numerically: a
  detector-like selector yields `\hat U ≈ 0`; a causal selector with injected
  `Var(tau)>0` yields `\hat U > 0.05`; budget-mismatch raises.

**Files a future implementation would EXTEND (additively):**
- `configs/baselines/baseline_registry.yaml` — add a `ciu_scored: true` flag per
  baseline (every baseline also reports `\hat U`). No baseline removed.
- `src/tracecausal/schemas.py` — `TraceSegment` already carries `intervention`
  and `selector`; add optional `edit_budget: int` field (default keeps back-
  compat). `Paradigm` Literal already supports the transfer paradigms.

**Contract invariants kept:** `validate_manifest` still requires
`server.authorized == false`, `>= 3` baselines, `>= 20` paper seeds;
`passes_intervention_gate` signature and thresholds unchanged; all
`required_docs` paths in `contracts.py` continue to exist (this file is
additional, not a replacement).

No `git commit`, no `git push`, no run. (Repo currently has no `.git`; nothing
to commit regardless.)

---

## 6. Open risks + honest limitations

- **(A1) is the load-bearing assumption.** Propriety (Theorem 2.5) holds only if
  the matched null `Pi` truly fixes every non-causal covariate. If masking a
  random segment is *systematically* easier/harder than masking the selected one
  (e.g. positional bias near the answer), `bar tau` is biased and a detector
  could leak a positive `\hat U`. Mitigation: the detector-floor check in
  Section 4 is an *empirical falsification test of (A1)* — if detectors clear
  `0.05`, we distrust `Pi`, not celebrate.
- **Reference-run dependence.** `patch`/`replay` need a factual/neutral reference
  whose construction (counterfactual question, retrieved evidence, or
  teacher-forced gold) can itself inject information. The utility guard catches
  sabotage but not *information smuggling*; we must ablate reference type and
  report `\hat U` per reference source.
- **Evaluator is the ground truth of `Y`.** All causal claims are relative to a
  hashed factuality evaluator; evaluator error bounds `\hat U` from below. We
  cannot claim more precision in `\hat U` than the evaluator's own agreement
  rate; this is a stated limitation, not hidden.
- **Diffusion transfer may fail.** That is now acceptable by design (G3),
  but it means the cross-paradigm story is *conditional*; we will not write
  "across all generation paradigms" unless G3 passes for diffusion.
- **Replay is expensive.** Roll-back + re-decode multiplies trace cost; the
  efficiency table may show CIU is an *audit-time* tool, not an inference-time
  one. We will frame it as offline causal auditing, not a live decoder.
- **Scope of the claim.** Per `AGENTS.md`, we never strengthen "causal process
  segments are intervention-useful" to "all hallucinations are causally
  explained." `\hat U > 0.05` certifies *existence and concentration* of causal
  segments, not coverage of every hallucination.
- **No empirical evidence exists.** Everything above is design + theory. Every
  claim-evidence row remains `pending` until a server run under the existing
  gates produces the artifacts; this document does not change any status label.
