# TraceCausal — EXECUTION PACKET v5 (BUILD-NOW / RUN-LATER)

> **STATUS: DO-NOT-RUN TEMPLATE.** `server.authorized: false` in every guarded
> config (`formal_tracecausal.yaml`, `redesign_v4_ar_lead.yaml`,
> `redesign_v5_ar_lead.yaml`). This packet is a *parameterized template* to be
> filled in and authorized later. It contains **NO authorized commands** and
> **NO empirical numbers**. Every results placeholder is `DATA_NEEDED`. Running
> anything here requires the explicit authorization gate in §1 (the user must
> supply `SERVER`, `GPU`, `CONDA_ENV`, flip `server.authorized`, and pin all
> hashes). These three projects (incl. `tracecausal`) have **no assigned server
> yet**; the busy server is off-limits. **Do not** infer a server from this
> document.
>
> **v5 scope (REDESIGN_v5).** This packet supersedes the v4 packet. The headline
> is no longer the in-place necessity statistic `U_hat` (now **screening**) but
> the **cross-example repair-transfer certification** `R_hat` (gate **G9**) with
> the baseline-conditional novelty gate **G9-NOV** (panel **B0–B5**, incl. the
> **TraceDet→AR span adapter B1**), valid dependent-pair inference (**two-way
> source×target cluster bootstrap + class-block permutation + Hájek cross-check**),
> the **selective-inference correction** (SI-1 split / SI-2 Bonferroni over
> `K_bin·K_op`), and the **adversarial oracle Axis X′** (detectable + a *proven*
> blind failure regime, NC-1/NC-2). See §§3.6–3.9.
>
> Pre-registered design source of truth: `docs/redesign/REDESIGN_v5.md` (the
> repair-transfer certification), `docs/redesign/REDESIGN_v4.md` (preserved CIU /
> matched-null / proper-scoring / G1–G8 cores),
> `configs/experiments/redesign_v5_ar_lead.yaml` (frozen v5 lead plan),
> `configs/experiments/redesign_v4_ar_lead.yaml`,
> `configs/experiments/first_gate.yaml`,
> `configs/baselines/baseline_registry.yaml`,
> `configs/compute/first_gate_budget.yaml`, `configs/seeds/paper_20.txt`.
> Governance: `docs/server_runbook.md`, `docs/experiment_protocol.md`,
> `docs/intervention_protocol.md`, `docs/reproducibility_ledger.md`,
> `docs/statistical_analysis_plan.md`, `docs/pre_registration.md`.
>
> **Phase-B gating rule (MF-8).** REDESIGN_v5 status is `design_frozen_stage1_RR`,
> **not** `stage1_ready`. Stage-1 readiness is earned only when every §9 v5 module
> is implemented and `tests/test_ciu_nulldata.py` is green on the v5 additions.
> The v5 modules now exist under `src/tracecausal/` (`repair_transfer.py`,
> `repair_ops.py`, `binning_selection.py`, `selective_inference.py`,
> `adversarial_oracle.py`, and the `ciu.py`/`nuisance.py`/`oracle_gen.py`
> extensions); this packet is written against their **actual** public interfaces.
> Authorization still requires the §1 ARIS gate and the explicit flip.

---

## 0. Placeholders (the user fills these AT AUTHORIZATION — not before)

All commands below are parameterized by shell variables. Nothing is runnable
until the user binds these and explicitly authorizes the run. This section defines
**every** `${...}` placeholder that appears anywhere in this packet. There are
three kinds: (0.1) run-binding variables, (0.2) per-cell iterator/derived
variables used inside the `for`-style command templates of §3, and (0.3)
lock-time analysis placeholders that are pinned only at analysis-lock from the
nuisance estimates (never fabricated here).

### 0.1 Run-binding variables (bound once at authorization)

| Placeholder | Meaning | Set when |
| --- | --- | --- |
| `${SERVER}` | hostname / SSH target of the **assigned** GPU server (none assigned yet; the busy server is off-limits) | at authorization |
| `${GPU}` | CUDA device id(s), e.g. `0` or `0,1` | at authorization |
| `${CONDA_ENV}` | conda/venv environment name with pinned deps | at authorization |
| `${REPO}` | absolute path of the repo checkout on `${SERVER}` | at authorization |
| `${OUT}` | output root on `${SERVER}` (resumable, empty-or-checkpointed) | at authorization |
| `${RUN_ID}` | run label, e.g. `v5_arlead_$(date +%Y%m%d)` | at authorization |
| `${QWEN_REV}` | exact pinned weight revision/commit for `Qwen/Qwen2.5-7B-Instruct` (PRIMARY model) | at authorization |
| `${LLAMA_REV}` | exact pinned weight revision/commit for `meta-llama/Llama-3.1-8B-Instruct` (SECONDARY model) | at authorization |
| `${TRIVIAQA_SPLIT_HASH}` | frozen `V_sel`/`V_inf`/test split hash for TriviaQA (3-way, SI-1) | at authorization |
| `${HOTPOTQA_SPLIT_HASH}` | frozen `V_sel`/`V_inf`/test split hash for HotpotQA (3-way, SI-1) | at authorization |
| `${PROMPT_TMPL_HASH}` | prompt-template hash (`prompt_template_hash` in config) | at authorization |
| `${EVAL_HASH}` | evaluator prompt + answer-key hash (intervention_protocol leakage rule) | at authorization |
| `${REPAIR_EVAL_HASH}` | **target-label** evaluator hash for the cross-example repair score `Y_j` (`kappa^repair`); distinct from `${EVAL_HASH}` (REDESIGN_v5 §4.1, Eq. m-R) | at authorization |
| `${TAXONOMY_HASH}` | frozen G3 class partition hash (`class_partition_hash`; the within-class estimand, A6) | at authorization |

### 0.2 Per-cell iterator / derived variables (loop variables inside §3 templates)

These are **not** bound once; each §3 command is a template iterated over them
(the queue manifest `experiments/queue_manifest.yaml` enumerates the concrete
expansion). They take values from the frozen design dimensions only.

| Placeholder | Meaning | Allowed values | Derived from |
| --- | --- | --- | --- |
| `${FAM}` | lead family id for the current cell | `ar_lead_qwen` (PRIMARY), `ar_lead_llama` (SECONDARY) | `redesign_v5_ar_lead.yaml.lead_models[*].id` |
| `${DS}` | lead dataset for the current cell | `triviaqa`, `hotpotqa` | config `lead_datasets` |
| `${S}` | seed | integers `0..19` | `configs/seeds/paper_20.txt` |
| `${METHOD}` | screening detector (§3.4) | `ciu_selector`, `random_segment`, `output_entropy`, `semantic_entropy`, `output_signature_detector`, `reasoning_consistency_detector`, `selfcheckgpt`, `inside_detector` | config `screening_methods` |
| `${SELECTOR}` | repair-panel selector (§3.6) | `B0`,`B1`,`B2`,`B3`,`B4`,`B5`,`PROPOSED` | config `repair_panel` |
| `${QWEN_REV_OR_LLAMA_REV}` | the §0.1 revision **matching `${FAM}`**: `${QWEN_REV}` when `${FAM}=ar_lead_qwen`, else `${LLAMA_REV}` | one of `${QWEN_REV}` / `${LLAMA_REV}` | §0.1 + current `${FAM}` |
| `${DS_SPLIT_HASH}` | the §0.1 split hash **matching `${DS}`**: `${TRIVIAQA_SPLIT_HASH}` when `${DS}=triviaqa`, else `${HOTPOTQA_SPLIT_HASH}` | one of `${TRIVIAQA_SPLIT_HASH}` / `${HOTPOTQA_SPLIT_HASH}` | §0.1 + current `${DS}` |

### 0.3 Lock-time analysis placeholders (pinned at analysis-lock, NOT fabricated here)

These literals appear in §3 commands as **explicit do-not-fabricate sentinels**.
They are pinned only at analysis-lock from the V_inf nuisance estimates, never
guessed in this packet.

| Sentinel literal | Replaced by (at analysis-lock) | Source |
| --- | --- | --- |
| `DATA_NEEDED_PIN_AT_LOCK` (the `--r-null` value in §3.6) | the variance-floored minimum `R_null` from the measured `sigma_MC` | §3.3 nuisance / §5; REDESIGN_v5 §6.3 |

> The example bindings shown in commands (e.g. `SERVER=PENDING-NO-SERVER-ASSIGNED`)
> are deliberately non-functional sentinels. Replace them; do not run with
> sentinels in place.

```bash
# === BIND AT AUTHORIZATION (example sentinels — REPLACE, do not run as-is) ===
export SERVER=PENDING-NO-SERVER-ASSIGNED        # none assigned; busy server off-limits
export GPU=PENDING
export CONDA_ENV=PENDING
export REPO=/PENDING/tracecausal
export OUT=/PENDING/outputs/tracecausal
export RUN_ID=v5_arlead_PENDING
export QWEN_REV=PENDING_PIN_AT_AUTH
export LLAMA_REV=PENDING_PIN_AT_AUTH
export TRIVIAQA_SPLIT_HASH=PENDING_PIN_AT_AUTH
export HOTPOTQA_SPLIT_HASH=PENDING_PIN_AT_AUTH
export PROMPT_TMPL_HASH=PENDING_PIN_AT_AUTH
export EVAL_HASH=PENDING_PIN_AT_AUTH
export REPAIR_EVAL_HASH=PENDING_PIN_AT_AUTH       # v5: target-label evaluator (kappa^repair)
export TAXONOMY_HASH=PENDING_PIN_AT_AUTH          # v5: frozen G3 class partition (A6)
```

---

## 1. Preflight checklist (ALL must pass before any arm runs)

Per `docs/server_runbook.md` "Before Any Server Run" + `docs/compute_budget.md`
"Feasibility Rule" + `configs/compute/first_gate_budget.yaml.preflight_required`,
extended for the v5 repair-transfer + selective-inference surfaces.

### 1.1 Authorization flip (HARD GATE)
- [ ] User has assigned a concrete `${SERVER}` / `${GPU}` / `${CONDA_ENV}` (none
      assigned at packet-write time; the busy server stays off-limits).
- [ ] ARIS experiment-plan review is on file and `>= 8` with no hard-rule
      violation (`docs/aris_experiment_plan_review.md`, `reports/aris_9_8_scorecard.md`),
      **re-reviewed for the v5 G9/G9-NOV/Axis X′ additions** (the v4 review does
      not cover the new gates).
- [ ] User explicitly approves the **exact** command, model, dataset, output
      path, and stop condition (server_runbook §"Before Any Server Run" item 5).
- [ ] Flip `server.authorized: false -> true` in **all three** guarded configs and
      re-run the local contract validator to confirm the *intended* flip
      (the validator currently **requires** `authorized: false`; flipping is a
      deliberate, logged, reviewed action — see note below).

```bash
# Confirm current frozen state BEFORE flip (must print authorized: false everywhere):
grep -rn "authorized:" configs/experiments/*.yaml
# Local contract validator (pure-python, no GPU) — passes while authorized:false:
python scripts/validate_project.py
# v5 record/gate harness (pure-python, no GPU) — must be GREEN before any v5 lock:
python -m pytest tests/test_ciu_nulldata.py -q
```

> **NOTE on the flip:** `scripts/validate_project.py` and
> `src/tracecausal/contracts.py::validate_manifest` enforce `authorized: false`
> on `formal_tracecausal.yaml`, `redesign_v4_ar_lead.yaml`, **and**
> `redesign_v5_ar_lead.yaml` (`AUTHORIZATION_GUARDED_CONFIGS`). Authorizing a run
> means consciously changing that guarded value, recording who/when/why in the
> reproducibility ledger, and accepting that the guard test will then fail by
> design. **Until that reviewed flip happens, treat this whole packet as inert.**

### 1.2 Dataset / model / hash pinning (reproducibility_ledger.md)
- [ ] Pin `${QWEN_REV}`, `${LLAMA_REV}` (exact weight revision/commit; the config
      pins only the family + instruct id, revision = `pin_exact_revision_at_run_authorization`).
- [ ] Pin `${TRIVIAQA_SPLIT_HASH}`, `${HOTPOTQA_SPLIT_HASH}` as **3-way**
      `V_sel`/`V_inf`/test split hashes, all disjoint from each other and from
      test (`selection_split: required`, SI-1 primary; REDESIGN_v5 §6.1). A 2-way
      v4 split hash is **not** sufficient for a v5 lock.
- [ ] Pin `${PROMPT_TMPL_HASH}` (`prompt_template_hash: pending_before_run`).
- [ ] Pin `${EVAL_HASH}` — necessity (`U_hat`) evaluator prompt + answer key
      hashed **before** the run (`intervention_protocol.md` "Evaluator Leakage").
- [ ] Pin `${REPAIR_EVAL_HASH}` — the **cross-example target-label** evaluator
      `Y_j` used by `R_hat` (its own `kappa^repair` re-estimated on target labels,
      Eq. m-R). Any post-hoc edit to either evaluator invalidates the run.
- [ ] Pin `${TAXONOMY_HASH}` — the frozen G3 class partition that defines the
      within-class estimand (A6); persisted as `class_partition_hash` in every
      `CIURecord` (REDESIGN_v5 §4.1, §10 risk 2).
- [ ] Leakage / contamination check executed and **passes**
      (`leakage_check: required_before_run`).
- [ ] Baseline readiness: every baseline marked
      `pending_before_server_run` / `verify_before_run` in
      `baseline_registry.yaml` has its `implementation_source`,
      `implementation_commit`, and `license` resolved + verified
      (`src/tracecausal/ciu.py::baseline_readiness`). For v5 this **includes the
      repair-panel selectors** B1 (TraceDet→AR adapter,
      `repair_ops.tracedet_ar_span_adapter`), B2 (entropy/perplexity peak), B3
      (latent probe) — each must select a span fed through the **identical**
      `repair_ops` pipeline (G9-NOV fairness).
- [ ] `diffusion_trace_detector` / `tdgnet_trace_detector` confirmed **excluded
      from the AR-lead table** (`ar_lead_inclusion: excluded_from_ar_lead_table`);
      they appear only in the diffusion transfer study, not these arms.

### 1.3 GPU / storage / process preflight (`first_gate_budget.yaml.preflight_required`)
- [ ] `gpu_available` — `${GPU}` visible, idle, not shared with another job.
- [ ] `output_dir_empty_or_resumable` — `${OUT}` empty OR holds a valid
      checkpoint state for resume (see §6).
- [ ] `disk_free_gb_at_least_390` — free storage `>= 390 GB` (300 GB raw traces +
      30% buffer). **v5 repair forwards add output volume** (per-target localized
      + `R_null` matched-null-repair + `R_int` repair-op repeats); re-check the
      realized surcharge `forwards_per_example` at lock (§5) and re-budget storage
      if the recomputed surcharge exceeds the v4 `18`.
- [ ] `compact_sync_path_ready` — local sync path for compact artifacts ready
      (`<2 GB`; raw traces never synced — server_runbook "Artifact Sync").
- [ ] Confirm **no active server process will be overwritten** (server_runbook
      item 4).

```bash
# Storage + GPU preflight (read-only checks; safe once a server is assigned):
ssh ${SERVER} 'nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv'
ssh ${SERVER} "df -BG ${OUT%/*} | awk 'NR==2{print \$4}'"   # must be >= 390G
```

### 1.4 Stop conditions armed (`first_gate.yaml.gates` + v5 G9 routing §3.9 / §8)
Arm the early-kill gates so a failing arm halts the queue (not just logs):
- `causal_margin_abs` < `0.05` (`targeted_delta - random_delta`, the **screening**
  `U_hat` margin G1) -> **stop causal claim (screening)**.
- `utility_drop_abs` > `0.02` -> **downgrade to diagnosis** (necessity G2).
- `repair_utility_drop_abs` (`D_util^repair`) > `0.02` -> **G9 reframes as
  abstention** (REDESIGN_v5 §5.1; never `invalidated`).
- `positivity_excluded_frac` >= `0.5` for any class -> that class routes to
  **"insufficient positivity"** (A7), NOT counted as a G9 null.
- `evaluator_leakage` audit != pass (either `${EVAL_HASH}` or `${REPAIR_EVAL_HASH}`)
  -> **invalidate run**.
- `storage_budget_exceeded` -> **stop**.
- invalid-intervention/invalid-repair rate > 5% -> **diagnostic-only**
  (`intervention_protocol.md`).
- **Axis X′-blind soundness (P5/NC-1):** if controls G7/G8 stay silent **and**
  `R_hat` stays certified under collinear confounding -> **certification withdrawn
  (R4)**; this is an oracle-fixture stop, decided in §3.8, before any lead claim.

> The G9/G9-NOV gates themselves are **fail-closed and never return
> `invalidated`** (they reuse the v4 `ciu_gate` template via
> `ciu.g9_repair_gate` / `ciu.g9_novelty_gate`): a failure **routes** per §3.9/§8,
> it does not silently bypass.

---

## 2. Run units (arms) and the arm × seed matrix

**Frozen lead (no diffusion in these arms):**
- Families (2): `ar_lead_qwen` (Qwen2.5-7B-Instruct @ `${QWEN_REV}`),
  `ar_lead_llama` (Llama-3.1-8B-Instruct @ `${LLAMA_REV}`).
- Lead datasets (2): `triviaqa`, `hotpotqa`.
- **Methods / repair-panel selectors (8):** the proposed `ciu_selector`
  (PROPOSED) **plus the B0–B5 repair panel** that feeds the **identical**
  `repair_ops` pipeline (REDESIGN_v5 §4.3):
  - `B0` no_op floor (target unedited),
  - `B1` TraceDet→AR-selected span (`repair_ops.tracedet_ar_span_adapter`),
  - `B2` entropy/perplexity-peak-selected span,
  - `B3` latent-probe-selected span,
  - `B4` random-same-class span (the matched-null repair control *inside* `g_ij`),
  - `B5` oracle-selected span (planted ground truth, oracle fixtures only; upper
    bound),
  - `PROPOSED` the CIU selector's `S*`.
  The **legacy v4 detector baselines** (`random_segment`, `output_entropy`,
  `semantic_entropy`, `output_signature_detector`,
  `reasoning_consistency_detector`, `selfcheckgpt`, `inside_detector`) remain in
  the **screening** (`U_hat`) detection/G1/G5′ tables (re-scored under
  `ciu_scored: true`). B1/B2/B3 are the **repair-transfer** counterparts of
  TraceDet / entropy / probe, run through `repair_ops`.
- Negative controls (run inside the relevant arms, not as separate methods):
  - necessity (`U_hat`): `random_non_causal_segment`, `shuffled_trace_segment`,
    `no_op_intervention`;
  - repair (`R_hat`): **NC-1** collinear-confounder (Axis X′-blind, controls
    silent / `R_hat` must collapse) and **NC-2** source-swap exchangeability
    (`g_ij` invariant under in-class source swap, tests A5/A6).
- Seeds (20): `configs/seeds/paper_20.txt` = `0..19`, shared by all methods
  (`strategy: sequential_0_to_19_shared_by_all_methods`).

**A "cell"** = one (family × dataset) pair. **A "selector-arm"** = one (panel
selector × cell). **A "job"** = one (arm × seed). The repair-transfer estimand
`R_hat` is computed **across examples within a class on the same cell** (Eq. R), so
its inference jobs are aggregation-stage jobs (§3.7), not per-example.

| Design | cells | screening arms (8 detectors × cells) | repair-panel arms (7 selectors × cells) | jobs/cell (extract+screen+repair) |
| --- | ---: | ---: | ---: | --- |
| **PRIMARY confirmatory (pre-registered feasible 2-cell point)** — `Qwen2.5-7B-Instruct` × {`triviaqa`, `hotpotqa`} | 2 | 8 × 2 = **16** | 7 × 2 = **14** | per §2.1; tagged `feasible_2cell` |
| **SECONDARY (budget-gated 4-cell grid)** — adds `Llama-3.1-8B-Instruct` on the same 2 datasets | 4 | 8 × 4 = **32** | 7 × 4 = **28** | tagged `full_grid_only` |

> **PRIMARY vs SECONDARY (identical wording in `paper/main.tex` §5.4
> "Sampling levels and the confirmatory family").** The **PRIMARY CONFIRMATORY**
> cells are the pre-registered feasible **2-cell point**:
> `ar_lead_qwen` (Qwen2.5-7B-Instruct) on **TriviaQA** and **HotpotQA** — one model
> family, two datasets. The headline `R_hat`/G9/G9-NOV claim and the Holm
> confirmatory family are adjudicated on exactly these two cells. The **4-cell grid**
> that *adds* `ar_lead_llama` (Llama-3.1-8B-Instruct) on the same two datasets is
> **SECONDARY and budget-gated**: it tests cross-family generalization, not the
> primary claim.
>
> Cell-count selection is governed by the **re-checked** budget identity with the
> *new* G9 forward surcharge (§5). Default at lock is the **PRIMARY 2-cell** point;
> the SECONDARY 4-cell grid is run **only if** the recomputed `c_fwd` and
> `forwards_per_example` clear the §5 ceiling
> (`redesign_v5_ar_lead.yaml.decision_order`). The queue manifest
> (`experiments/queue_manifest.yaml`) enumerates the full job set (screening +
> repair-transfer + adversarial-oracle stages) and tags each job with `cell_tier`
> (`feasible_2cell` = PRIMARY vs `full_grid_only` = SECONDARY) so the PRIMARY 2-cell
> subset runs first and the rest is gated behind the c_fwd / surcharge decision.

### 2.1 Pipeline order per cell (jobs are NOT independent)
Per `reports/experiment_plan/aris_plan.md` + `intervention_protocol.md` +
REDESIGN_v5 §§4–7:
1. **trace extraction** (once per cell × seed) — produces the trace manifest
   (`schemas/trace_manifest.schema.json`).
2. **timing calibration** — measure `c_fwd` on the **`V_sel`** split (a *timing
   measurement, not an experiment*); feeds the §5 cell-count decision **with the
   v5 forward surcharge**.
3. **selection split + binning-as-code** — `selective_inference.validate_selection_split`
   confirms `V_sel`/`V_inf`/test disjoint + floors;
   `binning_selection.select_binning(V_sel)` freezes `Delta_pos`/`displaced_mass`
   edges and emits the `selection_event` that sizes `K_bin` (REDESIGN_v5 §6.4).
4. **nuisance estimation** (validation-only) — `sigma_u` (`>=200` paired ex),
   `kappa` (`>=300` double-scored), `kappa^repair` (target labels), `m_pool`
   (`>=8`); freeze `U_target`/`R_power` at analysis-lock.
5. **operator-selection freeze (OS-1)** — `repair_ops.select_operator(grid, ...)`
   freezes `rho = (op, alpha, L_patch, budget_k, ref_type, anchor_rule)` and the
   class weights `w_c` **on `V_sel` only**, recording `k_op` from the actual grid
   cardinality (REDESIGN_v5 §4.7).
6. **screening interventions** (`ciu_selector` + 7 v4 detectors) — `U_hat`:
   targeted vs random vs negative controls; `R_int = 16` repeats.
7. **repair-transfer interventions** (panel B0–B5 + PROPOSED) — `g_ij` localized
   repair on target via `repair_ops.transport` (Variant C) + `R_null` matched-null
   repairs + `R_int` repair-op repeats per target (Eq. g-ij).
8. **adversarial oracle Axis X′** (fixture-only, no lead data) — `xi`-sweep,
   detectable + blind regimes, NC-1/NC-2 (REDESIGN_v5 §7).
9. **detection + repair scoring** — AUROC/AUPRC/FPR@95TPR (screening) and `R_hat`
   per class (repair).
10. **gate evaluation** — screening G1/G2/G5′/G6/G7/G8; headline **G9** + **G9-NOV**
    with two-way cluster bootstrap + class-block permutation + Hájek cross-check;
    SI Holm family over `m'` with `K_bin·K_op` fold (REDESIGN_v5 §6.3).

> **Run-skeleton note.** The pipeline entrypoints below now **exist on disk** as
> **thin do-not-run CLI wrappers**: `scripts/extract_traces.py`,
> `scripts/select_binning.py`, `scripts/run_intervention.py`,
> `scripts/run_repair_transfer.py`, `scripts/run_adversarial_oracle.py`,
> `scripts/score_detection.py`, `scripts/eval_gates.py` (plus the shared guard
> helper `scripts/_runpacket_common.py`). Each has an argparse CLI matching the
> commands in this section and a **hard authorization guard**: heavy work
> (model load / GPU / forwards) runs ONLY when BOTH (a) the `--config` has
> `server.authorized: true` AND (b) the explicit `--i-have-authorization` flag is
> passed. **Without both, every script is a DRY-RUN: it prints the fully-resolved
> plan as JSON, writes an inert `STATUS.json`, and exits 0 having loaded NO model,
> NO GPU, and not even the `tracecausal` kernel** (heavy and kernel imports are
> lazy, inside the guarded branch). The analysis kernels they call
> (`tracecausal.repair_transfer`, `.repair_ops`, `.binning_selection`,
> `.selective_inference`, `.adversarial_oracle`, `.nuisance`, `.ciu`) are
> implemented and unit-tested (113 passing pure-Python tests, no model/GPU). The
> actual extraction/forward/analysis loops inside each guarded branch are the
> authorized build-out (the wrappers `raise NotImplementedError` there, reachable
> only after the §1 flip); commands are written against the implemented kernel
> APIs so the packet is fill-in-ready. Resumability is concrete: every job writes a
> per-job `STATUS.json` checkpoint and every command accepts `--resume` (a job
> whose `STATUS.json.state == done` is skipped); `eval_gates.py --reconcile-queue`
> rebuilds the remaining pending set from the on-disk checkpoints in v5 dependency
> order (§6).

---

## 3. Per-step commands (template — bind §0 vars first)

> All commands below are **inert until §1 passes**. Run from `${REPO}` on
> `${SERVER}` inside `${CONDA_ENV}`. `--config` is always
> `configs/experiments/redesign_v5_ar_lead.yaml` (the frozen v5 lead plan). Each
> command is **idempotent + resumable** (`--resume`): a job whose `STATUS.json.state
> == done` with a matching `output_hash` is skipped (§6).
>
> **Authorization flag.** As written, every command below is a **dry-run** (prints
> the resolved plan, loads nothing, exits 0). To actually execute after the §1
> flip, append `--i-have-authorization`; the script then *additionally* checks that
> `--config` has `server.authorized: true` and only then loads the model/GPU. Both
> gates are required — the flag alone, or the config flip alone, still dry-runs.
> The `--i-have-authorization` flag is intentionally **omitted** from the templates
> below so the packet stays do-not-run.

### 3.0 Environment + provenance capture (once per run)
```bash
ssh ${SERVER}
conda activate ${CONDA_ENV}
cd ${REPO}
# Reproducibility ledger capture (reproducibility_ledger.md): git, env, hashes:
git rev-parse HEAD; git status --porcelain        # commit + dirty status
python -c "import torch,transformers,platform;print(torch.__version__,transformers.__version__,platform.platform())"
# v5 kernel harness must be green before any GPU step (no model/GPU):
python -m pytest tests/test_ciu_nulldata.py -q
mkdir -p ${OUT}/${RUN_ID}/{traces,binning,nuisance,operator_freeze,interventions,repair_transfer,adversarial_oracle,detection,gates,provenance,logs}
```

### 3.1 Trace extraction (once per cell × seed) — 8 GPU-hr planned (+30% buffer)
```bash
# Per (family ${FAM} in {ar_lead_qwen,ar_lead_llama}) x (dataset ${DS} in {triviaqa,hotpotqa}) x (seed ${S} in 0..19)
python scripts/extract_traces.py \
  --config configs/experiments/redesign_v5_ar_lead.yaml \
  --family ${FAM} --model-revision ${QWEN_REV_OR_LLAMA_REV} \
  --dataset ${DS} --split-hash ${DS_SPLIT_HASH} \
  --prompt-template-hash ${PROMPT_TMPL_HASH} \
  --taxonomy-hash ${TAXONOMY_HASH} \
  --seed ${S} \
  --run-tier paper_candidate \
  --output ${OUT}/${RUN_ID}/traces/${FAM}__${DS}__seed${S} \
  --device cuda:${GPU} \
  --resume
```
**Expected artifacts:** `traces/${FAM}__${DS}__seed${S}/trace_manifest.json`
(validates `schemas/trace_manifest.schema.json`, `server_authorized:false` const),
per-example residual-state cache for `patch`/`replay` reference construction,
atomic-claim spans tagged with `${TAXONOMY_HASH}` G3 class + proximity bin, and a
`STATUS.json`.

### 3.2 Timing calibration + selection split + binning-as-code (validation-only; not an experiment)
```bash
# (a) c_fwd timing calibration on V_sel (feeds the §5 cell-count decision WITH the v5 surcharge):
python scripts/extract_traces.py --config configs/experiments/redesign_v5_ar_lead.yaml \
  --calibrate-cfwd --family ${FAM} --dataset ${DS} --split v_sel \
  --output ${OUT}/${RUN_ID}/provenance/cfwd_${FAM}__${DS}.json --device cuda:${GPU}

# (b) SI-1 selection split validation + frozen binning (PURE CPU; selection-as-code, §6.4):
#     wraps tracecausal.selective_inference.validate_selection_split +
#     tracecausal.binning_selection.select_binning(V_sel) and records the selection_event.
python scripts/select_binning.py --config configs/experiments/redesign_v5_ar_lead.yaml \
  --family ${FAM} --dataset ${DS} \
  --v-sel ${OUT}/${RUN_ID}/traces \
  --delta-pos-ladder 1,2,4,8,16 \
  --displaced-mass-edges 0.0,0.05,0.1,0.2,0.4,1.0 \
  --pool-floor 8 \
  --output ${OUT}/${RUN_ID}/binning/${FAM}__${DS}.json
```
**Expected artifacts:** `provenance/cfwd_${FAM}__${DS}.json` (measured `c_fwd`,
`DATA_NEEDED` until run); `binning/${FAM}__${DS}.json` carrying the chosen
`Binning` (`delta_pos`, `displaced_mass_edges`, `meets_pool_floor`) **and** the
`SelectionEvent` (`rungs_walked`, `k_bin`) that sizes the `K_bin` Holm fold.

### 3.3 Nuisance estimation + operator-selection freeze (validation-only; frozen at analysis-lock)
```bash
# (a) Nuisance estimators on V_inf only — sigma_u, kappa, kappa^repair, m_pool, sigma_R:
python scripts/eval_gates.py --config configs/experiments/redesign_v5_ar_lead.yaml \
  --estimate-nuisance --split v_inf \
  --n-val-sigma 200 --n-val-kappa 300 --proximity-pool-min 8 \
  --repair-eval-hash ${REPAIR_EVAL_HASH} \
  --traces ${OUT}/${RUN_ID}/traces \
  --binning ${OUT}/${RUN_ID}/binning/${FAM}__${DS}.json \
  --output ${OUT}/${RUN_ID}/nuisance/${FAM}__${DS}.json
# -> wraps nuisance.estimate_sigma_u / estimate_kappa (necessity + kappa^repair) /
#    pool_inflation; sigma_R is estimated post-repair in §3.7 (needs g_ij draws).

# (b) Operator-selection freeze (OS-1) on V_sel ONLY — freeze rho + class weights w_c:
python scripts/run_repair_transfer.py --config configs/experiments/redesign_v5_ar_lead.yaml \
  --freeze-operator --split v_sel \
  --family ${FAM} --dataset ${DS} \
  --op-grid-ops patch,replay \
  --op-grid-alpha 0.1,0.25,0.5,0.75,1.0 \
  --op-grid-ref-type factual,neutral \
  --select-objective r_hat_proposed_minus_B4 \
  --traces ${OUT}/${RUN_ID}/traces \
  --output ${OUT}/${RUN_ID}/operator_freeze/${FAM}__${DS}.json --device cuda:${GPU}
# -> wraps repair_ops.select_operator(grid, score_on_v_sel=...); records the frozen
#    RepairPolicy, the policy_hash, and k_op = operator_grid_cardinality(grid).
```
**Expected artifacts:** `nuisance/${FAM}__${DS}.json` (`sigma_u` upper-CI, `kappa`
lower-CI, `kappa^repair` lower-CI, `m_pool`, all values `DATA_NEEDED`);
`operator_freeze/${FAM}__${DS}.json` (frozen `rho`, `repair_policy_hash`,
`transport_map_hash`, `k_op`, `w_c`). **These freeze on `V_sel`/`V_inf` and are
never re-touched on test** (SI-1).

### 3.4 Screening interventions (`U_hat`, per detector × cell × seed) — necessity arms
```bash
# ${METHOD} in {ciu_selector,random_segment,output_entropy,semantic_entropy,
#               output_signature_detector,reasoning_consistency_detector,
#               selfcheckgpt,inside_detector}
python scripts/run_intervention.py \
  --config configs/experiments/redesign_v5_ar_lead.yaml \
  --stage screening \
  --method ${METHOD} \
  --family ${FAM} --model-revision ${QWEN_REV_OR_LLAMA_REV} \
  --dataset ${DS} --split-hash ${DS_SPLIT_HASH} \
  --seed ${S} \
  --traces ${OUT}/${RUN_ID}/traces/${FAM}__${DS}__seed${S} \
  --binning ${OUT}/${RUN_ID}/binning/${FAM}__${DS}.json \
  --operator mask \
  --r-int 16 \
  --negative-controls random_non_causal_segment,shuffled_trace_segment,no_op_intervention \
  --evaluator-hash ${EVAL_HASH} \
  --ciu-scored \
  --output ${OUT}/${RUN_ID}/interventions/${METHOD}__${FAM}__${DS}__seed${S} \
  --device cuda:${GPU} \
  --resume
```
**Expected artifacts:** per-arm `CIURecord` rows (necessity `U_hat`, `u_deflated`,
matched-null pool draws, negative-control deltas) under
`interventions/...`; `STATUS.json`. These feed screening gates G1/G2/G5′/G6/G7/G8.

### 3.5 Matched-null pool construction (per cell; the repair control `Pi_j`)
```bash
# Build the per-TARGET matched-null pool Pi_j (proximity+budget-stratified) used by
# BOTH the screening necessity control AND the repair control B4 (A9). Pure CPU on
# the extracted traces (no new forwards); wraps tracecausal.nullpool.
python scripts/run_repair_transfer.py \
  --config configs/experiments/redesign_v5_ar_lead.yaml \
  --build-matched-null-pool \
  --family ${FAM} --dataset ${DS} \
  --traces ${OUT}/${RUN_ID}/traces \
  --binning ${OUT}/${RUN_ID}/binning/${FAM}__${DS}.json \
  --proximity-pool-min 8 \
  --output ${OUT}/${RUN_ID}/repair_transfer/${FAM}__${DS}__nullpool.json
```
**Expected artifacts:** `repair_transfer/${FAM}__${DS}__nullpool.json` — per target
`x_j` the in-budget, proximity-matched, length-matched span set `Pi_j` (the B4
matched-null repair candidates). Positivity exclusions are recorded here (A7).

### 3.6 Repair-transfer interventions (`g_ij`, panel B0–B5 + PROPOSED) — the v5 forwards
```bash
# ${SELECTOR} in {B0,B1,B2,B3,B4,B5,PROPOSED}. Variant C transport: the SOURCE-derived
# policy rho is applied on the TARGET's own run via repair_ops.transport (anchor map T).
# Per target: 1 localized-repair fwd + R_null matched-null-repair fwds + R_int repair-op repeats.
python scripts/run_repair_transfer.py \
  --config configs/experiments/redesign_v5_ar_lead.yaml \
  --stage repair_transfer \
  --selector ${SELECTOR} \
  --family ${FAM} --model-revision ${QWEN_REV_OR_LLAMA_REV} \
  --dataset ${DS} --split-hash ${DS_SPLIT_HASH} \
  --seed ${S} \
  --traces ${OUT}/${RUN_ID}/traces \
  --operator-freeze ${OUT}/${RUN_ID}/operator_freeze/${FAM}__${DS}.json \
  --nullpool ${OUT}/${RUN_ID}/repair_transfer/${FAM}__${DS}__nullpool.json \
  --transport-variant C \
  --taxonomy-hash ${TAXONOMY_HASH} \
  --r-null DATA_NEEDED_PIN_AT_LOCK \
  --r-int 16 \
  --repair-eval-hash ${REPAIR_EVAL_HASH} \
  --source-neq-target --within-class-only \
  --output ${OUT}/${RUN_ID}/repair_transfer/${SELECTOR}__${FAM}__${DS}__seed${S} \
  --device cuda:${GPU} \
  --resume
```
**Expected artifacts:** per (selector × cell × seed) the per-pair `RepairGain`
rows (`g_ij`, `mc_var`, `source_id`, `target_id`, `g3_class`) from
`repair_transfer.repair_gain` (Eq. g-ij), the localized `Y_j(do(...))`, the `no_op`
`Y_j`, and the `R_null` matched-null repair `Y_j` draws; positivity-fail exclusions
recorded; `STATUS.json`. `--r-null` is **`DATA_NEEDED`**, pinned at lock to the
variance-floored minimum from `sigma_MC` (§5); never fabricated here.

### 3.7 Repair-transfer estimation + dependent-pair inference (per cell, then global)
```bash
# Pure CPU on the collected g_ij rows (no new forwards). Wraps:
#   repair_transfer.r_hat (within-class U-statistic, Eq. R),
#   repair_transfer.two_way_cluster_bootstrap (MF-4 primary variance, >=10,000 reps),
#   repair_transfer.class_block_permutation (MF-4 null; tests A6),
#   repair_transfer.hajek_projection_var (U-statistic analytic cross-check),
#   repair_transfer.target_clustered_mc_var (nested matched-null MC, in quadrature),
#   nuisance.estimate_sigma_r (Eq. R-VAR design effect).
python scripts/run_repair_transfer.py \
  --config configs/experiments/redesign_v5_ar_lead.yaml \
  --estimate-r-hat \
  --family ${FAM} --dataset ${DS} \
  --pairs ${OUT}/${RUN_ID}/repair_transfer \
  --class-weights ${OUT}/${RUN_ID}/operator_freeze/${FAM}__${DS}.json \
  --bootstrap 10000 \
  --permutations 10000 \
  --output ${OUT}/${RUN_ID}/repair_transfer/${FAM}__${DS}__rhat.json
```
**Expected artifacts:** `repair_transfer/${FAM}__${DS}__rhat.json` — per-class and
weighted `R_hat` (`RHatEstimate`), the two-way-cluster-bootstrap CI
`(ci_lo, ci_hi)`, the class-block-permutation `p`, the Hájek ordered-kernel
two-projection variance (`zeta_10/n_source + zeta_01/n_target`), the nested
matched-null MC variance, the per-baseline `R_hat(B0..B5)`, and the
`SigmaREstimate` (`zeta_10`, `zeta_01`, `n_source`, `n_target`, `n_eff`, `sigma_MC`,
`sigma_op`, `D_eff`; `zeta_1_max_reporting_only` is reporting-only, NOT the variance)
— all numbers `DATA_NEEDED`.

### 3.8 Adversarial oracle Axis X′ (fixture-only; no lead data; no model forwards on lead)
```bash
# xi-sweep, detectable + blind regimes, NC-1/NC-2. Reproduces v4 clean oracle at xi=0.
# Wraps adversarial_oracle.axis_x_confounded / negative_control_collinear / source_swap.
python scripts/run_adversarial_oracle.py \
  --config configs/experiments/redesign_v5_ar_lead.yaml \
  --axis x_prime \
  --regimes detectable,blind \
  --xi-grid 0.0,0.25,0.5,0.75,1.0 \
  --negative-controls NC-1_collinear_confounder,NC-2_source_swap \
  --output ${OUT}/${RUN_ID}/adversarial_oracle/axis_x_prime.json
```
**Expected artifacts:** `adversarial_oracle/axis_x_prime.json` — per `(xi, regime)`
the planted `c_i`, the misspecified reference, the G7/G8 control readouts, the
`R_hat` readout, and the **registered P5 expectation**: detectable ⇒ controls trip
*before* `R_hat` certifies; blind ⇒ controls stay silent while `R_hat -> 0` (P5);
NC-1 (controls silent / `R_hat` must collapse) and NC-2 (`g_ij` invariant under
in-class source swap). If blind shows controls silent AND `R_hat` certified ⇒
**R4 unsound** (§1.4 stop).

### 3.9 Gate evaluation + Holm/SI family (screening G1–G8, headline G9/G9-NOV)
```bash
python scripts/eval_gates.py \
  --config configs/experiments/redesign_v5_ar_lead.yaml \
  --family ${FAM} --dataset ${DS} \
  --interventions ${OUT}/${RUN_ID}/interventions \
  --repair-transfer ${OUT}/${RUN_ID}/repair_transfer \
  --adversarial-oracle ${OUT}/${RUN_ID}/adversarial_oracle/axis_x_prime.json \
  --detection ${OUT}/${RUN_ID}/detection \
  --nuisance ${OUT}/${RUN_ID}/nuisance/${FAM}__${DS}.json \
  --binning ${OUT}/${RUN_ID}/binning/${FAM}__${DS}.json \
  --operator-freeze ${OUT}/${RUN_ID}/operator_freeze/${FAM}__${DS}.json \
  --g1-on u_deflated --g7-on leakage_bound_upper_ci \
  --g9 --g9-nov \
  --si-path-rule SI1_if_floors_met_else_SI2 \
  --holm-family g1,g2,g5prime,g6,g7,g8,g9_cells,g9_nov_B1,g9_nov_B2,g9_nov_B3 \
  --bootstrap 10000 --permutations 10000 \
  --require-v5 \
  --output ${OUT}/${RUN_ID}/gates/${FAM}__${DS}
```
**Expected artifacts:** `gates/${FAM}__${DS}/` —
- screening verdicts G1 (on `u_deflated`), G2, G5′, G6, G7 (upper-CI leakage
  bound), G8 (SHAM null + bounded OOD slope);
- **G9** (`ciu.g9_repair_gate`): Holm/SI-corrected two-way-cluster-bootstrap CI
  lower bound vs `m_R`, class-block permutation `p < alpha_1'`, `D_util^repair <=
  0.02`, positivity `< 0.5` per class, source≠target/no-self-leakage — verdict
  routes per §8, **never `invalidated`**;
- **G9-NOV** (`ciu.g9_novelty_gate` / `repair_transfer.g9_nov_margin_simultaneous`):
  `R_hat(PROPOSED) − max(R_hat(B1),R_hat(B2),R_hat(B3))` simultaneous lower CI > 0;
- the SI family record: `m'`, `K_bin`, `K_op`, `alpha_1'`
  (`selective_inference.holm_alpha`, Eq. SI-ALPHA), the chosen SI path
  (`choose_si_path`);
- a validated v5 `CIURecord` per cell (`validate_ciu_record(require_v5=True)` must
  pass: G9/G9-NOV + SI + transport/positivity fields present, hashes non-empty,
  `server_authorized:false`).
All result numbers `DATA_NEEDED`.

---

## 4. Per-step expected-artifact summary (DATA_NEEDED slots the paper requires)

| Step | Script | Output | Result slot (paper) |
| --- | --- | --- | --- |
| 3.1 trace extraction | `extract_traces.py` | `traces/.../trace_manifest.json` | provenance / N per cell |
| 3.2 timing + binning | `extract_traces.py --calibrate-cfwd`, `select_binning.py` | `cfwd_*.json`, `binning/*.json` | `c_fwd`, `K_bin`, pool-floor table |
| 3.3 nuisance + OS-1 | `eval_gates.py --estimate-nuisance`, `run_repair_transfer.py --freeze-operator` | `nuisance/*.json`, `operator_freeze/*.json` | `sigma_u`, `kappa`, `kappa^repair`, frozen `rho`, `K_op` |
| 3.4 screening `U_hat` | `run_intervention.py --stage screening` | `interventions/*` | **Tab G1 / G5′** (necessity, detection AUROC) |
| 3.5 matched-null pool | `run_repair_transfer.py --build-matched-null-pool` | `*__nullpool.json` | positivity / A7 exclusion table |
| 3.6 repair `g_ij` | `run_repair_transfer.py --stage repair_transfer` | `repair_transfer/${SELECTOR}__*` | per-pair `g_ij` rows (B0–B5 + PROPOSED) |
| 3.7 `R_hat` + inference | `run_repair_transfer.py --estimate-r-hat` | `*__rhat.json` | **Tab G9** (`R_hat`, CI, perm-p), `sigma_R`, **Tab G9-NOV** |
| 3.8 Axis X′ | `run_adversarial_oracle.py` | `axis_x_prime.json` | **Fig oracle / Axis X′** (P5 detectable+blind, NC-1/NC-2) |
| 3.9 gates | `eval_gates.py --g9 --g9-nov --require-v5` | `gates/*` | all gate verdicts + SI family `m'`/`alpha_1'`; routing §8 |

> Every cell of every table/figure above is `DATA_NEEDED`. No number in this
> packet is empirical; the only arithmetic is the §5 budget identity (labelled
> "formula evaluation, not evidence").

---

## 5. Compute budget — expected wall-clock + GPU-hours (v5 re-checked, not assumed)

> **Source:** `configs/compute/first_gate_budget.yaml` (component GPU-hours, 30%
> buffer, 48h wall-clock) and `configs/experiments/redesign_v5_ar_lead.yaml`
> `feasible_point` / `g9_power` (the budget + variance identities). These are
> **planning envelopes from the frozen configs** (formula/config arithmetic),
> **not measured runtimes** — actual hours are `DATA_NEEDED`, recorded at run time
> into the reproducibility ledger. The v4 feasible point is **RE-CHECKED against
> the new G9 forward surcharge and the new `sigma_R` variance model**, not assumed
> (REDESIGN_v5 §6.3).

### 5.1 First-gate component envelope (`first_gate_budget.yaml`)
| Component | GPU-hr (raw) | +30% buffer | Storage | Wall-clock |
| --- | ---: | ---: | ---: | ---: |
| Trace extraction | 8 | 10.4 | 300 GB raw | — |
| Screening interventions (`U_hat`) | 6 | 7.8 | — | — |
| Screening baselines | 8 | 10.4 | — | — |
| **v4 first-gate subtotal** | **22** | **28.6** | **300 GB (+30% ⇒ ≥390 GB free)** | **48 h** |
| **v5 repair-transfer forwards (G9)** | `DATA_NEEDED` | `DATA_NEEDED` | re-budget | — |
| Local sync | — | — | `<2 GB` compact | — |

> The v5 repair-transfer line is **`DATA_NEEDED`**: it is governed by the
> re-checked forward surcharge below and is **not** assumed equal to the v4
> intervention line. The Axis X′ oracle is **CPU-only fixtures** (no lead forwards),
> as are binning-as-code (§3.2b), matched-null pool construction (§3.5), and all of
> §3.7 inference.

### 5.2 Budget identity (`redesign_v5_ar_lead.yaml`) — the binding ceiling, re-checked
The single arithmetic everything must satisfy:
`cells * n * forwards_per_example * c_fwd <= budget_gpu_hr`,
with frozen `n = 850`, `budget_gpu_hr = 14` (intervention+baseline line). The v4
screening surcharge `forwards_per_example = 18` (1 targeted + 1 no_op + `R_int`=16).

**v5 re-check (§6.3).** The G9 surcharge per target is **a localized-repair
forward + `R_null` matched-null-repair forwards + `R_int` repair-op repeats**:
`forwards_per_example_v5 = 1 + R_null + R_int` (per repair-panel selector). Because
`R_null` is **`DATA_NEEDED`** (pinned at lock to the `sigma_MC` variance floor), the
total v5 surcharge and the realized `c_fwd` ceiling are **`DATA_NEEDED`**; the
`decision_order` (request budget line → reduce cells → reduce `R_null`/`R_int` to
the variance-floored minimum) applies and is recorded.

| Design | c_fwd ceiling (screening, exact) | locked screening ceiling | v5 repair surcharge |
| --- | --- | --- | --- |
| **2 cells (pre-registered feasible)** | `14/(2·850·18)=4.5752e-4` | `4.57e-4` (~1.65 s/fwd) | re-checked at `1 + R_null + R_int`; `DATA_NEEDED` |
| 4 cells (full grid) | `14/(4·850·18)=2.2876e-4` | `2.28e-4` | re-checked; `DATA_NEEDED` |

- **Screening MDE** (`U_hat`): `n >= (z·sigma_hi/0.03)^2 · infl =
  (2.734·0.30/0.03)^2 · 1.125 = 747.5 · 1.125 = 841` ⇒ `n = 850`/cell.
- **Screening attenuation**: `kappa = 0.92` ⇒ `2·kappa-1 = 0.84` ⇒
  `U_target = 0.05/0.84 = 0.0595 <= 0.08`.
- **G9 power model (`R_power`, NOT reused from screening; Eq. R-VAR/R-POWER):**
  `Var(R_hat) ≈ (zeta_10/n_source + zeta_01/n_target) + (1/N_pair)·(sigma_MC^2/R_null + sigma_op^2/R_int)`
  — the ordered-kernel **two-projection** variance (findings 4, 10), NOT the
  superseded symmetric `(4/n_eff)·zeta_1` shorthand;
  `R_power := ceil( (z·sigma_R_hi/m_R)^2 · D_eff )` via `nuisance.r_power_repair`,
  with `m_R := m_R0/(2·kappa_lo^repair − 1)` (Eq. m-R). `zeta_10`, `zeta_01`,
  `n_source`, `n_target`, `n_eff`,
  `sigma_MC`, `sigma_op`, `D_eff`, `m_R0`, `kappa_lo^repair` are all
  **`DATA_NEEDED`**, estimated on `V_sel`/`V_inf` at lock. **No number fabricated.**
- **The `5.4e-3` figure is NON-BINDING** (a forward-timing scale note, not a
  feasibility ceiling): `2·850·18·5.4e-3 = 165.2` GPU-hr ≫ 14. Do not use it as a
  ceiling.

### 5.3 GPU-hour estimate for the full v5 matrix
> Per-forward cost `c_fwd` and the repair surcharge are **`DATA_NEEDED`** (measured
> in §3.2 calibration; `R_null` pinned at lock). The numbers below are the frozen
> budget *ceilings* the design pre-commits to, not a measured total.

- **2-cell feasible point:** screening intervention+baseline line ≤ **14 GPU-hr** at
  `c_fwd <= 4.57e-4`; trace-extraction envelope ≤ 10.4 GPU-hr buffered; **v5
  repair-transfer line `DATA_NEEDED`** (re-checked against `1+R_null+R_int`). If the
  recomputed total exceeds the v4 point, follow `decision_order`.
- **4-cell full grid:** gated on the recomputed `c_fwd` + surcharge clearing
  `2.28e-4`; else run the 2-cell subset and request a budget increase.
- **Per-arm / per-job split:** `DATA_NEEDED` — fill from §3.2:
  `per_job_gpu_hr ≈ n_job · forwards_per_example_v5 · c_fwd`.

> **Caption for the budget table in the paper:** *"Pre-registered compute envelope
> from `configs/compute/first_gate_budget.yaml` and the `redesign_v5_ar_lead.yaml`
> budget + `R_power` identities. The v5 repair-transfer forward surcharge
> (`1+R_null+R_int`), measured `c_fwd`, `R_power`, realized GPU-hours, and realized
> wall-clock are reported after the authorized run (`DATA_NEEDED`)."*

---

## 6. Resume procedure

Resumability is a hard preflight (`output_dir_empty_or_resumable`) and the queue
manifest (`experiments/queue_manifest.yaml`) is the durable state file. Every v5
stage (extraction, binning, nuisance, operator-freeze, screening, repair-transfer,
Axis X′, scoring, gates) is **idempotent and checkpoint/restartable**.

### 6.1 State model
- Each job has a state in the manifest: `pending -> running -> done` (or
  `failed` / `oom_retry` / `blocked`).
- Every script writes a per-job `STATUS.json` under its `--output` dir
  (`{state, started_at, finished_at, output_hash, row_count, git_commit,
  stage, depends_on}`) and appends to `${OUT}/${RUN_ID}/logs/queue.log`.
- A job is **complete** iff its `STATUS.json.state == done` AND its `output_hash`
  matches the recorded row-count check (reproducibility_ledger: "output hashes and
  row counts").

### 6.2 Resume after interruption / preemption
1. **Do not wipe `${OUT}`.** Re-run preflight §1.3 (`output_dir_empty_or_resumable`
   now takes the *resumable* branch).
2. Reconcile manifest ↔ disk:
   ```bash
   python scripts/eval_gates.py --reconcile-queue \
     --manifest experiments/queue_manifest.yaml \
     --out ${OUT}/${RUN_ID}
   # marks done/failed/partial per STATUS.json; emits the remaining pending set,
   # respecting v5 dependency order (extract -> binning -> nuisance -> operator_freeze
   # -> screening / matched_null_pool -> repair_transfer -> r_hat_inference -> gates).
   ```
3. Re-issue each command from §3 with `--resume`; jobs whose `STATUS.json.state
   == done` are skipped. Partial trace / intervention / repair dirs resume from the
   last checkpointed seed / example / target pair. CPU aggregation stages (binning,
   matched-null pool, `--estimate-r-hat`, Axis X′) re-run cheaply and idempotently
   from their inputs.
4. **Hash re-pin guard:** on resume, re-verify `${*_REV}`, `${*_SPLIT_HASH}`,
   `${PROMPT_TMPL_HASH}`, `${EVAL_HASH}`, `${REPAIR_EVAL_HASH}`, `${TAXONOMY_HASH}`
   are byte-identical to the locked values. A hash mismatch on resume **invalidates**
   the affected cell (re-extract, do not mix). A change to the binning
   `selection_event` or the frozen `rho`/`policy_hash` after lock **invalidates the
   SI correction** (re-freeze on `V_sel`, do not silently re-select on test).

### 6.3 OOM retry policy (mirrors `queue_manifest.yaml.oom_retry`)
- On CUDA out-of-memory: retry up to **2** times with the manifest's degradation
  ladder (1) halve `intervention_batch_size` / `repair_batch_size`; (2) if still
  out-of-memory, set `gradient_free_microbatch=1` / enable activation offload.
- `R_int = 16`, `n`, the frozen `R_null`, and the seeds are **frozen** — never
  reduce them to fit memory (that changes the pre-registered design / variance
  floor). Only batch/microbatch knobs may move.
- After **2** failed retries: mark job `failed`, leave it in the ledger
  (statistical_analysis_plan "Failure Handling"), and continue the queue. Do **not**
  silently drop.

### 6.4 Stop-condition halt vs resume
If a §1.4 stop condition fires (screening causal margin `<0.05`, utility drop
`>0.02`, repair utility drop `>0.02`, positivity `>=0.5`, leakage audit fail on
either evaluator, storage exceeded, invalid-rate `>5%`, or the **Axis X′-blind R4
unsound** condition): the queue **halts** and the run is routed per the gate's
`failure_action` / REDESIGN_v5 §8 (R1–R4). Resuming after a stop-condition halt
requires a fresh iterate-or-pivot review (`experiment_protocol.md` "Early Kill
Gates"), not a bare `--resume`.

---

## 7. What this packet does NOT do
- It authorizes nothing. `server.authorized: false` stays false in all three
  guarded configs.
- It compiles no LaTeX and reports no numbers — all results are `DATA_NEEDED`.
- It assigns no server. The user must bind §0 and pass §1 (incl. the v5-aware ARIS
  re-review) first.
- It runs no experiment, model load, training, or GPU job locally or on a server.
  The `scripts/*.py` entrypoints exist but **default to dry-run**: they refuse to
  load any model/GPU unless BOTH `server.authorized: true` (in `--config`) and
  `--i-have-authorization` are set, neither of which holds in this packet.
- It claims no Stage-1 readiness: REDESIGN_v5 is `design_frozen_stage1_RR`; the
  v5 modules are implemented and unit-tested, but the executable-preregistration
  run remains gated on the §1 authorization flip that has not happened.

---

## 8. v6 verified-repo baselines — run approach + single-GPU schedule

> **v6 delta (REDESIGN_v6, ARIS v6).** The headline matched model is
> **Qwen2.5-7B-Instruct** (`ar_lead_qwen`, PRIMARY; Llama-3.1-8B-Instruct
> SECONDARY; 1.5B = smoke/contract only). The comparison locks a `>=8`
> OFFICIAL-REPO verified-repo suite. This section adds **how each baseline is run**
> (re-score to segment-level `Û` + identical repair pipeline) and the
> **single-RTX-4090 inference-only schedule**. All numbers `DATA_NEEDED`;
> `server.authorized: false`. Roster + repos: `BASELINES.md`,
> `configs/baselines/baseline_registry.yaml`.

### 8.1 Per-baseline run approach (all on Qwen2.5-7B-Instruct / TriviaQA + HotpotQA)

Each verified-repo baseline runs under the **identical** setup as PROPOSED
(matched model, datasets, splits, evaluator, seeds; §0/§1). The only per-baseline
difference is its **native signal** and a **declared, hashed segment adapter**.
Two steps for every baseline:

1. **Re-score to segment-level `Û`** (`ciu_scored: true`). The baseline's native
   output-/representation-/sampling-level signal is mapped to a per-segment score
   via its registry `segment_adaptation` (identical-across-baselines adapter
   family, `adapter_hash` pinned at authorization). This feeds the screening /
   detection tables (AUROC for DETECTION comparators; the segment-localization
   margin G1/G6/G7 for the LOCALIZATION controls).
2. **Identical repair pipeline (G9-NOV).** The baseline's selected span is fed
   through the **same** `repair_ops` pipeline PROPOSED uses (Variant C
   source-derived policy + frozen anchor map `T`), so `R̂(baseline)` is computed on
   the same footing as `R̂(PROPOSED)`. A strong baseline therefore *helps* itself —
   the fairness incentive is symmetric.

| Baseline | Venue | class | Native signal → segment `Û` re-score | Repair (G9-NOV) |
| --- | --- | --- | --- | --- |
| ReDeEP | ICLR 2025 | detection | decoupled external/parametric knowledge → localize to segment layers/heads | identical pipeline |
| RACE | AAAI 2026 | detection | answer-reasoning consistency → per reasoning segment | identical pipeline |
| LapEigvals | EMNLP 2025 | detection | Laplacian-eigenvalue attention features → segment attention submatrix | identical pipeline |
| HaloScope | NeurIPS 2024 | detection | unlabeled-gen membership subspace → segment hidden states | identical pipeline |
| Lookback-Lens | EMNLP 2024 | detection | attention context-vs-generation ratio → per segment | identical pipeline |
| MIND | ACL 2024 (F) | detection | unsupervised internal-state signal → segment hidden states | identical pipeline |
| INSIDE/EigenScore | ICLR 2024 | detection | internal-state covariance eigenscore → segment hidden states | identical pipeline |
| Semantic Entropy | Nature 2024 | detection | meaning-level sampling uncertainty → segment-conditioned resamples | identical pipeline |
| SelfCheckGPT | EMNLP 2023 | detection | NLI sampling consistency → segment-aligned resamples | identical pipeline |
| Captum IG | attribution control | **localization** | integrated-gradient token attributions → segment importance; **scored on G1/G6/G7 margin, not AUROC** | identical pipeline |
| Causal Mediation | NeurIPS 2020 | **localization** | indirect-effect over components → segment components; **scored on G1/G6/G7 margin, not AUROC** | identical pipeline |

- **DETECTION baselines** map onto the §3.4 screening stage (`run_intervention.py
  --stage screening`) for `Û`/AUROC, then onto the §3.6 repair-transfer stage as
  B2/B3-style selectors (`run_repair_transfer.py --stage repair_transfer`,
  `--selector ...`) for `R̂`.
- **LOCALIZATION controls** (Captum-IG, Causal Mediation) are the attribution
  neighbours the necessity certification competes with most directly: scored on the
  **segment-localization margin** (G1 necessity / G6-G7 leakage), not AUROC, and
  also fed through the identical repair pipeline for G9-NOV.
- **B1 = TraceDet-derived AR span adapter** (`repair_ops.tracedet_ar_span_adapter`)
  is **our own re-cast span selector** (not the TraceDet diffusion detector, which is
  cited-not-reproduced; see `BASELINES.md` §4). It runs exactly as a §3.6 selector.
- **Preflight (extends §1.2).** Every verified-repo baseline must have its
  `implementation_source: official_repo` commit pinned and license verified, and an
  audited `segment_adapter` hash, before any arm runs; the run is **blocked** while
  any baseline carries `pending_before_server_run`
  (`ciu.baseline_readiness`). A baseline without an audited adapter is AUROC-only and
  excluded from the `R̂` table.

### 8.2 Single-RTX-4090 inference-only schedule

The v6 hardware tier is a **single RTX 4090 (24 GB), inference-only** (no training).
Schedule:

1. **Recompute `c_fwd` at 7B FIRST** (before committing cell count). Run the §3.2
   timing calibration (`extract_traces.py --calibrate-cfwd ... --split v_sel`) on
   **Qwen2.5-7B-Instruct**; the v5 feasible point (`c_fwd <= 4.57e-4` at the 2-cell
   screening line) was sized at an earlier scale and **must be re-measured** at 7B.
   This is a timing measurement, not an experiment.
2. **Re-check the budget identity** `cells * n * forwards_per_example * c_fwd <=
   budget_gpu_hr` with the re-measured 7B `c_fwd` AND the v5 G9 repair surcharge
   `forwards_per_example_v5 = 1 + R_null + R_int` (per repair-panel selector). If
   the recomputed total exceeds budget, apply the v5 `decision_order` (request budget
   → reduce cells → reduce `R_null`/`R_int` to the variance-floored minimum).
3. **PRIMARY 2-cell first.** Run Qwen2.5-7B-Instruct × {TriviaQA, HotpotQA} (the
   pre-registered confirmatory point) first; the SECONDARY Llama-3.1-8B 4-cell grid
   is **gated** behind the re-checked `c_fwd` + surcharge clearing the 4-cell
   ceiling.
4. **Memory fit.** Qwen2.5-7B-Instruct in bf16/fp16 (≈14-15 GB weights) + the
   `patch`/`replay` activation cache fits a 24 GB 4090 at batch-1/small-batch. The
   §6.3 OOM ladder (halve batch → microbatch=1 → activation offload) applies;
   `R_int`, `n`, frozen `R_null`, and seeds are **never** reduced to fit memory —
   only batch/microbatch knobs move.
5. **Inference-only forwards.** Generation, hidden-state reads, `patch`/`replay`,
   Captum-IG backward attributions (frozen model, no weight update), and Causal
   Mediation restore-and-read forwards are all inference-time; no gradient step
   updates any weight.
6. **Method-iteration loop budget (REDESIGN_v6 §5).** If G1/G9-NOV do not clear at
   7B, the capped method-iteration loop re-runs within this single-GPU envelope (cap
   = `method_iteration_cap`, `DATA_NEEDED`, pinned at authorization); each iteration
   re-freezes operators on `V_sel` and is paid for in the SI `K_op` factor. Never
   alter data / splits / evaluator / seeds / baselines to pass; never fabricate.

All schedule numbers (`c_fwd` at 7B, `forwards_per_example`, GPU-hours, wall-clock,
`method_iteration_cap`) are **`DATA_NEEDED`**, measured/pinned at the authorized run.
`server.authorized: false`; this section authorizes no run.
