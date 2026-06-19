# -*- coding: utf-8 -*-
"""Generator for the v5 resumable queue manifest (BUILD-NOW / RUN-LATER).

Emits experiments/queue_manifest.yaml: every job state: pending; server.authorized:
false; dependency-ordered; idempotent/checkpoint-restartable. ZERO fabricated
numbers (R_null and the v5 forward surcharge are DATA_NEEDED).
"""
from io import StringIO

FAMILIES = [
    ("ar_lead_qwen", "qwen2.5", "Qwen/Qwen2.5-7B-Instruct"),
    ("ar_lead_llama", "llama3.1", "meta-llama/Llama-3.1-8B-Instruct"),
]
DATASETS = ["triviaqa", "hotpotqa"]
SEEDS = list(range(20))
SCREEN_METHODS = [
    "ciu_selector", "random_segment", "output_entropy", "semantic_entropy",
    "output_signature_detector", "reasoning_consistency_detector",
    "selfcheckgpt", "inside_detector",
]
REPAIR_SELECTORS = ["B0", "B1", "B2", "B3", "B4", "B5", "PROPOSED"]


def cell_tier(fam):
    return "feasible_2cell" if fam == "ar_lead_qwen" else "full_grid_only"


HEADER = """# TraceCausal - RESUMABLE QUEUE MANIFEST v5 (BUILD-NOW / RUN-LATER)
# =============================================================================
# DO-NOT-RUN TEMPLATE. Every job is `state: pending`; `server.authorized: false`.
# This manifest enumerates the FULL v5 pipeline (trace extraction, binning-as-code,
# nuisance, operator-selection freeze, matched-null pool, screening U_hat arms,
# repair-transfer R_hat arms B0..B5+PROPOSED, R_hat dependent-pair inference, the
# adversarial oracle Axis X', detection + repair scoring, and the G1..G8 + G9/G9-NOV
# gate evaluation). It is the durable resume state file referenced by
# reports/run_packet.md section 6. Every cell is idempotent + checkpoint/restartable
# and dependency-ordered. It authorizes nothing.
#
# Source of truth (frozen):
#   configs/experiments/redesign_v5_ar_lead.yaml   (v5 lead plan: repair-transfer, SI, Axis X')
#   configs/experiments/redesign_v4_ar_lead.yaml   (preserved CIU/matched-null/G1-G8)
#   configs/experiments/first_gate.yaml            (stop conditions, negative controls)
#   configs/baselines/baseline_registry.yaml       (AR-lead + repair-panel selectors)
#   configs/compute/first_gate_budget.yaml         (GPU-hr / storage / wall-clock)
#   configs/seeds/paper_20.txt                     (seeds 0..19)
#   docs/redesign/REDESIGN_v5.md                   (G9 certification, analysis plan)
#   docs/redesign/REDESIGN_v4.md                   (screening analysis plan, budget identity)
#
# ZERO fabricated numbers: R_null and the v5 forward surcharge are DATA_NEEDED,
# pinned at lock to the sigma_MC variance floor (REDESIGN_v5 section 6.3).
# =============================================================================

project: tracecausal
run_tier: paper_candidate
manifest_version: v5_arlead
redesign: v5
config: configs/experiments/redesign_v5_ar_lead.yaml

server:
  authorized: false        # HARD GATE - no job may run while this is false
  reason: >-
    design_frozen_stage1_RR only; Stage-1 readiness not claimed; v5 ARIS re-review
    (G9/G9-NOV/Axis X') not complete; no server assigned; busy server off-limits.

# --- run binding: filled by the user AT AUTHORIZATION (run_packet.md section 0) --
authorization_placeholders:
  SERVER: PENDING_NO_SERVER_ASSIGNED      # none assigned; busy server off-limits
  GPU: PENDING
  CONDA_ENV: PENDING
  REPO: PENDING
  OUT: PENDING
  RUN_ID: PENDING
  qwen_revision: PENDING_PIN_AT_AUTHORIZATION       # Qwen/Qwen2.5-7B-Instruct
  llama_revision: PENDING_PIN_AT_AUTHORIZATION      # meta-llama/Llama-3.1-8B-Instruct
  triviaqa_split_hash: PENDING_PIN_AT_AUTHORIZATION # 3-way V_sel/V_inf/test (SI-1)
  hotpotqa_split_hash: PENDING_PIN_AT_AUTHORIZATION # 3-way V_sel/V_inf/test (SI-1)
  prompt_template_hash: PENDING_PIN_AT_AUTHORIZATION
  evaluator_hash: PENDING_PIN_AT_AUTHORIZATION          # necessity U_hat evaluator
  repair_evaluator_hash: PENDING_PIN_AT_AUTHORIZATION   # v5 cross-example target-label Y_j (kappa^repair)
  taxonomy_hash: PENDING_PIN_AT_AUTHORIZATION           # frozen G3 class partition (A6)

# --- design dimensions (frozen) ----------------------------------------------
dimensions:
  families:                 # 2 AR-LLM lead families (diffusion excluded from AR lead)
    - {id: ar_lead_qwen, family: qwen2.5, checkpoint: "Qwen/Qwen2.5-7B-Instruct"}
    - {id: ar_lead_llama, family: llama3.1, checkpoint: "meta-llama/Llama-3.1-8B-Instruct"}
  datasets: [triviaqa, hotpotqa]            # lead datasets
  screening_methods:        # U_hat: proposed selector + 7 AR-lead detectors (ciu_scored)
    - ciu_selector
    - random_segment
    - output_entropy
    - semantic_entropy
    - output_signature_detector
    - reasoning_consistency_detector
    - selfcheckgpt
    - inside_detector
  repair_panel:             # R_hat: B0..B5 + PROPOSED, all through identical repair_ops (section 4.3)
    - {id: B0, kind: no_op_floor}
    - {id: B1, kind: tracedet_ar_span_adapter}     # repair_ops.tracedet_ar_span_adapter (refinement 3)
    - {id: B2, kind: entropy_perplexity_peak}
    - {id: B3, kind: latent_probe}
    - {id: B4, kind: random_same_class}            # matched-null repair control inside g_ij (A9)
    - {id: B5, kind: oracle_selected}              # planted GT; oracle fixtures only; upper bound
    - {id: PROPOSED, kind: ciu_selector}
  excluded_from_ar_lead:    # diffusion transfer study only - NOT in these arms
    - diffusion_trace_detector
    - tdgnet_trace_detector
  screening_negative_controls:    # necessity arms (U_hat)
    - random_non_causal_segment
    - shuffled_trace_segment
    - no_op_intervention
  repair_negative_controls:       # repair arms (R_hat); REDESIGN_v5 section 7.4
    - NC-1_collinear_confounder    # Axis X'-blind: controls silent, R_hat MUST collapse (tests A8)
    - NC-2_source_swap             # g_ij invariant under in-class source swap (tests A5/A6)
  seeds: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]   # paper_20.txt

# --- v5 transport / estimand (REDESIGN_v5 section 4) -------------------------
repair_transfer:
  transport_variant: C                       # source-derived repair POLICY (not a state)
  estimand: within_class_source_neq_target_u_statistic   # Eq. R
  variance_primary: two_way_source_target_cluster_bootstrap   # MF-4 CONFIRMATORY, >=10,000 reps
  null_test: class_block_source_block_signflip_diagnostic     # G9-FIX: DIAGNOSTIC, not confirmatory; probes A6 sign-symmetry (NOT exact for R=0)
  variance_crosscheck: hajek_projection_zeta10_over_nsource_plus_zeta01_over_ntarget  # both ordered-kernel projections (findings 4, 10)
  nested_matched_null_mc: target_clustered_covariance
  positivity_excluded_max_per_class: 0.5     # A7
  max_repair_utility_drop: 0.02              # G9 bounded repair utility cost (D_util^repair)
  r_null: DATA_NEEDED                        # matched-null repair draws; pin at lock to sigma_MC floor
  r_int: 16                                  # repair-op repeats (frozen)
  kernels:                                   # implemented + unit-tested pure-Python analysis kernels
    repair_gain: tracecausal.repair_transfer.repair_gain                 # Eq. g-ij
    r_hat: tracecausal.repair_transfer.r_hat                             # Eq. R
    two_way_cluster_bootstrap: tracecausal.repair_transfer.two_way_cluster_bootstrap
    class_block_permutation: tracecausal.repair_transfer.class_block_permutation
    hajek_projection_var: tracecausal.repair_transfer.hajek_projection_var
    g9_nov_margin: tracecausal.repair_transfer.g9_nov_margin_simultaneous
    transport: tracecausal.repair_ops.transport                         # anchor map T (Variant C)
    select_operator: tracecausal.repair_ops.select_operator             # OS-1 freeze
    tracedet_ar_span_adapter: tracecausal.repair_ops.tracedet_ar_span_adapter  # B1

# --- v5 selective inference (REDESIGN_v5 section 6) --------------------------
selective_inference:
  primary: SI-1_selection_split              # V_sel / V_inf / test disjoint
  fallback: SI-2_bonferroni_over_k_bin_times_k_op
  path_rule: SI1_if_n_val_floors_met_on_both_splits_else_SI2
  holm_alpha_identity: "alpha_1' = 0.05 / (m' * K_bin * K_op)"   # Eq. SI-ALPHA
  k_bin_ladder: [1, 2, 4, 8, 16]             # Delta_pos coarsening (finest first); bounds K_bin
  displaced_mass_edges: [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
  k_op_grid: {op: [patch, replay], alpha: [0.1, 0.25, 0.5, 0.75, 1.0], ref_type: [factual, neutral]}
  operator_freeze_on: V_sel                  # OS-1
  kernels:
    holm_alpha: tracecausal.selective_inference.holm_alpha
    validate_selection_split: tracecausal.selective_inference.validate_selection_split
    choose_si_path: tracecausal.selective_inference.choose_si_path
    select_binning: tracecausal.binning_selection.select_binning   # selection-as-code (section 6.4)

# --- v5 adversarial oracle Axis X' (REDESIGN_v5 section 7) -------------------
adversarial_oracle:
  axis: x_prime
  regimes: [detectable, blind]
  xi_grid: [0.0, 0.25, 0.5, 0.75, 1.0]       # xi=0 reproduces the v4 clean oracle
  p5_detectable: controls_trip_before_r_hat_certifies
  p5_blind: controls_silent_and_r_hat_collapses_else_R4_unsound
  negative_controls: [NC-1_collinear_confounder, NC-2_source_swap]
  kernels:
    axis_x_confounded: tracecausal.adversarial_oracle.axis_x_confounded
    negative_control_collinear: tracecausal.adversarial_oracle.negative_control_collinear
    source_swap: tracecausal.adversarial_oracle.source_swap

# --- v5 G9 power model (REDESIGN_v5 section 4.6; Eq. R-VAR / R-POWER) --------
# Formula identity only; all inputs DATA_NEEDED, estimated on V_sel/V_inf at lock.
g9_power:
  # Ordered/asymmetric-kernel two-projection variance (findings 4, 10); matches
  # tracecausal.nuisance.estimate_sigma_r and repair_transfer.hajek_projection_var.
  # The symmetric (4/n_eff)*zeta_1 shorthand collapsed both margins and is NOT used.
  sigma_r_decomposition: "Var(R_hat) = zeta_10/n_source + zeta_01/n_target + (1/N_pair)*(sigma_MC^2/R_null + sigma_op^2/R_int)"
  r_power_identity: "R_power = ceil( (z * sigma_R_hi / m_R)^2 * D_eff )"
  m_r_identity: "m_R = m_R0 / (2*kappa_lo^repair - 1)"           # Eq. m-R
  estimate_sigma_r: tracecausal.nuisance.estimate_sigma_r
  r_power_repair: tracecausal.nuisance.r_power_repair            # NOT v4 r_power (MF-5)
  zeta_10: DATA_NEEDED                                           # source projection var
  zeta_01: DATA_NEEDED                                           # target-margin projection var
  n_source: DATA_NEEDED
  n_target: DATA_NEEDED
  n_eff: DATA_NEEDED                                             # = min(n_source, n_target), bookkeeping only
  sigma_mc: DATA_NEEDED
  sigma_op: DATA_NEEDED
  d_eff: DATA_NEEDED
  m_r0: DATA_NEEDED
  kappa_lo_repair: DATA_NEEDED
  r_power: DATA_NEEDED
  forward_surcharge_per_example: DATA_NEEDED   # 1 localized-repair + R_null null-repair + R_int repeats

# --- cell-tier selection (redesign_v5_ar_lead.yaml decision_order, RE-CHECKED) -
# Default at lock = 2-cell feasible point (hold ar_lead_qwen across both datasets).
# The 4-cell full grid (jobs tagged cell_tier: full_grid_only) runs ONLY if the
# RE-CHECKED c_fwd AND the v5 forward surcharge (1+R_null+R_int) clear the ceiling;
# else run feasible_2cell and request a budget increase. c_fwd from section 3.2.
cell_tiers:
  feasible_2cell:
    cells: 2
    family: ar_lead_qwen
    datasets: [triviaqa, hotpotqa]
    c_fwd_max_screening: 4.57e-4   # 2-cell screening budget-binding ceiling (~1.65 s/fwd)
    c_fwd_max_repair: DATA_NEEDED  # re-checked at forwards_per_example_v5 = 1+R_null+R_int
  full_grid_only:
    cells: 4
    families: [ar_lead_qwen, ar_lead_llama]
    datasets: [triviaqa, hotpotqa]
    c_fwd_max_screening: 2.28e-4   # 4-cell screening ceiling
    c_fwd_max_repair: DATA_NEEDED  # full grid gated on re-checked c_fwd + surcharge <= this

# --- frozen compute envelope (configs/compute/first_gate_budget.yaml + v5 section 5) -
# Planning envelope from frozen configs (config arithmetic, NOT measured).
# Realized GPU-hr / wall-clock / c_fwd / repair surcharge are DATA_NEEDED.
compute_budget:
  first_gate_component_gpu_hr: {trace_extraction: 8, screening_intervention: 6, screening_baseline: 8}
  repair_transfer_gpu_hr: DATA_NEEDED        # v5 line; NOT assumed equal to v4 intervention line
  buffer_percent: 30
  v4_first_gate_subtotal_gpu_hr_buffered: 28.6     # 22 * 1.30 (screening only)
  storage_gb_raw: 300
  storage_gb_free_required: 390              # 300 + 30%; re-budget for repair forwards at lock
  wall_clock_hours: 48
  budget_identity: "cells * n * forwards_per_example * c_fwd <= budget_gpu_hr"
  n: 850
  forwards_per_example_screening: 18         # 1 targeted + 1 no_op + R_int(16)
  forwards_per_example_repair: DATA_NEEDED   # 1 localized-repair + R_null + R_int (per panel selector)
  budget_gpu_hr: 14                          # screening intervention+baseline line
  c_fwd_max_2cell: 4.57e-4
  c_fwd_max_4cell: 2.28e-4
  c_fwd_legacy_nonbinding: 5.4e-3            # NOT a feasibility ceiling (165.2 GPU-hr); scale note only
  c_fwd_measured: DATA_NEEDED
  realized_gpu_hr: DATA_NEEDED
  realized_wall_clock_h: DATA_NEEDED

# --- pipeline preconditions / dependency DAG (jobs are NOT independent) -------
# Stage order per cell; every downstream stage lists its upstream dependency by id.
pipeline_order:
  - extract_traces            # per cell x seed -> trace manifest
  - calibrate_c_fwd           # validation-only timing (not an experiment) -> cell-tier decision
  - select_binning            # SI-1 split validate + binning-as-code (V_sel); emits K_bin (CPU)
  - estimate_nuisance         # V_inf only: sigma_u(n>=200), kappa(n>=300), kappa^repair, m_pool (CPU)
  - operator_freeze           # OS-1 freeze rho + w_c on V_sel; emits K_op
  - build_matched_null_pool   # Pi_j per target (A9 / B4 control) (CPU)
  - run_screening             # U_hat arms (per detector x seed)
  - run_repair_transfer       # g_ij arms (per panel selector x seed)
  - estimate_r_hat            # within-class U-statistic + two-way cluster bootstrap + perm (CPU)
  - run_adversarial_oracle    # Axis X' xi-sweep + NC-1/NC-2 (CPU fixtures; once, global)
  - score_detection           # auroc/auprc/fpr_at_95_tpr (screening) + R_hat tables (repair)
  - eval_gates                # G1..G8 (screening) + G9/G9-NOV + SI Holm family (require_v5)

# --- global precondition gate (ALL must hold before ANY job leaves pending) ----
global_preconditions:
  - server_authorized_true              # currently FALSE - blocks everything
  - aris_v5_review_ge_8_no_hard_violation   # re-reviewed for G9/G9-NOV/Axis X'
  - v5_kernel_harness_green             # pytest tests/test_ciu_nulldata.py green (no GPU)
  - all_hashes_pinned                   # model revs, 3-way split hashes, prompt+both evaluator+taxonomy hashes
  - selection_split_disjoint_v_sel_v_inf_test   # SI-1 (validate_selection_split)
  - leakage_check_pass                  # both EVAL_HASH and REPAIR_EVAL_HASH
  - baseline_readiness_resolved         # incl. B1/B2/B3 repair-panel selectors through identical repair_ops
  - gpu_available
  - output_dir_empty_or_resumable
  - disk_free_gb_at_least_390
  - compact_sync_path_ready
  - no_active_server_process_overwritten

# --- OOM retry policy (run_packet.md section 6.3) ----------------------------
# CUDA out-of-memory handling. R_int, n, R_null, and seeds are FROZEN - never
# reduced to fit memory (that would change the pre-registered design / variance
# floor). Only batch/microbatch knobs may move. Field uses the literal o-o-m token.
oom_retry_policy:
  max_retries: 2
  ladder:
    - halve_intervention_or_repair_batch_size
    - microbatch_1_with_activation_offload
  never_reduce: [r_int, n, r_null, seeds]   # frozen design knobs / variance floor
  after_max_retries: mark_failed_keep_in_ledger_continue_queue   # never silently drop

# --- stop conditions (first_gate.yaml.gates + v5 G9 routing section 8) --------
# Firing any of these HALTS the queue (not just logs) and applies failure_action.
stop_conditions:
  - {metric: causal_margin_abs, threshold: 0.05, comparator: below, action: stop_causal_claim_screening}
  - {metric: utility_drop_abs, threshold: 0.02, comparator: above, action: downgrade_to_diagnosis}
  - {metric: repair_utility_drop_abs, threshold: 0.02, comparator: above, action: g9_reframe_as_abstention}
  - {metric: positivity_excluded_frac_per_class, threshold: 0.5, comparator: at_or_above, action: route_insufficient_positivity}
  - {metric: evaluator_leakage, threshold: pass, comparator: not_equal, action: invalidate_run}
  - {metric: repair_evaluator_leakage, threshold: pass, comparator: not_equal, action: invalidate_run}
  - {metric: invalid_intervention_or_repair_rate, threshold: 0.05, comparator: above, action: diagnostic_only}
  - {metric: storage_budget, threshold: exceeded, comparator: equal, action: stop}
  - {metric: axis_x_prime_blind_soundness, threshold: controls_silent_and_r_hat_certified, comparator: equal, action: R4_certification_withdrawn}

# --- Stage-2 decision routing (hardened, REDESIGN_v5 section 8) ---------------
stage2_routing:
  R1_full_certification: g9_and_g9_nov_and_p5_and_p6_pass        # headline causal claim (scoped A5-A9)
  R2_necessity_plus_protocol: g9_fails_but_protocol_validated_and_B1_B3_lose
  R3_necessity_only_gated: only_if_g9_null_powered_AND_theory_informative   # not auto-publishable
  R4_unsound: p5_fails_in_x_prime_blind_certification_withdrawn

# --- state machine + resume (run_packet.md section 6) ------------------------
state_model:
  states: [pending, running, done, failed, oom_retry, blocked]
  durable_state_file: this_manifest_plus_per_job_STATUS_json
  done_criteria: "STATUS.json.state == done AND output_hash matches row_count check"
  idempotent: true                          # every stage re-runs cleanly from its inputs
  reconcile_command: "scripts/eval_gates.py --reconcile-queue --manifest experiments/queue_manifest.yaml --out ${OUT}/${RUN_ID}"
  resume_command_suffix: "--resume"         # done jobs skipped; partials resume from last checkpoint
  hash_repin_guard: >-
    re-verify all pinned hashes byte-identical on resume; a model/split/eval/taxonomy
    mismatch invalidates the cell; a binning selection_event or frozen rho/policy_hash
    change after lock invalidates the SI correction (re-freeze on V_sel, never on test).

# =============================================================================
# JOBS - all `state: pending`. Dependency-ordered v5 pipeline. Counts at end.
# Run cell_tier: feasible_2cell first; gate full_grid_only behind the re-checked
# c_fwd + repair surcharge decision (decision_order).
# =============================================================================
jobs:
"""


def main():
    out = StringIO()
    out.write(HEADER)

    state = {"job_id": 0}
    counts = {}

    def emit(stage, fields, deps):
        state["job_id"] += 1
        jid = "job_%04d" % state["job_id"]
        counts[stage] = counts.get(stage, 0) + 1
        out.write("  - id: %s\n" % jid)
        out.write("    stage: %s\n" % stage)
        for k, v in fields.items():
            out.write("    %s: %s\n" % (k, v))
        out.write("    state: pending\n")
        out.write("    depends_on: [%s]\n" % ", ".join(deps))
        out.write("    idempotent: true\n")
        out.write("    oom_retry: {max_retries: 2, ladder: [halve_batch, microbatch_1_offload], frozen: [r_int, n, r_null, seeds]}\n")
        out.write("    on_fail: keep_in_ledger_continue_queue\n")

    # Stage A: trace extraction
    out.write("  # --- Stage A: trace extraction (per cell x seed) ---------------------------\n")
    for fam, _f, _c in FAMILIES:
        for ds in DATASETS:
            for s in SEEDS:
                emit("extract_traces",
                     {"arm": "extract__%s__%s" % (fam, ds),
                      "family": fam, "dataset": ds, "seed": s,
                      "cell_tier": cell_tier(fam),
                      "output_artifact": "traces/%s__%s__seed%d/trace_manifest.json" % (fam, ds, s)},
                     ["server_authorized_true", "hashes_pinned", "preflight_pass"])

    # Stage B: c_fwd calibration
    out.write("  # --- Stage B: c_fwd timing calibration (per cell, V_sel; not an experiment) -\n")
    for fam, _f, _c in FAMILIES:
        for ds in DATASETS:
            emit("calibrate_c_fwd",
                 {"arm": "cfwd__%s__%s" % (fam, ds),
                  "family": fam, "dataset": ds, "split": "v_sel",
                  "cell_tier": cell_tier(fam),
                  "output_artifact": "provenance/cfwd_%s__%s.json" % (fam, ds)},
                 ["extract__%s__%s_all_seeds" % (fam, ds)])

    # Stage C: select_binning
    out.write("  # --- Stage C: SI-1 split validate + binning-as-code (per cell, V_sel; CPU) -\n")
    for fam, _f, _c in FAMILIES:
        for ds in DATASETS:
            emit("select_binning",
                 {"arm": "binning__%s__%s" % (fam, ds),
                  "family": fam, "dataset": ds, "split": "v_sel",
                  "cell_tier": cell_tier(fam),
                  "kernel": "tracecausal.binning_selection.select_binning",
                  "emits": "K_bin_and_selection_event",
                  "output_artifact": "binning/%s__%s.json" % (fam, ds)},
                 ["extract__%s__%s_all_seeds" % (fam, ds)])

    # Stage D: estimate_nuisance
    out.write("  # --- Stage D: nuisance (per cell, V_inf; sigma_u/kappa/kappa^repair/m_pool) -\n")
    for fam, _f, _c in FAMILIES:
        for ds in DATASETS:
            emit("estimate_nuisance",
                 {"arm": "nuisance__%s__%s" % (fam, ds),
                  "family": fam, "dataset": ds, "split": "v_inf",
                  "cell_tier": cell_tier(fam),
                  "n_val_sigma": 200, "n_val_kappa": 300, "proximity_pool_min": 8,
                  "kernels": "estimate_sigma_u, estimate_kappa, kappa_repair, pool_inflation",
                  "output_artifact": "nuisance/%s__%s.json" % (fam, ds)},
                 ["binning__%s__%s" % (fam, ds)])

    # Stage E: operator_freeze
    out.write("  # --- Stage E: operator-selection freeze OS-1 (per cell, V_sel; emits K_op) --\n")
    for fam, _f, _c in FAMILIES:
        for ds in DATASETS:
            emit("operator_freeze",
                 {"arm": "operator_freeze__%s__%s" % (fam, ds),
                  "family": fam, "dataset": ds, "split": "v_sel",
                  "cell_tier": cell_tier(fam),
                  "kernel": "tracecausal.repair_ops.select_operator",
                  "select_objective": "r_hat_proposed_minus_B4",
                  "emits": "frozen_rho_policy_hash_K_op_w_c",
                  "output_artifact": "operator_freeze/%s__%s.json" % (fam, ds)},
                 ["nuisance__%s__%s" % (fam, ds)])

    # Stage F: build_matched_null_pool
    out.write("  # --- Stage F: matched-null pool Pi_j (per cell; A9 / B4 control; CPU) -------\n")
    for fam, _f, _c in FAMILIES:
        for ds in DATASETS:
            emit("build_matched_null_pool",
                 {"arm": "nullpool__%s__%s" % (fam, ds),
                  "family": fam, "dataset": ds,
                  "cell_tier": cell_tier(fam),
                  "kernel": "tracecausal.nullpool",
                  "proximity_pool_min": 8,
                  "records": "positivity_A7_exclusions",
                  "output_artifact": "repair_transfer/%s__%s__nullpool.json" % (fam, ds)},
                 ["binning__%s__%s" % (fam, ds)])

    # Stage G: screening interventions
    out.write("  # --- Stage G: screening U_hat arms (per detector x cell x seed) ------------\n")
    for fam, _f, _c in FAMILIES:
        for ds in DATASETS:
            for m in SCREEN_METHODS:
                for s in SEEDS:
                    emit("run_screening",
                         {"arm": "%s__%s__%s" % (m, fam, ds),
                          "method": m, "family": fam, "dataset": ds, "seed": s,
                          "cell_tier": cell_tier(fam), "operator": "mask", "r_int": 16,
                          "negative_controls": "[random_non_causal_segment, shuffled_trace_segment, no_op_intervention]",
                          "ciu_scored": "true",
                          "output_artifact": "interventions/%s__%s__%s__seed%d" % (m, fam, ds, s)},
                         ["extract__%s__%s__seed%d" % (fam, ds, s),
                          "operator_freeze__%s__%s" % (fam, ds)])

    # Stage H: repair-transfer interventions
    out.write("  # --- Stage H: repair-transfer g_ij arms (panel B0..B5+PROPOSED x cell x seed) -\n")
    for fam, _f, _c in FAMILIES:
        for ds in DATASETS:
            for sel in REPAIR_SELECTORS:
                for s in SEEDS:
                    emit("run_repair_transfer",
                         {"arm": "%s__%s__%s" % (sel, fam, ds),
                          "selector": sel, "family": fam, "dataset": ds, "seed": s,
                          "cell_tier": cell_tier(fam),
                          "transport_variant": "C", "r_null": "DATA_NEEDED", "r_int": 16,
                          "within_class_only": "true", "source_neq_target": "true",
                          "kernel": "tracecausal.repair_transfer.repair_gain",
                          "output_artifact": "repair_transfer/%s__%s__%s__seed%d" % (sel, fam, ds, s)},
                         ["nullpool__%s__%s" % (fam, ds),
                          "operator_freeze__%s__%s" % (fam, ds),
                          "extract__%s__%s__seed%d" % (fam, ds, s)])

    # Stage I: estimate_r_hat
    out.write("  # --- Stage I: R_hat U-statistic + two-way cluster bootstrap + perm (per cell; CPU) -\n")
    for fam, _f, _c in FAMILIES:
        for ds in DATASETS:
            emit("estimate_r_hat",
                 {"arm": "rhat__%s__%s" % (fam, ds),
                  "family": fam, "dataset": ds,
                  "cell_tier": cell_tier(fam),
                  "bootstrap": 10000, "permutations": 10000,
                  "kernels": "r_hat, two_way_cluster_bootstrap, class_block_permutation, hajek_projection_var, estimate_sigma_r",
                  "output_artifact": "repair_transfer/%s__%s__rhat.json" % (fam, ds)},
                 ["%s__%s__%s_all_seeds" % (sel, fam, ds) for sel in REPAIR_SELECTORS])

    # Stage J: adversarial oracle (global)
    out.write("  # --- Stage J: adversarial oracle Axis X' (global; CPU fixtures; no lead forwards) -\n")
    emit("run_adversarial_oracle",
         {"arm": "axis_x_prime",
          "axis": "x_prime", "regimes": "[detectable, blind]",
          "xi_grid": "[0.0, 0.25, 0.5, 0.75, 1.0]",
          "negative_controls": "[NC-1_collinear_confounder, NC-2_source_swap]",
          "kernel": "tracecausal.adversarial_oracle.axis_x_confounded",
          "p5": "detectable_controls_trip_before_r_hat; blind_controls_silent_r_hat_collapses_else_R4",
          "cell_tier": "global",
          "output_artifact": "adversarial_oracle/axis_x_prime.json"},
         ["v5_kernel_harness_green"])

    # Stage K: score_detection
    out.write("  # --- Stage K: detection + repair scoring (per cell) ------------------------\n")
    for fam, _f, _c in FAMILIES:
        for ds in DATASETS:
            emit("score_detection",
                 {"arm": "score__%s__%s" % (fam, ds),
                  "family": fam, "dataset": ds,
                  "cell_tier": cell_tier(fam),
                  "metrics": "[auroc, auprc, fpr_at_95_tpr, r_hat, repair_factuality_gain, repair_utility_delta]",
                  "output_artifact": "detection/%s__%s" % (fam, ds)},
                 ["%s__%s__%s_all_seeds" % (SCREEN_METHODS[0], fam, ds),
                  "rhat__%s__%s" % (fam, ds)])

    # Stage L: eval_gates (per cell)
    out.write("  # --- Stage L: gate evaluation G1..G8 + G9/G9-NOV + SI Holm family (per cell) -\n")
    for fam, _f, _c in FAMILIES:
        for ds in DATASETS:
            emit("eval_gates",
                 {"arm": "gates__%s__%s" % (fam, ds),
                  "family": fam, "dataset": ds,
                  "cell_tier": cell_tier(fam),
                  "screening_gates": "[g1_on_u_deflated, g2, g5prime, g6, g7_upper_ci, g8_sham_ood]",
                  "headline_gates": "[g9_repair_transfer, g9_novelty]",
                  "si_holm_family": "[g1, g2, g5prime, g6, g7, g8, g9_cells, g9_nov_B1, g9_nov_B2, g9_nov_B3]",
                  "si_path_rule": "SI1_if_floors_met_else_SI2",
                  "require_v5": "true",
                  "bootstrap": 10000, "permutations": 10000,
                  "output_artifact": "gates/%s__%s" % (fam, ds)},
                 ["score__%s__%s" % (fam, ds),
                  "axis_x_prime",
                  "binning__%s__%s" % (fam, ds),
                  "operator_freeze__%s__%s" % (fam, ds)])

    # Stage M: global rollup
    out.write("  # --- Stage M: global SI Holm rollup + Stage-2 routing (single; after all cells) -\n")
    emit("eval_gates_global",
         {"arm": "gates_global_rollup",
          "scope": "all_cells",
          "rollup": "holm_si_family_over_m_prime_with_k_bin_times_k_op",
          "kernel": "tracecausal.selective_inference.holm_alpha",
          "routing": "[R1_full, R2_necessity_plus_protocol, R3_necessity_only_gated, R4_unsound]",
          "validates": "validate_ciu_record(require_v5=True)_per_cell",
          "cell_tier": "global",
          "output_artifact": "gates/global_rollup.json"},
         ["gates__%s__%s" % (fam, ds) for fam, _f, _c in FAMILIES for ds in DATASETS])

    # footer
    out.write("\n# --- job counts (emitted by the generator; for resume/reconcile sanity) ----\n")
    out.write("job_counts:\n")
    out.write("  total: %d\n" % state["job_id"])
    order = ["extract_traces", "calibrate_c_fwd", "select_binning", "estimate_nuisance",
             "operator_freeze", "build_matched_null_pool", "run_screening",
             "run_repair_transfer", "estimate_r_hat", "run_adversarial_oracle",
             "score_detection", "eval_gates", "eval_gates_global"]
    for st in order:
        out.write("  %s: %d\n" % (st, counts.get(st, 0)))
    out.write("  by_cell_tier:\n")
    out.write("    note: feasible_2cell = PRIMARY (ar_lead_qwen cells); full_grid_only = SECONDARY (ar_lead_llama cells); global = cross-cell aggregation jobs\n")
    out.write("\n# --- final discipline marker -------------------------------------------------\n")
    out.write("authorizes_nothing: true\n")
    out.write("server_authorized: false   # re-affirmed at file end; no job runs while false\n")

    with open(r"D:/Research/tracecausal/experiments/queue_manifest.yaml",
              "w", encoding="utf-8", newline="\n") as fh:
        fh.write(out.getvalue())

    print("wrote queue_manifest.yaml; total jobs =", state["job_id"])
    print("counts:", counts)


if __name__ == "__main__":
    main()
