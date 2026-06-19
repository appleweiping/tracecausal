"""Null-data harness for the v5 cross-example repair-transfer core (REDESIGN_v5 §9).

DO NOT RUN ON THE SERVER. Pure-Python unit fixtures (no model, no GPU, no network,
no server run). They prove the v5 estimator and gates behave correctly on *planted*
synthetic data before any real run is authorized. The properties asserted are the
falsifiable design predictions of REDESIGN_v5:

* G9 **certifies** a transferable repair and **fails** a non-transferable one;
* the B4 matched-null repair (within-``g`` control) **also-passing** routes to
  ``diagnostic`` (repair is generic, not localized), never ``invalidated``;
* G9-NOV passes **only when** PROPOSED out-transfers B1-B3 (refinement 8 simultaneous
  CI for the data-selected max baseline);
* the two-way (source x target) cluster bootstrap and class-block permutation give
  consistent CIs, and the Hajek cross-check recovers the U-statistic variance using
  **both** zeta projections (refinement 5);
* common-support / positivity accounting restricts every arm to the same estimand
  (refinement 4) and the A7 positivity event is recorded (refinement 1/2);
* Axis X'-detectable controls trip **before** ``R_hat``; Axis X'-blind controls stay
  silent while ``R_hat -> 0`` as ``xi -> 1`` (P5; refinement 9);
* NC-2 source-swap invariance (A5/A6);
* SI-1/SI-2 path selection + ``K_bin``/``K_op`` Holm folding reproduce the feasible
  point (refinements 6/8 / §6).

Run later only when the user authorizes it: ``pytest tests/test_repair_transfer.py``.
``server.authorized`` stays false; nothing here loads a model.
"""

from __future__ import annotations

import math

from tracecausal.adversarial_oracle import (
    AXIS_X_XI_GRID,
    axis_x_confounded,
    negative_control_collinear,
    source_swap,
    structural_equations,
)
from tracecausal.binning_selection import POOL_MIN, select_binning
from tracecausal.ciu import (
    POSITIVITY_EXCLUDED_MAX,
    REPAIR_UTILITY_BOUND,
    CIURecord,
    calibrate_m_r,
    g9_novelty_gate,
    g9_repair_gate,
    validate_ciu_record,
)
from tracecausal.interventions import Span
from tracecausal.nuisance import estimate_sigma_r, r_power_repair
from tracecausal.repair_ops import (
    OperatorGrid,
    PositivityFail,
    TargetClaimSpan,
    TargetEdit,
    apply,
    localized_repair,
    operator_grid_cardinality,
    policy_hash,
    select_operator,
    tracedet_ar_span_adapter,
    transport,
    transport_map_hash,
)
from tracecausal.repair_transfer import (
    RepairGain,
    class_block_permutation,
    common_support_pairs,
    g9_nov_margin_simultaneous,
    hajek_projection_var,
    r_hat,
    repair_gain,
    target_clustered_mc_var,
    two_way_cluster_bootstrap,
)
from tracecausal.selective_inference import (
    SiFloors,
    choose_si_path,
    holm_alpha,
    validate_selection_split,
)


# ---------------------------------------------------------------------------
# Helpers: build a synthetic in-class pair set with a controllable transfer signal.
# ---------------------------------------------------------------------------


def _make_pairs(
    n_sources: int,
    n_targets: int,
    *,
    base_gain: float,
    g3_class: str = "claimA",
    noise: float = 0.0,
    seed: int = 0,
    mc_var: float = 0.0,
) -> list[RepairGain]:
    """Build ordered source != target in-class pairs with a planted mean gain.

    ``base_gain`` is the planted ``g_{ij}``; ``noise`` adds a small deterministic
    per-pair perturbation (zero-mean over the grid) so variance estimators have
    something to chew on without being random.
    """
    import random as _r

    rng = _r.Random(seed)
    pairs: list[RepairGain] = []
    for i in range(n_sources):
        for j in range(n_targets):
            if i == j:
                continue  # source != target
            g = base_gain + (rng.uniform(-noise, noise) if noise else 0.0)
            pairs.append(
                RepairGain(
                    g=g, mc_var=mc_var, source_id=f"s{i}", target_id=f"t{j}",
                    g3_class=g3_class,
                )
            )
    return pairs


# ---------------------------------------------------------------------------
# 1. repair_gain (Eq. g-ij): localized minus matched-null repair, nested MC var
# ---------------------------------------------------------------------------


def test_repair_gain_subtracts_matched_null_and_records_mc_var():
    # localized gain = 0.6 - 0.1 = 0.5; matched-null mean repair = mean([0.1,0.2,0.0]) = 0.1
    rg = repair_gain(
        y_localized=0.6, y_noop=0.1,
        matched_null_repair_samples=[0.1, 0.2, 0.0],
        source_id="s0", target_id="t1", g3_class="claimA",
    )
    assert math.isclose(rg.g, 0.5 - 0.1, abs_tol=1e-12)
    # nested matched-null MC variance = sample_var / R_null > 0 for >1 draw
    assert rg.mc_var > 0.0
    # empty matched-null samples -> the null term is unestimated (must raise)
    try:
        repair_gain(0.6, 0.1, [], source_id="s0", target_id="t1", g3_class="claimA")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("empty matched-null samples must raise")


# ---------------------------------------------------------------------------
# 2. r_hat U-statistic + two-way cluster bootstrap + Hajek cross-check
# ---------------------------------------------------------------------------


def test_r_hat_is_within_class_pair_mean():
    pairs = _make_pairs(5, 5, base_gain=0.2)
    est = r_hat(pairs)
    assert math.isclose(est.r_hat, 0.2, abs_tol=1e-12)
    assert est.n_source_clusters == 5 and est.n_target_clusters == 5


def test_two_way_cluster_bootstrap_ci_brackets_point_and_hajek_floor():
    # A transferable repair: planted mean 0.3 with mild noise. The two-way cluster
    # bootstrap CI must bracket the point estimate, and the Hajek analytic variance
    # (BOTH zeta projections) must be a floor the bootstrap SD does not undershoot.
    pairs = _make_pairs(8, 8, base_gain=0.3, noise=0.05, seed=1)
    point = r_hat(pairs).r_hat
    ci_lo, ci_hi = two_way_cluster_bootstrap(
        pairs, n_bootstrap=400, seed=2, propagate_mc=False
    )
    assert ci_lo <= point <= ci_hi, "bootstrap CI must bracket the point estimate"

    hv = hajek_projection_var(pairs, per_class=True)
    # both projections are reported and the floor uses both (refinement 5)
    assert hv.zeta_10 >= 0.0 and hv.zeta_01 >= 0.0
    assert math.isclose(
        hv.var_hat, hv.zeta_10 / hv.n_source + hv.zeta_01 / hv.n_target, rel_tol=1e-9
    )
    # the bootstrap SE should be of the same order as (not far below) the analytic floor
    boot_se = (ci_hi - ci_lo) / (2 * 1.96)
    hajek_se = math.sqrt(hv.var_hat)
    # not a strict equality (finite B), but the bootstrap must not collapse far below
    # the analytic floor (the cross-check the design demands).
    assert boot_se >= 0.25 * hajek_se


def test_hajek_uses_both_zeta_projections_for_asymmetric_kernel():
    # Construct a kernel whose SOURCE margin carries variance but the TARGET margin
    # is (near) constant, and a second where the roles flip. A floor using only the
    # source projection would understate the second case; using both does not.
    # Case A: gain depends only on source index.
    a = [
        RepairGain(g=float(i), mc_var=0.0, source_id=f"s{i}", target_id=f"t{j}", g3_class="c")
        for i in range(4) for j in range(4) if i != j
    ]
    hv_a = hajek_projection_var(a)
    assert hv_a.zeta_10 > hv_a.zeta_01  # source margin dominates
    # Case B: gain depends only on target index.
    b = [
        RepairGain(g=float(j), mc_var=0.0, source_id=f"s{i}", target_id=f"t{j}", g3_class="c")
        for i in range(4) for j in range(4) if i != j
    ]
    hv_b = hajek_projection_var(b)
    assert hv_b.zeta_01 > hv_b.zeta_10  # target margin dominates
    # If we had used only zeta_10 (4*zeta_1 the design warns against), Case B's floor
    # would be ~0; including zeta_01 keeps it positive (refinement 5).
    assert hv_b.var_hat > 0.0


def test_bootstrap_propagates_nested_matched_null_mc_noise():
    # With nested matched-null MC variance present, the propagated CI must be WIDER
    # than without it (refinement 6: the inner sampling noise is added).
    pairs_no_mc = _make_pairs(8, 8, base_gain=0.3, seed=3, mc_var=0.0)
    pairs_mc = _make_pairs(8, 8, base_gain=0.3, seed=3, mc_var=0.04)
    lo0, hi0 = two_way_cluster_bootstrap(pairs_no_mc, n_bootstrap=400, seed=4, propagate_mc=True)
    lo1, hi1 = two_way_cluster_bootstrap(pairs_mc, n_bootstrap=400, seed=4, propagate_mc=True)
    assert (hi1 - lo1) > (hi0 - lo0), "MC-noise propagation must widen the CI"


# ---------------------------------------------------------------------------
# 3. class-block permutation: A6 association test, NOT composite-null exactness
# ---------------------------------------------------------------------------


def test_class_block_permutation_detects_association_and_states_scope():
    # A strong transferable signal: the observed R_hat should be far in the tail of
    # the source-block sign-flip diagnostic (small p). Crucially the result must be
    # honestly scoped as a DIAGNOSTIC, never an exact significance certificate (G9-FIX):
    # it is NOT confirmatory and claims NO sharp-null / composite-null exactness.
    pairs = _make_pairs(6, 6, base_gain=0.5, noise=0.02, seed=5)
    res = class_block_permutation(pairs, n_permutations=500, seed=6)
    assert res.p_value <= 0.05, "strong association should be detected by the diagnostic"
    assert res.confirmatory is False
    assert res.exact_under_registered_sharp_null is False
    assert res.exact_for_composite_null is False
    assert "DIAGNOSTIC" in res.tests and "NOT" in res.tests


def test_class_block_permutation_null_signal_is_not_significant():
    # A null transfer (mean 0, symmetric noise) should NOT be flagged significant.
    pairs = _make_pairs(6, 6, base_gain=0.0, noise=0.3, seed=7)
    res = class_block_permutation(pairs, n_permutations=500, seed=8)
    assert res.p_value > 0.05


def test_two_way_bootstrap_and_permutation_agree_on_signal_presence():
    pairs = _make_pairs(8, 8, base_gain=0.4, noise=0.03, seed=9)
    ci_lo, _ = two_way_cluster_bootstrap(pairs, n_bootstrap=400, seed=10, propagate_mc=False)
    perm = class_block_permutation(pairs, n_permutations=400, seed=11)
    # both must agree the signal is present: CI clears 0 AND permutation is significant
    assert ci_lo > 0.0
    assert perm.p_value <= 0.05


# ---------------------------------------------------------------------------
# 4. G9 certify vs fail (transferable vs non-transferable); B4-also-passes -> diagnostic
# ---------------------------------------------------------------------------


def test_g9_certifies_transferable_repair():
    # transferable: R_hat CI lower bound > m_R, perm significant, utility/positivity OK,
    # no leakage, and the B4 matched-null CI brackets 0 (repair is localized).
    verdict = g9_repair_gate(
        r_hat_estimate=0.30,
        r_hat_ci=(0.12, 0.48),          # lower bound 0.12 > m_R 0.05
        perm_p=0.001,                   # < alpha_1'
        d_util_repair=0.01,             # <= 0.02
        positivity_excluded_frac=0.10,  # < 0.5
        class_leakage_ok=True,
        matched_null_repair_ci=(-0.02, 0.03),  # brackets 0 -> localized, not generic
        alpha_1_prime=0.00625,
        m_r=0.05,
    )
    assert verdict == "useful_candidate"


def test_g9_fails_non_transferable_repair():
    # non-transferable: R_hat CI lower bound does NOT clear m_R -> diagnostic.
    verdict = g9_repair_gate(
        r_hat_estimate=0.02,
        r_hat_ci=(-0.04, 0.08),         # lower bound -0.04 < m_R
        perm_p=0.30,
        d_util_repair=0.01,
        positivity_excluded_frac=0.10,
        class_leakage_ok=True,
        matched_null_repair_ci=(-0.02, 0.03),
        alpha_1_prime=0.00625,
        m_r=0.05,
    )
    assert verdict == "diagnostic"  # never 'invalidated'
    assert verdict != "invalidated"


def test_g9_b4_also_passes_routes_to_diagnostic_generic_repair():
    # R_hat clears the margin, but the B4 matched-null repair CI sits ABOVE 0 -> the
    # repair is GENERIC (any matched span transfers), not localized -> diagnostic.
    verdict = g9_repair_gate(
        r_hat_estimate=0.30,
        r_hat_ci=(0.12, 0.48),
        perm_p=0.001,
        d_util_repair=0.01,
        positivity_excluded_frac=0.10,
        class_leakage_ok=True,
        matched_null_repair_ci=(0.04, 0.10),  # off-zero ABOVE 0 -> generic repair
        alpha_1_prime=0.00625,
        m_r=0.05,
    )
    assert verdict == "diagnostic"


def test_g9_bounded_utility_and_positivity_and_leakage_routes():
    base = dict(
        r_hat_estimate=0.30, r_hat_ci=(0.12, 0.48), perm_p=0.001,
        positivity_excluded_frac=0.10, class_leakage_ok=True,
        matched_null_repair_ci=(-0.02, 0.03), alpha_1_prime=0.00625, m_r=0.05,
    )
    # utility cost over the bound -> diagnostic
    assert g9_repair_gate(d_util_repair=0.05, **base) == "diagnostic"
    # positivity excluded fraction at/over 0.5 -> diagnostic
    bad_pos = dict(base); bad_pos["positivity_excluded_frac"] = 0.6
    assert g9_repair_gate(d_util_repair=0.01, **bad_pos) == "diagnostic"
    # self/class leakage -> diagnostic
    bad_leak = dict(base); bad_leak["class_leakage_ok"] = False
    assert g9_repair_gate(d_util_repair=0.01, **bad_leak) == "diagnostic"
    # permutation not significant at the level -> diagnostic
    bad_perm = dict(base); bad_perm["perm_p"] = 0.5
    assert g9_repair_gate(d_util_repair=0.01, **bad_perm) == "diagnostic"


# ---------------------------------------------------------------------------
# 5. G9-NOV: passes only when PROPOSED out-transfers B1-B3 (simultaneous CI)
# ---------------------------------------------------------------------------


def test_g9_nov_passes_only_when_proposed_beats_baselines():
    # PROPOSED transfers strongly (0.4); detector baselines transfer weakly (~0.1).
    proposed = _make_pairs(8, 8, base_gain=0.40, noise=0.02, seed=12)
    b1 = _make_pairs(8, 8, base_gain=0.10, noise=0.02, seed=12)
    b2 = _make_pairs(8, 8, base_gain=0.08, noise=0.02, seed=13)
    b3 = _make_pairs(8, 8, base_gain=0.05, noise=0.02, seed=14)
    margin = g9_nov_margin_simultaneous(
        proposed, {"B1": b1, "B2": b2, "B3": b3}, n_bootstrap=400, seed=15
    )
    assert margin.clears_zero, "PROPOSED out-transfers the detector baselines"
    assert margin.argmax_baseline == "B1"  # the strongest baseline
    assert g9_novelty_gate(
        r_hat(proposed).r_hat,
        {b: r_hat(ps).r_hat for b, ps in {"B1": b1, "B2": b2, "B3": b3}.items()},
        margin_ci_low=margin.margin_ci_low,
    ) == "useful_candidate"


def test_g9_nov_downgrades_when_baseline_matches_proposed():
    # A detector-selected span licenses an equally transferable repair: novelty
    # downgrades (NOT an identification failure, never 'invalidated').
    proposed = _make_pairs(8, 8, base_gain=0.20, noise=0.02, seed=16)
    b1 = _make_pairs(8, 8, base_gain=0.21, noise=0.02, seed=16)  # matches/beats PROPOSED
    margin = g9_nov_margin_simultaneous(
        proposed, {"B1": b1}, n_bootstrap=400, seed=17
    )
    assert not margin.clears_zero
    verdict = g9_novelty_gate(
        r_hat(proposed).r_hat, {"B1": r_hat(b1).r_hat}, margin_ci_low=margin.margin_ci_low
    )
    assert verdict == "not_novel"
    assert verdict != "invalidated"


# ---------------------------------------------------------------------------
# 6. common-support / positivity accounting (refinement 4 + A7 event)
# ---------------------------------------------------------------------------


def test_common_support_restricts_all_arms_to_same_pairs():
    # PROPOSED has a pair (s0,t9) that the baseline lacks (positivity-excluded for it).
    proposed = _make_pairs(3, 3, base_gain=0.3, seed=18)
    proposed.append(RepairGain(0.9, 0.0, "s0", "t9", "claimA"))  # extra pair
    baseline = _make_pairs(3, 3, base_gain=0.1, seed=18)
    restricted = common_support_pairs({"PROPOSED": proposed, "B1": baseline})
    # the extra (s0,t9) pair is dropped from PROPOSED so both arms share the estimand
    prop_keys = {(p.source_id, p.target_id) for p in restricted["PROPOSED"]}
    base_keys = {(p.source_id, p.target_id) for p in restricted["B1"]}
    assert prop_keys == base_keys
    assert ("s0", "t9") not in prop_keys


def test_transport_positivity_event_is_recorded_not_raised():
    # No in-class, proximity/budget-matched target span -> PositivityFail (A7), not raise.
    policy = localized_repair(
        Span(0, 1), op="patch", alpha=0.5, layer_set=[3, 4],
        ref_type="factual", source_proximity_bin=0, source_example_id="s0",
    )
    # target has only an out-of-class span -> positivity fails
    target_spans = [TargetClaimSpan(Span(5, 6), g3_class="OTHER", distance_to_answer=2, budget_k=4)]
    res = transport(policy, "t1", target_spans, g3_class="claimA", proximity_bin_width=4)
    assert isinstance(res, PositivityFail)
    assert res.target_example_id == "t1"


# ---------------------------------------------------------------------------
# 7. repair_ops Variant C: source-derived policy, target-own state, collapse guard
# ---------------------------------------------------------------------------


def test_transport_anchors_target_own_span_and_records_source_content():
    # budget for patch = |span| * |layers| = 2 * 2 = 4; source proximity bin 0 (dist 2 // 4).
    policy = localized_repair(
        Span(0, 1), op="patch", alpha=0.5, layer_set=[3, 4],
        ref_type="factual", source_proximity_bin=0, source_example_id="s0",
    )
    assert policy.budget_k == 4
    target_spans = [
        TargetClaimSpan(Span(2, 3), g3_class="claimA", distance_to_answer=2, budget_k=4),
        TargetClaimSpan(Span(8, 9), g3_class="claimA", distance_to_answer=20, budget_k=4),  # wrong prox bin
    ]
    edit = transport(policy, "t1", target_spans, g3_class="claimA", proximity_bin_width=4)
    assert isinstance(edit, TargetEdit)
    assert edit.target_span == Span(2, 3)          # the in-bin, budget-matched span
    assert edit.source_proximity_bin == 0           # source-instance-derived content recorded
    assert edit.source_budget_k == 4
    assert edit.target_example_id == "t1"


def test_transport_collapse_guard_refuses_target_own_oracle_span():
    # The only in-class candidate is the target's OWN designated/oracle span -> the
    # collapse guard refuses to anchor it (refinement 1); positivity fails instead.
    policy = localized_repair(
        Span(0, 0), op="replay", alpha=1.0, layer_set=[],
        ref_type="factual", source_proximity_bin=0, source_example_id="s0",
    )
    target_spans = [
        TargetClaimSpan(Span(1, 1), g3_class="claimA", distance_to_answer=1, budget_k=1,
                        is_target_designated=True),
    ]
    res = transport(policy, "t1", target_spans, g3_class="claimA", proximity_bin_width=4)
    assert isinstance(res, PositivityFail)
    assert "within-target oracle repair" in res.reason


def test_apply_patch_uses_target_own_states_no_model_call():
    policy = localized_repair(
        Span(0, 1), op="patch", alpha=0.25, layer_set=[5, 6],
        ref_type="factual", source_proximity_bin=0,
    )
    target_spans = [TargetClaimSpan(Span(2, 3), g3_class="claimA", distance_to_answer=2, budget_k=4)]
    edit = transport(policy, "t1", target_spans, g3_class="claimA", proximity_bin_width=4)
    plan = apply(
        edit,
        target_residual_states=[[1.0, 1.0], [1.0, 1.0]],  # the TARGET's own states
        target_reference_states=[[0.0, 0.0], [0.0, 0.0]],
    )
    assert plan.no_model_call is True
    assert plan.op == "patch"
    # convex interpolation computed from the TARGET's own states: (1-0.25)*1 + 0.25*0
    assert plan.result.patched_states[0] == (0.75, 0.75)
    # Variant C patch without the target's own states must raise (no synthetic state)
    try:
        apply(edit)  # no states supplied
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Variant C patch without target states must raise")


def test_tracedet_ar_span_adapter_selects_longest_high_entropy_run():
    # B1 executable on AR: longest contiguous run at/above threshold -> claim span.
    entropy = [0.1, 0.9, 0.95, 0.92, 0.2, 0.8, 0.85, 0.1]  # run [1..3] len3, run [5..6] len2
    span = tracedet_ar_span_adapter(entropy, threshold=0.8)
    assert span == Span(1, 3)
    # no high-entropy step -> the baseline abstains (raises)
    try:
        tracedet_ar_span_adapter([0.1, 0.2, 0.3], threshold=0.8)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("B1 must abstain when no sub-trace clears threshold")


def test_policy_and_transport_hashes_are_deterministic_and_recipe_addressed():
    p1 = localized_repair(Span(0, 1), op="patch", alpha=0.5, layer_set=[3],
                          ref_type="factual", source_proximity_bin=0, source_example_id="s0")
    p2 = localized_repair(Span(0, 1), op="patch", alpha=0.5, layer_set=[3],
                          ref_type="factual", source_proximity_bin=0, source_example_id="sZ")
    # hash addresses the RECIPE, not which source induced it -> p1 == p2
    assert policy_hash(p1) == policy_hash(p2)
    assert transport_map_hash(p1, 4) == transport_map_hash(p2, 4)
    # a different recipe -> different hash
    p3 = localized_repair(Span(0, 1), op="replay", alpha=1.0, layer_set=[3],
                          ref_type="factual", source_proximity_bin=0)
    assert policy_hash(p3) != policy_hash(p1)


# ---------------------------------------------------------------------------
# 8. Axis X': detectable controls trip before R_hat; blind controls silent, R_hat->0
# ---------------------------------------------------------------------------


def test_axis_x_zero_reproduces_clean_oracle():
    fx = axis_x_confounded(0.0, regime="detectable")
    # xi=0 -> clean oracle examples; controls at zero; R_hat ~ 1 (no confounding)
    assert math.isclose(fx.readout.r_hat_expected, 1.0, abs_tol=1e-12)
    assert fx.readout.controls_trip is False


def test_axis_x_detectable_controls_trip_and_rhat_collapses():
    prev_g7 = prev_r = None
    for xi in AXIS_X_XI_GRID:
        ro = structural_equations(xi, "detectable")
        # controls rise monotonically with xi (they trip as confounding grows)
        if prev_g7 is not None:
            assert ro.g7_leakage_slope >= prev_g7 - 1e-12
            assert ro.r_hat_expected <= prev_r + 1e-12  # R_hat collapses
        prev_g7, prev_r = ro.g7_leakage_slope, ro.r_hat_expected
        if xi > 0:
            assert ro.controls_trip is True  # controls trip BEFORE certification
    # at full confounding R_hat has collapsed to 0
    assert math.isclose(structural_equations(1.0, "detectable").r_hat_expected, 0.0, abs_tol=1e-12)


def test_axis_x_blind_controls_silent_while_rhat_collapses():
    # The PROVEN failure regime: c_i collinear with the controls' regressors, so
    # G7/G8 slopes are UNCHANGED (provably do not trip), yet R_hat collapses (P5).
    for xi in AXIS_X_XI_GRID:
        ro = structural_equations(xi, "blind")
        assert ro.controls_trip is False, "X'-blind controls provably do not trip"
        assert math.isclose(ro.g7_leakage_slope, 0.0, abs_tol=1e-12)
        assert math.isclose(ro.g8_ood_slope, 0.0, abs_tol=1e-12)
    # but R_hat -> 0 as xi -> 1 (the falsifiable test), while a correlational score stays high
    assert math.isclose(structural_equations(1.0, "blind").r_hat_expected, 0.0, abs_tol=1e-12)
    assert math.isclose(
        structural_equations(1.0, "blind").correlational_score_expected, 1.0, abs_tol=1e-12
    )


def test_nc1_collinear_confounder_negative_control():
    # NC-1: on X'-blind, controls silent AND R_hat collapses at full confounding -> passes.
    nc = negative_control_collinear(1.0)
    assert nc.controls_silent is True
    assert math.isclose(nc.r_hat_expected, 0.0, abs_tol=1e-12)
    assert nc.passes is True


def test_nc2_source_swap_invariance():
    # A5/A6: swapping the in-class source leaves g_ij invariant up to MC noise.
    inv = source_swap(0.300, 0.305, mc_tol=0.02)
    assert inv.invariant is True
    # a significant source-identity effect falsifies A6 -> re-stratify (not silent fix)
    not_inv = source_swap(0.30, 0.55, mc_tol=0.02)
    assert not_inv.invariant is False


# ---------------------------------------------------------------------------
# 9. Selective inference: Holm fold (K_bin/K_op), SI path, selection-split validate
# ---------------------------------------------------------------------------


def test_holm_alpha_folds_k_bin_and_k_op_under_si2_not_si1():
    # SI-1 (selection split): K_bin = K_op = 1 for the inference family.
    a_si1 = holm_alpha(8, k_bin=3, k_op=10, selection_split_used=True)
    assert math.isclose(a_si1, 0.05 / 8, rel_tol=1e-12)
    # SI-2 (fallback): pays the Bonferroni factor K_bin * K_op.
    a_si2 = holm_alpha(8, k_bin=3, k_op=10, selection_split_used=False)
    assert math.isclose(a_si2, 0.05 / (8 * 3 * 10), rel_tol=1e-12)
    assert a_si2 < a_si1  # SI-2 is the conservative, selection-honest level


def test_choose_si_path_deterministic_rule():
    floors = SiFloors(sigma_min=200, kappa_min=300)
    # both splits meet the stricter floor (300) -> SI-1
    assert choose_si_path(350, 320, floors=floors) == "SI-1"
    # one split below -> SI-2
    assert choose_si_path(350, 250, floors=floors) == "SI-2"


def test_si1_holm_folding_reproduces_feasible_point():
    # SI-1 with the v4 m=8 Holm family reproduces the per-test level alpha_1 = 0.00625
    # (the feasible point's confirmatory level; the z=2.734 the config records derives
    # from it). This is the "reproduce the feasible point" check (formula, not evidence).
    alpha_1 = holm_alpha(8, k_bin=5, k_op=20, selection_split_used=True)
    assert math.isclose(alpha_1, 0.00625, rel_tol=1e-12)


def test_validate_selection_split_disjointness_and_floor():
    v_sel = list(range(0, 400))
    v_inf = list(range(400, 800))
    test = list(range(800, 900))
    assert validate_selection_split(v_sel, v_inf, test) == []
    # overlap between V_sel and V_inf is a violation
    bad = validate_selection_split(list(range(0, 400)), list(range(390, 790)), test)
    assert any("overlap" in e for e in bad)
    # an undersized V_sel routes to SI-2 (flagged as below the eligibility floor)
    small = validate_selection_split(list(range(0, 50)), v_inf, test)
    assert any("below SI-1 eligibility floor" in e for e in small)


def test_validate_split_floor_matches_choose_si_path_router():
    # finding 13: the validator's eligibility bar must equal the router's bar (max of
    # the two floors). A split in [min, max) the router sends to SI-2 must NOT be
    # accepted as a valid SI-1 config by the validator.
    floors = SiFloors(sigma_min=200, kappa_min=300)
    test = list(range(10_000, 10_100))
    # sizes 250 are >= min (200) but < max (300): the router routes to SI-2 ...
    assert choose_si_path(250, 250, floors=floors) == "SI-2"
    # ... so the validator must flag them as below the SI-1 eligibility floor.
    v_sel = list(range(0, 250))
    v_inf = list(range(250, 500))
    errs = validate_selection_split(v_sel, v_inf, test, floors=floors)
    assert any("below SI-1 eligibility floor" in e for e in errs), (
        "validator must agree with choose_si_path's stricter (max) floor"
    )


# ---------------------------------------------------------------------------
# 10. binning_selection: coarsening ladder + emitted selection event (K_bin)
# ---------------------------------------------------------------------------


def test_select_binning_coarsens_until_pool_floor_and_emits_event():
    # Many spans clustered near the answer: a fine Delta_pos (1) over-shrinks the pool;
    # the rule coarsens until the mean in-bin pool clears POOL_MIN (8).
    distances = [d for d in range(0, 40)]  # 40 spans spread 0..39
    binning, event = select_binning(distances)
    assert binning.meets_pool_floor is True
    assert binning.mean_pool_at_choice >= POOL_MIN
    # the event records every rung walked and a K_bin >= 1 cardinality
    assert event.chosen_delta_pos in event.delta_pos_ladder
    # K_bin pays for the FULL pre-enumerated ladder, NOT the data-dependent prefix
    # actually walked (findings 14, 20): k_bin = |full ladder| x |edge candidate sets|.
    assert event.k_bin == len(event.delta_pos_ladder) * len(event.displaced_mass_edge_candidates)
    assert event.k_bin >= 1
    # and when the rule short-circuits before the coarsest rung, K_bin is STRICTLY
    # larger than the walked-prefix count (the anti-conservative undercount it fixes).
    if len(event.rungs_walked) < len(event.delta_pos_ladder):
        assert event.k_bin > len(event.rungs_walked) * len(event.displaced_mass_edge_candidates)


def test_select_binning_k_bin_pays_full_ladder_on_early_shortcircuit():
    # finding 20: when the rule short-circuits at the FINEST rung, K_bin must still
    # pay for the full pre-enumerated ladder (not the 1-rung walked prefix).
    # 10 spans at distance 0: at delta_pos=1 the bin has 10 members -> matched-null
    # pool (excluding self) = 9 >= POOL_MIN(8), so the rule stops at the first rung.
    distances = [0] * 10
    binning, event = select_binning(distances)
    assert binning.delta_pos == event.delta_pos_ladder[0]  # finest rung, short-circuit
    assert len(event.rungs_walked) == 1
    # K_bin uses the FULL ladder length, strictly larger than the 1-rung walked prefix.
    assert event.k_bin == len(event.delta_pos_ladder) * len(event.displaced_mass_edge_candidates)
    assert event.k_bin > len(event.rungs_walked) * len(event.displaced_mass_edge_candidates)


def test_select_binning_pool_excludes_the_member_itself():
    # finding 15: the matched-null pool excludes the example itself. With 8 spans in a
    # single bin the INCLUDING-self count is 8 (== POOL_MIN, would falsely clear), but
    # the real matched-null pool is 7 (< 8) -> must NOT clear the floor at that width.
    distances = [0] * 8  # one bin of 8 at delta_pos=1
    binning, event = select_binning(distances, delta_pos_ladder=(1,))
    # pool seen by a member = 8 - 1 = 7 < POOL_MIN(8): the floor is NOT met (finding 15)
    assert math.isclose(event.mean_pool_per_rung[0], 7.0, abs_tol=1e-12)
    assert binning.meets_pool_floor is False


def test_select_binning_records_underpowered_when_no_rung_clears_floor():
    # Very few, spread-out distances: even the coarsest rung cannot reach POOL_MIN.
    distances = [0, 50, 100]
    binning, event = select_binning(distances)
    assert binning.meets_pool_floor is False  # routed to 'insufficient pool', never silent
    assert binning.delta_pos == event.delta_pos_ladder[-1]  # coarsest rung selected


# ---------------------------------------------------------------------------
# 11. nuisance sigma_R (Eq. R-VAR) and r_power_repair (Eq. R-POWER) — MF-5
# ---------------------------------------------------------------------------


def test_estimate_sigma_r_assembles_design_effect_with_both_projections():
    pairs = _make_pairs(6, 6, base_gain=0.3, noise=0.05, seed=20)
    g_by_pair = [(p.source_id, p.target_id, p.g) for p in pairs]
    mc = [0.01 for _ in pairs]
    op = [0.02 for _ in pairs]
    est = estimate_sigma_r(g_by_pair, mc_var_per_pair=mc, op_var_per_pair=op,
                           r_null=16, r_int=16, ci_inflation=1.2)
    # the variance is assembled from Eq. R-VAR; n_eff is the binding cluster margin
    assert est.n_eff == 6.0
    assert est.n_source == 6 and est.n_target == 6
    assert est.var_r_hat > 0.0
    # the dominant term is the ORDERED-PAIR two-way projection variance using BOTH
    # zeta projections, each over its own margin (findings 4, 10) -- NOT (4/n_eff)*zeta_1.
    assert math.isclose(
        est.proj_var, est.zeta_10 / est.n_source + est.zeta_01 / est.n_target, rel_tol=1e-12
    )
    assert math.isclose(est.var_r_hat, est.proj_var
                        + (1.0 / est.n_pair) * (est.mc_term + est.op_term), rel_tol=1e-9)
    assert math.isclose(est.sigma_r, math.sqrt(est.var_r_hat), rel_tol=1e-12)
    assert math.isclose(est.sigma_r_hi, est.sigma_r * 1.2, rel_tol=1e-12)


def test_estimate_sigma_r_matches_hajek_projection_var_floor():
    # The power-path projection variance must equal the standalone Hajek cross-check
    # floor (BOTH zeta projections, each over its own margin) -- findings 4, 10.
    from tracecausal.repair_transfer import hajek_projection_var
    pairs = _make_pairs(7, 5, base_gain=0.25, noise=0.07, seed=21)
    g_by_pair = [(p.source_id, p.target_id, p.g) for p in pairs]
    est = estimate_sigma_r(g_by_pair)
    hv = hajek_projection_var(pairs)
    assert math.isclose(est.proj_var, hv.var_hat, rel_tol=1e-12)
    assert math.isclose(est.zeta_10, hv.zeta_10, rel_tol=1e-12)
    assert math.isclose(est.zeta_01, hv.zeta_01, rel_tol=1e-12)


def test_r_power_repair_uses_sigma_r_not_sigma_u_and_returns_forwards():
    # R_power = ceil( (z * sigma_r_hi / m_R)^2 * D_eff ); forwards = R_power * surcharge.
    r, forwards = r_power_repair(0.20, z=2.734, m_r=0.05, d_eff=1.125, forward_surcharge=20)
    expected = math.ceil((2.734 * 0.20 / 0.05) ** 2 * 1.125)
    assert r == expected
    assert forwards == r * 20


# ---------------------------------------------------------------------------
# 12. CIURecord v5 fields + validate_ciu_record(require_v5=...) round trip
# ---------------------------------------------------------------------------


def _v5_complete_record(**over) -> CIURecord:
    """A v5-complete record: v4 lock fields + v5 fields, all passing."""
    kwargs = dict(
        selector_id="ciu_causal", operator="patch", reference_type="factual",
        edit_budget=4, null_pool_hash="np", noop_run_hash="noop",
        evaluator_hash="ev", evaluator_kappa=0.92, ref_hash="ref",
        n_examples=850, r_int=16, b_boot=10_000, s_seed=20,
        u_hat=0.12, ci_low=0.07, ci_high=0.17, d_util=0.01,
        tau_per_example=(0.12, 0.12, 0.12, 0.12),
        pi_mean_per_example=(0.0, 0.0, 0.0, 0.0),
        matched_control_provenance=("c0", "c1", "c2", "c3"),
        # v4 lock fields
        beta_hi=0.01, proximity_bin_width=2.0,
        ood_slope_ci=(0.0, 0.05), displaced_mass_range=0.4,
        sham_u_ci=(-0.01, 0.01), noop_u_ci=(-0.01, 0.01),
        oracle_pass=True, graded_curve_pass=True, m_pool_mean=8.0,
        # v5 fields
        r_hat=0.30, r_hat_ci=(0.12, 0.48), r_hat_perm_p=0.001,
        d_util_repair=0.01, matched_null_repair_ci=(-0.02, 0.03),
        positivity_excluded=0.10,
        positivity_excluded_by_class={"claimA": 0.10, "claimB": 0.20},
        baseline_r_hats=(0.0, 0.1, 0.08, 0.05, 0.02, 0.9),
        repair_policy_hash="rp", transport_map_hash="tm", class_partition_hash="cp",
        selection_event="ev_serialised", k_bin=5, k_op=20,
        xi_axis_x=1.0, axis_x_regime="blind",
    )
    kwargs.update(over)
    return CIURecord(**kwargs)


def test_v5_complete_record_validates_and_v3_v4_still_pass():
    rec = _v5_complete_record()
    assert validate_ciu_record(rec, require_v5=True) == []
    # the same record (without require_v5) also validates (back-compatible)
    assert validate_ciu_record(rec) == []


def test_require_v5_hard_requires_v5_fields():
    # A v4-complete record with NO v5 fields fails the v5 lock (hard-required).
    rec = _v5_complete_record(
        r_hat=None, r_hat_ci=None, r_hat_perm_p=None, d_util_repair=None,
        matched_null_repair_ci=None, positivity_excluded=None,
        positivity_excluded_by_class=None, baseline_r_hats=None,
        repair_policy_hash=None, transport_map_hash=None, class_partition_hash=None,
        selection_event=None, k_bin=None, k_op=None, xi_axis_x=None, axis_x_regime=None,
    )
    errs = validate_ciu_record(rec, require_v5=True)
    assert any("v5 lock requires r_hat" in e for e in errs)
    assert any("repair_policy_hash" in e for e in errs)
    # the v4 record itself is still valid WITHOUT require_v5 (additive, back-compatible)
    assert validate_ciu_record(rec) == []


def test_require_v5_enforces_positivity_and_utility_bounds():
    # positivity excluded fraction at/over 0.5 -> v5 lock fails (A7)
    over_pos = _v5_complete_record(positivity_excluded=0.6)
    assert any("positivity_excluded" in e and str(POSITIVITY_EXCLUDED_MAX) in e
               for e in validate_ciu_record(over_pos, require_v5=True))
    # repair utility cost over 0.02 -> v5 lock fails (reframe as abstention)
    over_util = _v5_complete_record(d_util_repair=0.05)
    assert any("d_util_repair" in e and str(REPAIR_UTILITY_BOUND) in e
               for e in validate_ciu_record(over_util, require_v5=True))


def test_v5_malformed_cis_flagged_regardless_of_lock():
    bad_ci = _v5_complete_record(r_hat=0.3, r_hat_ci=(0.5, 0.1))  # lo > hi
    assert any("r_hat_ci must be a (lo, hi)" in e for e in validate_ciu_record(bad_ci))
    out_of_ci = _v5_complete_record(r_hat=0.9, r_hat_ci=(0.1, 0.5))  # r_hat outside CI
    assert any("r_hat must lie within r_hat_ci" in e for e in validate_ciu_record(out_of_ci))


def test_v5_record_is_back_compatible_with_v4_lock():
    # A v5-complete record still satisfies the v4 lock (G9 sits on top of G1-G8).
    rec = _v5_complete_record()
    assert validate_ciu_record(rec, require_v4=True) == []


# ---------------------------------------------------------------------------
# 13. Code-review fixes: transport self/cross-class guards (findings 1/2/6/7)
# ---------------------------------------------------------------------------


def test_transport_rejects_self_repair_source_equals_target():
    # findings 1/6: source != target must be enforced fail-closed at transport.
    policy = localized_repair(
        Span(0, 1), op="patch", alpha=0.5, layer_set=[3, 4],
        ref_type="factual", source_proximity_bin=0, source_example_id="ex7",
        source_g3_class="claimA",
    )
    target_spans = [TargetClaimSpan(Span(2, 3), g3_class="claimA", distance_to_answer=2, budget_k=4)]
    # target id == source id -> PositivityFail (self-repair refused), not a TargetEdit.
    res = transport(policy, "ex7", target_spans, g3_class="claimA", proximity_bin_width=4)
    assert isinstance(res, PositivityFail)
    assert "source != target" in res.reason or "self" in res.reason.lower()
    # a different target id is accepted as before.
    ok = transport(policy, "ex8", target_spans, g3_class="claimA", proximity_bin_width=4)
    assert isinstance(ok, TargetEdit)


def test_transport_rejects_cross_class_when_source_class_recorded():
    # findings 2/7: class(source) == class(target) enforced fail-closed at transport.
    policy = localized_repair(
        Span(0, 1), op="patch", alpha=0.5, layer_set=[3, 4],
        ref_type="factual", source_proximity_bin=0, source_example_id="s0",
        source_g3_class="claimA",
    )
    target_spans = [TargetClaimSpan(Span(2, 3), g3_class="claimB", distance_to_answer=2, budget_k=4)]
    # target class B != source class A -> PositivityFail (within-class estimand).
    res = transport(policy, "t1", target_spans, g3_class="claimB", proximity_bin_width=4)
    assert isinstance(res, PositivityFail)
    assert "within-class" in res.reason or "class(source)" in res.reason


def test_repair_policy_rejects_off_grid_patch_alpha():
    # finding 16 (OS-1): patch alpha must be a frozen PATCH_RHO_LEVELS value.
    import pytest
    with pytest.raises(ValueError):
        localized_repair(
            Span(0, 1), op="patch", alpha=0.37, layer_set=[3],  # 0.37 not on the grid
            ref_type="factual", source_proximity_bin=0,
        )
    # an on-grid alpha is fine.
    p = localized_repair(
        Span(0, 1), op="patch", alpha=0.25, layer_set=[3],
        ref_type="factual", source_proximity_bin=0,
    )
    assert p.alpha == 0.25


# ---------------------------------------------------------------------------
# 14. OS-1 frozen operator selection + derived K_op (finding 16)
# ---------------------------------------------------------------------------


def test_operator_grid_cardinality_derives_k_op():
    # finding 16: K_op is DERIVED from the enumerated grid, not caller-declared.
    grid = OperatorGrid(
        ops=("patch", "replay"),
        alphas=(0.1, 0.25, 0.5, 0.75, 1.0),  # PATCH_RHO_LEVELS
        layer_sets=((0,), (1, 2)),
        ref_types=("factual", "neutral"),
    )
    # patch: 5 alphas x 2 layer_sets x 2 ref_types = 20; replay: 1 x 2 x 2 = 4 -> 24
    assert operator_grid_cardinality(grid) == 24


def test_select_operator_freezes_argmax_and_derives_k_op():
    # finding 16: the frozen freeze rule picks argmax of the registered V_sel score
    # and returns a DERIVED k_op (== grid cardinality), not a declared one.
    grid = OperatorGrid(
        ops=("patch", "replay"), alphas=(0.5, 1.0),
        layer_sets=((0,),), ref_types=("factual",),
    )
    # patch:2 alphas x1 x1 = 2 ; replay:1 -> k_op = 3
    def score(policy):
        # prefer the patch at alpha=1.0 (highest score)
        return policy.alpha if policy.op == "patch" else -1.0
    sel = select_operator(
        grid, score, source_proximity_bin=0, source_span_length=2,
        source_example_id="s0", source_g3_class="claimA",
    )
    assert sel.k_op == 3
    assert sel.policy.op == "patch"
    assert sel.policy.alpha == 1.0
    assert sel.policy.source_g3_class == "claimA"


# ---------------------------------------------------------------------------
# 15. target-clustered matched-null MC variance (findings 9/19)
# ---------------------------------------------------------------------------


def test_target_clustered_mc_var_exceeds_naive_per_pair_quadrature():
    # findings 9/19: one Pi_j estimate is reused across all source pairs sharing a
    # target, so the MC error is perfectly correlated WITHIN a target. The clustered
    # variance must exceed the naive independent-per-pair quadrature whenever a target
    # appears in more than one pair.
    pairs = _make_pairs(4, 4, base_gain=0.3, seed=30, mc_var=0.04)
    clustered = target_clustered_mc_var(pairs)
    n = len(pairs)
    # naive independent quadrature for the equal-weight single-class mean:
    naive = sum((1.0 / n) ** 2 * p.mc_var for p in pairs)
    assert clustered > naive


def test_two_way_bootstrap_target_clustered_noise_widens_more_than_independent():
    # The target-clustered propagation (shared z_t per target) must widen the CI at
    # least as much as it did before; with multiple pairs per target the within-target
    # correlation makes the propagated CI strictly wider than the no-MC CI.
    pairs = _make_pairs(8, 8, base_gain=0.3, seed=31, mc_var=0.05)
    lo0, hi0 = two_way_cluster_bootstrap(pairs, n_bootstrap=400, seed=2, propagate_mc=False)
    lo1, hi1 = two_way_cluster_bootstrap(pairs, n_bootstrap=400, seed=2, propagate_mc=True)
    assert (hi1 - lo1) > (hi0 - lo0)


# ---------------------------------------------------------------------------
# 16. permutation primary-null exactness flag (findings 3/8)
# ---------------------------------------------------------------------------


def test_permutation_is_demoted_to_diagnostic_not_confirmatory():
    # G9-FIX: the source-block sign-flip is a DIAGNOSTIC, not the confirmatory test.
    # Matched-null centring + A6 give mean-zero + within-class exchangeability, which
    # do NOT imply the distributional sign-symmetry a +-1 flip assumes, so NO exactness
    # for the registered sharp null is claimed; the confirmatory burden is the two-way
    # cluster-bootstrap CI.
    pairs = _make_pairs(6, 6, base_gain=0.4, noise=0.02, seed=33)
    res = class_block_permutation(pairs, n_permutations=300, seed=34)
    assert res.confirmatory is False
    assert res.exact_under_registered_sharp_null is False
    # the over-broad arbitrary-composite-null exactness was never and is not claimed.
    assert res.exact_for_composite_null is False


# ---------------------------------------------------------------------------
# 17. g9_nov identical-pair-keys enforcement (findings 5/17)
# ---------------------------------------------------------------------------


def test_g9_nov_raises_on_mismatched_pair_keys_across_arms():
    import pytest
    proposed = _make_pairs(4, 4, base_gain=0.4, seed=35)
    # baseline missing one PROPOSED pair (common support NOT enforced upstream)
    baseline_short = proposed[:-1]
    baseline_short = [RepairGain(0.1, 0.0, p.source_id, p.target_id, p.g3_class)
                      for p in baseline_short]
    with pytest.raises(ValueError):
        g9_nov_margin_simultaneous(
            proposed, {"B1": baseline_short}, n_bootstrap=50, seed=36
        )
    # after aligning via common_support_pairs the identical-key guard is satisfied.
    aligned = common_support_pairs({"PROPOSED": proposed, "B1": baseline_short})
    margin = g9_nov_margin_simultaneous(
        aligned["PROPOSED"], {"B1": aligned["B1"]}, n_bootstrap=50, seed=37
    )
    assert margin.argmax_baseline == "B1"


# ---------------------------------------------------------------------------
# 18. m_R required + calibrated, NOT inherited from necessity (finding 11)
# ---------------------------------------------------------------------------


def test_g9_repair_gate_requires_explicit_m_r():
    import pytest
    # finding 11: m_r has no default -> calling without it is a TypeError (the caller
    # MUST pass the V_sel-calibrated repair margin, never the silent necessity 0.05).
    with pytest.raises(TypeError):
        g9_repair_gate(
            0.30, (0.12, 0.48), 0.001, 0.01, 0.10, True, (-0.02, 0.03),
            alpha_1_prime=0.00625,
        )  # no m_r


def test_calibrate_m_r_uses_repair_kappa_not_necessity_margin():
    # Eq. m-R: m_R = m_R0 / (2*kappa_lo_repair - 1). With a repair kappa != 1 the
    # calibrated margin differs from the bare m_R0 (and from the necessity 0.05).
    m_r = calibrate_m_r(0.05, 0.90)
    assert math.isclose(m_r, 0.05 / (2 * 0.90 - 1.0), rel_tol=1e-12)
    assert m_r != 0.05  # NOT silently inherited from the necessity margin
    import pytest
    with pytest.raises(ValueError):
        calibrate_m_r(0.05, 0.50)  # agreement at chance -> no attenuation band


# ---------------------------------------------------------------------------
# 19. per-class positivity (finding 12): a catastrophic class is not hidden
# ---------------------------------------------------------------------------


def test_g9_per_class_positivity_blocks_catastrophic_class():
    base = dict(
        r_hat_estimate=0.30, r_hat_ci=(0.12, 0.48), perm_p=0.001, d_util_repair=0.01,
        class_leakage_ok=True, matched_null_repair_ci=(-0.02, 0.03),
        alpha_1_prime=0.00625, m_r=0.05,
    )
    # overall average is fine (0.20) but one class is catastrophic (0.7 >= 0.5):
    verdict = g9_repair_gate(
        positivity_excluded_frac=0.20,
        positivity_excluded_by_class={"claimA": 0.10, "claimB": 0.70},
        **base,
    )
    assert verdict == "diagnostic"  # the catastrophic class blocks certification
    # all classes under the bound -> certifies.
    ok = g9_repair_gate(
        positivity_excluded_frac=0.20,
        positivity_excluded_by_class={"claimA": 0.10, "claimB": 0.30},
        **base,
    )
    assert ok == "useful_candidate"


def test_require_v5_enforces_per_class_positivity():
    # finding 12: the v5 lock fails if ANY class exceeds the per-class A7 bound, even
    # when the overall scalar is fine.
    bad = _v5_complete_record(
        positivity_excluded=0.20,
        positivity_excluded_by_class={"claimA": 0.10, "claimB": 0.60},
    )
    errs = validate_ciu_record(bad, require_v5=True)
    assert any("positivity_excluded_by_class" in e and "claimB" in e for e in errs)


# ---------------------------------------------------------------------------
# 20. NC-1 collapse is NOT tautological: a non-collapsing R_hat fails (finding 21)
# ---------------------------------------------------------------------------


def test_nc1_fails_when_observed_rhat_does_not_collapse():
    # finding 21: NC-1 must exercise the R4 "method unsound" branch. Feeding a
    # non-collapsing OBSERVED R_hat at full confounding must make NC-1 FAIL.
    from tracecausal.adversarial_oracle import negative_control_collinear
    nc_bad = negative_control_collinear(1.0, r_hat_observed=0.8)  # stays certified
    assert nc_bad.controls_silent is True            # controls still provably silent
    assert nc_bad.passes is False                    # but NC-1 fails -> route R4
    # the structurally-collapsing default still passes (the certifiable case).
    nc_ok = negative_control_collinear(1.0)
    assert nc_ok.passes is True


def test_axis_x_transferred_effect_is_derived_from_planted_tau():
    # finding 18: the designated span's transferred tau is DERIVED from tau_designated
    # (not a discarded computation). At xi=1 the misspecified reference cancels the
    # planted effect -> every designated transferred tau is exactly 0.
    fx = axis_x_confounded(1.0, n_examples=4, regime="blind")
    designated = [s.tau for ex in fx.examples for s in ex.spans if s.is_designated]
    assert designated  # there ARE designated spans
    assert all(math.isclose(t, 0.0, abs_tol=1e-12) for t in designated)
    # and the structural readout's r_hat_expected equals the mean of those (derived).
    assert math.isclose(fx.readout.r_hat_expected, sum(designated) / len(designated),
                        abs_tol=1e-12)
