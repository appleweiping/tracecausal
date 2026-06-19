"""Null-data harness for the CIU Phase-2 core (REDESIGN_v4 §5 test plan).

DO NOT RUN ON THE SERVER. These are **pure-Python unit fixtures** (no model, no
GPU, no server run): they prove the estimator and gates behave correctly on
*planted* synthetic data before any real run is authorized. The properties
asserted here are the falsifiable design predictions of REDESIGN_v4:

* a **random selector** yields ``U_hat ~= 0`` (Prop. 2.5a: the matched-null arm
  is unbiased for the pool mean, so selecting at random recovers no contrast);
* a **SHAM-MASK** (do-nothing renormalisation over the empty set on an inert,
  answer-disjoint span) yields ``U_hat ~= 0`` (G8 null);
* the **G5' novelty path** fires (selector beating / not beating the best adapted
  detector routes to ``useful_candidate`` / ``not_novel``, **never**
  ``invalidated``);
* the **G7 graded-oracle path** fires: Axis P degrades linearly and crosses the
  ``0.03`` floor at the planted point, Axis M under-recovers to ``1/m_c``, Axis D
  stays flat-at-zero for the detector-on-distractor.

Run later only when the user authorizes it: ``pytest tests/test_ciu_nulldata.py``.
``server.authorized`` stays false; nothing here loads a model.
"""

from __future__ import annotations

import math

from tracecausal.ciu import (
    LOCALIZATION_MARGIN,
    NECESSITY_MARGIN,
    NOVELTY_MARGIN,
    OOD_INERT_MANUFACTURE_BOUND,
    CIURecord,
    ciu_gate,
    leakage_slope_regression,
    ood_deflation,
    ood_slope_ci_excludes_manufacture,
    validate_ciu_record,
)
from tracecausal.interventions import Span, mask, patch, replay
from tracecausal.nuisance import (
    apply_kappa_fallback,
    claim_level_aggregate,
    estimate_kappa,
    estimate_sigma_u,
    pool_inflation,
    r_power,
    u_target,
)
from tracecausal.nullpool import (
    CandidateSpan,
    build_null_pool,
    pool_hash,
    proximity_bin,
    sample_matched_null,
)
from tracecausal.oracle_gen import (
    AXIS_D_DISTRACTOR_GRID,
    AXIS_M_CAUSE_GRID,
    AXIS_P_LEAKAGE_GRID,
    expected_u_hat,
    graded_oracle_family,
)


# ---------------------------------------------------------------------------
# Helpers: a tiny in-test estimator that mirrors U = mean_i(tau_i - bar_tau_i)
# on the planted oracle fixtures. No model; pure arithmetic on planted tau.
# ---------------------------------------------------------------------------


def _estimate_u_on_fixture(fixture, *, selector: str) -> float:
    """Compute U_hat on a graded-oracle fixture for a given selector.

    ``selector`` in {"oracle", "topk", "detector_on_distractor", "random"}. This
    reproduces the closed-form contrast the real CIU estimator targets, so the
    test asserts the estimator tracks the planted ground truth.
    """
    contrasts: list[float] = []
    for ex in fixture.examples:
        bar = ex.bar_tau_pool
        if selector == "oracle":
            tau_sel = max(s.tau for s in ex.spans if s.is_designated)
        elif selector == "topk":
            tau_sel = sum(s.tau for s in ex.spans if s.is_designated)
        elif selector == "detector_on_distractor":
            # The Axis-D probe: the detector that selects the *distractor* span
            # (inert, high detector signal). Identified as the highest-detector
            # span among the NON-designated spans, so it stays the distractor
            # regardless of the swept distractor strength d (REDESIGN_v4 §2.10G).
            inert = ex.inert()
            chosen = max(inert, key=lambda s: s.detector_signal)
            tau_sel = chosen.tau
        elif selector == "random":
            # a random selector recovers the pool mean in expectation
            inert = ex.inert()
            tau_sel = sum(s.tau for s in inert) / len(inert) if inert else 0.0
        else:  # pragma: no cover - guard
            raise ValueError(selector)
        contrasts.append(expected_u_hat(tau_sel, bar))
    return sum(contrasts) / len(contrasts)


# ---------------------------------------------------------------------------
# 1. Random selector -> U_hat ~= 0  (Prop. 2.5a)
# ---------------------------------------------------------------------------


def test_random_selector_u_hat_is_zero_on_clean_oracle():
    fixture = graded_oracle_family("clean", 0.0)
    u_random = _estimate_u_on_fixture(fixture, selector="random")
    assert abs(u_random) < 1e-9, "random selector must recover ~0 contrast"
    # and the oracle selector recovers ~1
    u_oracle = _estimate_u_on_fixture(fixture, selector="oracle")
    assert abs(u_oracle - 1.0) < 1e-9


def test_sample_matched_null_is_unbiased_for_pool_mean():
    # Build a small per-example pool of inert candidates and confirm a uniform
    # draw recovers the pool-mean structure (matched on budget/length/layers/ref).
    target = Span(0, 1)
    candidates = [
        CandidateSpan(Span(4, 5), layer_set=(3,), ref_hash="ref0", distance_to_answer=10),
        CandidateSpan(Span(6, 7), layer_set=(3,), ref_hash="ref0", distance_to_answer=11),
        CandidateSpan(Span(8, 9), layer_set=(3,), ref_hash="ref0", distance_to_answer=12),
        # wrong layer-set / length / ref must be excluded:
        CandidateSpan(Span(10, 11), layer_set=(4,), ref_hash="ref0", distance_to_answer=10),
        CandidateSpan(Span(12, 14), layer_set=(3,), ref_hash="ref0", distance_to_answer=10),
        CandidateSpan(Span(16, 17), layer_set=(3,), ref_hash="refX", distance_to_answer=10),
    ]
    pool = build_null_pool(
        "ex0", target, target_layer_set=(3,), target_ref_hash="ref0",
        target_distance_to_answer=10, candidates=candidates, proximity_bin_width=8,
    )
    assert pool.size == 3, "only budget/length/layer/ref/proximity-matched spans admitted"
    draws = sample_matched_null(pool, 50, seed=1)
    assert len(draws) == 50
    assert all(d in pool.members for d in draws)
    # deterministic given seed
    draws2 = sample_matched_null(pool, 50, seed=1)
    assert draws == draws2


def test_proximity_stratification_shrinks_pool_and_is_recorded():
    target = Span(0, 0)
    near = [
        CandidateSpan(Span(j, j), layer_set=(), ref_hash="r", distance_to_answer=1)
        for j in range(2, 8)
    ]
    far = [
        CandidateSpan(Span(j, j), layer_set=(), ref_hash="r", distance_to_answer=40)
        for j in range(10, 16)
    ]
    pool_strat = build_null_pool(
        "ex", target, target_layer_set=(), target_ref_hash="r",
        target_distance_to_answer=1, candidates=near + far, proximity_bin_width=4,
    )
    pool_unstrat = build_null_pool(
        "ex", target, target_layer_set=(), target_ref_hash="r",
        target_distance_to_answer=1, candidates=near + far, proximity_bin_width=0,
    )
    assert pool_strat.size < pool_unstrat.size
    assert pool_strat.proximity_bin == proximity_bin(1, 4)
    assert pool_strat.proximity_bin_width == 4
    # hash is stable and content-addressed
    assert pool_hash(pool_strat) == pool_hash(pool_strat)
    assert pool_hash(pool_strat) != pool_hash(pool_unstrat)


# ---------------------------------------------------------------------------
# 2. SHAM-MASK -> U_hat ~= 0  (G8 null); mask mechanics exact
# ---------------------------------------------------------------------------


def test_mask_renormalises_and_measures_displaced_mass():
    # Two query rows; mask key positions [1,2].
    weights = [
        [0.5, 0.2, 0.2, 0.1],
        [0.4, 0.1, 0.3, 0.2],
    ]
    res = mask(weights, Span(1, 2))
    # displaced mass = mean over rows of (w1 + w2)
    assert math.isclose(res.displaced_mass, ((0.2 + 0.2) + (0.1 + 0.3)) / 2)
    # surviving weights renormalise to 1 per row
    for row in res.renormalised_weights:
        assert math.isclose(sum(row), 1.0)
        assert row[1] == 0.0 and row[2] == 0.0  # masked positions zeroed
    assert res.edit_budget == 2
    assert not res.invalid


def test_sham_mask_empty_set_is_vacuous_and_yields_zero_effect():
    # SHAM-MASK do-nothing renormalisation: mask the empty set by masking a span
    # whose positions carry ~0 attention -> displaced_mass ~ 0, near_vacuous flag.
    weights = [[0.97, 0.0, 0.0, 0.03], [0.95, 0.0, 0.0, 0.05]]
    res = mask(weights, Span(1, 2))
    assert res.displaced_mass < 1e-3
    assert res.near_vacuous, "near-vacuous mask must be flagged (REDESIGN_v4 §2.12 guard)"
    # The renorm path still runs (isolating operator footprint) and rows still sum to 1.
    for row in res.renormalised_weights:
        assert math.isclose(sum(row), 1.0)


def test_mask_empty_attendable_set_is_invalid():
    weights = [[0.0, 0.6, 0.4, 0.0]]
    res = mask(weights, Span(1, 2))
    assert res.invalid and res.reason_code == "empty_attendable_set"


def test_patch_convex_interpolation_and_budget():
    res = patch(
        residual_states=[[1.0, 1.0], [1.0, 1.0]],
        reference_states=[[0.0, 0.0], [0.0, 0.0]],
        span=Span(0, 1),
        rho=0.25,
        layer_set=[5, 6],
    )
    assert res.patched_states[0] == (0.75, 0.75)  # (1-0.25)*1 + 0.25*0
    assert res.edit_budget == 2 * 2  # |span| * |layers|
    assert res.rho == 0.25 and res.n_layers == 2


def test_replay_plans_rollback_without_running():
    res = replay(Span(3, 5), reference_type="neutral", suffix_length=4)
    assert res.rollback_index == 3
    assert res.redecoded_positions == (3, 4, 5)
    assert res.edit_budget == 3
    assert not res.invalid


# ---------------------------------------------------------------------------
# 3. G5' novelty path fires; never returns invalidated for positive-U detector
# ---------------------------------------------------------------------------


def _base_record(**over) -> CIURecord:
    kwargs = dict(
        selector_id="ciu_causal",
        operator="mask",
        reference_type="factual",
        edit_budget=3,
        null_pool_hash="np",
        noop_run_hash="noop",
        evaluator_hash="ev",
        evaluator_kappa=0.92,
        ref_hash="ref",
        n_examples=850,
        r_int=16,
        b_boot=10_000,
        s_seed=20,
        u_hat=0.12,
        ci_low=0.07,
        ci_high=0.17,
        d_util=0.01,
        # Full paper-tier provenance (all three vectors aligned) so a require_v4
        # lock is satisfiable; the fail-closed provenance check rejects records
        # that omit any of these at lock.
        tau_per_example=(0.12, 0.12, 0.12, 0.12),
        pi_mean_per_example=(0.0, 0.0, 0.0, 0.0),
        matched_control_provenance=("c0", "c1", "c2", "c3"),
    )
    kwargs.update(over)
    return CIURecord(**kwargs)


def test_g5prime_useful_when_selector_beats_best_detector():
    # 0.12 - 0.05 = 0.07 >= 0.03; proximity_bin_width present so the leakage bound
    # (beta_hi * Delta_pos) is computable.
    rec = _base_record(best_detector_u=0.05, proximity_bin_width=1.0)
    # All four required v4 controls supplied -> useful_candidate (review fix 1:
    # required controls are HARD-required; a useful verdict needs the full set).
    verdict = ciu_gate(rec, best_detector_u=0.05, sham_u=0.0,
                       s_ood_ci=(0.0, 0.05), displaced_mass_range=0.4,
                       beta_hi=0.01, oracle_pass=True, graded_curve_pass=True)
    assert verdict == "useful_candidate"


def test_useful_candidate_requires_all_controls_present():
    # Review fix 1: omitting a required control must withhold useful_candidate
    # (downgrade to diagnostic), even when the scalar + novelty gates pass.
    rec = _base_record(best_detector_u=0.05, proximity_bin_width=1.0)
    # Missing OOD slope CI + leakage bound -> diagnostic, not useful.
    assert ciu_gate(rec, best_detector_u=0.05, sham_u=0.0,
                    oracle_pass=True, graded_curve_pass=True) == "diagnostic"
    # Missing graded-oracle pass -> diagnostic.
    assert ciu_gate(rec, best_detector_u=0.05, sham_u=0.0,
                    s_ood_ci=(0.0, 0.05), displaced_mass_range=0.4,
                    beta_hi=0.01) == "diagnostic"
    # Full control set present -> useful.
    assert ciu_gate(rec, best_detector_u=0.05, sham_u=0.0,
                    s_ood_ci=(0.0, 0.05), displaced_mass_range=0.4,
                    beta_hi=0.01, oracle_pass=True,
                    graded_curve_pass=True) == "useful_candidate"
    # require_controls=False restores the isolated-probe behaviour for focused
    # unit tests (no control omission downgrade).
    assert ciu_gate(rec, best_detector_u=0.05, sham_u=0.0,
                    oracle_pass=True, graded_curve_pass=True,
                    require_controls=False) == "useful_candidate"


def test_g5prime_not_novel_when_detector_matches_within_margin():
    rec = _base_record(best_detector_u=0.11)  # 0.12 - 0.11 = 0.01 < 0.03
    verdict = ciu_gate(rec, best_detector_u=0.11)
    assert verdict == "not_novel", "novelty downgrade, NOT an identification failure"


def test_positive_u_detector_never_invalidates():
    # A detector with strong positive U beating us: expected, never 'invalidated'.
    rec = _base_record(best_detector_u=0.30)
    verdict = ciu_gate(rec, best_detector_u=0.30)
    assert verdict in {"not_novel", "diagnostic", "useful_candidate"}
    assert verdict != "invalidated"  # there is no such verdict by design


def test_scalar_gate_failure_routes_to_diagnostic():
    rec = _base_record(u_hat=0.02, ci_low=-0.01, ci_high=0.05, best_detector_u=0.0)
    verdict = ciu_gate(rec, best_detector_u=0.0)
    assert verdict == "diagnostic"  # below necessity margin


def test_sham_positive_downgrades_to_diagnostic():
    rec = _base_record(best_detector_u=0.0)
    verdict = ciu_gate(rec, best_detector_u=0.0, sham_u=0.06,
                       oracle_pass=True, graded_curve_pass=True)
    assert verdict == "diagnostic"  # SHAM manufactured a signal -> operator artifact


def test_g1_evaluated_on_deflated_estimate():
    # Raw u_hat clears the margin but the deflated value does not -> diagnostic.
    rec = _base_record(u_hat=0.10, u_deflated=0.02, ci_low=0.05, ci_high=0.15,
                       best_detector_u=0.0)
    assert rec.gated_u == 0.02
    verdict = ciu_gate(rec, best_detector_u=0.0)
    assert verdict == "diagnostic"


def test_g1_uses_propagated_deflated_ci_low_not_raw_shift():
    # review fix 2: the un-widened shift would CLEAR necessity, but the PROPAGATED
    # deflated lower CI (slope uncertainty in quadrature) does not -> diagnostic.
    # Build the propagated CI via ood_deflation and persist it on the record.
    u_def, ci = ood_deflation(
        0.12, displaced_mass=0.4, slope=0.05, intercept=0.0,
        u_hat_ci=(0.07, 0.17), slope_ci=(0.0, 0.30),  # wide slope CI -> wide deflated CI
    )
    assert ci is not None
    # The bare shift of ci_low would be 0.07 - (0.12 - u_def) = above the margin,
    # but the propagated lower CI is pulled below NECESSITY_MARGIN.
    shifted_low = 0.07 - (0.12 - u_def)
    assert shifted_low >= NECESSITY_MARGIN  # un-widened shift would have cleared
    assert ci[0] < NECESSITY_MARGIN          # propagated lower CI does not clear
    rec = _base_record(u_hat=0.12, u_deflated=u_def, u_deflated_ci_low=ci[0],
                       ci_low=0.07, ci_high=0.17, best_detector_u=0.0)
    assert rec.gated_ci_low == ci[0]  # gate reads the propagated bound
    assert ciu_gate(rec, best_detector_u=0.0, require_controls=False) == "diagnostic"
    # And the runtime arg path uses the propagated lower CI too.
    rec_raw = _base_record(u_hat=0.12, ci_low=0.07, ci_high=0.17, best_detector_u=0.0)
    assert ciu_gate(rec_raw, best_detector_u=0.0, u_deflated_ci_low=ci[0],
                    require_controls=False) == "diagnostic"


# ---------------------------------------------------------------------------
# 4. G7 graded-oracle path: planted curves on all three axes
# ---------------------------------------------------------------------------


def test_axis_p_degrades_linearly_and_crosses_floor():
    prev = None
    for leak in AXIS_P_LEAKAGE_GRID:
        fx = graded_oracle_family("partial_leakage", leak)
        u = _estimate_u_on_fixture(fx, selector="oracle")
        # pre-registered: U_hat(oracle) = 1 - tau_inert
        assert math.isclose(u, 1.0 - leak, abs_tol=1e-9)
        assert math.isclose(fx.expected_u_hat_selector, 1.0 - leak, abs_tol=1e-9)
        if prev is not None:
            assert u <= prev + 1e-12, "Axis P must be monotone non-increasing"
        prev = u
        # crossing flag matches the planted 0.03 point
        assert fx.crosses_gate_floor == (leak >= LOCALIZATION_MARGIN)


def test_axis_m_underrecovers_to_one_over_mc():
    for m_c in AXIS_M_CAUSE_GRID:
        fx = graded_oracle_family("multi_cause", m_c)
        u_single = _estimate_u_on_fixture(fx, selector="oracle")  # single-span pick
        u_topk = _estimate_u_on_fixture(fx, selector="topk")
        assert math.isclose(u_single, 1.0 / m_c, abs_tol=1e-9), "single-span -> 1/m_c"
        assert math.isclose(u_topk, 1.0, abs_tol=1e-9), "top-k selector recovers 1"


def test_axis_d_detector_on_distractor_stays_zero():
    for d in AXIS_D_DISTRACTOR_GRID:
        fx = graded_oracle_family("distractor", d)
        u_detector = _estimate_u_on_fixture(fx, selector="detector_on_distractor")
        u_selector = _estimate_u_on_fixture(fx, selector="oracle")
        assert math.isclose(u_detector, 0.0, abs_tol=1e-9), "detector picks inert distractor"
        assert math.isclose(u_selector, 1.0, abs_tol=1e-9)
        # G5' selector-minus-detector difference stays >= 0.03 across strengths
        assert (u_selector - u_detector) >= NOVELTY_MARGIN


# ---------------------------------------------------------------------------
# 5. validate_ciu_record: contract + v4 G7/G8 checks fire only when populated
# ---------------------------------------------------------------------------


def test_validate_rejects_server_authorized_and_empty_hashes():
    rec = _base_record(server_authorized=True, null_pool_hash="")
    errors = validate_ciu_record(rec)
    assert any("server_authorized" in e for e in errors)
    assert any("null_pool_hash" in e for e in errors)


def test_validate_rejects_seed_floor_and_mde():
    rec = _base_record(s_seed=10)
    assert any("s_seed" in e for e in validate_ciu_record(rec))
    rec2 = _base_record(n_examples=100)
    assert any("MDE" in e for e in validate_ciu_record(rec2, mde_min_n=850))


def test_validate_v4_g7_upper_ci_leakage_bound():
    # beta_hi * Delta_pos must be < 0.03
    bad = _base_record(beta_hi=0.05, proximity_bin_width=1.0)  # 0.05 >= 0.03 -> fail
    assert any("G7" in e for e in validate_ciu_record(bad))
    good = _base_record(beta_hi=0.01, proximity_bin_width=2.0)  # 0.02 < 0.03 -> ok
    assert not any("G7" in e for e in validate_ciu_record(good))


def test_validate_v3_record_without_v4_fields_still_passes():
    rec = _base_record()  # no v4 fields populated
    assert validate_ciu_record(rec) == []


# ---------------------------------------------------------------------------
# 6. OOD deflation + leakage-slope regression (v4 additive helpers)
# ---------------------------------------------------------------------------


def test_ood_deflation_subtracts_calibrated_footprint():
    # artifact line: art = 0.1 * displaced_mass + 0.0; at d*=0.4 -> art=0.04
    u_def, ci = ood_deflation(
        0.10, displaced_mass=0.4, slope=0.1, intercept=0.0,
        u_hat_ci=(0.06, 0.14), slope_ci=(0.08, 0.12),
    )
    assert math.isclose(u_def, 0.10 - 0.04, abs_tol=1e-12)
    assert ci is not None and ci[0] < u_def < ci[1]


def test_leakage_slope_regression_recovers_planted_slope():
    # planted: delta = 0.02 * proximity + 0.01
    proximity = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    delta = [0.02 * p + 0.01 for p in proximity]
    beta_hat, beta_lo, beta_hi = leakage_slope_regression(
        delta, proximity, n_bootstrap=500, seed=0
    )
    assert math.isclose(beta_hat, 0.02, abs_tol=1e-9)
    assert beta_lo <= beta_hat <= beta_hi


# ---------------------------------------------------------------------------
# 7. Nuisance estimators + the §4.7 exhibited feasible point arithmetic
# ---------------------------------------------------------------------------


def test_r_power_reproduces_exhibited_feasible_point():
    # REDESIGN_v4 §4.7: (z*sigma/margin)^2 * infl, sigma=0.30, z=2.734, margin=0.03,
    # infl=1.125 -> 747.5 * 1.125 = 841 (rounded to n=850 in config).
    n = r_power(0.30, z=2.734, margin=0.03, infl=1.125)
    assert n == 841
    # attenuation target: 0.05 / (2*0.92 - 1) = 0.0595
    assert math.isclose(u_target(0.05, 0.92), 0.0595, abs_tol=1e-4)


def test_pool_inflation_matches_one_plus_one_over_mean():
    assert math.isclose(pool_inflation([8, 8, 8, 8]), 1.125)


def test_sigma_and_kappa_estimators_return_point_and_ci():
    contrasts = [0.1, -0.2, 0.0, 0.3, -0.1, 0.2, 0.05, -0.15, 0.1, 0.0]
    se = estimate_sigma_u(contrasts, n_bootstrap=300, seed=0)
    assert se.sigma_lo <= se.sigma_hat <= se.sigma_hi
    assert se.n_val == len(contrasts) and not se.meets_min_n  # below 200
    a = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1]
    b = [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]
    ke = estimate_kappa(a, b, n_bootstrap=300, seed=0)
    assert ke.kappa_lo <= ke.kappa_hat <= ke.kappa_hi
    assert not ke.meets_min_n  # below 300


# ---------------------------------------------------------------------------
# 8. Review fixes: lower-CI gate, SHAM CI conjunct, G8 kill-gate, v4 hard fields,
#    crosses-floor disambiguation, kappa claim-level fallback.
# ---------------------------------------------------------------------------


def test_gate_requires_ci_lower_bound_not_point_estimate():
    # Point estimate clears 0.05 but the CI LOWER bound does not -> diagnostic
    # (review fix 1: gate on the Holm/MDE lower bound, not the point).
    rec = _base_record(u_hat=0.12, ci_low=0.02, ci_high=0.22, best_detector_u=0.0)
    assert rec.gated_ci_low == 0.02
    assert ciu_gate(rec, best_detector_u=0.0) == "diagnostic"
    # Lower bound clears -> useful (focused on the scalar gate; control-presence
    # enforcement is isolated out via require_controls=False — see the dedicated
    # test_useful_candidate_requires_all_controls_present).
    rec2 = _base_record(u_hat=0.12, ci_low=0.06, ci_high=0.18, best_detector_u=0.0)
    assert ciu_gate(rec2, best_detector_u=0.0, sham_u_ci=(-0.01, 0.01),
                    oracle_pass=True, graded_curve_pass=True,
                    require_controls=False) == "useful_candidate"


def test_gate_sham_ci_conjunct_blocks_artifact_in_open_interval():
    # An operator artifact CI sitting entirely in (0, 0.03) must NOT pass silently
    # (review fix 4: test the SHAM CI lo<=0<=hi, not sham_u >= 0.03).
    rec = _base_record(u_hat=0.12, ci_low=0.07, ci_high=0.17, best_detector_u=0.0)
    # CI entirely above 0 -> artifact -> diagnostic
    assert ciu_gate(rec, best_detector_u=0.0, sham_u_ci=(0.005, 0.02)) == "diagnostic"
    # CI brackets 0 -> clean null -> useful (focused on the SHAM CI conjunct;
    # control-presence enforcement isolated via require_controls=False).
    assert ciu_gate(rec, best_detector_u=0.0, sham_u_ci=(-0.01, 0.02),
                    oracle_pass=True, graded_curve_pass=True,
                    require_controls=False) == "useful_candidate"


def test_ood_slope_kill_gate_helper_and_gate():
    # range D=0.4; bound 0.05. A CI of (0.0, 0.1) -> worst 0.1*0.4=0.04 < 0.05 OK.
    assert ood_slope_ci_excludes_manufacture((0.0, 0.1), 0.4) is True
    # CI (0.0, 0.2) -> 0.2*0.4 = 0.08 >= 0.05 -> NOT excluded (kill).
    assert ood_slope_ci_excludes_manufacture((0.0, 0.2), 0.4) is False
    # Negative-side slope counts too.
    assert ood_slope_ci_excludes_manufacture((-0.2, 0.0), 0.4) is False
    # Runtime gate applies it (review fix 5: s_ood_ci arg on ciu_gate).
    rec = _base_record(u_hat=0.12, ci_low=0.07, ci_high=0.17, best_detector_u=0.0)
    assert ciu_gate(rec, best_detector_u=0.0, sham_u_ci=(-0.01, 0.01),
                    s_ood_ci=(0.0, 0.2), displaced_mass_range=0.4) == "diagnostic"
    # Focused on the OOD kill-gate; other-control presence isolated out.
    assert ciu_gate(rec, best_detector_u=0.0, sham_u_ci=(-0.01, 0.01),
                    s_ood_ci=(0.0, 0.05), displaced_mass_range=0.4,
                    oracle_pass=True, graded_curve_pass=True,
                    require_controls=False) == "useful_candidate"


def test_gate_g7_leakage_upper_ci_bound_runtime_arg():
    rec = _base_record(u_hat=0.12, ci_low=0.07, ci_high=0.17, best_detector_u=0.0,
                       proximity_bin_width=1.0)
    # beta_hi*Delta_pos = 0.05 >= 0.03 -> diagnostic
    assert ciu_gate(rec, best_detector_u=0.0, sham_u_ci=(-0.01, 0.01),
                    beta_hi=0.05) == "diagnostic"
    # 0.02 < 0.03 -> useful (focused on the leakage bound; other-control presence
    # isolated via require_controls=False).
    assert ciu_gate(rec, best_detector_u=0.0, sham_u_ci=(-0.01, 0.01), beta_hi=0.02,
                    oracle_pass=True, graded_curve_pass=True,
                    require_controls=False) == "useful_candidate"


def test_validate_v4_g8_kill_gate_in_record_path():
    # ood_slope_ci admits a manufacturing slope over displaced_mass_range -> G8 fail.
    bad = _base_record(ood_slope_ci=(0.0, 0.2), displaced_mass_range=0.4)
    errors = validate_ciu_record(bad)
    assert any("G8 fail" in e for e in errors)
    good = _base_record(ood_slope_ci=(0.0, 0.05), displaced_mass_range=0.4)
    assert not any("G8 fail" in e for e in validate_ciu_record(good))


def test_validate_v4_fields_all_or_nothing_and_hard_required():
    # Partial G8 conjunct (slope CI without displaced_mass_range) is a violation.
    partial = _base_record(ood_slope_ci=(0.0, 0.01))
    assert any("G8 OOD conjunct partially populated" in e for e in validate_ciu_record(partial))
    # require_v4 hard-requires the full G7/G8 field set.
    rec = _base_record()
    errs = validate_ciu_record(rec, require_v4=True)
    assert any("v4 lock requires" in e for e in errs)
    # review fix 3: require_v4 also hard-requires the persisted control OUTCOMES
    # (SHAM/no-op null CIs, graded-oracle pass, m_pool floor), not just leakage/OOD.
    g7g8_only = _base_record(
        beta_hi=0.01, proximity_bin_width=2.0,
        ood_slope_ci=(0.0, 0.05), displaced_mass_range=0.4,
    )
    errs2 = validate_ciu_record(g7g8_only, require_v4=True)
    assert any("sham_u_ci" in e for e in errs2)
    assert any("graded_curve_pass" in e for e in errs2)
    assert any("m_pool_mean" in e for e in errs2)
    # A complete v4 record (G7/G8 fields + all controls passing) passes require_v4.
    full = _base_record(
        beta_hi=0.01, proximity_bin_width=2.0,
        ood_slope_ci=(0.0, 0.05), displaced_mass_range=0.4,
        sham_u_ci=(-0.01, 0.01), noop_u_ci=(-0.01, 0.01),
        oracle_pass=True, graded_curve_pass=True, m_pool_mean=8.0,
    )
    assert validate_ciu_record(full, require_v4=True) == []
    # A v4 record whose SHAM null CI sits off zero FAILS the lock (control enforced).
    sham_off = _base_record(
        beta_hi=0.01, proximity_bin_width=2.0,
        ood_slope_ci=(0.0, 0.05), displaced_mass_range=0.4,
        sham_u_ci=(0.01, 0.03), noop_u_ci=(-0.01, 0.01),
        oracle_pass=True, graded_curve_pass=True, m_pool_mean=8.0,
    )
    assert any("SHAM-MASK null CI must bracket zero" in e
               for e in validate_ciu_record(sham_off, require_v4=True))


def test_v4_lock_requires_nonempty_per_example_provenance():
    # Paper-tier / lockable provenance is FAIL-CLOSED: a v4 lock that omits any of
    # the per-example provenance vectors must be rejected (review fix 2). Start from
    # an otherwise-complete passing v4 record and strip provenance.
    full_kwargs = dict(
        beta_hi=0.01, proximity_bin_width=2.0,
        ood_slope_ci=(0.0, 0.05), displaced_mass_range=0.4,
        sham_u_ci=(-0.01, 0.01), noop_u_ci=(-0.01, 0.01),
        oracle_pass=True, graded_curve_pass=True, m_pool_mean=8.0,
    )
    # Empty matched-control provenance -> lock fails closed.
    no_prov = _base_record(matched_control_provenance=(), **full_kwargs)
    errs = validate_ciu_record(no_prov, require_v4=True)
    assert any("non-empty matched_control_provenance" in e for e in errs)
    # Empty tau / pi vectors also fail the lock.
    no_tau = _base_record(tau_per_example=(), **full_kwargs)
    assert any("non-empty tau_per_example" in e
               for e in validate_ciu_record(no_tau, require_v4=True))
    no_pi = _base_record(pi_mean_per_example=(), **full_kwargs)
    assert any("non-empty pi_mean_per_example" in e
               for e in validate_ciu_record(no_pi, require_v4=True))
    # Misaligned vector lengths are caught regardless of the lock.
    misaligned = _base_record(matched_control_provenance=("c0", "c1"))
    assert any("provenance vectors must align in length" in e
               for e in validate_ciu_record(misaligned))
    # A v3 record (require_v4=False) with empty provenance still validates
    # (back-compatible: the fail-closed requirement only fires at lock).
    v3 = _base_record(
        tau_per_example=(), pi_mean_per_example=(), matched_control_provenance=()
    )
    assert validate_ciu_record(v3) == []


def test_crosses_gate_floor_disambiguated_leakage_vs_selector():
    # Axis P: crosses_gate_floor is the LEAKAGE/bias crossing (tau_inert >= 0.03).
    fx = graded_oracle_family("partial_leakage", 0.04)
    assert fx.crosses_gate_floor is True            # bias term 0.04 >= 0.03
    assert fx.crosses_selector_floor is False        # U_hat = 0.96, well above 0.03
    fx0 = graded_oracle_family("partial_leakage", 0.0)
    assert fx0.crosses_gate_floor is False
    # Axis M: NO leakage term -> crosses_gate_floor must be False; the 1/m_c
    # selector-curve crossing is tracked separately (was conflated before).
    fx_m4 = graded_oracle_family("multi_cause", 4)   # U_hat = 0.25
    assert fx_m4.crosses_gate_floor is False
    assert fx_m4.crosses_selector_floor is False     # 0.25 > 0.03
    # (No m_c in the grid drives 1/m_c <= 0.03, but the field exists and is honest.)


def test_kappa_counts_degenerate_resamples_not_silently_dropped():
    # Mostly-agreeing labels with one disagreement: the point kappa is defined,
    # but some bootstrap resamples draw an all-identical sample (chance agreement
    # 1 -> kappa undefined). Those degenerate resamples must be COUNTED, not
    # silently dropped (review fix 9).
    a = [1, 1, 1, 1, 1, 0]
    b = [1, 1, 1, 1, 1, 1]
    ke = estimate_kappa(a, b, n_bootstrap=200, seed=0)
    assert ke.n_bootstrap == 200
    assert ke.n_degenerate_resamples > 0  # some resamples degenerated and were counted
    # The CI still came from the non-degenerate resamples (not silently empty).
    assert ke.kappa_lo <= ke.kappa_hat <= ke.kappa_hi


def test_kappa_claim_level_fallback_recovers_and_blocks():
    # Item-level kappa low; claim-level aggregation raises agreement above 0.90.
    # 4 claims x 3 items: raters disagree on a few items but agree per-claim.
    labels_a = [1, 1, 0,  1, 1, 1,  0, 0, 0,  1, 0, 1]
    labels_b = [1, 0, 0,  1, 1, 1,  0, 1, 0,  1, 0, 1]
    claim_ids = ["c1", "c1", "c1", "c2", "c2", "c2", "c3", "c3", "c3", "c4", "c4", "c4"]
    agg_a, agg_b = claim_level_aggregate(labels_a, labels_b, claim_ids)
    assert len(agg_a) == len(agg_b) == 4  # one outcome per claim
    item = estimate_kappa(labels_a, labels_b, n_bootstrap=200, seed=0)
    fb = apply_kappa_fallback(
        item, labels_a=labels_a, labels_b=labels_b, claim_ids=claim_ids,
        n_bootstrap=200, seed=0,
    )
    assert fb.used_claim_level == (item.kappa_lo < 0.90)
    assert isinstance(fb.blocks_lock, bool)
    # Missing claim data with low item kappa BLOCKS the lock (no silent proceed).
    low = estimate_kappa([1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], n_bootstrap=200, seed=0)
    assert low.kappa_lo < 0.90
    fb2 = apply_kappa_fallback(low)
    assert fb2.blocks_lock is True


def test_necessity_margin_constant_is_005():
    assert NECESSITY_MARGIN == 0.05
    assert OOD_INERT_MANUFACTURE_BOUND == 0.05


def test_kappa_handles_constant_label_inputs_gracefully():
    # Opus minor: constant-label inputs (chance agreement 1.0) must NOT raise.
    # Perfect agreement on a constant label -> kappa_hat = 1.0.
    ke = estimate_kappa([1, 1, 1, 1], [1, 1, 1, 1], n_bootstrap=50, seed=0)
    assert ke.kappa_hat == 1.0
    # One rater constant, the other not: defined, does not raise.
    ke2 = estimate_kappa([1, 1, 1, 1], [1, 0, 1, 1], n_bootstrap=50, seed=0)
    assert -1.0 <= ke2.kappa_hat <= 1.0
    # Both constant but disagreeing across raters -> graceful 0.0 (no raise).
    ke3 = estimate_kappa([0, 0, 0, 0], [1, 1, 1, 1], n_bootstrap=50, seed=0)
    assert ke3.kappa_hat == 0.0


def test_validate_evaluator_kappa_cross_check_above_chance():
    # Opus minor: evaluator_kappa must be > 0.5 (above chance) so the (2*kappa-1)
    # attenuation band is defined; <= 0.5 is flagged.
    at_chance = _base_record(evaluator_kappa=0.5)
    assert any("above chance" in e for e in validate_ciu_record(at_chance))
    below = _base_record(evaluator_kappa=0.3)
    assert any("above chance" in e for e in validate_ciu_record(below))
    ok = _base_record(evaluator_kappa=0.92)
    assert not any("above chance" in e for e in validate_ciu_record(ok))


def test_necessity_gate_inclusive_at_margin():
    # Opus minor (ge vs gt): a lower-CI bound landing EXACTLY on 0.05 clears G1
    # (inclusive convention). pi_mean is 0 so gated necessity == gated_ci_low.
    rec = _base_record(u_hat=0.10, ci_low=0.05, ci_high=0.15, best_detector_u=0.0,
                       proximity_bin_width=1.0)
    assert rec.gated_ci_low == 0.05
    assert ciu_gate(rec, best_detector_u=0.0, sham_u_ci=(-0.01, 0.01),
                    s_ood_ci=(0.0, 0.05), displaced_mass_range=0.4,
                    beta_hi=0.01,
                    oracle_pass=True, graded_curve_pass=True) == "useful_candidate"
