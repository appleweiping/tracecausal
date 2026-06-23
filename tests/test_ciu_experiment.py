"""Pure-python unit harness for the REAL CIU primary-cell runner (NO GPU/model/network).

DO NOT RUN ON THE SERVER for the heavy path. These tests exercise ONLY the
no-GPU pieces of ``scripts/run_ciu_experiment.py``:

* the matched-null span sampler (budget/length/position matching + disjointness),
  which wraps the implemented ``tracecausal.nullpool``;
* the CIU ``U_hat`` estimand + the **G1 necessity gate** (via the implemented
  ``tracecausal.ciu.ciu_gate``) on synthetic per-example ``(tau_targeted, tau_null)``
  arrays with hand-checkable values;
* the factuality normalizer / EM scorer on crafted strings;
* the salience-proxy span selector.

The model load, dataset load, and activation-patching forward pass are NEVER
imported here (they are lazy-imported only inside the authorized branch of the
runner), so this whole module runs with no torch/transformers/datasets present.

Run: ``cd /d/Research/tracecausal && C:/Python314/python -m pytest -q tests/test_ciu_experiment.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts/ and src/ importable (the runner lives in scripts/, kernels in src/).
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "scripts", _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_ciu_experiment as rce  # noqa: E402


# =========================================================================== #
# (A) Factuality normalizer / answer-recall scorer on crafted strings.
#
# nq_open's gold ``answer`` is a LIST of acceptable short-answer strings; the model
# answers in free text. The scorer is the standard open-domain-QA ANSWER-RECALL
# metric: a case is factual iff ANY normalized gold is a whitespace-token block
# (contiguous token substring) of the normalized prediction -- i.e. the prediction
# CONTAINS a gold. This credits correct-but-differently-phrased / superset answers
# instead of demanding a brittle exact match.
# =========================================================================== #
def test_normalize_answer_lowercases_strips_punct_articles_whitespace():
    assert rce.normalize_answer("The Beatles!") == "beatles"
    assert rce.normalize_answer("  A   Tale,  of   Two Cities. ") == "tale of two cities"
    assert rce.normalize_answer("AN apple") == "apple"
    # idempotent
    once = rce.normalize_answer("The Quick Brown Fox.")
    assert rce.normalize_answer(once) == once


def test_is_factual_answer_recall_contains_gold():
    # the load-bearing relaxed-scorer cases from the bug report:
    # a free-text answer that CONTAINS a gold short answer is factual.
    assert rce.is_factual("The capital is Paris.", ["Paris"]) is True
    assert rce.is_factual("London", ["Paris"]) is False
    # superset / differently-phrased free-text answers still recall the gold
    assert rce.is_factual("The author is J.K. Rowling", ["J. K. Rowling"]) is True
    assert rce.is_factual("new york city", ["New York"]) is True
    assert rce.is_factual("It was written in 1997.", ["1997"]) is True


def test_is_factual_recall_against_alias_list():
    # nq_open-style alias list: factual iff ANY alias is recalled.
    assert rce.is_factual("the beatles", ["Beatles", "The Beatles"]) is True
    assert rce.is_factual("The   BEATLES.", "the beatles") is True
    assert rce.is_factual("the answer is the rolling stones", ["The Beatles"]) is False
    # one matching alias in a list of several suffices
    assert rce.is_factual("born in Honolulu, Hawaii", ["Hawaii", "Kenya"]) is True


def test_is_factual_token_substring_not_subword():
    # answer-recall is at the WHITESPACE-TOKEN level, not raw str.__contains__, so a
    # gold must occur as a full contiguous token block -- never a sub-word fragment.
    assert rce.is_factual("warsaw is the capital", ["war"]) is False
    assert rce.is_factual("the answer is category", ["cat"]) is False
    # multi-token gold must appear contiguously and in order
    assert rce.is_factual("york new is wrong order", ["New York"]) is False
    assert rce.is_factual("the answer is New York", ["New York"]) is True


def test_is_factual_blank_is_never_factual():
    assert rce.is_factual("", ["anything"]) is False
    assert rce.is_factual("   ", ["anything"]) is False
    assert rce.is_factual("...", ["anything"]) is False  # punctuation-only -> blank


def test_is_factual_handles_empty_or_whitespace_golds():
    assert rce.is_factual("paris", ["", "  ", "Paris"]) is True
    assert rce.is_factual("paris", ["", "  "]) is False
    # a blank gold must never trivially "contain"-match a non-blank prediction
    assert rce.is_factual("any answer", ["", "   ", "..."]) is False


def test_factuality_score_is_binary_proper_score():
    assert rce.factuality_score("The capital is Paris", ["paris"]) == 1.0
    assert rce.factuality_score("London", ["paris"]) == 0.0
    assert isinstance(rce.factuality_score("Paris", ["paris"]), float)


# =========================================================================== #
# (B) Matched-null span sampling (budget / length / position matching).
# =========================================================================== #
def _candidates(specs):
    return [rce.SpanCandidate(a=a, b=b, distance_to_answer=d) for (a, b, d) in specs]


def test_matched_null_pool_matches_length_and_position_and_excludes_target():
    # target S* = [10, 13] (length 4), distance-to-answer 2 -> bin 2//4 = 0
    target_a, target_b, target_dist = 10, 13, 2
    candidates = _candidates([
        (0, 3, 1),    # len 4, dist 1 -> bin 0 : MATCH (disjoint from S*)
        (4, 7, 3),    # len 4, dist 3 -> bin 0 : MATCH
        (20, 23, 5),  # len 4, dist 5 -> bin 1 : reject (wrong proximity bin)
        (0, 2, 1),    # len 3        : reject (wrong length/budget)
        (12, 15, 2),  # len 4 but overlaps S* [10,13] : reject (not disjoint)
    ])
    drawn = rce.sample_matched_null_spans(
        target_a, target_b, target_dist, candidates,
        proximity_bin_width=4, n_draws=50, seed=1,
    )
    # every drawn span must be one of the two valid matched controls
    drawn_keys = {(s.a, s.b) for s in drawn}
    assert drawn_keys.issubset({(0, 3), (4, 7)})
    # both valid members appear across 50 draws (uniform-with-replacement)
    assert drawn_keys == {(0, 3), (4, 7)}
    # length / budget matched exactly to S*
    for s in drawn:
        assert s.length == 4


def test_matched_null_empty_pool_raises():
    # only a wrong-length candidate -> empty matched pool -> ValueError
    target_a, target_b, target_dist = 10, 13, 2
    candidates = _candidates([(0, 2, 1)])  # length 3, never matches budget 4
    with pytest.raises(ValueError):
        rce.sample_matched_null_spans(
            target_a, target_b, target_dist, candidates,
            proximity_bin_width=4, n_draws=4, seed=0,
        )


def test_matched_null_draw_is_deterministic_given_seed():
    target_a, target_b, target_dist = 10, 13, 2
    candidates = _candidates([(0, 3, 1), (4, 7, 3)])
    d1 = rce.sample_matched_null_spans(
        target_a, target_b, target_dist, candidates,
        proximity_bin_width=4, n_draws=10, seed=7,
    )
    d2 = rce.sample_matched_null_spans(
        target_a, target_b, target_dist, candidates,
        proximity_bin_width=4, n_draws=10, seed=7,
    )
    assert [(s.a, s.b) for s in d1] == [(s.a, s.b) for s in d2]


# --------------------------------------------------------------------------- #
# (B2) Empty-null-pool fix: NON-OVERLAPPING candidate grid + coarsened proximity
# bin yield a NON-EMPTY matched-null pool for a realistic-length tokenized prompt.
# The bug: the old fully-overlapping (stride-1) enumeration meant every same-bin
# candidate overlapped S*, so the disjointness filter deleted the whole pool -> the
# runner skipped every example as "skipped_empty_null_pool".
# --------------------------------------------------------------------------- #
def test_effective_proximity_bin_width_coarsens_to_hold_disjoint_controls():
    # auto (<=0) -> 4*budget; an explicit too-tight width is coarsened up to 2*budget;
    # an already-wide explicit width is used as-is.
    assert rce.effective_proximity_bin_width(0, 4) == 16
    assert rce.effective_proximity_bin_width(4, 4) == 8     # 4 < 2*budget -> 8
    assert rce.effective_proximity_bin_width(20, 4) == 20   # already >= 2*budget


def test_enumerate_candidate_spans_nonoverlapping_grid_is_default():
    # default stride == budget -> non-overlapping windows [0,1],[2,3],[4,5]
    cands = rce.enumerate_candidate_spans(prompt_len=6, answer_index=5, budget=2)
    assert [(c.a, c.b) for c in cands] == [(0, 1), (2, 3), (4, 5)]
    # neighbouring grid windows are pairwise disjoint (the property the null pool needs)
    for x, y in zip(cands, cands[1:]):
        assert x.b < y.a
    # stride=1 reproduces the old dense fully-overlapping enumeration
    dense = rce.enumerate_candidate_spans(prompt_len=6, answer_index=5, budget=2, stride=1)
    assert len(dense) == 5


def test_span_enumeration_yields_nonempty_matched_null_pool_for_realistic_prompt():
    # A realistic short closed-book QA prompt: ~35 tokens after the chat template.
    prompt_len = 35
    budget = 4
    answer_index = prompt_len - 1
    # S* picked by the salience proxy somewhere mid-prompt (here a fixed peak window).
    s_a, s_b = 16, 19
    target_dist = abs(answer_index - s_b)
    # the runner's exact wiring: non-overlapping grid + coarsened proximity bin.
    candidates = rce.enumerate_candidate_spans(
        prompt_len - 1, answer_index, budget=budget, prompt_start=0, stride=budget
    )
    eff_bin = rce.effective_proximity_bin_width(0, budget)  # auto
    drawn = rce.sample_matched_null_spans(
        s_a, s_b, target_dist, candidates,
        proximity_bin_width=eff_bin, n_draws=8, seed=3, layer_set=(12, 13, 14, 15),
    )
    # NON-EMPTY pool: several matched controls drawn (the bug was 0 -> all skipped).
    assert len(drawn) == 8
    # every drawn control is budget-length, disjoint from S*, and a real other location
    for s in drawn:
        assert s.length == budget
        assert s.b < s_a or s.a > s_b  # disjoint from S*=[16,19]
    assert len({(s.a, s.b) for s in drawn}) >= 1


def test_old_buggy_config_empties_the_pool():
    # Regression guard reproducing the ORIGINAL bug: the dense stride-1 enumeration
    # together with the original TIGHT proximity bin (width 4 == budget) empties the
    # pool for this realistic prompt -- every same-bin window is a one-token shift of
    # S* and so overlaps it, and the disjointness filter deletes them all. This is
    # exactly the "skipped_empty_null_pool" path the fix removes (the fix coarsens the
    # bin via effective_proximity_bin_width AND uses a non-overlapping grid).
    prompt_len = 35
    budget = 4
    answer_index = prompt_len - 1
    s_a, s_b = 16, 19
    target_dist = abs(answer_index - s_b)
    dense = rce.enumerate_candidate_spans(
        prompt_len - 1, answer_index, budget=budget, prompt_start=0, stride=1
    )
    # original tight bin (width == budget) -> empty pool (the bug)
    with pytest.raises(ValueError):
        rce.sample_matched_null_spans(
            s_a, s_b, target_dist, dense,
            proximity_bin_width=budget, n_draws=8, seed=3,
        )
    # and the coarsened effective bin width RESCUES it (non-empty), proving the fix
    # is what populates the pool.
    eff_bin = rce.effective_proximity_bin_width(0, budget)
    rescued = rce.sample_matched_null_spans(
        s_a, s_b, target_dist, dense,
        proximity_bin_width=eff_bin, n_draws=8, seed=3,
    )
    assert len(rescued) == 8


# =========================================================================== #
# (C) U_hat + G1 gate on hand-checkable synthetic per-example arrays.
# =========================================================================== #
def test_u_hat_point_estimate_is_mean_paired_contrast():
    # tau_targeted - tau_null, per example: [0.8, 0.6, 1.0] - [0.1, 0.1, 0.0]
    tau_t = [0.8, 0.6, 1.0]
    tau_p = [0.1, 0.1, 0.0]
    res = rce.compute_u_hat(tau_t, tau_p, seed=0)
    # u_i = [0.7, 0.5, 1.0] -> mean = 2.2/3
    assert res.u_hat == pytest.approx(2.2 / 3.0)
    assert res.mean_targeted == pytest.approx(2.4 / 3.0)
    assert res.mean_null == pytest.approx(0.2 / 3.0)
    assert res.n_examples == 3
    assert res.ci_low <= res.u_hat <= res.ci_high


def test_u_hat_length_mismatch_raises():
    with pytest.raises(ValueError):
        rce.compute_u_hat([0.5, 0.5], [0.1], seed=0)


def test_g1_gate_clears_when_targeted_beats_null_by_margin():
    # Strong, consistent targeted effect well above the matched null and the 0.05
    # necessity margin -> the registered ciu_gate certifies (useful_candidate).
    tau_t = [0.9, 0.8, 1.0, 0.85, 0.95, 0.9, 0.8, 1.0]
    tau_p = [0.1, 0.05, 0.0, 0.1, 0.05, 0.1, 0.0, 0.05]
    from tracecausal.ciu import NECESSITY_MARGIN
    res = rce.compute_u_hat(tau_t, tau_p, seed=0)
    verdict = rce.g1_necessity_verdict(res, utility_drop=0.0, edit_budget=4)
    assert verdict == "useful_candidate"
    # a useful_candidate must clear the registered necessity margin on the CI lower bound
    assert res.ci_low >= NECESSITY_MARGIN


def test_g1_gate_withholds_when_targeted_matches_null():
    # Targeted effect indistinguishable from the matched null -> U_hat ~ 0, below the
    # necessity margin -> the gate withholds the necessity reading (diagnostic).
    tau_t = [0.2, 0.1, 0.15, 0.1, 0.2, 0.1, 0.15, 0.1]
    tau_p = [0.2, 0.1, 0.15, 0.1, 0.2, 0.1, 0.15, 0.1]
    res = rce.compute_u_hat(tau_t, tau_p, seed=0)
    verdict = rce.g1_necessity_verdict(res, utility_drop=0.0, edit_budget=4)
    assert verdict == "diagnostic"


def test_g1_gate_withholds_when_utility_cost_too_high():
    # Even a large necessity signal is downgraded if the utility drop blows the G2
    # bound (0.02) -> diagnostic (the registered conjunct).
    tau_t = [0.9, 0.8, 1.0, 0.85, 0.95, 0.9, 0.8, 1.0]
    tau_p = [0.1, 0.05, 0.0, 0.1, 0.05, 0.1, 0.0, 0.05]
    res = rce.compute_u_hat(tau_t, tau_p, seed=0)
    verdict = rce.g1_necessity_verdict(res, utility_drop=0.5, edit_budget=4)
    assert verdict == "diagnostic"


def test_g1_uses_registered_necessity_margin():
    from tracecausal.ciu import NECESSITY_MARGIN
    assert NECESSITY_MARGIN == 0.05


# =========================================================================== #
# (D) Salience-proxy span selector (pure python).
# =========================================================================== #
def test_select_salience_span_picks_max_mass_window():
    # budget-2 window of max attention mass: positions [3,4] (0.4+0.5 = 0.9)
    att = [0.05, 0.05, 0.1, 0.4, 0.5, 0.05]
    a, b = rce.select_salience_span(att, budget=2)
    assert (a, b) == (3, 4)


def test_select_salience_span_respects_prompt_start():
    att = [0.9, 0.9, 0.1, 0.2, 0.3]
    # excluding the first two high positions, the best budget-2 window is [3,4]
    a, b = rce.select_salience_span(att, budget=2, prompt_start=2)
    assert (a, b) == (3, 4)


def test_select_salience_span_budget_longer_than_vector_raises():
    with pytest.raises(ValueError):
        rce.select_salience_span([0.1, 0.2], budget=4)


def test_enumerate_candidate_spans_lengths_and_distance():
    # stride=1 (dense) checks the distance-key logic over every window.
    cands = rce.enumerate_candidate_spans(prompt_len=6, answer_index=5, budget=2, stride=1)
    # windows: [0,1],[1,2],[2,3],[3,4],[4,5] -> 5 candidates, each length 2
    assert len(cands) == 5
    assert all(c.length == 2 for c in cands)
    # distance to answer_index=5 from span end b
    assert cands[0].distance_to_answer == abs(5 - 1)  # span [0,1], b=1 -> 4
    assert cands[-1].distance_to_answer == abs(5 - 5)  # span [4,5], b=5 -> 0


# =========================================================================== #
# (E) Plan / dry-run wiring (no model).
# =========================================================================== #
def test_resolve_plan_lists_implemented_kernels_and_no_gpu_by_default():
    parser = rce.build_parser()
    args = parser.parse_args([])  # all defaults; no --i-have-authorization
    plan = rce.resolve_plan(args)
    assert plan["loads_model_or_gpu"] is False
    assert plan["edit_budget_k"] == args.budget
    assert any("ciu_gate" in k for k in plan["kernels"])
    assert any("build_null_pool" in k for k in plan["kernels"])


def test_dry_run_main_loads_nothing_and_exits_zero(capsys):
    rc = rce.main(["--n-examples", "5"])  # no --i-have-authorization
    assert rc == 0
    out = capsys.readouterr().out
    assert '"mode": "DRY_RUN"' in out
    assert "i-have-authorization" in out
    # torch/transformers/datasets must NOT have been imported by the dry-run path
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert "datasets" not in sys.modules
