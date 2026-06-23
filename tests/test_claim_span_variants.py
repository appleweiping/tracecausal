"""Pure-python unit tests for the §8a proxy-robustness ablation hook.

REDESIGN_v5 §8a pre-registers a ``--claim-span-variant`` hook on
``scripts/repair_forward_provider.py`` that selects the claim-span construction
rule so the headline G9 / G9-NOV verdict can be shown invariant to the
documented PROXY claim segmentation (§4.2a). This pins, with **no model, no GPU,
no network**, that:

* the registered DEFAULT (``salience_grid``) is **byte-identical** to the
  pre-hook ``build_target_claim_spans`` output (the existing G9 path is unchanged
  when no variant / the default is selected);
* each of the >=3 alternative segmentations the ablation registers
  (``sentence``, ``stride_half``, ``salience_threshold``) returns a well-formed
  ``TargetClaimSpan`` list of the same shape, with exactly one designated span,
  and is **distinct** from the default inventory;
* every variant snaps to the same budget-``k`` grid (length / budget / proximity
  key unchanged across variants -- only WHICH windows enter changes);
* the CLI flag resolves through ``_resolve_claim_span_variant`` and fails closed
  on an unknown value (so a typo cannot silently run the headline path).

All synthetic; the GPU provider is never constructed and no model is loaded.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
_SRC = _ROOT / "src"
for _p in (_SCRIPTS, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest  # noqa: E402

import repair_forward_provider as rfp  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic target geometry (hand-built; deterministic; no model).
# --------------------------------------------------------------------------- #
_KW = dict(
    prompt_len=40,
    answer_index=39,
    designated_span=(20, 23),
    g3_class="factoid",
    budget=4,
    proximity_bin_width=16,
    prompt_start=8,
)


def _graded_salience() -> list[float]:
    """A per-token salience vector where each budget window has distinct mass.

    The designated window ``[20, 23]`` is the argmax; the 0.5 quantile cuts the
    grid so ``salience_threshold`` keeps a STRICT subset (a genuinely
    content-selective inventory, not the full grid).
    """
    att = [0.0] * 40
    per_window_weight = {8: 1, 12: 2, 16: 3, 20: 9, 24: 4, 28: 5, 32: 6, 36: 7}
    for start, w in per_window_weight.items():
        for i in range(start, start + 4):
            att[i] = float(w)
    return att


def _keys(spans) -> list[tuple[int, int]]:
    return sorted((s.span.a, s.span.b) for s in spans)


def _assert_well_formed(spans) -> None:
    """Every variant must return a valid TargetClaimSpan list of the same shape."""
    assert spans, "expected a non-empty atomic-claim inventory"
    # exactly one designated/oracle span -> the collapse guard has one to drop.
    assert sum(1 for s in spans if s.is_target_designated) == 1
    # all spans carry the cell class and the (unchanged) edit budget.
    assert all(s.g3_class == _KW["g3_class"] for s in spans)
    assert all(s.budget_k == _KW["budget"] for s in spans)
    # proximity key is the same distance-to-answer rule (window end -> answer_index).
    for s in spans:
        assert s.distance_to_answer == abs(_KW["answer_index"] - s.span.b)
    # every span is a budget-length window (length / budget unchanged across variants).
    assert all(s.span.b - s.span.a + 1 == _KW["budget"] for s in spans)


# --------------------------------------------------------------------------- #
# 1. DEFAULT is byte-identical to the pre-hook output.
# --------------------------------------------------------------------------- #
def test_default_variant_is_byte_identical_to_unhooked():
    # no variant kwarg == explicit 'salience_grid' == the registered DEFAULT.
    no_kw = rfp.build_target_claim_spans(**_KW)
    explicit = rfp.build_target_claim_spans(variant="salience_grid", **_KW)
    assert no_kw == explicit
    # and identical even when a salience vector is (irrelevantly) supplied.
    with_att = rfp.build_target_claim_spans(
        variant="salience_grid", received_attention=_graded_salience(), **_KW
    )
    assert no_kw == with_att
    # the constant the hook advertises as the default really is salience_grid.
    assert rfp.CLAIM_SPAN_VARIANT_DEFAULT == "salience_grid"
    default_again = rfp.build_target_claim_spans(
        variant=rfp.CLAIM_SPAN_VARIANT_DEFAULT, **_KW
    )
    assert no_kw == default_again


# --------------------------------------------------------------------------- #
# 2. The registered variant set is exactly default + >=3 alternatives.
# --------------------------------------------------------------------------- #
def test_registered_variant_set_matches_redesign_8a():
    assert rfp.CLAIM_SPAN_VARIANTS == (
        "salience_grid",
        "sentence",
        "stride_half",
        "salience_threshold",
    )
    # default + at least three alternatives (the §8a invariance criterion needs >=3).
    alternatives = [v for v in rfp.CLAIM_SPAN_VARIANTS if v != "salience_grid"]
    assert len(alternatives) >= 3


# --------------------------------------------------------------------------- #
# 3. Each variant is well-formed AND distinct from the default inventory.
# --------------------------------------------------------------------------- #
def test_each_variant_well_formed_and_distinct():
    att = _graded_salience()
    default = rfp.build_target_claim_spans(variant="salience_grid", **_KW)
    _assert_well_formed(default)

    inventories = {"salience_grid": _keys(default)}
    for variant in ("sentence", "stride_half", "salience_threshold"):
        spans = rfp.build_target_claim_spans(
            variant=variant, received_attention=att, **_KW
        )
        _assert_well_formed(spans)
        inventories[variant] = _keys(spans)
        # each ALTERNATIVE must differ from the default grid (it is a real ablation).
        assert inventories[variant] != inventories["salience_grid"], (
            f"variant {variant!r} did not change the inventory vs the default"
        )

    # all four inventories are pairwise distinct (no two rules collapse together).
    distinct = {tuple(v) for v in inventories.values()}
    assert len(distinct) == len(inventories)


def test_stride_half_is_denser_and_overlapping():
    # alt 2: budget-k windows on the denser stride-k/2 grid -> strictly MORE windows,
    # and they overlap (start step == budget/2 < budget).
    default = rfp.build_target_claim_spans(variant="salience_grid", **_KW)
    stride_half = rfp.build_target_claim_spans(variant="stride_half", **_KW)
    assert len(stride_half) > len(default)
    starts = sorted(s.span.a for s in stride_half)
    assert any(b - a < _KW["budget"] for a, b in zip(starts, starts[1:]))


def test_sentence_is_a_coarser_partition_with_answer_window():
    # alt 1: coarser clause-length partition snapped to the grid; strictly FEWER
    # windows than the full grid, and the answer-bearing designated window survives.
    default = rfp.build_target_claim_spans(variant="salience_grid", **_KW)
    sentence = rfp.build_target_claim_spans(variant="sentence", **_KW)
    assert len(sentence) < len(default)
    assert (20, 23) in _keys(sentence)  # the answer-bearing / designated window


def test_salience_threshold_is_content_selective_subset():
    # alt 3: only windows clearing the salience quantile (plus the argmax) survive;
    # a strict subset of the full grid, always including the top-salience window.
    att = _graded_salience()
    default = rfp.build_target_claim_spans(variant="salience_grid", **_KW)
    sal = rfp.build_target_claim_spans(
        variant="salience_threshold", received_attention=att, **_KW
    )
    default_keys = set(_keys(default))
    sal_keys = set(_keys(sal))
    assert sal_keys < default_keys  # strict subset (content-selective)
    assert (20, 23) in sal_keys     # argmax-salience window is retained


def test_salience_threshold_falls_back_to_full_grid_without_signal():
    # no salience signal -> degenerate to the full grid rather than fabricating a
    # selection (the §8a "do not invent a number" discipline at the inventory level).
    default = rfp.build_target_claim_spans(variant="salience_grid", **_KW)
    sal_flat = rfp.build_target_claim_spans(
        variant="salience_threshold", received_attention=None, **_KW
    )
    assert _keys(sal_flat) == _keys(default)


# --------------------------------------------------------------------------- #
# 4. Off-grid designated window stays present under every variant.
# --------------------------------------------------------------------------- #
def test_off_grid_designated_span_is_injected_for_all_variants():
    off_kw = dict(_KW)
    off_kw["designated_span"] = (21, 24)  # not on the stride-4 grid from prompt_start=8
    att = _graded_salience()
    for variant in rfp.CLAIM_SPAN_VARIANTS:
        spans = rfp.build_target_claim_spans(
            variant=variant, received_attention=att, **off_kw
        )
        designated = [s for s in spans if s.is_target_designated]
        assert len(designated) == 1
        assert (designated[0].span.a, designated[0].span.b) == (21, 24)


# --------------------------------------------------------------------------- #
# 5. Unknown variant fails closed (function + CLI resolver).
# --------------------------------------------------------------------------- #
def test_unknown_variant_raises_in_builder():
    with pytest.raises(ValueError):
        rfp.build_target_claim_spans(variant="not_a_real_variant", **_KW)


def test_cli_resolver_defaults_and_validates():
    # default (no flag) -> the registered salience_grid.
    assert rfp._resolve_claim_span_variant(Namespace(), {}) == "salience_grid"
    # explicit valid flag is passed through.
    args = Namespace(claim_span_variant="stride_half")
    assert rfp._resolve_claim_span_variant(args, {}) == "stride_half"
    # plan-level value is honored when the flag is absent.
    assert rfp._resolve_claim_span_variant(
        Namespace(), {"claim_span_variant": "sentence"}
    ) == "sentence"
    # unknown value fails closed (never silently falls back to the default).
    with pytest.raises(ValueError):
        rfp._resolve_claim_span_variant(Namespace(claim_span_variant="bogus"), {})
