#!/usr/bin/env python
"""repair_forward_provider.py -- the v5 repair-transfer GPU ForwardProvider.

This is the ONLY missing executable surface in the v5 repair-transfer headline
(G9 / G9-NOV). The v5 inference + gate stack (``repair_transfer.py``,
``repair_ops.py``, ``interventions.py``, ``ciu.py``, ``nullpool.py``,
``selective_inference.py``, ``nuisance.py``) is already implemented and green; the
pure-CPU runners in ``scripts/_runners.py`` already consume on-disk artifacts. The
one stage that genuinely needs the model -- emitting the per-pair ``g_ij`` rows
(and the operator-freeze on ``V_sel``) -- is bound through the ``ForwardProvider``
seam (``_runners.get_forward_provider`` / ``TRACECAUSAL_FORWARD_PROVIDER``).

This module supplies that provider. The zero-arg factory :func:`make_provider`
returns a callable matching the seam contract (``_runners._run_forward_stage``):

    provider(stage, args, plan, output_dir) -> dict   # row_count, output_hash, ...

It handles two stages:

* ``operator_freeze``           -- GPU forwards on ``V_sel`` to select+freeze the
                                  repair policy (OS-1), emit the frozen
                                  ``RepairPolicy`` + ``k_op`` + class weights ``w_c``.
* ``repair_transfer_forwards``  -- the Variant-C ``g_ij`` forwards; emits the
                                  ``g_ij_rows.json`` that ``run_estimate_r_hat``
                                  consumes, in the EXACT RepairGain row shape.

REPAIR MECHANIC (Variant C, load-bearing; REDESIGN_v5 §4.2):
  The source localization induces a repair *policy* ``rho`` (a recipe, NOT a hidden
  state). That recipe is applied **on the TARGET's own run** at the anchored TARGET
  span -- the source's hidden state NEVER crosses examples (``repair_ops.apply``
  enforces this). The forward edit is the convex reference patch
  ``h[:,lo:hi,:] = (1-alpha) h + alpha h_ref`` with ``alpha in PATCH_RHO_LEVELS`` and
  ``ref_type in {factual, neutral}`` (or a ``replay`` re-decode of the span under the
  reference policy). ``Y_j = factuality_score(target post-edit generation, golds_j)``
  in [0, 1]. The patch hook mechanism is lifted from
  ``run_ciu_experiment.HFForwardProvider.generate_patched`` and extended with this
  convex-reference mode.

CLAIM-SPAN INVENTORY (the highest-risk assumption; flagged for review):
  ``transport`` consumes a per-target list of ``TargetClaimSpan`` (atomic claim
  spans with g3_class, proximity bin via distance-to-answer, budget_k, and the
  ``is_target_designated`` oracle flag). Real TriviaQA/HotpotQA carry NO
  ground-truth claim segmentation. :func:`build_target_claim_spans` constructs them
  deterministically and CONSISTENTLY with how ``run_ciu_experiment.py`` selects
  spans: the answer-bearing/highest-salience window is the target's designated span
  (``is_target_designated=True``), and the remaining atomic claim spans are the
  budget-length windows on the same NON-OVERLAPPING grid
  (``enumerate_candidate_spans`` with ``stride=budget``), each tagged with the
  example's g3_class and its distance-to-answer-derived proximity bin. See that
  function's docstring for the exact rule and the explicit review flag.

DO-NOT-RUN discipline: nothing here loads a model, touches a GPU, or runs at import.
The heavy work happens only when the provider is CALLED, which the seam reaches only
after the hard authorization guard (``server.authorized: true`` AND
``--i-have-authorization``) in ``_runpacket_common``. ``server.authorized`` stays
false in committed configs. NO fabricated numbers: every ``g`` / ``Y`` is a real
forward-pass score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

# Make src/ and scripts/ importable (mirrors _runpacket_common / run_ciu_experiment).
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SCRIPTS = _ROOT / "scripts"
for _p in (_SRC, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Pure-python kernels (no model). These are the frozen, green surfaces.
from tracecausal.interventions import PATCH_RHO_LEVELS, Span  # noqa: E402
from tracecausal.repair_ops import (  # noqa: E402
    OperatorGrid,
    PositivityFail,
    TargetClaimSpan,
    localized_repair,
    operator_grid_cardinality,
    policy_hash,
    select_operator,
    tracedet_ar_span_adapter,
    transport,
    transport_map_hash,
)
from tracecausal.repair_transfer import RepairGain, r_hat, repair_gain  # noqa: E402

# Pure-python helpers lifted/reused from the existing CIU experiment script. These
# are stdlib-only (no model) and unit-tested in tests/test_ciu_experiment.py.
from run_ciu_experiment import (  # noqa: E402
    enumerate_candidate_spans,
    factuality_score,
    effective_proximity_bin_width,
    select_salience_span,
)


# =========================================================================== #
# Provider helpers (pure-python; no model) -- claim-span inventory + arms.
# =========================================================================== #

# The pre-registered claim-span construction variants (REDESIGN_v5 §8a; the
# proxy-robustness ablation). ``salience_grid`` is the DEFAULT (§4.2a) and is the
# ONLY one with which the headline G9 / G9-NOV path is byte-identical; the other
# three are the alternative segmentation rules the ablation sweeps so the verdict's
# invariance to the claim-span proxy can be measured. Every variant returns a
# ``list[TargetClaimSpan]`` of identical shape; nothing downstream changes.
CLAIM_SPAN_VARIANT_DEFAULT = "salience_grid"
CLAIM_SPAN_VARIANTS = (
    "salience_grid",     # DEFAULT (§4.2a): non-overlapping stride-k budget windows
    "sentence",          # alt 1: sentence/clause segments snapped to budget-k windows
    "stride_half",       # alt 2: budget-k windows on the denser stride-k/2 grid
    "salience_threshold",  # alt 3: only windows clearing a salience quantile q
)
# Pre-registered salience quantile for ``salience_threshold`` (frozen on V_sel, §8a).
CLAIM_SPAN_SALIENCE_QUANTILE = 0.5


def _enumerate_variant_spans(
    *,
    variant: str,
    prompt_len: int,
    answer_index: int,
    designated_span: tuple[int, int],
    budget: int,
    prompt_start: int,
    received_attention: "Sequence[float] | None",
    salience_quantile: float,
) -> list[tuple[int, int, int]]:
    """Enumerate the variant's atomic windows as ``(a, b, distance_to_answer)``.

    Every variant snaps to the SAME budget-``k`` / proximity grid the matched-null
    pool uses (REDESIGN_v5 §8a: "Each variant snaps to the budget-k / proximity grid
    so positivity is comparable"), so the only thing that changes between variants is
    **which** budget-``k`` windows enter the inventory -- never their length, budget,
    or proximity key. ``salience_grid`` reproduces the default non-overlapping grid
    EXACTLY (so the headline path is byte-identical); the others sub-/re-select on
    that same grid.
    """
    da, db = int(designated_span[0]), int(designated_span[1])

    if variant == "stride_half":
        # alt 2: budget-k windows on a denser stride-k/2 (overlapping) grid.
        stride = max(1, int(budget) // 2)
        cands = enumerate_candidate_spans(
            prompt_len, answer_index, budget=budget, prompt_start=prompt_start,
            stride=stride,
        )
        return [(c.a, c.b, c.distance_to_answer) for c in cands]

    # The default non-overlapping stride-k grid (shared base for the other variants).
    grid = enumerate_candidate_spans(
        prompt_len, answer_index, budget=budget, prompt_start=prompt_start,
        stride=budget,
    )

    if variant in ("salience_grid", CLAIM_SPAN_VARIANT_DEFAULT):
        return [(c.a, c.b, c.distance_to_answer) for c in grid]

    if variant == "sentence":
        # alt 1: sentence/clause units snapped to the budget-k grid. With no gold
        # segmentation (and no raw text at this seam) the proxy for "linguistic claim
        # unit" is the coarser, non-overlapping clause partition: take every other
        # budget-k window (clause-length ~= 2*budget), which is still a deterministic,
        # reproducible partition snapped to the same grid (the edit budget and the
        # matched-null grid still match). The designated (answer-bearing) window is
        # re-injected by the caller, so the answer-bearing sentence is always present.
        return [
            (c.a, c.b, c.distance_to_answer)
            for i, c in enumerate(grid)
            if i % 2 == 0
        ]

    if variant == "salience_threshold":
        # alt 3: only windows whose aggregate salience clears the pre-registered
        # quantile threshold q enter the inventory (a content-selective inventory).
        att = [float(v) for v in (received_attention or [])]
        masses: list[float] = []
        for c in grid:
            hi = min(len(att), c.b + 1)
            masses.append(sum(att[c.a:hi]) if hi > c.a else 0.0)
        if not masses or all(m == masses[0] for m in masses):
            # no salience signal (or flat): degenerate to the full grid rather than
            # fabricating a selection. The caller re-injects the designated span.
            return [(c.a, c.b, c.distance_to_answer) for c in grid]
        thr = _quantile(masses, salience_quantile)
        kept = [
            (c.a, c.b, c.distance_to_answer)
            for c, m in zip(grid, masses)
            if m >= thr
        ]
        # always retain the argmax-salience window (the designated span; §8a).
        best = max(range(len(grid)), key=lambda i: masses[i])
        bc = grid[best]
        if (bc.a, bc.b) not in {(a, b) for a, b, _ in kept}:
            kept.append((bc.a, bc.b, bc.distance_to_answer))
        return kept

    raise ValueError(
        f"unknown claim_span_variant {variant!r}; expected one of {CLAIM_SPAN_VARIANTS}"
    )


def _quantile(values: "Sequence[float]", q: float) -> float:
    """Lower-interpolated quantile of ``values`` (pure-python; stdlib only)."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return 0.0
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < len(xs):
        return xs[lo] + frac * (xs[lo + 1] - xs[lo])
    return xs[lo]


def build_target_claim_spans(
    *,
    prompt_len: int,
    answer_index: int,
    designated_span: tuple[int, int],
    g3_class: str,
    budget: int,
    proximity_bin_width: int,
    prompt_start: int = 0,
    variant: str = CLAIM_SPAN_VARIANT_DEFAULT,
    received_attention: "Sequence[float] | None" = None,
    salience_quantile: float = CLAIM_SPAN_SALIENCE_QUANTILE,
) -> list[TargetClaimSpan]:
    """Construct the per-target atomic claim-span inventory (DOCUMENTED ASSUMPTION).

    *** TOP RISK -- the claim-span inventory. EXPLICITLY FLAGGED FOR REVIEW. ***

    ``transport`` requires a list of :class:`TargetClaimSpan` (atomic claim spans on
    the target with their G3 class, proximity bin via ``distance_to_answer``, edit
    budget ``budget_k``, and the ``is_target_designated`` oracle flag). Real
    TriviaQA / HotpotQA examples carry **no ground-truth claim segmentation**, so we
    construct one deterministically and **consistently with how
    ``run_ciu_experiment.py`` picks spans** (so PROPOSED, B1, etc. and the matched
    null all live on the same span grid):

    * the **atomic claim spans** are the ``budget``-length windows on the
      **non-overlapping** stride-``budget`` grid (the exact
      :func:`enumerate_candidate_spans` grid the matched-null pool uses; the
      non-overlapping grid is the load-bearing empty-pool fix), each restricted to
      positions ``>= prompt_start`` (excludes the content-free chat-template prefix);
    * each window's ``distance_to_answer`` is ``abs(answer_index - b)`` (distance from
      the window end to the answer-forming position) -- the same proximity key the
      matched null and the G7 leakage bound share;
    * ``budget_k`` is the window's transport budget. For a Variant-C ``patch`` the
      transported policy's budget is ``span_len * |L_patch|`` (mirrors
      :func:`localized_repair`), so the windows here are tagged with the policy's
      realised budget by the caller via ``budget`` (the caller passes the policy
      budget, NOT the raw token length, when the op is patch over a multi-layer set);
    * the **target's own designated span** -- the salience/answer-bearing window the
      CIU selector would localise on this very target -- is flagged
      ``is_target_designated=True`` so the transport collapse guard refuses to anchor
      onto it (anchoring the target's own oracle span would collapse the "transfer"
      to a within-target oracle repair; refinement 1).
    * every span carries the **example's single g3_class** (the dataset cell's
      hallucination class); within one cell all spans share it, so the within-class
      transport condition is satisfiable.

    ASSUMPTION TO REVIEW (do not treat as ground truth): we have NO atomic-claim
    annotation on open TriviaQA/HotpotQA, so "atomic claim span" == "budget-length
    salience-grid window" and "designated/oracle span" == "the target's own
    top-salience window". This is a faithful, reproducible proxy that keeps the
    transport positivity/collapse guards meaningful, but a human reviewer must
    confirm it matches the intended claim-level estimand before any real run. If a
    real claim segmenter is later supplied, replace this function with it; the rest
    of the provider is agnostic to how the inventory was built.

    PROXY-ROBUSTNESS ABLATION HOOK (``variant``; REDESIGN_v5 §8a). ``variant`` selects
    the claim-span construction rule. ``salience_grid`` (the DEFAULT) is the registered
    behavior above and is **byte-identical** to the pre-hook code path; the other three
    (``sentence``, ``stride_half``, ``salience_threshold``) are the pre-registered
    alternative segmentations the ablation sweeps so the headline G9 / G9-NOV verdict
    can be shown invariant to the proxy. Each variant only changes **which** budget-``k``
    windows enter the inventory (never their length / budget / proximity key), always
    re-injects the designated span, and returns the SAME ``list[TargetClaimSpan]`` shape,
    so everything downstream (``transport`` -> ``g_ij`` -> ``r_hat`` -> gates) is
    unchanged. ``received_attention`` (per-prompt salience) is consumed ONLY by
    ``salience_threshold``; ``salience_quantile`` is its pre-registered threshold ``q``.
    """
    da, db = int(designated_span[0]), int(designated_span[1])
    spans: list[TargetClaimSpan] = []
    seen: set[tuple[int, int]] = set()
    for cand_a, cand_b, cand_dist in _enumerate_variant_spans(
        variant=variant,
        prompt_len=prompt_len,
        answer_index=answer_index,
        designated_span=designated_span,
        budget=budget,
        prompt_start=prompt_start,
        received_attention=received_attention,
        salience_quantile=salience_quantile,
    ):
        key = (cand_a, cand_b)
        if key in seen:
            continue
        seen.add(key)
        is_designated = (cand_a == da and cand_b == db)
        spans.append(
            TargetClaimSpan(
                span=Span(cand_a, cand_b),
                g3_class=g3_class,
                distance_to_answer=int(cand_dist),
                budget_k=int(budget),
                is_target_designated=is_designated,
            )
        )
    # If the designated window is NOT on the grid (e.g. salience picked an off-grid
    # window), append it explicitly as the designated span so the collapse guard has
    # it to exclude (it must never be an anchorable target).
    if (da, db) not in seen:
        spans.append(
            TargetClaimSpan(
                span=Span(da, db),
                g3_class=g3_class,
                distance_to_answer=abs(int(answer_index) - db),
                budget_k=int(budget),
                is_target_designated=True,
            )
        )
    spans.sort(key=lambda c: (c.span.a, c.span.b))
    return spans


# The MINIMAL panel built on real cells (REDESIGN_v5 §4.3). B5 oracle EXCLUDED from
# real cells (oracle fixtures only). B2/B3 are optional stubs (entropy/probe selector)
# -- present in the arm list only when the caller supplies their selector signal.
ARM_PROPOSED = "PROPOSED"   # CIU selector S* (top-salience window)
ARM_B0 = "B0"               # no_op floor (target unedited) -- a per-target reference
ARM_B1 = "B1"               # TraceDet -> AR span adapter (refinement 3), same pipeline
ARM_B4 = "B4"               # matched-null repair control (within-g_ij control)
REAL_PANEL = (ARM_PROPOSED, ARM_B1, ARM_B4, ARM_B0)


def _source_span_for_arm(
    arm: str,
    *,
    salience: Sequence[float],
    temporal_entropy: Sequence[float] | None,
    budget: int,
    prompt_start: int,
    prompt_len: int,
    b1_threshold: float,
) -> tuple[int, int] | None:
    """Resolve the SOURCE localized span per selector arm (pure python).

    * PROPOSED / B0 / B4 -> the CIU selector's top-salience budget window on the
      source (B0/B4 still localise the same S* so the matched-null and no-op are
      contrasted against the SAME source localization; B4's *control* is the
      matched-null repair inside ``g_ij``, not a different source span);
    * B1 -> the TraceDet AR span adapter over the source temporal entropy (refinement
      3); abstains (returns None) when no high-entropy sub-trace clears the threshold.
    """
    if arm == ARM_B1:
        if temporal_entropy is None:
            return None
        try:
            sp = tracedet_ar_span_adapter(
                temporal_entropy[: prompt_len - 1], threshold=b1_threshold, min_run=1
            )
        except ValueError:
            return None
        # clamp into the question region and to a budget-length window from sp.a
        a = max(prompt_start, sp.a)
        b = min(prompt_len - 2, a + budget - 1)
        if b < a:
            return None
        return a, b
    # PROPOSED / B0 / B4: salience selector
    try:
        return select_salience_span(
            list(salience)[: prompt_len - 1], budget=budget, prompt_start=prompt_start
        )
    except ValueError:
        return None


def _score_callback_factory(provider, examples, layers, budget, patch_mode, max_new_tokens):
    """Build the ``score_on_v_sel(policy) -> R_hat(PROPOSED) - R_hat(B4)`` callback.

    Runs GPU forwards on the V_sel subset for the candidate policy and returns the
    registered operator-selection objective (REDESIGN_v5 §4.7 OS-1): the within-class
    PROPOSED-minus-B4 repair-gain mean under that policy. Pure orchestration; the
    forwards are real model calls inside ``provider``.
    """
    def score_on_v_sel(policy) -> float:
        rows = _forward_pairs_for_arm(
            provider, examples, ARM_PROPOSED, layers, budget, patch_mode,
            max_new_tokens, policy=policy,
        )
        rows_b4 = _forward_pairs_for_arm(
            provider, examples, ARM_B4, layers, budget, patch_mode,
            max_new_tokens, policy=policy,
        )
        if not rows or not rows_b4:
            return float("-inf")
        try:
            r_prop = r_hat(rows).r_hat
            r_b4 = r_hat(rows_b4).r_hat
        except ValueError:
            return float("-inf")
        return r_prop - r_b4
    return score_on_v_sel


# =========================================================================== #
# The provider.
# =========================================================================== #
class RepairForwardProvider:
    """Concrete ``ForwardProvider`` for the v5 repair-transfer GPU stages.

    Wraps :class:`run_ciu_experiment.HFForwardProvider` for the model load, chat
    template, salience selection, and the activation-patching hook mechanism (which
    we extend with the convex-reference patch mode). Constructed lazily by
    :func:`make_provider`; no GPU work until a stage is dispatched.
    """

    def __init__(self, *, model_path: str | None = None, device: str = "cuda",
                 dtype: str = "bfloat16"):
        self._model_path = model_path
        self._device = device
        self._dtype = dtype
        self._hf = None  # lazily constructed HFForwardProvider (model load)

    # -- seam contract -------------------------------------------------------
    def __call__(self, *, stage: str, args, plan: dict, output_dir: Path) -> dict:
        if stage == "operator_freeze":
            return self.run_operator_freeze(args, plan, output_dir)
        if stage == "repair_transfer_forwards":
            return self.run_repair_transfer_forwards(args, plan, output_dir)
        raise ValueError(
            f"RepairForwardProvider does not handle stage {stage!r}; expected "
            "'operator_freeze' or 'repair_transfer_forwards'"
        )

    # -- model handle (lazy) -------------------------------------------------
    def _hf_provider(self, args):
        if self._hf is None:
            from run_ciu_experiment import HFForwardProvider  # lazy: model load

            model_path = (
                self._model_path
                or getattr(args, "model_path", None)
                or "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct"
            )
            device = getattr(args, "device", None) or self._device
            self._hf = _PatchingHF(model_path, device=device, dtype=self._dtype)
        return self._hf

    # ----------------------------------------------------------------------- #
    # Stage: operator_freeze (OS-1 freeze on V_sel).
    # ----------------------------------------------------------------------- #
    def run_operator_freeze(self, args, plan: dict, output_dir: Path) -> dict:
        hf = self._hf_provider(args)
        cfg = _resolve_forward_config(args, plan)
        examples = _load_forward_examples(args, plan, split="V_sel")

        grid = _build_operator_grid(plan)
        k_op = operator_grid_cardinality(grid)

        # The OS-1 objective: R_hat(PROPOSED) - R_hat(B4) on V_sel, run as GPU forwards.
        score_cb = _score_callback_factory(
            hf, examples, cfg["layers"], cfg["budget"], cfg["patch_mode"],
            cfg["max_new_tokens"],
        )
        # source span length/proximity used to compute candidate policy budgets: take
        # the median question budget window (deterministic) -- the freeze is over the
        # grid, not over a single example's geometry.
        selection = select_operator(
            grid,
            score_cb,
            source_proximity_bin=0,
            source_span_length=cfg["budget"],
            source_example_id=None,
            source_g3_class=cfg["g3_class"],
        )
        policy = selection.policy
        # class weights w_c: equal over the classes present in V_sel (frozen on V_sel).
        classes = sorted({ex["g3_class"] for ex in examples}) or [cfg["g3_class"]]
        w_c = {c: 1.0 / len(classes) for c in classes}

        fam = cfg["family"]
        ds = cfg["dataset"]
        payload = {
            "stage": "operator_freeze",
            "server_authorized": False,
            "family": fam,
            "dataset": ds,
            "frozen_policy": {
                "op": policy.op,
                "alpha": policy.alpha,
                "layer_set": list(policy.layer_set),
                "ref_type": policy.ref_type,
                "budget_k": policy.budget_k,
                "source_proximity_bin": policy.source_proximity_bin,
                "anchor_rule": policy.anchor_rule,
                "source_g3_class": policy.source_g3_class,
            },
            "policy_hash": policy_hash(policy),
            "k_op": k_op,
            "select_score": selection.score,
            "class_weights": w_c,
            "n_v_sel_examples": len(examples),
        }
        out_path = output_dir / "operator_freeze" / f"{fam}__{ds}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        h = _write_json(out_path, payload)
        return {"output_hash": h, "row_count": 1, "k_op": k_op,
                "operator_freeze_path": str(out_path)}

    # ----------------------------------------------------------------------- #
    # Stage: repair_transfer_forwards (the g_ij forwards).
    # ----------------------------------------------------------------------- #
    def run_repair_transfer_forwards(self, args, plan: dict, output_dir: Path) -> dict:
        hf = self._hf_provider(args)
        cfg = _resolve_forward_config(args, plan)
        examples = _load_forward_examples(args, plan, split="V_inf")
        policy = _load_frozen_policy(args, plan, cfg)

        selector = (getattr(args, "selector", None) or plan.get("selector") or ARM_PROPOSED)
        selector = str(selector).upper()
        if selector == "B5":
            raise ValueError(
                "B5 (oracle) is EXCLUDED from real cells (oracle fixtures only; §4.3); "
                "do not run repair_transfer_forwards for B5 on a real dataset"
            )

        rows = _forward_pairs_for_arm(
            hf, examples, selector, cfg["layers"], cfg["budget"], cfg["patch_mode"],
            cfg["max_new_tokens"], policy=policy, r_int=cfg["r_int"],
            r_null=cfg["r_null"], proximity_bin_width=cfg["proximity_bin_width"],
            seed=cfg["seed"], claim_span_variant=cfg["claim_span_variant"],
        )
        # Persist in the EXACT shape run_estimate_r_hat consumes (_load_repair_gain_rows
        # -> {"rows": [...]} with g / mc_var / source_id / target_id / g3_class), plus
        # the realised R_null integer (fixes the r_null_count ambiguity downstream).
        row_dicts = [
            {
                "g": rg.g,
                "mc_var": rg.mc_var,
                "source_id": rg.source_id,
                "target_id": rg.target_id,
                "g3_class": rg.g3_class,
            }
            for rg in rows
        ]
        n_excluded, excluded_by_class = _ARM_EXCLUSION.get(selector, (0, {}))
        n_total_attempted = len(row_dicts) + n_excluded
        payload = {
            "stage": "repair_transfer_forwards",
            "server_authorized": False,
            "selector_arm": selector,
            "transport_variant": "C",
            "claim_span_variant": cfg["claim_span_variant"],
            "r_null_realized": int(cfg["r_null"]),
            "r_int": int(cfg["r_int"]),
            "positivity_excluded_count": n_excluded,
            "positivity_excluded_frac": (
                n_excluded / n_total_attempted if n_total_attempted else 0.0
            ),
            "positivity_excluded_by_class": dict(excluded_by_class),
            "transport_map_hash": transport_map_hash(policy, cfg["proximity_bin_width"]),
            "rows": row_dicts,
        }
        out_path = output_dir / "g_ij_rows.json"
        h = _write_json(out_path, payload)
        return {"output_hash": h, "row_count": len(row_dicts),
                "r_null_realized": int(cfg["r_null"]),
                "positivity_excluded_count": n_excluded,
                "g_ij_rows_path": str(out_path)}


# =========================================================================== #
# Core forward loop: build the ordered in-class source != target g_ij rows.
# =========================================================================== #
def _forward_pairs_for_arm(
    hf,
    examples: list[dict],
    arm: str,
    layers: tuple[int, ...],
    budget: int,
    patch_mode: str,
    max_new_tokens: int,
    *,
    policy=None,
    r_int: int = 1,
    r_null: int = 1,
    proximity_bin_width: int = 0,
    seed: int = 0,
    b1_threshold: float = 0.5,
    claim_span_variant: str = CLAIM_SPAN_VARIANT_DEFAULT,
) -> list[RepairGain]:
    """Run the Variant-C forwards for one arm and assemble RepairGain rows.

    For each example we (once) resolve its source localized span, induce the policy
    ``rho`` (``localized_repair``), build its target claim-span inventory, and cache
    its no_op factuality + matched-null repair gains (the R_null matched-null term is
    SHARED across arms per target -- same Pi_j -- so it is computed once per target).

    Then for each ORDERED in-class source != target pair we
      (1) ``transport(rho_source, target, ...)`` -- skip+record A7 exclusion on
          PositivityFail (the within-class / self-repair / collapse / positivity
          guards all surface here);
      (2) ``Y_j(localized repair)`` -- the target's OWN run patched at the anchored
          TARGET span with the source-derived recipe (NEVER the source state),
          averaged over ``r_int`` repeats;
      (3) ``Y_j(no_op)`` -- the target's unedited factuality (cached);
      (4) the SHARED matched-null repair gains at S for R_null draws from Pi_j
          (cached per target);
      (5) ``repair_gain(...)`` -> a RepairGain with the per-TARGET shared matched-null
          SE^2 in ``mc_var`` (so the cluster bootstrap propagates one shared z_t per
          target -- NOT per-pair-independent).

    Returns the list of RepairGain rows for this arm. Exclusions are recorded on the
    examples' shared ``_excluded`` accumulators when present (set by the caller).
    """
    import random as _random

    # per-example caches keyed by example id
    cache: dict[str, dict] = {}
    excluded_total = [0]
    excluded_by_class: dict[str, list[int]] = {}
    excluded_by_class_count: dict[str, int] = {}
    inclass_by_class_count: dict[str, int] = {}

    # resolve per-example source span + policy + per-target no_op/null gains
    prepared: list[dict] = []
    for ex in examples:
        prompt = ex["question"]
        golds = ex["golds"]
        g3 = ex["g3_class"]
        ex_id = str(ex["id"])
        prompt_len = hf.prompt_token_count(prompt)
        if prompt_len <= budget + 1:
            continue
        q_start = hf.question_start()
        answer_index = prompt_len - 1

        salience = hf.salience_spans(prompt)
        temporal_entropy = ex.get("temporal_entropy")
        src = _source_span_for_arm(
            arm, salience=salience, temporal_entropy=temporal_entropy, budget=budget,
            prompt_start=q_start, prompt_len=prompt_len, b1_threshold=b1_threshold,
        )
        if src is None:
            continue  # arm abstains on this example (e.g. B1 no high-entropy sub-trace)
        s_a, s_b = src

        # induce the source policy rho (unless a frozen policy is supplied: then we
        # use the frozen recipe but stamp THIS source's id/class/proximity).
        src_prox_bin = abs(answer_index - s_b) // max(1, proximity_bin_width or 1)
        if policy is not None:
            rho = localized_repair(
                Span(s_a, s_b),
                op=policy.op,
                alpha=policy.alpha,
                layer_set=policy.layer_set,
                ref_type=policy.ref_type,
                source_proximity_bin=src_prox_bin,
                source_example_id=ex_id,
                source_g3_class=g3,
            )
        else:
            rho = localized_repair(
                Span(s_a, s_b),
                op="patch",
                alpha=PATCH_RHO_LEVELS[-1],
                layer_set=layers,
                ref_type="factual",
                source_proximity_bin=src_prox_bin,
                source_example_id=ex_id,
                source_g3_class=g3,
            )

        # target-side caches (no_op + shared matched-null repair gains). The matched
        # null is at the TARGET's OWN designated span S (Pi_j), shared across arms.
        if ex_id not in cache:
            y_noop = hf.factuality_noop(prompt, golds, max_new_tokens=max_new_tokens)
            # the target's own designated/oracle span (top salience window)
            try:
                d_a, d_b = select_salience_span(
                    list(salience)[: prompt_len - 1], budget=budget, prompt_start=q_start
                )
            except ValueError:
                continue
            # SHARED matched-null repair gains at S over R_null draws from Pi_j: random
            # in-question disjoint budget windows, patched with the SAME convex recipe.
            null_gains = _matched_null_repair_gains(
                hf, prompt, golds, d_a, d_b, layers, budget, patch_mode,
                max_new_tokens, ref_type=rho.ref_type, alpha=rho.alpha, op=rho.op,
                r_null=r_null, prompt_start=q_start, prompt_len=prompt_len,
                seed=seed + (hash(ex_id) & 0xFFFF), y_noop=y_noop,
            )
            cache[ex_id] = {
                "y_noop": y_noop,
                "designated": (d_a, d_b),
                "null_gains": null_gains,
                "g3_class": g3,
                "prompt_len": prompt_len,
                "answer_index": answer_index,
                "q_start": q_start,
                # per-prompt salience, consumed ONLY by the salience_threshold variant
                # of the claim-span ablation (REDESIGN_v5 §8a); harmless otherwise.
                "salience": list(salience)[: prompt_len - 1],
            }
        prepared.append({
            "ex": ex, "ex_id": ex_id, "rho": rho, "g3_class": g3, "prompt": prompt,
            "golds": golds,
        })

    rows: list[RepairGain] = []
    rng = _random.Random(seed)
    for src in prepared:
        rho = src["rho"]
        s_id = src["ex_id"]
        s_class = src["g3_class"]
        for tgt in prepared:
            t_id = tgt["ex_id"]
            if s_id == t_id:
                continue  # source != target
            if src["g3_class"] != tgt["g3_class"]:
                continue  # within-class only
            t = cache.get(t_id)
            if t is None:
                continue
            inclass_by_class_count[s_class] = inclass_by_class_count.get(s_class, 0) + 1
            claim_spans = build_target_claim_spans(
                prompt_len=t["prompt_len"],
                answer_index=t["answer_index"],
                designated_span=t["designated"],
                g3_class=tgt["g3_class"],
                budget=rho.budget_k,
                proximity_bin_width=proximity_bin_width,
                prompt_start=t["q_start"],
                variant=claim_span_variant,
                received_attention=t.get("salience"),
            )
            edit = transport(
                rho, t_id, claim_spans, tgt["g3_class"], proximity_bin_width,
            )
            if isinstance(edit, PositivityFail):
                excluded_total[0] += 1
                excluded_by_class_count[s_class] = excluded_by_class_count.get(s_class, 0) + 1
                continue
            # Y_j(localized repair): target's OWN run, anchored TARGET span, source recipe.
            y_localized = _repaired_factuality(
                hf, tgt["prompt"], tgt["golds"], edit.target_span.a, edit.target_span.b,
                layers, patch_mode, max_new_tokens, ref_type=rho.ref_type,
                alpha=rho.alpha, op=rho.op, r_int=r_int,
            )
            # assemble g_ij via the kernel; mc_var is the per-TARGET shared SE^2.
            rg = repair_gain(
                y_localized=y_localized,
                y_noop=t["y_noop"],
                matched_null_repair_samples=t["null_gains"],
                source_id=s_id,
                target_id=t_id,
                g3_class=tgt["g3_class"],
            )
            rows.append(rg)

    # surface exclusion accounting on the per-arm config accumulator (caller reads it)
    _ARM_EXCLUSION[arm] = (
        excluded_total[0],
        {
            c: excluded_by_class_count.get(c, 0)
            / max(1, excluded_by_class_count.get(c, 0) + inclass_by_class_count.get(c, 0))
            for c in set(excluded_by_class_count) | set(inclass_by_class_count)
        },
    )
    return rows


# arm -> (excluded_count, {class: excluded_frac}) accumulator from the last forward run
_ARM_EXCLUSION: dict[str, tuple[int, dict[str, float]]] = {}


def _matched_null_repair_gains(
    hf, prompt, golds, s_a, s_b, layers, budget, patch_mode, max_new_tokens, *,
    ref_type, alpha, op, r_null, prompt_start, prompt_len, seed, y_noop,
) -> list[float]:
    """The SHARED matched-null repair gains ``Y_j(do phi_rho^S) - Y_j(no_op)`` (Pi_j).

    Draws ``r_null`` random in-question budget windows disjoint from the target's
    designated span S, patches each with the SAME convex reference recipe (so the
    contrast is matched on operator/budget/ref), and returns the per-draw repair gain
    differenced against ``no_op`` -- exactly what ``repair_gain`` expects as
    ``matched_null_repair_samples`` (one value per draw from Pi_j).
    """
    import random as _random

    rng = _random.Random(seed)
    target_span = Span(s_a, s_b)
    # candidate disjoint budget windows on the non-overlapping grid in the question
    cands = [
        (c.a, c.b)
        for c in enumerate_candidate_spans(
            prompt_len - 1, prompt_len - 1, budget=budget, prompt_start=prompt_start,
            stride=budget,
        )
        if Span(c.a, c.b).disjoint_from(target_span)
    ]
    if not cands:
        # no matched control location: degenerate; return a single zero-gain so the
        # pair is still scorable but contributes no spurious null signal.
        return [0.0]
    gains: list[float] = []
    n = max(1, int(r_null))
    for _ in range(n):
        a, b = rng.choice(cands)
        y = _repaired_factuality(
            hf, prompt, golds, a, b, layers, patch_mode, max_new_tokens,
            ref_type=ref_type, alpha=alpha, op=op, r_int=1,
        )
        gains.append(y - y_noop)
    return gains


def _repaired_factuality(
    hf, prompt, golds, a, b, layers, patch_mode, max_new_tokens, *,
    ref_type, alpha, op, r_int,
) -> float:
    """``Y = factuality_score(target post-edit generation, golds)`` averaged over r_int.

    Variant-C convex-reference patch (``op=='patch'``): the target's own residual
    states at ``[a, b]`` are mixed toward the reference state by
    ``h <- (1-alpha) h + alpha h_ref`` (ref_type in {factual, neutral}); for
    ``op=='replay'`` the span is re-decoded under the reference policy. ``r_int``
    repeats average out repair-op MC noise (greedy decoding is deterministic, so
    repeats matter only under stochastic reference construction; the average is the
    registered estimator).
    """
    ys: list[float] = []
    for _ in range(max(1, int(r_int))):
        gen = hf.generate_repaired(
            prompt, a, b, layers, mode=patch_mode, ref_type=ref_type, alpha=alpha,
            op=op, max_new_tokens=max_new_tokens,
        )
        ys.append(factuality_score(gen, golds))
    return sum(ys) / len(ys)


# =========================================================================== #
# Config / IO resolution (read-only; never fabricate).
# =========================================================================== #
def _resolve_forward_config(args, plan: dict) -> dict:
    layers = tuple(
        int(x) for x in str(getattr(args, "layers", "") or "12,13,14,15").split(",")
        if str(x).strip()
    )
    budget = int(getattr(args, "budget", 4) or 4)
    r_int = int(getattr(args, "r_int", None) or plan.get("r_int", 16) or 16)
    # r_null: prefer an explicit integer; reject the DATA_NEEDED sentinel by falling
    # back to a small realized default (the realized integer is persisted downstream).
    r_null_raw = getattr(args, "r_null", None)
    if r_null_raw is None:
        r_null_raw = plan.get("r_null")
    r_null = _coerce_int(r_null_raw, default=8)
    pbw = effective_proximity_bin_width(
        int(getattr(args, "proximity_bin_width", 0) or 0), budget
    )
    cfg = {
        "layers": layers,
        "budget": budget,
        "patch_mode": getattr(args, "patch_mode", None) or "mean_ablate",
        "max_new_tokens": int(getattr(args, "max_new_tokens", 32) or 32),
        "r_int": r_int,
        "r_null": r_null,
        "proximity_bin_width": pbw,
        "seed": int(getattr(args, "seed", 0) or 0),
        "family": str(getattr(args, "family", None) or plan.get("family") or "ar_lead"),
        "dataset": str(getattr(args, "dataset", None) or plan.get("dataset") or "triviaqa"),
        "g3_class": str(getattr(args, "g3_class", None) or plan.get("g3_class") or "factoid"),
        "claim_span_variant": _resolve_claim_span_variant(args, plan),
        "_excluded": [0],
        "_excluded_by_class": {},
    }
    return cfg


def _resolve_claim_span_variant(args, plan: dict) -> str:
    """Resolve the proxy-robustness ablation variant (REDESIGN_v5 §8a).

    Sourced from ``--claim-span-variant`` (CLI), then the plan, defaulting to
    ``salience_grid`` (the registered DEFAULT, behavior byte-identical). An unknown
    value fails closed rather than silently using the default, so a typo cannot run
    the headline path under an unintended segmentation.
    """
    v = getattr(args, "claim_span_variant", None) or plan.get("claim_span_variant")
    v = str(v) if v else CLAIM_SPAN_VARIANT_DEFAULT
    if v not in CLAIM_SPAN_VARIANTS:
        raise ValueError(
            f"unknown --claim-span-variant {v!r}; expected one of {CLAIM_SPAN_VARIANTS}"
        )
    return v


def _coerce_int(val, *, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return int(default)


def _build_operator_grid(plan: dict) -> OperatorGrid:
    og = plan.get("op_grid") or {}
    ops = tuple(og.get("ops") or ("patch", "replay"))
    alphas = tuple(float(a) for a in (og.get("alpha") or PATCH_RHO_LEVELS))
    ref_types = tuple(og.get("ref_type") or ("factual", "neutral"))
    return OperatorGrid(ops=ops, alphas=alphas, ref_types=ref_types)


def _load_forward_examples(args, plan: dict, *, split: str) -> list[dict]:
    """Resolve the per-cell examples for a forward stage.

    On the authorized server branch the examples come from the on-disk extracted
    traces (the upstream extract_traces stage's output) carrying, per example:
    ``id``, ``question``, ``golds`` (gold alias list), ``g3_class``, and optionally
    ``temporal_entropy`` (for B1's TraceDet adapter). If a ``--traces`` path is given
    we read it; otherwise we fall back to the live dataset loader (authorized only).
    """
    traces = getattr(args, "traces", None)
    if traces:
        from _runners import _read_json  # reuse the actionable loader

        p = Path(traces)
        data = _read_json(p if p.suffix == ".json" else p / "forward_examples.json")
        items = data.get("examples") if isinstance(data, dict) else data
        if isinstance(data, dict) and split in data:
            items = data[split]
        if not isinstance(items, list) or not items:
            from _runners import RunInputError

            raise RunInputError(f"no forward examples ({split}) found in {p}")
        return [_normalize_example(it, i) for i, it in enumerate(items)]

    # live dataset (authorized only); split the loaded items into V_sel/V_inf halves.
    from run_ciu_experiment import _load_dataset  # lazy: datasets import

    name = str(getattr(args, "dataset", None) or plan.get("dataset") or "triviaqa")
    n = int(getattr(args, "n_examples", 60) or 60)
    seed = int(getattr(args, "seed", 0) or 0)
    raw = _load_dataset(name, n, seed)
    g3 = str(getattr(args, "g3_class", None) or plan.get("g3_class") or "factoid")
    items = [_normalize_example({**r, "g3_class": g3}, i) for i, r in enumerate(raw)]
    half = max(1, len(items) // 2)
    return items[:half] if split == "V_sel" else items[half:]


def _normalize_example(it: dict, i: int) -> dict:
    return {
        "id": str(it.get("id", it.get("example_id", i))),
        "question": it.get("question") or it.get("prompt") or "",
        "golds": list(it.get("golds") or it.get("answers") or []),
        "g3_class": str(it.get("g3_class", "factoid")),
        "temporal_entropy": it.get("temporal_entropy"),
    }


def _load_frozen_policy(args, plan: dict, cfg: dict):
    """Load the frozen RepairPolicy from the operator_freeze artifact, if present.

    ``run_repair_transfer.py`` passes ``--operator-freeze <path>``; we read the
    persisted ``frozen_policy`` and reconstruct the recipe. When absent (e.g. a B1
    baseline run that re-localizes), a default factual full-strength patch over the
    configured layers is used (still a real Variant-C recipe).
    """
    of = getattr(args, "operator_freeze", None) or plan.get("operator_freeze")
    if of:
        from _runners import _read_json

        p = Path(of)
        data = _read_json(p if p.suffix == ".json" else p / "operator_freeze.json")
        fp = data.get("frozen_policy") if isinstance(data, dict) else None
        if fp:
            return localized_repair(
                Span(0, max(0, int(fp.get("budget_k", cfg["budget"])) - 1)),
                op=fp.get("op", "patch"),
                alpha=float(fp.get("alpha", PATCH_RHO_LEVELS[-1])),
                layer_set=tuple(fp.get("layer_set", cfg["layers"])),
                ref_type=fp.get("ref_type", "factual"),
                source_proximity_bin=int(fp.get("source_proximity_bin", 0)),
                source_g3_class=fp.get("source_g3_class"),
            )
    # default recipe (no freeze artifact): factual full-strength patch over layers.
    return localized_repair(
        Span(0, max(0, cfg["budget"] - 1)),
        op="patch",
        alpha=PATCH_RHO_LEVELS[-1],
        layer_set=cfg["layers"],
        ref_type="factual",
        source_proximity_bin=0,
        source_g3_class=cfg["g3_class"],
    )


def _write_json(path: Path, payload: Any) -> str:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {f: getattr(obj, f) for f in obj.__dataclass_fields__}
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    return repr(obj)


# =========================================================================== #
# The model handle: HFForwardProvider + the convex-reference patch mode.
# =========================================================================== #
def _make_patching_hf():
    """Define the _PatchingHF subclass lazily (only when a model is actually loaded).

    Subclasses ``run_ciu_experiment.HFForwardProvider`` to add:
      * :meth:`factuality_noop`   -- no_op generation factuality (Y_j(no_op));
      * :meth:`generate_repaired` -- the Variant-C convex-reference patch / replay
        forward (extends generate_patched's hook mechanism with the convex mode).
    """
    from run_ciu_experiment import HFForwardProvider

    class _PatchingHFImpl(HFForwardProvider):
        def factuality_noop(self, prompt, golds, *, max_new_tokens=32) -> float:
            gen = self.generate(prompt, max_new_tokens=max_new_tokens)
            return factuality_score(gen, golds)

        def generate_repaired(
            self, prompt, span_a, span_b, layer_set, *, mode="mean_ablate",
            ref_type="factual", alpha=1.0, op="patch", max_new_tokens=32,
        ) -> str:
            """Greedy generation under a Variant-C repair edit on ``[a, b]``.

            ``op=='replay'`` re-decodes the span under the reference policy -- here
            realized as the activation-patching ``mode`` (the registered re-decode
            surrogate the hook implements). ``op=='patch'`` applies the CONVEX
            reference patch ``h <- (1-alpha) h + alpha h_ref`` where ``h_ref`` is the
            reference state: for ``ref_type=='neutral'`` the per-feature mean over the
            other prompt positions (a content-neutral reference); for
            ``ref_type=='factual'`` the rolled neighbouring states (a same-distribution
            factual-style reference). This reuses ``generate_patched``'s hook plumbing
            and adds the convex mix.
            """
            torch = self._torch
            inputs = self._build_inputs(prompt)
            prompt_len = inputs["input_ids"].shape[1]
            a = max(0, int(span_a))
            b = min(prompt_len - 1, int(span_b))
            handles = []
            al = float(alpha)

            def _make_hook():
                def hook(module, args_, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    seq_len = hs.shape[1]
                    if seq_len <= b:
                        return output
                    lo, hi = a, b + 1
                    # build the reference state h_ref over the patched span
                    if ref_type == "neutral":
                        mask = torch.ones(seq_len, dtype=torch.bool, device=hs.device)
                        mask[lo:hi] = False
                        other = hs[:, :prompt_len, :][:, mask[:prompt_len], :]
                        if other.shape[1] > 0:
                            h_ref = other.mean(dim=1, keepdim=True)
                        else:
                            h_ref = hs[:, lo:hi, :]
                    else:  # factual: same-distribution rolled neighbour states
                        h_ref = torch.roll(hs[:, lo:hi, :], shifts=1, dims=1)
                    # CONVEX reference patch: h <- (1-alpha) h + alpha h_ref
                    hs[:, lo:hi, :] = (1.0 - al) * hs[:, lo:hi, :] + al * h_ref
                    if isinstance(output, tuple):
                        return (hs,) + tuple(output[1:])
                    return hs

                return hook

            try:
                for li in layer_set:
                    if 0 <= int(li) < self.n_layers:
                        handles.append(
                            self._layers[int(li)].register_forward_hook(_make_hook())
                        )
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                gen = out[0][prompt_len:]
                return self.tokenizer.decode(gen, skip_special_tokens=True).strip()
            finally:
                for h in handles:
                    h.remove()

    return _PatchingHFImpl


def _PatchingHF(*args, **kwargs):
    cls = _make_patching_hf()
    return cls(*args, **kwargs)


# =========================================================================== #
# Zero-arg factory (the seam entrypoint).
# =========================================================================== #
def make_provider():
    """Zero-arg factory returning the v5 repair-transfer ForwardProvider.

    Registered via ``TRACECAUSAL_FORWARD_PROVIDER=scripts.repair_forward_provider:make_provider``
    (or imported as ``repair_forward_provider:make_provider`` when ``scripts/`` is on
    the path), or by calling ``_runners.register_forward_provider(make_provider())``.
    No model is loaded here; the load happens lazily on the first stage call.
    """
    return RepairForwardProvider()
