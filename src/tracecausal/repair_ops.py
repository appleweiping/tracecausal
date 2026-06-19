"""Source-derived repair policy + frozen source->target transport (REDESIGN_v5 §4.2).

This module pins the **Variant C** transport protocol the v5 design freezes as its
headline (REDESIGN_v5 §4.2, MF-2): the source localization induces a **repair
*policy*** ``rho`` (NOT a state), and that policy is applied **on the target's own
run** at the target's own claim span identified by the frozen anchor map ``T``.
Cross-example content is the *policy + anchor*, which is what "the localization
licenses a reusable repair" means.

Design discipline (BUILD-NOW / RUN-LATER), preserved from v4:

* **Nothing here loads a model, touches a GPU, or runs an experiment.**
  ``localized_repair`` builds a typed, frozen ``RepairPolicy``; ``transport``
  resolves the frozen anchor map ``T`` to a concrete target span (or a typed
  ``PositivityFail``); ``apply`` wraps :func:`interventions.patch` /
  :func:`interventions.replay` into an :class:`EditedTargetPlan` — the *typed plan*
  a server run would consume. **No forward pass is executed.**

The nine design-gate refinements honored here:

* **(1) Variant C is a real source->target transfer.** ``transport`` patches the
  target on the target's **own** reference state (``ref_type`` carried by the
  policy) at the target's **own** claim span; it never injects the source's
  residual state. The *source-instance-derived content* (proximity bin + edit
  budget read by the anchor map ``T``) is recorded on the resolved edit
  (``source_proximity_bin``, ``source_budget_k``). A **target-side
  oracle-repair collapse guard** rejects a policy whose anchor would resolve the
  target's *own designated/oracle* span (which would make the "transfer" a
  trivial within-target oracle repair, not a transferred policy).
* **(2) Target-label leakage guard.** The anchor map restricts to **atomic claim
  spans of the same G3 class** on the target. ``TargetClaimSpan`` carries only the
  class label and structural keys (proximity, budget) the matched null is matched
  on — never the target's gold/factuality label — so localization cannot leak the
  target-side label.
* **(3) TraceDet -> AR span adapter.** ``tracedet_ar_span_adapter`` maps a base
  TraceDet temporal-entropy sub-trace (a contiguous run of high-entropy decode
  steps on the source) to a source claim span, so baseline **B1** runs through the
  *identical* ``repair_ops`` pipeline as PROPOSED.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

from .interventions import (
    PATCH_RHO_LEVELS,
    PatchResult,
    ReferenceType,
    ReplayResult,
    Span,
    patch,
    replay,
)

__all__ = [
    "RepairPolicy",
    "TargetClaimSpan",
    "TargetEdit",
    "PositivityFail",
    "EditedTargetPlan",
    "OperatorGrid",
    "OperatorSelection",
    "localized_repair",
    "transport",
    "apply",
    "operator_grid_cardinality",
    "select_operator",
    "policy_hash",
    "transport_map_hash",
    "tracedet_ar_span_adapter",
    "REPAIR_OP_CHOICES",
    "REF_TYPE_CHOICES",
]

RepairOp = Literal["patch", "replay"]

# The discrete operator grid the freeze rule (OS-1) selects over; its cardinality
# is the K_op selection event (REDESIGN_v5 §4.7 OS-2).
REPAIR_OP_CHOICES: tuple[RepairOp, ...] = ("patch", "replay")
REF_TYPE_CHOICES: tuple[ReferenceType, ...] = ("factual", "neutral")


@dataclass(frozen=True)
class RepairPolicy:
    """A **source-derived repair policy** ``rho`` (Variant C, REDESIGN_v5 §4.2).

    The policy carries the *recipe* — operator, mixing strength, layer set,
    reference type, edit budget, proximity bin, and the anchor tie-break rule —
    **not** a residual state. It is induced by the source localization
    ``S*(x_i)`` and then applied on a *different* target's own run.

    Attributes
    ----------
    op:
        ``"patch"`` (convex reference-state injection) or ``"replay"`` (rollback +
        re-decode under a reference policy). One of :data:`REPAIR_OP_CHOICES`.
    alpha:
        The ``patch`` mixing weight ``rho`` (one of
        :data:`interventions.PATCH_RHO_LEVELS`); ignored for ``replay`` but
        recorded for provenance.
    layer_set:
        ``L_patch`` — the layer set the patch touches. This is the closest
        tracecausal analogue of the orchestration template's "layer_function"
        surface (REDESIGN_v5 §9 note): a *parameter* of the repair policy, not a
        routing module.
    ref_type:
        ``"factual"`` | ``"neutral"`` — which reference run builds the policy and
        is patched on the **target's own** run (the cross-example analogue of A8).
    budget_k:
        The edit budget ``k`` (length / coordinate count) the anchor map matches
        **exactly** on the target (REDESIGN_v5 §4.2 (iii)).
    source_proximity_bin:
        The source span's proximity-to-answer bin (the same ``Delta_pos`` grid the
        matched null uses); the anchor map matches the target span's proximity bin
        to this (REDESIGN_v5 §4.2 (ii)).
    anchor_rule:
        The deterministic tie-break when multiple in-class, budget-matched,
        proximity-matched target spans remain ("first_by_position", §4.2 (iv)).
    source_example_id:
        The source ``x_i`` the policy was localized on (used to enforce
        source != target and the A5/NC-2 source-swap diagnostic).
    source_g3_class:
        The **frozen G3 class of the source** ``class(x_i)`` (REDESIGN_v5 §4.1).
        The estimand ``R`` (Eq. R) is the within-class, source != target mean, so
        ``class(source) == class(target)`` must be enforceable at ``transport``
        time. Carrying it on the policy (rather than relying on a caller-supplied
        target class) makes the within-class condition **fail-closed** (findings
        2/7): ``transport`` refuses any target whose class differs from this.
    alpha:
        For ``op="patch"`` the mixing weight is constrained to
        :data:`interventions.PATCH_RHO_LEVELS` (OS-1: ``alpha in PATCH_RHO_LEVELS``,
        REDESIGN_v5 §4.7) — an arbitrary ``alpha in (0, 1]`` is rejected so the
        operator-selection grid the ``K_op`` factor pays for is the only reachable
        set (finding 16).
    """

    op: RepairOp
    alpha: float
    layer_set: tuple[int, ...]
    ref_type: ReferenceType
    budget_k: int
    source_proximity_bin: int
    anchor_rule: str = "first_by_position"
    source_example_id: str | None = None
    source_g3_class: str | None = None

    def __post_init__(self) -> None:
        if self.op not in REPAIR_OP_CHOICES:
            raise ValueError(f"op must be one of {REPAIR_OP_CHOICES}, got {self.op!r}")
        if self.ref_type not in REF_TYPE_CHOICES:
            raise ValueError(f"ref_type must be one of {REF_TYPE_CHOICES}, got {self.ref_type!r}")
        if self.op == "patch":
            # OS-1 (REDESIGN_v5 §4.7): patch alpha must be a frozen grid level, not an
            # arbitrary (0, 1] value, so the reachable operator grid (K_op) is bounded
            # and enumerated before data (finding 16).
            if not any(math.isclose(self.alpha, lvl, abs_tol=1e-12) for lvl in PATCH_RHO_LEVELS):
                raise ValueError(
                    f"patch alpha (rho) must be one of PATCH_RHO_LEVELS {PATCH_RHO_LEVELS} "
                    f"(OS-1, REDESIGN_v5 §4.7), got {self.alpha}"
                )
        if self.budget_k <= 0:
            raise ValueError(f"budget_k must be a positive int, got {self.budget_k}")
        if self.source_proximity_bin < 0:
            raise ValueError("source_proximity_bin must be >= 0")


@dataclass(frozen=True)
class TargetClaimSpan:
    """An atomic claim span on a *target* trace ``x_j`` (REDESIGN_v5 §4.2, refinement 2).

    Carries ONLY the class label and the structural match keys the matched null is
    matched on (proximity bin via distance-to-answer, edit budget). It deliberately
    does **not** carry the target's gold/factuality label, so resolving the anchor
    map cannot leak target-side localization (refinement 2).

    ``is_target_designated`` marks the target's *own* oracle/designated span; the
    transport's collapse guard refuses to anchor onto it (refinement 1), because
    repairing the target's own oracle span would be a within-target oracle repair,
    not a transferred policy.
    """

    span: Span
    g3_class: str
    distance_to_answer: int
    budget_k: int
    is_target_designated: bool = False


@dataclass(frozen=True)
class TargetEdit:
    """A resolved target-side edit: the transported policy anchored on ``x_j``.

    ``source_proximity_bin`` / ``source_budget_k`` record the
    **source-instance-derived content** the anchor map ``T`` read (refinement 1's
    documentation requirement): exactly the proximity bin and budget that crossed
    examples.
    """

    target_example_id: str
    target_span: Span
    policy: RepairPolicy
    g3_class: str
    target_proximity_bin: int
    # source-instance-derived content read by T (refinement 1 documentation):
    source_proximity_bin: int
    source_budget_k: int


@dataclass(frozen=True)
class PositivityFail:
    """The A7 positivity event: no in-class, budget/proximity-matched target span.

    Returned (not raised) by :func:`transport` so the caller can record the
    excluded pair and report the per-class excluded fraction (REDESIGN_v5 §4.4 A7).
    """

    target_example_id: str
    g3_class: str
    reason: str


@dataclass(frozen=True)
class EditedTargetPlan:
    """The typed plan a server run consumes — **no model call is made** (§9).

    Wraps the underlying :class:`interventions.PatchResult` /
    :class:`interventions.ReplayResult` *specification*. The ``patched_states`` of
    a ``PatchResult`` here are computed from the **target's own** residual and
    reference states (Variant C), never the source's.
    """

    target_example_id: str
    target_span: Span
    op: RepairOp
    edit_budget: int
    result: PatchResult | ReplayResult
    g3_class: str
    no_model_call: bool = True


def localized_repair(
    source_span: Span,
    *,
    op: RepairOp,
    alpha: float,
    layer_set: Sequence[int],
    ref_type: ReferenceType = "factual",
    source_proximity_bin: int,
    source_example_id: str | None = None,
    source_g3_class: str | None = None,
    anchor_rule: str = "first_by_position",
) -> RepairPolicy:
    """Induce the **source-derived repair policy** ``rho`` (Variant C, §4.2; MF-2).

    The policy is the recipe ``(op, alpha, L_patch, budget_k, ref_type,
    anchor_rule)`` plus the source proximity bin **and the source G3 class** — *not*
    a residual state. The edit budget is the source span's length
    (``mask``/``replay`` budget) or ``length * |L_patch|`` for ``patch`` (mirrors
    :func:`interventions.patch`'s ``edit_budget``), so the anchor map matches the
    target on the same ``k``. ``source_g3_class`` carries ``class(x_i)`` so the
    within-class estimand condition ``class(source) == class(target)`` is
    enforceable (fail-closed) at :func:`transport` (findings 2/7).
    """
    layers = tuple(layer_set)
    if op == "patch":
        budget_k = source_span.length * max(1, len(layers))
    else:
        budget_k = source_span.length
    return RepairPolicy(
        op=op,
        alpha=alpha,
        layer_set=layers,
        ref_type=ref_type,
        budget_k=budget_k,
        source_proximity_bin=source_proximity_bin,
        anchor_rule=anchor_rule,
        source_example_id=source_example_id,
        source_g3_class=source_g3_class,
    )


def transport(
    policy: RepairPolicy,
    target_example_id: str,
    target_claim_spans: Sequence[TargetClaimSpan],
    g3_class: str,
    proximity_bin_width: int,
    *,
    guard_target_oracle_collapse: bool = True,
) -> TargetEdit | PositivityFail:
    """The frozen anchor/alignment map ``T(S*(x_i), x_j)`` (REDESIGN_v5 §4.2).

    Resolves the target span ``[a_j, b_j]`` by, in order:

    (i)   restricting to atomic claim spans of the **same G3 class** on ``x_j``
          (taxonomy-matched; refinement 2 — no target-label leakage);
    (ii)  selecting the in-class span whose **proximity bin matches** the source
          span's proximity bin (same ``Delta_pos`` grid the matched null uses);
    (iii) matching the **edit budget** ``k`` exactly;
    (iv)  if multiple candidates remain, the **first by position** (deterministic
          tie-break ``anchor_rule``).

    If no in-class, budget-matched, proximity-matched target span exists, the
    **positivity guard fails** and a :class:`PositivityFail` is returned (the A7
    positivity event, §4.4) — *not raised*, so the caller records the excluded
    pair and reports the per-class excluded fraction.

    The **target-side oracle-repair collapse guard** (refinement 1): when
    ``guard_target_oracle_collapse`` is ``True`` (default), candidate spans flagged
    ``is_target_designated`` are excluded, so the transported policy can never
    resolve the target's *own* oracle span (which would make the "transfer" a
    trivial within-target oracle repair rather than a transferred policy).

    The transported edit is computed on the target's **own** reference state (the
    ``ref_type`` carried by ``policy``); no source residual state crosses examples.

    **Self-repair guard (findings 1/6).** The estimand ``R`` (Eq. R) is the
    ``source != target`` mean; G9 requires "an example is never repaired by a policy
    localized on itself" (§5.1). When the policy records its ``source_example_id``,
    ``transport`` **refuses** a target whose id equals it (returns a
    :class:`PositivityFail`, fail-closed) rather than silently admitting a self-pair
    that would inflate ``R_hat`` via a trivial within-example repair.

    **Within-class guard (findings 2/7).** ``R`` is a *within-class* mean, so
    ``class(source) == class(target)`` must hold. When the policy records its
    ``source_g3_class``, ``transport`` refuses any target ``g3_class`` that differs
    from it (fail-closed); the within-class condition is no longer left to a
    caller-supplied target class.
    """
    if proximity_bin_width <= 0:
        raise ValueError(f"proximity_bin_width (Delta_pos) must be positive, got {proximity_bin_width}")

    # self-repair guard: source != target (findings 1/6; G9 §5.1 "never self-repaired").
    if policy.source_example_id is not None and policy.source_example_id == target_example_id:
        return PositivityFail(
            target_example_id,
            g3_class,
            "source_example_id == target_example_id: an example is never repaired by a "
            "policy localized on itself (source != target, G9 §5.1; findings 1/6)",
        )

    # within-class guard: class(source) == class(target) (findings 2/7; Eq. R within-class).
    if policy.source_g3_class is not None and policy.source_g3_class != g3_class:
        return PositivityFail(
            target_example_id,
            g3_class,
            f"class(source)={policy.source_g3_class!r} != class(target)={g3_class!r}: "
            "the repair-transfer estimand R is within-class (Eq. R); cross-class "
            "transport is refused fail-closed (findings 2/7)",
        )

    # (i) same G3 class.
    in_class = [c for c in target_claim_spans if c.g3_class == g3_class]
    if not in_class:
        return PositivityFail(target_example_id, g3_class, "no in-class atomic claim span on target")

    # collapse guard: drop the target's own designated/oracle span (refinement 1).
    if guard_target_oracle_collapse:
        in_class = [c for c in in_class if not c.is_target_designated]
        if not in_class:
            return PositivityFail(
                target_example_id,
                g3_class,
                "only the target's own designated/oracle span is in-class; "
                "anchoring it would collapse transfer to a within-target oracle repair",
            )

    # (ii) proximity-bin match (same Delta_pos grid as the matched null).
    def _prox_bin(d: int) -> int:
        return d // proximity_bin_width

    prox_matched = [
        c for c in in_class if _prox_bin(c.distance_to_answer) == policy.source_proximity_bin
    ]
    if not prox_matched:
        return PositivityFail(
            target_example_id,
            g3_class,
            f"no in-class span in source proximity bin {policy.source_proximity_bin}",
        )

    # (iii) exact edit-budget match.
    budget_matched = [c for c in prox_matched if c.budget_k == policy.budget_k]
    if not budget_matched:
        return PositivityFail(
            target_example_id,
            g3_class,
            f"no in-class, proximity-matched span at budget k={policy.budget_k}",
        )

    # (iv) deterministic tie-break: first by position.
    budget_matched.sort(key=lambda c: (c.span.a, c.span.b))
    chosen = budget_matched[0]

    return TargetEdit(
        target_example_id=target_example_id,
        target_span=chosen.span,
        policy=policy,
        g3_class=g3_class,
        target_proximity_bin=_prox_bin(chosen.distance_to_answer),
        source_proximity_bin=policy.source_proximity_bin,
        source_budget_k=policy.budget_k,
    )


def apply(
    target_edit: TargetEdit,
    *,
    target_residual_states: Sequence[Sequence[float]] | None = None,
    target_reference_states: Sequence[Sequence[float]] | None = None,
    target_suffix_length: int = 0,
) -> EditedTargetPlan:
    """Wrap :func:`interventions.patch` / :func:`interventions.replay` (§9).

    Returns the typed :class:`EditedTargetPlan` a server run would consume. **No
    model call is made**; for ``patch`` the convex interpolation is computed from
    the **target's own** residual and reference states (Variant C, refinement 1),
    and for ``replay`` only the rollback/redecode plan is recorded (the re-decode
    is a forward pass that is NOT executed here, mirroring v4 ``replay``).

    Raises
    ------
    ValueError
        If a ``patch`` plan is requested without the target's own residual /
        reference states (Variant C requires the target's own state — refusing to
        synthesize one is the positivity/collapse safeguard at the apply layer).
    """
    policy = target_edit.policy
    span = target_edit.target_span
    if policy.op == "patch":
        if target_residual_states is None or target_reference_states is None:
            raise ValueError(
                "Variant C patch requires the TARGET's own residual + reference "
                "states; no source state crosses examples (refinement 1)"
            )
        result = patch(
            target_residual_states,
            target_reference_states,
            span,
            rho=policy.alpha,
            reference_type=policy.ref_type,
            layer_set=policy.layer_set,
        )
    else:  # replay
        result = replay(
            span,
            reference_type=policy.ref_type,
            suffix_length=target_suffix_length,
        )
    return EditedTargetPlan(
        target_example_id=target_edit.target_example_id,
        target_span=span,
        op=policy.op,
        edit_budget=result.edit_budget,
        result=result,
        g3_class=target_edit.g3_class,
        no_model_call=True,
    )


@dataclass(frozen=True)
class OperatorGrid:
    """The frozen, pre-enumerated operator-selection grid (OS-2, REDESIGN_v5 §4.7).

    Every degree of freedom in the repair policy ``rho`` is a selection layer
    (OS-1). The set of policies the freeze rule **could** have chosen — the discrete
    Cartesian grid ``ops x alphas x L_patch sets x ref_types`` — has a bounded
    cardinality ``K_op`` that must be **enumerated before data** (§4.7 OS-2). ``K_op``
    is **derived** from this grid (:func:`operator_grid_cardinality` /
    :func:`select_operator`), never caller-declared (finding 16). ``alphas`` defaults
    to the frozen :data:`interventions.PATCH_RHO_LEVELS` (OS-1: ``alpha in
    PATCH_RHO_LEVELS``); ``alphas`` applies only to ``patch`` (``replay`` ignores it),
    so the cardinality counts ``patch`` policies over the alpha grid plus ``replay``
    policies once per (L_patch, ref_type).
    """

    ops: tuple[RepairOp, ...] = REPAIR_OP_CHOICES
    alphas: tuple[float, ...] = PATCH_RHO_LEVELS
    layer_sets: tuple[tuple[int, ...], ...] = ((0,),)
    ref_types: tuple[ReferenceType, ...] = REF_TYPE_CHOICES

    def __post_init__(self) -> None:
        if not self.ops or any(o not in REPAIR_OP_CHOICES for o in self.ops):
            raise ValueError(f"ops must be a non-empty subset of {REPAIR_OP_CHOICES}")
        if "patch" in self.ops:
            if not self.alphas:
                raise ValueError("alphas must be non-empty when 'patch' is in the grid")
            for a in self.alphas:
                if not any(math.isclose(a, lvl, abs_tol=1e-12) for lvl in PATCH_RHO_LEVELS):
                    raise ValueError(
                        f"grid alpha {a} not in frozen PATCH_RHO_LEVELS {PATCH_RHO_LEVELS} (OS-1)"
                    )
        if not self.layer_sets:
            raise ValueError("layer_sets must be non-empty")
        if not self.ref_types or any(r not in REF_TYPE_CHOICES for r in self.ref_types):
            raise ValueError(f"ref_types must be a non-empty subset of {REF_TYPE_CHOICES}")

    def enumerate_policies(self) -> list[tuple[RepairOp, float, tuple[int, ...], ReferenceType]]:
        """Enumerate the reachable ``(op, alpha, layer_set, ref_type)`` recipes.

        For ``replay`` the ``alpha`` is fixed to ``1.0`` (ignored by the operator) so
        a single ``replay`` recipe is counted per ``(layer_set, ref_type)`` rather than
        once per redundant alpha — the honest reachable-recipe count.
        """
        recipes: list[tuple[RepairOp, float, tuple[int, ...], ReferenceType]] = []
        for op in self.ops:
            for ls in self.layer_sets:
                for rt in self.ref_types:
                    if op == "patch":
                        for a in self.alphas:
                            recipes.append((op, a, tuple(ls), rt))
                    else:  # replay ignores alpha -> count once (alpha fixed to 1.0)
                        recipes.append((op, 1.0, tuple(ls), rt))
        return recipes


@dataclass(frozen=True)
class OperatorSelection:
    """The OS-1 freeze outcome on ``V_sel`` + the derived ``K_op`` selection event."""

    policy: RepairPolicy
    k_op: int
    score: float


def operator_grid_cardinality(grid: OperatorGrid) -> int:
    """Derive ``K_op`` from the frozen operator grid (OS-2; finding 16).

    ``K_op`` is the number of reachable policy recipes the freeze rule could have
    chosen — **derived** from the enumerated grid, not declared by the caller. This is
    the factor the SI-2 Bonferroni level pays (:func:`selective_inference.holm_alpha`).
    """
    return len(grid.enumerate_policies())


def select_operator(
    grid: OperatorGrid,
    score_on_v_sel,
    *,
    source_proximity_bin: int,
    source_span_length: int,
    source_example_id: str | None = None,
    source_g3_class: str | None = None,
) -> OperatorSelection:
    """Frozen operator-selection function (OS-1 freeze on ``V_sel``; finding 16).

    Walks the **pre-enumerated** :class:`OperatorGrid`, scores each reachable recipe
    with ``score_on_v_sel(policy) -> float`` (the registered objective, e.g.
    ``R_hat(PROPOSED) - B4`` on ``V_sel``), and returns the **argmax** policy with
    ties broken by the frozen rule: **smallest budget**, then **patch over replay**,
    then **smallest alpha**, then **first ref_type** (deterministic, §4.7 OS-1). The
    returned ``k_op`` is **derived** from the grid cardinality
    (:func:`operator_grid_cardinality`) — never caller-declared — so the ``K_op`` Holm
    fold is enforced from the actual selection event (finding 16).

    Pure Python; no model call. ``source_span_length`` is used to compute each
    candidate policy's ``budget_k`` exactly as :func:`localized_repair` does.
    """
    recipes = grid.enumerate_policies()
    if not recipes:
        raise ValueError("empty operator grid")
    k_op = len(recipes)

    best: tuple | None = None
    best_policy: RepairPolicy | None = None
    best_score = float("-inf")
    for (op, alpha, layer_set, ref_type) in recipes:
        if op == "patch":
            budget_k = source_span_length * max(1, len(layer_set))
        else:
            budget_k = source_span_length
        policy = RepairPolicy(
            op=op,
            alpha=alpha,
            layer_set=tuple(layer_set),
            ref_type=ref_type,
            budget_k=budget_k,
            source_proximity_bin=source_proximity_bin,
            source_example_id=source_example_id,
            source_g3_class=source_g3_class,
        )
        score = float(score_on_v_sel(policy))
        # frozen tie-break key: maximize score; then smallest budget; then patch(0) <
        # replay(1); then smallest alpha; then ref_type order.
        op_rank = 0 if op == "patch" else 1
        rt_rank = REF_TYPE_CHOICES.index(ref_type)
        key = (-score, budget_k, op_rank, alpha, rt_rank)
        if best is None or key < best:
            best = key
            best_policy = policy
            best_score = score
    assert best_policy is not None
    return OperatorSelection(policy=best_policy, k_op=k_op, score=best_score)


def tracedet_ar_span_adapter(
    temporal_entropy: Sequence[float],
    *,
    threshold: float,
    min_run: int = 1,
) -> Span:
    """Base-TraceDet -> AR claim-span adapter for baseline **B1** (refinement 3).

    Base TraceDet (an information-bottleneck sub-trace) localizes a D-LLM
    *temporal-entropy* sub-trace. On an AR-LLM there is no per-step temporal
    entropy schedule, so this adapter specifies the AR token/claim-span mapping:
    select the **longest contiguous run** of decode steps whose per-step entropy is
    at or above ``threshold`` (the high-entropy sub-trace), and return it as a
    contiguous token :class:`Span`. The selected source span then runs through the
    **identical** ``localized_repair`` / ``transport`` / ``apply`` pipeline as
    PROPOSED, so B1 is executable on the AR lead (REDESIGN_v5 §4.3 B1).

    Ties (equal-length runs) break toward the **earliest** run (deterministic).

    Raises
    ------
    ValueError
        If no decode step clears ``threshold`` (no high-entropy sub-trace; the
        baseline abstains on this example — the caller records it as a B1
        positivity/abstention event).
    """
    ent = [float(v) for v in temporal_entropy]
    if not ent:
        raise ValueError("temporal_entropy is empty; no AR sub-trace to adapt")
    best_start = best_len = -1
    cur_start = cur_len = 0
    in_run = False
    for i, e in enumerate(ent):
        if e >= threshold:
            if not in_run:
                in_run = True
                cur_start = i
                cur_len = 1
            else:
                cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            in_run = False
    if best_len < min_run:
        raise ValueError(
            f"no high-entropy sub-trace clears threshold {threshold} for min_run {min_run}; "
            "B1 abstains on this example"
        )
    return Span(best_start, best_start + best_len - 1)


def _policy_payload(policy: RepairPolicy) -> dict:
    return {
        "op": policy.op,
        "alpha": policy.alpha,
        "layer_set": list(policy.layer_set),
        "ref_type": policy.ref_type,
        "budget_k": policy.budget_k,
        "source_proximity_bin": policy.source_proximity_bin,
        "anchor_rule": policy.anchor_rule,
    }


def policy_hash(policy: RepairPolicy) -> str:
    """SHA-256 ``repair_policy_hash`` for ``CIURecord`` (REDESIGN_v5 §9 / §5).

    Excludes ``source_example_id`` so the hash addresses the *policy recipe*
    (which is what crosses examples), not which source induced it.
    """
    payload = json.dumps(_policy_payload(policy), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def transport_map_hash(
    policy: RepairPolicy,
    proximity_bin_width: int,
    *,
    guard_target_oracle_collapse: bool = True,
) -> str:
    """SHA-256 ``transport_map_hash`` for ``CIURecord`` (frozen anchor map ``T``).

    Addresses the anchor/alignment rule (the policy recipe + the ``Delta_pos`` grid
    + the collapse guard + the tie-break) so the realised transport is reproducible
    (REDESIGN_v5 §9, §10).
    """
    payload = json.dumps(
        {
            "policy": _policy_payload(policy),
            "proximity_bin_width": proximity_bin_width,
            "guard_target_oracle_collapse": guard_target_oracle_collapse,
            "transport_variant": "C",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
