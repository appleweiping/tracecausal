"""Intervention operators for the CIU (Counterfactual Intervention Utility) core.

This module is **additive** and implements the operational ``mask`` / ``patch`` /
``replay`` operators specified in ``docs/redesign/REDESIGN_v3.md`` §2.2 and refined
by ``docs/redesign/REDESIGN_v4.md`` §2.12 (the ``displaced_mass`` dose-response).

Design discipline (BUILD-NOW / RUN-LATER):

* **Nothing here loads a model, touches a GPU, or runs an experiment.** The
  operators are *pure, framework-agnostic specifications* that compute the exact
  edit a real decoder would apply, given the already-extracted attention
  weights / residual states / decoder trajectory passed in by the caller. The
  numerical transforms (softmax renormalisation, convex residual interpolation,
  trajectory rollback bookkeeping) are exact and deterministic, so they are
  unit-testable on synthetic fixtures without a server run.
* Each operator returns a **typed, frozen result** carrying the realised edit,
  the edit budget ``c(I_S^theta) = k`` (which the matched null pool must match,
  §2.3), and — for ``mask`` — the ``displaced_mass`` measure (§2.2 sanity field,
  §2.12 OOD dose-response axis).

The exact KV/attention/logit behaviour follows §2.2 verbatim: ``mask`` excludes
keys ``[a, b]`` from the attention mixture by sending their pre-softmax logits to
``-inf`` and renormalising the softmax over the surviving keys, **without**
re-indexing surviving positions (``mask_position_policy: keep_absolute``) and
**without** editing the residual stream, RoPE ids, or the logit head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

__all__ = [
    "Span",
    "InterventionResult",
    "MaskResult",
    "PatchResult",
    "ReplayResult",
    "mask",
    "patch",
    "replay",
    "MASK_POSITION_POLICY",
]

# Position policy recorded with every ``mask`` (REDESIGN_v3 §2.2): downstream
# tokens keep their original absolute indices; only the *attendable set* shrinks.
MASK_POSITION_POLICY: Literal["keep_absolute"] = "keep_absolute"

ReferenceType = Literal["factual", "neutral"]

# The five reference-injection strengths rho for ``patch`` (REDESIGN_v3 §2.2/§2.7).
# rho is the convex mixing weight alpha in h <- (1-alpha) h + alpha h_ref; the
# five levels sweep from a vanishing nudge to a full reference swap.
PATCH_RHO_LEVELS: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Span:
    """A contiguous, half-open-by-convention token span ``[a, b]`` (inclusive ``b``).

    ``a`` and ``b`` are absolute token indices into the trace. ``length`` is the
    number of positions ``b - a + 1`` (the ``mask`` edit budget; see ``mask``).
    """

    a: int
    b: int

    def __post_init__(self) -> None:
        if self.a < 0:
            raise ValueError(f"span start must be >= 0, got a={self.a}")
        if self.b < self.a:
            raise ValueError(f"span must satisfy b >= a, got a={self.a}, b={self.b}")

    @property
    def length(self) -> int:
        """Number of positions in ``[a, b]`` (inclusive)."""
        return self.b - self.a + 1

    def positions(self) -> tuple[int, ...]:
        return tuple(range(self.a, self.b + 1))

    def disjoint_from(self, other: "Span") -> bool:
        return self.b < other.a or other.b < self.a


@dataclass(frozen=True)
class InterventionResult:
    """Common base for every operator result.

    Attributes
    ----------
    operator:
        ``"mask"`` | ``"patch"`` | ``"replay"`` | ``"no_op"``.
    span:
        The edited span ``[a, b]``.
    edit_budget:
        ``k = c(I_S^theta)``. The matched null pool ``Pi_i`` must contain only
        spans with the *same* ``edit_budget`` (REDESIGN_v3 §2.3).
    invalid:
        ``True`` if the edit was degenerate (e.g. empty attendable set, NaN
        logits). Persisted into ``CIURecord.invalid_count`` (REDESIGN_v3 §2.2).
    reason_code:
        Optional reason string when ``invalid`` is ``True`` (per
        ``intervention_protocol.md``).
    """

    operator: str
    span: Span
    edit_budget: int
    invalid: bool = False
    reason_code: str | None = None


@dataclass(frozen=True)
class MaskResult(InterventionResult):
    """Result of a ``mask`` (necessity) edit.

    Attributes
    ----------
    renormalised_weights:
        The post-mask attention distribution over the *surviving* keys for the
        observed query rows. Each row sums to ``1.0`` (or is empty / flagged
        invalid when the attendable set is empty).
    displaced_mass:
        The pre-mask softmax mass that *would have* gone to ``[a, b]``, averaged
        over the observed query rows. Near-zero ``displaced_mass`` means ``S`` was
        barely attended to and the mask is near-vacuous (REDESIGN_v3 §2.2 sanity
        field; REDESIGN_v4 §2.12 dose-response abscissa).
    near_vacuous:
        ``True`` when ``displaced_mass`` is below the near-vacuous guard threshold
        (REDESIGN_v4 §2.12 "near-vacuous-mask guard").
    """

    renormalised_weights: tuple[tuple[float, ...], ...] = field(default_factory=tuple)
    displaced_mass: float = 0.0
    near_vacuous: bool = False


@dataclass(frozen=True)
class PatchResult(InterventionResult):
    """Result of a ``patch`` (sufficiency, reference-state injection) edit."""

    reference_type: ReferenceType = "factual"
    rho: float = 1.0
    n_layers: int = 1
    patched_states: tuple[tuple[float, ...], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ReplayResult(InterventionResult):
    """Result of a ``replay`` (trajectory rollback + re-decode) edit."""

    reference_type: ReferenceType = "factual"
    rollback_index: int = 0
    redecoded_positions: tuple[int, ...] = field(default_factory=tuple)


# Numerical guard below which a mask is treated as near-vacuous (§2.12 guard).
_NEAR_VACUOUS_DISPLACED_MASS = 1e-3


def mask(
    attention_weights: Sequence[Sequence[float]],
    span: Span,
    *,
    near_vacuous_threshold: float = _NEAR_VACUOUS_DISPLACED_MASS,
) -> MaskResult:
    """Apply the exact KV/attention ``mask`` of REDESIGN_v3 §2.2.

    For each query row ``alpha_t`` (a probability vector over key positions),
    the keys in ``[a, b]`` are excluded from the attention mixture (their
    pre-softmax logits are sent to ``-inf``) and the distribution is
    **renormalised over the surviving keys** ``j not in [a, b]``. This matches
    "produce the suffix as if positions ``[a, b]`` were absent from the context"
    *without* shifting surviving position ids (``keep_absolute``), editing the
    residual stream, or touching the logit head.

    The caller supplies the *post-softmax* attention weights (rows already sum to
    ``1``); renormalisation is therefore the exact operation a real decoder's
    post-mask softmax would yield, computed here without a model.

    Parameters
    ----------
    attention_weights:
        ``[n_query_rows][n_key_positions]`` post-softmax attention weights.
    span:
        The key span ``[a, b]`` to mask out. ``edit_budget = span.length``.
    near_vacuous_threshold:
        ``displaced_mass`` below this flags ``near_vacuous=True`` (§2.12 guard).

    Returns
    -------
    MaskResult
        With ``renormalised_weights``, the ``displaced_mass`` measure, and the
        ``near_vacuous`` flag. If *any* query row has an empty attendable set
        (all surviving weights sum to ``0``), the result is flagged ``invalid``
        with ``reason_code="empty_attendable_set"`` (REDESIGN_v3 §2.2).
    """
    masked = set(span.positions())
    renormalised: list[tuple[float, ...]] = []
    displaced_total = 0.0
    n_rows = 0
    empty_attendable = False

    for row in attention_weights:
        n_rows += 1
        displaced = sum(w for j, w in enumerate(row) if j in masked)
        surviving = [(0.0 if j in masked else float(w)) for j, w in enumerate(row)]
        survive_sum = sum(surviving)
        if survive_sum <= 0.0:
            # Entire attention mass was on [a, b]; the suffix has nothing to
            # attend to -> degenerate edit (empty attendable set).
            empty_attendable = True
            renormalised.append(tuple(surviving))
        else:
            renormalised.append(tuple(w / survive_sum for w in surviving))
        displaced_total += displaced

    displaced_mass = displaced_total / n_rows if n_rows else 0.0

    return MaskResult(
        operator="mask",
        span=span,
        edit_budget=span.length,
        invalid=empty_attendable,
        reason_code="empty_attendable_set" if empty_attendable else None,
        renormalised_weights=tuple(renormalised),
        displaced_mass=displaced_mass,
        near_vacuous=displaced_mass < near_vacuous_threshold,
    )


def patch(
    residual_states: Sequence[Sequence[float]],
    reference_states: Sequence[Sequence[float]],
    span: Span,
    *,
    rho: float,
    reference_type: ReferenceType = "factual",
    layer_set: Sequence[int] | None = None,
) -> PatchResult:
    """Apply the exact ``patch`` (sufficiency) edit of REDESIGN_v3 §2.2.

    For ``t in [a, b]`` and each layer ``l`` in the patch layer-set:
    ``h_t^(l) <- (1 - alpha) h_t^(l) + alpha h_t^(l),ref`` with ``alpha = rho``.
    The convex interpolation injects a factual/neutral reference state at one of
    the five ``rho`` levels (``PATCH_RHO_LEVELS``). Edit budget is the number of
    patched coordinates ``|[a, b]| * |L_patch|``.

    ``residual_states`` and ``reference_states`` are ``[n_positions][hidden]``
    arrays aligned to ``span.positions()`` (one row per patched position). The
    method is pure; a real run would supply model states.
    """
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"patch rho (alpha) must be in (0, 1], got {rho}")
    if len(residual_states) != span.length:
        raise ValueError(
            "residual_states must have one row per patched position "
            f"({span.length}), got {len(residual_states)}"
        )
    if len(reference_states) != len(residual_states):
        raise ValueError("reference_states must align with residual_states row count")

    layers = tuple(layer_set) if layer_set is not None else (0,)
    n_layers = max(1, len(layers))

    patched: list[tuple[float, ...]] = []
    invalid = False
    for h_row, ref_row in zip(residual_states, reference_states):
        if len(h_row) != len(ref_row):
            raise ValueError("residual and reference state widths must match")
        mixed = tuple((1.0 - rho) * float(h) + rho * float(r) for h, r in zip(h_row, ref_row))
        if any(v != v for v in mixed):  # NaN guard
            invalid = True
        patched.append(mixed)

    return PatchResult(
        operator="patch",
        span=span,
        edit_budget=span.length * n_layers,
        invalid=invalid,
        reason_code="nan_state" if invalid else None,
        reference_type=reference_type,
        rho=rho,
        n_layers=n_layers,
        patched_states=tuple(patched),
    )


def replay(
    span: Span,
    *,
    reference_type: ReferenceType = "factual",
    suffix_length: int = 0,
) -> ReplayResult:
    """Specify the ``replay`` (trajectory) edit of REDESIGN_v3 §2.2.

    Roll back the decoder state to *before* position ``a``, re-decode ``[a, b]``
    under a reference policy, then free-run the suffix. This function records the
    rollback/redecode bookkeeping (which positions are re-decoded, the edit
    budget = number of re-decoded positions). The actual re-decode is a model
    forward pass and is **not** executed here; this returns the typed plan that a
    server run would consume.
    """
    if suffix_length < 0:
        raise ValueError(f"suffix_length must be >= 0, got {suffix_length}")
    redecoded = span.positions()
    return ReplayResult(
        operator="replay",
        span=span,
        edit_budget=span.length,
        invalid=False,
        reason_code=None,
        reference_type=reference_type,
        rollback_index=span.a,
        redecoded_positions=redecoded,
    )
