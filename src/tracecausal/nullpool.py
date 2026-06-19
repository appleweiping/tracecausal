"""Proximity-stratified matched null-pool sampler ``Pi_i`` (REDESIGN_v3 §2.3, §2.10;
REDESIGN_v4 §4.6 B3).

The per-example matched null pool is

    Pi_i(S) := Uniform{ S' in example x_i :  c(I_{S'}^theta) = k,
                                             len(S') = len(S),
                                             layer-set L_patch identical,
                                             S' disjoint from S,
                                             reference run identical (same ref_hash),
                                             same example x_i }

REDESIGN_v4 §2.10/§4.6 additionally requires the draw to be **proximity-
stratified**: a candidate ``S'`` must share ``S*``'s distance-to-answer bin, so
the positional-leakage term the leakage bound ``beta * Delta_pos`` controls
(§2.11, G7) is held fixed across the contrast. The proximity bin width
``Delta_pos`` is the shared knob between the G7 leakage bound and the §4.6(B3)
pool-shrinkage variance.

This module is **additive** and **pure**: it constructs the candidate pool and a
deterministic-seeded uniform draw over residual-stream positions. It does not
load a model or run anything. ``null_pool_hash`` (used by ``CIURecord``) is
produced here by ``serialize_pool`` so the realised estimand is reproducible
(REDESIGN_v3 §2.3, §5).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Sequence

from .interventions import Span

__all__ = [
    "CandidateSpan",
    "NullPool",
    "build_null_pool",
    "sample_matched_null",
    "serialize_pool",
    "pool_hash",
    "proximity_bin",
    "PROXIMITY_POOL_MIN",
]

# Floor on the mean in-bin pool size below which the proximity bin is coarsened
# one step (REDESIGN_v4 §4.6 B3): if mean m_pool < 8 the stratifier shrank the
# pool too far and inflates the random-arm variance unacceptably.
PROXIMITY_POOL_MIN: int = 8


@dataclass(frozen=True)
class CandidateSpan:
    """A candidate null span ``S'`` inside one example, with its match keys.

    Attributes
    ----------
    span:
        The candidate ``[a, b]``.
    layer_set:
        The patch layer-set ``L_patch`` (must equal ``S*``'s for a valid match).
    ref_hash:
        Reference-run identity (must equal ``S*``'s, REDESIGN_v3 §2.3).
    distance_to_answer:
        Non-negative token distance from the span to the answer region; used for
        proximity stratification (REDESIGN_v4 §2.10).
    """

    span: Span
    layer_set: tuple[int, ...]
    ref_hash: str
    distance_to_answer: int


@dataclass(frozen=True)
class NullPool:
    """A realised per-example matched null pool ``Pi_i(S*)``.

    Attributes
    ----------
    example_id:
        The example ``x_i`` this pool belongs to.
    target_budget:
        The edit budget ``k`` every member matches (REDESIGN_v3 §2.3).
    target_length:
        The span length every member matches.
    target_layer_set:
        ``L_patch`` every member matches.
    target_ref_hash:
        Reference-run identity every member matches.
    proximity_bin:
        The distance-to-answer bin index every member shares with ``S*``
        (``None`` when proximity stratification is disabled).
    proximity_bin_width:
        ``Delta_pos`` — the bin width used (token units). Feeds the G7 leakage
        bound ``beta_hi * Delta_pos`` (REDESIGN_v4 §2.11).
    members:
        The candidate spans that satisfy every match constraint, disjoint from
        ``S*``.
    """

    example_id: str
    target_budget: int
    target_length: int
    target_layer_set: tuple[int, ...]
    target_ref_hash: str
    proximity_bin: int | None
    proximity_bin_width: int
    members: tuple[CandidateSpan, ...]

    @property
    def size(self) -> int:
        """``m_pool(i)`` — the realised in-bin pool size (REDESIGN_v4 §4.6 B3)."""
        return len(self.members)


def proximity_bin(distance_to_answer: int, bin_width: int) -> int:
    """Return the proximity bin index for ``distance_to_answer`` (REDESIGN_v4 §2.10).

    Bins are ``[0, w), [w, 2w), ...`` so a smaller ``bin_width`` = finer
    stratification = tighter leakage bound but smaller pools (the §4.6 B3 trade).
    """
    if bin_width <= 0:
        raise ValueError(f"bin_width (Delta_pos) must be positive, got {bin_width}")
    if distance_to_answer < 0:
        raise ValueError(f"distance_to_answer must be >= 0, got {distance_to_answer}")
    return distance_to_answer // bin_width


def build_null_pool(
    example_id: str,
    target: Span,
    target_layer_set: Sequence[int],
    target_ref_hash: str,
    target_distance_to_answer: int,
    candidates: Sequence[CandidateSpan],
    *,
    proximity_bin_width: int = 0,
) -> NullPool:
    """Construct the realised matched null pool ``Pi_i(S*)`` (REDESIGN_v3 §2.3).

    A candidate ``S'`` is admitted iff it matches ``S*`` on **all** of: edit
    budget ``k`` (here ``target.length``), span length, layer-set ``L_patch``,
    reference identity ``ref_hash``, and is **disjoint** from ``S*`` — and, when
    ``proximity_bin_width > 0``, additionally shares ``S*``'s distance-to-answer
    bin (REDESIGN_v4 §2.10).

    Parameters
    ----------
    proximity_bin_width:
        ``Delta_pos`` in token units; ``0`` disables proximity stratification
        (back-compatible with the v3 unstratified pool).

    Notes
    -----
    The pool is built deterministically (it is a *filter*, not a sample); the
    random draw happens in ``sample_matched_null``. The constraints exactly
    mirror REDESIGN_v3 §2.3 so the contrast is matched on operator, budget,
    length, layers, reference, and example — only *which positions are edited*
    varies.
    """
    target_layers = tuple(target_layer_set)
    target_bin: int | None
    if proximity_bin_width > 0:
        target_bin = proximity_bin(target_distance_to_answer, proximity_bin_width)
    else:
        target_bin = None

    members: list[CandidateSpan] = []
    for cand in candidates:
        if cand.span.length != target.length:
            continue  # length + budget match (budget == length for mask)
        if tuple(cand.layer_set) != target_layers:
            continue  # identical layer-set
        if cand.ref_hash != target_ref_hash:
            continue  # identical reference run
        if not cand.span.disjoint_from(target):
            continue  # disjointness from S*
        if target_bin is not None:
            if proximity_bin(cand.distance_to_answer, proximity_bin_width) != target_bin:
                continue  # proximity stratification
        members.append(cand)

    # Stable ordering by span position makes the serialisation/hash deterministic.
    members.sort(key=lambda c: (c.span.a, c.span.b))

    return NullPool(
        example_id=example_id,
        target_budget=target.length,
        target_length=target.length,
        target_layer_set=target_layers,
        target_ref_hash=target_ref_hash,
        proximity_bin=target_bin,
        proximity_bin_width=proximity_bin_width,
        members=tuple(members),
    )


def sample_matched_null(pool: NullPool, n_draws: int, *, seed: int) -> tuple[CandidateSpan, ...]:
    """Draw ``n_draws`` matched random controls ``tilde S ~ Pi_i`` (REDESIGN_v3 §2.3).

    Uniform **with replacement** over the pool members (the estimand is the pool
    mean ``bar tau_i(Pi)``; with-replacement keeps each draw an i.i.d. uniform
    sample so the MC average is unbiased for the pool mean, Prop. 2.5a). The draw
    is deterministic given ``seed`` for reproducibility.

    Raises
    ------
    ValueError
        If the pool is empty (no matched control exists; the caller must coarsen
        the proximity bin or flag the example invalid).
    """
    if pool.size == 0:
        raise ValueError(
            f"empty matched null pool for example {pool.example_id!r}; "
            "coarsen proximity bin (REDESIGN_v4 §4.6 B3) or flag invalid"
        )
    if n_draws <= 0:
        raise ValueError(f"n_draws must be positive, got {n_draws}")
    rng = random.Random(seed)
    return tuple(rng.choice(pool.members) for _ in range(n_draws))


def serialize_pool(pool: NullPool) -> str:
    """Serialise a pool to canonical JSON for hashing (REDESIGN_v3 §2.3, §5)."""
    payload = {
        "example_id": pool.example_id,
        "target_budget": pool.target_budget,
        "target_length": pool.target_length,
        "target_layer_set": list(pool.target_layer_set),
        "target_ref_hash": pool.target_ref_hash,
        "proximity_bin": pool.proximity_bin,
        "proximity_bin_width": pool.proximity_bin_width,
        "members": [
            {
                "a": c.span.a,
                "b": c.span.b,
                "layer_set": list(c.layer_set),
                "ref_hash": c.ref_hash,
                "distance_to_answer": c.distance_to_answer,
            }
            for c in pool.members
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def pool_hash(pool: NullPool) -> str:
    """SHA-256 ``null_pool_hash`` for ``CIURecord`` (REDESIGN_v3 §2.3, §5)."""
    return hashlib.sha256(serialize_pool(pool).encode("utf-8")).hexdigest()
