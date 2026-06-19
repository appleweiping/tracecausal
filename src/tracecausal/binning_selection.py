"""Data-adaptive binning selection as code, not prose (REDESIGN_v5 §6.4).

v4 *adapted* the proximity-bin width ``Delta_pos`` (coarsen until ``bar m_pool >=
POOL_MIN``) and the ``displaced_mass`` bin edges, but the *selection procedure* was
not a frozen, unit-tested function emitting the selection event the SI correction
must condition on (REDESIGN_v5 §1.2 / §6.4 — the "prose-ahead-of-code" gap).

This module closes that gap. ``select_binning(v_sel) -> (binning, selection_event)``
walks a **finite, ordered coarsening ladder** for ``Delta_pos`` and selects the
``displaced_mass`` edges, returning BOTH the chosen value AND the
:class:`SelectionEvent` (the ladder walked / the candidate edge set), so the SI
multiplicity factors ``K_bin`` are computable from the recorded event rather than
narrated. **All choices are made on ``V_sel`` only** (SI-1 split discipline).

Pure Python; no model, no GPU, no run. Mirrors ``nullpool.POOL_MIN`` /
``ciu.PROXIMITY_POOL_MIN`` (the §4.6 B3 floor of 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "Binning",
    "SelectionEvent",
    "select_binning",
    "DEFAULT_DELTA_POS_LADDER",
    "DEFAULT_DISPLACED_MASS_EDGES",
    "POOL_MIN",
]

# The §4.6 B3 mean in-bin pool floor; mirrors nullpool.POOL_MIN / ciu.PROXIMITY_POOL_MIN.
POOL_MIN: int = 8

# The finite, ordered coarsening ladder for Delta_pos (token units), FINEST first.
# Coarsening walks toward larger widths until the mean in-bin pool clears POOL_MIN.
# Its length bounds K_bin (REDESIGN_v5 §6.2 / OS-2 analogue for binning).
DEFAULT_DELTA_POS_LADDER: tuple[int, ...] = (1, 2, 4, 8, 16)

# The candidate displaced_mass edge set (REDESIGN_v4 §2.12; carried into v5). The
# number of candidate edge sets the selection could choose contributes to K_bin.
DEFAULT_DISPLACED_MASS_EDGES: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4, 1.0)


@dataclass(frozen=True)
class Binning:
    """The frozen binning chosen on ``V_sel`` (REDESIGN_v5 §6.4)."""

    delta_pos: int
    displaced_mass_edges: tuple[float, ...]
    mean_pool_at_choice: float
    meets_pool_floor: bool


@dataclass(frozen=True)
class SelectionEvent:
    """The recorded selection event the SI correction conditions on (§6.4 / §6.2).

    ``k_bin`` is the number of binning configurations the data-adaptive rule **could**
    have chosen over the **full pre-enumerated** ladder (the entire
    ``delta_pos_ladder`` length times the candidate edge sets), enumerated **before**
    data so the SI-2 Bonferroni factor is honest (REDESIGN_v5 §6.2: "K_bin/K_op
    bounded and enumerated BEFORE data"). It is deliberately NOT the data-dependent
    ``rungs_walked`` prefix (which short-circuits at the realised pool floor and would
    under-penalise multiplicity — findings 14, 20). Under SI-1 (selection split) the
    inference family uses ``k_bin = 1`` because the choice was made on ``V_sel``;
    ``k_bin`` here is the *full reachable cardinality*, surfaced for the SI-2 fallback
    and for the record. ``rungs_walked`` is still recorded (for provenance/the chosen
    width), but it does **not** size ``k_bin``.
    """

    delta_pos_ladder: tuple[int, ...]
    rungs_walked: tuple[int, ...]  # the Delta_pos values examined, in order
    mean_pool_per_rung: tuple[float, ...]
    displaced_mass_edge_candidates: tuple[tuple[float, ...], ...]
    k_bin: int
    chosen_delta_pos: int
    chosen_edges: tuple[float, ...]


def _mean_pool(delta_pos: int, distances_to_answer: Sequence[int]) -> float:
    """Mean **matched-null** pool size at width ``delta_pos`` over the ``V_sel`` distances.

    Each example's proximity bin is ``distance // delta_pos``. The matched-null pool
    ``Pi`` for a given example is the set of *other* in-budget spans sharing its bin —
    an example is **never its own matched null**. So the pool size seen by a member is
    its bin's count **minus one** (itself); a bin of count 1 yields an **empty** pool
    (0), not a pool of 1 (finding 15). Folding the member into the count over-states
    the pool by exactly 1 per example and could let a width clear the §4.6 B3 floor
    that the real matched-null pool does not. A coarser width pools more spans per bin,
    raising the mean — the §4.6 B3 trade.
    """
    if delta_pos <= 0:
        raise ValueError(f"delta_pos must be positive, got {delta_pos}")
    if not distances_to_answer:
        return 0.0
    bins = [d // delta_pos for d in distances_to_answer]
    counts: dict[int, int] = {}
    for bdx in bins:
        counts[bdx] = counts.get(bdx, 0) + 1
    # mean matched-null pool size as seen by a member = mean over members of
    # (their bin's count - 1) -- the member itself is excluded (finding 15).
    return sum(counts[bdx] - 1 for bdx in bins) / len(bins)


def select_binning(
    v_sel_distances_to_answer: Sequence[int],
    *,
    delta_pos_ladder: Sequence[int] = DEFAULT_DELTA_POS_LADDER,
    displaced_mass_edge_candidates: Sequence[Sequence[float]] = (DEFAULT_DISPLACED_MASS_EDGES,),
    pool_floor: int = POOL_MIN,
) -> tuple[Binning, SelectionEvent]:
    """Frozen data-adaptive binning selection on ``V_sel`` (REDESIGN_v5 §6.4).

    Walks ``delta_pos_ladder`` (finest first) and picks the **first (finest)**
    ``Delta_pos`` whose mean in-bin pool clears ``pool_floor`` (the §4.6 B3 rule:
    coarsen only as far as needed). If no rung clears the floor, the **coarsest**
    rung is selected and ``meets_pool_floor=False`` is recorded (the caller routes
    that to "insufficient pool / under-powered", never a silent pass). The
    ``displaced_mass`` edges are taken from the first candidate set (the others are
    recorded only to size ``K_bin``).

    Returns ``(binning, selection_event)``; the event records every rung walked and
    the reachable cardinality ``k_bin`` so the SI correction is computable from the
    event, not narrated.
    """
    ladder = tuple(int(w) for w in delta_pos_ladder)
    if not ladder:
        raise ValueError("delta_pos_ladder must be non-empty")
    if any(ladder[i] >= ladder[i + 1] for i in range(len(ladder) - 1)):
        raise ValueError("delta_pos_ladder must be strictly increasing (finest first)")
    edge_candidates = tuple(tuple(float(e) for e in es) for es in displaced_mass_edge_candidates)
    if not edge_candidates:
        raise ValueError("displaced_mass_edge_candidates must be non-empty")

    rungs_walked: list[int] = []
    mean_pool_per_rung: list[float] = []
    chosen: int | None = None
    chosen_mean = 0.0
    for w in ladder:
        mp = _mean_pool(w, v_sel_distances_to_answer)
        rungs_walked.append(w)
        mean_pool_per_rung.append(mp)
        if mp >= pool_floor:
            chosen = w
            chosen_mean = mp
            break
    meets_floor = chosen is not None
    if chosen is None:
        # no rung cleared the floor: take the coarsest (last) rung walked.
        chosen = ladder[-1]
        chosen_mean = mean_pool_per_rung[-1]

    chosen_edges = edge_candidates[0]

    # K_bin must pay for the FULL pre-enumerated candidate ladder, NOT the
    # data-dependent prefix actually walked (findings 14, 20; REDESIGN_v5 §6.2:
    # "K_bin/K_op bounded and enumerated BEFORE data"). The short-circuit
    # ``rungs_walked`` depends on the V_sel realisation, so folding its length into
    # the SI-2 Bonferroni factor under-penalises multiplicity (on other V_sel draws
    # the adaptive rule could have walked the full ladder). Use the full ladder length
    # x the candidate edge-set count. (Under SI-1 the level forces K_bin = 1, so this
    # only bites the SI-2 fallback the code supports — exactly where it must be honest.)
    k_bin = len(ladder) * len(edge_candidates)

    binning = Binning(
        delta_pos=chosen,
        displaced_mass_edges=chosen_edges,
        mean_pool_at_choice=chosen_mean,
        meets_pool_floor=meets_floor,
    )
    event = SelectionEvent(
        delta_pos_ladder=ladder,
        rungs_walked=tuple(rungs_walked),
        mean_pool_per_rung=tuple(mean_pool_per_rung),
        displaced_mass_edge_candidates=edge_candidates,
        k_bin=k_bin,
        chosen_delta_pos=chosen,
        chosen_edges=chosen_edges,
    )
    return binning, event
