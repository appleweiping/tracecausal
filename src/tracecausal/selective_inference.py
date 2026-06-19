"""Selective-inference correction folding K_bin and K_op (REDESIGN_v5 §6).

The principle: **every data-adaptive choice is a selection event that must be paid
for.** v5 corrects inference for two selection layers — the data-adaptive *binning*
(``K_bin``, from :mod:`binning_selection`) and the *operator/policy* selection
(``K_op``, from the :mod:`repair_ops` policy grid, OS-2). This module provides:

* ``holm_alpha(m_prime, k_bin, k_op, selection_split_used) -> alpha_1_prime``
  (Eq. SI-ALPHA, §6.2): the confirmatory per-test level, folding ``K_bin`` and
  ``K_op``. Under **SI-1** (selection split) the inference family uses
  ``K_bin = K_op = 1`` (the choice was made on ``V_sel``); under **SI-2** the
  fallback pays the Bonferroni factor ``K_bin * K_op``.
* ``validate_selection_split(v_sel, v_inf, test) -> errors`` (SI-1): disjointness
  of the three splits + the ``n_val`` floor check on both ``V_sel`` and ``V_inf``.
* ``choose_si_path(n_val_sel, n_val_inf, floors) -> "SI-1" | "SI-2"``: the
  **deterministic** rule (SI-1 iff the ``n_val`` floors are met on both splits,
  else SI-2), frozen so the power cost is known before lock, not discovered after.

Pure Python; no model, no GPU, no run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

__all__ = [
    "SiPath",
    "SiFloors",
    "holm_alpha",
    "validate_selection_split",
    "choose_si_path",
    "BASE_ALPHA",
]

SiPath = Literal["SI-1", "SI-2"]

# The base family-wise level (REDESIGN_v4 §4.2 Holm family; carried into v5).
BASE_ALPHA: float = 0.05


@dataclass(frozen=True)
class SiFloors:
    """Minimum validation sizes for SI-1 eligibility (mirror nuisance N_VAL_MIN_*)."""

    sigma_min: int = 200  # paired examples per cell for the variance estimate (B1)
    kappa_min: int = 300  # double-scored items per cell for kappa (B2)


def holm_alpha(
    m_prime: int,
    *,
    k_bin: int = 1,
    k_op: int = 1,
    selection_split_used: bool = True,
) -> float:
    """Confirmatory per-test level ``alpha_1'`` (Eq. SI-ALPHA, REDESIGN_v5 §6.2).

    ``alpha_1' = BASE_ALPHA / (m' * K_bin * K_op)``.

    * ``m_prime`` — the Holm family size at the v5 lock (v4 ``m`` plus the G9 cells
      and the G9-NOV baseline contrasts, §6.3).
    * Under **SI-1** (``selection_split_used=True``, the primary path) the inference
      family is computed on ``V_inf``/test with selections frozen on ``V_sel``, so
      ``K_bin = K_op = 1`` for the inference family **regardless** of the supplied
      values — the selection was already paid for by the split. This function
      enforces that: when ``selection_split_used`` is ``True`` it folds
      ``K_bin = K_op = 1`` (a passed-in cardinality is ignored for the level,
      though the caller still records it on the event).
    * Under **SI-2** (``selection_split_used=False``, the fallback) it folds the
      Bonferroni factor ``k_bin * k_op`` (each must be >= 1).

    Raises
    ------
    ValueError
        If ``m_prime < 1`` or, under SI-2, any cardinality ``< 1``.
    """
    if m_prime < 1:
        raise ValueError(f"m_prime must be >= 1, got {m_prime}")
    if selection_split_used:
        eff_k_bin = eff_k_op = 1
    else:
        if k_bin < 1 or k_op < 1:
            raise ValueError("under SI-2, K_bin and K_op must each be >= 1")
        eff_k_bin, eff_k_op = k_bin, k_op
    return BASE_ALPHA / (m_prime * eff_k_bin * eff_k_op)


def validate_selection_split(
    v_sel: Sequence[object],
    v_inf: Sequence[object],
    test: Sequence[object],
    *,
    floors: SiFloors | None = None,
) -> list[str]:
    """SI-1 disjointness + floor check (REDESIGN_v5 §6.1).

    Returns the list of violations (empty == valid SI-1 configuration):

    * the three splits ``V_sel`` / ``V_inf`` / ``test`` must be pairwise **disjoint**
      (no example may appear in two — selection independence requires it);
    * ``V_sel`` and ``V_inf`` must each meet the ``n_val`` floor (so SI-1 is even
      eligible; the §6.2 deterministic rule otherwise routes to SI-2);
    * ``test`` must be non-empty (sealed but present).

    Membership is by element identity/equality (e.g. example ids). Pure Python.
    """
    floors = floors or SiFloors()
    errors: list[str] = []
    s_sel, s_inf, s_test = set(v_sel), set(v_inf), set(test)

    sel_inf = s_sel & s_inf
    sel_test = s_sel & s_test
    inf_test = s_inf & s_test
    if sel_inf:
        errors.append(f"V_sel and V_inf overlap (must be disjoint): {sorted(map(repr, sel_inf))}")
    if sel_test:
        errors.append(f"V_sel and test overlap (must be disjoint): {sorted(map(repr, sel_test))}")
    if inf_test:
        errors.append(f"V_inf and test overlap (must be disjoint): {sorted(map(repr, inf_test))}")

    # The eligibility bar MUST match the deterministic SI-1-vs-SI-2 router
    # (:func:`choose_si_path`, which uses the STRICTER ``max`` floor); otherwise the
    # validator would accept a split in ``[min, max)`` that the router sends to SI-2
    # (SI gate / multiplicity inconsistency, finding 13). Use the same stricter bar so
    # an SI-1-validated config is exactly the set the router actually routes to SI-1.
    floor = max(floors.sigma_min, floors.kappa_min)
    if len(s_sel) < floor:
        errors.append(
            f"V_sel size {len(s_sel)} below SI-1 eligibility floor {floor} "
            "(route to SI-2 per the deterministic rule, §6.2)"
        )
    if len(s_inf) < floor:
        errors.append(
            f"V_inf size {len(s_inf)} below SI-1 eligibility floor {floor} "
            "(route to SI-2 per the deterministic rule, §6.2)"
        )
    if not s_test:
        errors.append("test split must be non-empty (sealed but present)")
    return errors


def choose_si_path(
    n_val_sel: int,
    n_val_inf: int,
    *,
    floors: SiFloors | None = None,
) -> SiPath:
    """The deterministic SI-1-vs-SI-2 rule (REDESIGN_v5 §6.2).

    **SI-1 iff** both ``n_val_sel`` and ``n_val_inf`` meet the ``n_val`` floors;
    else **SI-2**. The rule is frozen so the power cost (SI-1 splits power; SI-2
    pays the ``K_bin * K_op`` Bonferroni) is known *before* lock, not discovered
    after (§6.2). Uses the stricter of the two floors as the bar on each split.
    """
    floors = floors or SiFloors()
    bar = max(floors.sigma_min, floors.kappa_min)
    if n_val_sel >= bar and n_val_inf >= bar:
        return "SI-1"
    return "SI-2"
