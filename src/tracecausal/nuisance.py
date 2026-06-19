"""Leakage-safe re-estimation of the nuisance parameters ``sigma_u`` and ``kappa``
(REDESIGN_v4 §4.6, fixing b) plus the powered-``n`` formula (§4.7).

The two nuisance parameters that set the power budget (``sigma_u``) and the
attenuation correction (``kappa``) are estimated **only on the validation split**,
with named estimators, CIs, minimum ``n_val``, and a **freeze-at-lock** rule. The
test split stays sealed and neither parameter is re-estimated after unlock. This
module enforces that discipline at the API surface: callers pass frozen
validation arrays; **there is no test-split access path**.

Key formulas (REDESIGN_v4 §4.6 / §4.7), all checked numerically in-session and
labelled *formula evaluation, not evidence*:

* ``sigma_u`` = sample sd of paired per-example contrasts
  ``u_i = delta_tgt_i - delta_rand_i``; **power uses the upper CI** ``sigma_hi``.
* ``kappa`` = Cohen/Fleiss agreement; the attenuation target uses the **lower CI**
  ``kappa_lo``: ``U_target = margin / (2*kappa_lo - 1)``.
* proximity pool-shrinkage inflation ``infl = 1 + 1/mean(m_pool)``.
* ``R_power = ceil( (z * sigma_hi / margin)^2 * infl )``.

The §4.7 exhibited feasible point evaluates to ``n = 850`` at ``sigma_hi = 0.30``,
``z = 2.734`` (m=8 family), ``margin = 0.03``, ``infl = 1.125`` — reproduced by
``r_power`` below. **No model, no GPU, no run.**
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from ._numerics import quantile as _quantile

__all__ = [
    "SigmaEstimate",
    "SigmaREstimate",
    "KappaEstimate",
    "KappaFallback",
    "estimate_sigma_u",
    "estimate_sigma_r",
    "estimate_kappa",
    "claim_level_aggregate",
    "apply_kappa_fallback",
    "pool_inflation",
    "r_power",
    "r_power_repair",
    "u_target",
    "N_VAL_MIN_SIGMA",
    "N_VAL_MIN_KAPPA",
    "POOL_MIN",
    "KAPPA_FALLBACK_THRESHOLD",
    "SIGMA_U_UPPER_BOUND",
]

# Minimum validation sizes (REDESIGN_v4 §4.6 B1/B2).
N_VAL_MIN_SIGMA: int = 200  # paired examples per cell for sigma_u
N_VAL_MIN_KAPPA: int = 300  # double-scored items per cell for kappa
POOL_MIN: int = 8  # mean in-bin pool floor (§4.6 B3)
KAPPA_FALLBACK_THRESHOLD: float = 0.90  # kappa_lo < this -> claim-level aggregation (B2)
# Bounded-Bernoulli-contrast variance cap (REDESIGN_v3 §4.5 E2; carried in v4).
SIGMA_U_UPPER_BOUND: float = 0.707


@dataclass(frozen=True)
class SigmaEstimate:
    """``sigma_u`` point + bootstrap CI (REDESIGN_v4 §4.6 B1)."""

    sigma_hat: float
    sigma_lo: float
    sigma_hi: float
    n_val: int
    meets_min_n: bool
    meets_precision: bool  # bootstrap CI half-width <= 0.05 * sigma_hat (B1 rule)


@dataclass(frozen=True)
class SigmaREstimate:
    """``sigma_R`` decomposition for the cross-example U-statistic (REDESIGN_v5 §4.6).

    The v4 ``sigma_u`` is a within-example paired-contrast SD; it is the WRONG scale
    for the cross-example U-statistic ``R_hat`` (MF-5). ``sigma_R`` is derived from
    Eq. R-VAR. The kernel ``g_{ij}`` is **asymmetric** (source role != target role),
    so the dominant pair-dependence term is the **ordered-pair two-way projection
    variance** using BOTH first-order projections (findings 4, 10):

        Var(R_hat) ~= ( zeta_10 / n_source + zeta_01 / n_target )
                    + (1 / N_pair) * ( sigma_MC^2 / R_null + sigma_op^2 / R_int )

    where ``zeta_10`` is the source-projection variance and ``zeta_01`` the
    target-margin projection variance — the same ordered two-way variance the
    standalone Hájek check (:func:`repair_transfer.hajek_projection_var`) computes,
    so the power path and the analytic cross-check now agree. The symmetric-kernel
    shorthand ``(4 / n_eff) * zeta_1`` of §4.6 collapses both margins to one and
    UNDERSTATES the variance for the asymmetric repair kernel; it is no longer used.

    Attributes
    ----------
    zeta_10:
        Variance of the **source** first projection ``E[g_{ij} | i]``.
    zeta_01:
        Variance of the **target-margin** first projection ``E[g_{ij} | j]``.
    zeta_1_max_reporting_only:
        ``max(zeta_10, zeta_01)`` — retained for back-reference / reporting ONLY; the
        variance no longer collapses to it (findings 4, 10). **Never read this as the
        variance**: it is the superseded symmetric-kernel shorthand's projection scale
        and is deliberately named to make that misuse loud. The variance uses the
        two-projection ``proj_var`` (``zeta_10/n_source + zeta_01/n_target``) below.
    n_source / n_target:
        The two cluster-margin counts; the ordered projection variance divides each
        projection by its own margin.
    n_eff:
        Effective number of independent examples per class after pair reuse
        (``min(n_source, n_target)``); retained for reporting (the design-effect
        bookkeeping) — the variance uses the per-margin counts, not this scalar.
    n_pair:
        Number of ordered in-class pairs ``N_pair``.
    mc_term:
        ``sigma_MC^2 / R_null`` averaged per pair (nested matched-null MC).
    op_term:
        ``sigma_op^2 / R_int`` averaged per pair (repair-operator stochasticity).
    proj_var:
        The ordered-pair two-way projection variance
        ``zeta_10 / n_source + zeta_01 / n_target`` (the dominant term).
    var_r_hat:
        The assembled ``Var(R_hat)`` from Eq. R-VAR.
    sigma_r:
        ``sqrt(var_r_hat)`` — the point SD of ``R_hat``.
    sigma_r_hi:
        A conservative **upper** SD used for power (``sigma_r`` inflated by
        ``ci_inflation``; the §4.6 "upper CI of sigma_R estimated on V_sel").
    """

    zeta_10: float
    zeta_01: float
    # NOT the variance: the superseded symmetric-kernel max(zeta_10, zeta_01) scale,
    # kept for back-reference/reporting and named to make a misread loud (findings 4, 10).
    zeta_1_max_reporting_only: float
    n_source: int
    n_target: int
    n_eff: float
    n_pair: int
    mc_term: float
    op_term: float
    proj_var: float
    var_r_hat: float
    sigma_r: float
    sigma_r_hi: float


@dataclass(frozen=True)
class KappaEstimate:
    """``kappa`` point + CI (REDESIGN_v4 §4.6 B2)."""

    kappa_hat: float
    kappa_lo: float
    kappa_hi: float
    n_val: int
    meets_min_n: bool
    below_fallback_threshold: bool  # kappa_lo < 0.90 -> claim-level fallback (B2)
    # Degenerate bootstrap resamples are COUNTED, not silently dropped (review fix):
    # a resample with chance agreement == 1 (all labels identical) gives an
    # undefined kappa. n_degenerate_resamples records how many of n_bootstrap were
    # degenerate; all_degenerate flags the pathological case where the CI collapsed
    # to the point estimate (which must BLOCK a lock — see apply_kappa_fallback).
    n_bootstrap: int = 0
    n_degenerate_resamples: int = 0
    all_degenerate: bool = False


@dataclass(frozen=True)
class KappaFallback:
    """Outcome of the §4.6 B2 deterministic kappa-fallback branch.

    The B2 rule (selected **at lock from the validation ``kappa_lo``**, removing
    v3 post-hoc discretion):

    1. If ``kappa_lo >= 0.90`` — no fallback; use ``kappa_lo`` directly.
    2. Else, aggregate to claim-level factuality proportion (raises effective
       agreement) and **re-measure** ``kappa_lo`` on the aggregated outcome.
    3. If the re-measured ``kappa_lo`` is still ``< 0.90`` — widen the margin to
       the attenuation-adjusted value ``0.05 / (2*kappa_lo - 1)``.

    The branch is deterministic given the validation data. A degenerate
    re-measurement (``all_degenerate`` / agreement at-or-below chance) sets
    ``blocks_lock = True``: the lock MUST NOT proceed on a non-informative kappa.
    """

    used_claim_level: bool  # step 2 fired
    kappa_lo_used: float  # the kappa_lo the design proceeds on
    widened_margin: float | None  # step 3 widened margin (None if not needed)
    blocks_lock: bool  # True -> assertion blocks the analysis lock
    reason: str


def estimate_sigma_u(
    paired_contrasts: Sequence[float],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
) -> SigmaEstimate:
    """Estimate ``sigma_u`` as the realised paired-contrast sd (REDESIGN_v4 §4.6 B1).

    ``paired_contrasts`` are the validation per-example contrasts
    ``u_i = delta_tgt_i - delta_rand_i`` (``delta_rand_i`` already MC-averaged over
    ``R_int`` draws). Returns ``sigma_hat = sample_sd(u_i)`` with a bootstrap 95%
    CI; **power uses ``sigma_hi``** so the MDE ``n`` is conservative against an
    underestimated variance.

    Flags whether the validation size meets ``N_VAL_MIN_SIGMA`` and whether the
    bootstrap CI half-width meets the 5%-relative-precision rule (B1). Pure Python.
    """
    u = [float(v) for v in paired_contrasts]
    n = len(u)
    if n < 2:
        raise ValueError("need at least 2 paired contrasts to estimate sigma_u")

    sigma_hat = _sample_sd(u)

    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(u) for _ in range(n)]
        boots.append(_sample_sd(sample))
    boots.sort()
    lo_q = (1.0 - ci) / 2.0
    sigma_lo = _quantile(boots, lo_q)
    sigma_hi = _quantile(boots, 1.0 - lo_q)

    half_width = (sigma_hi - sigma_lo) / 2.0
    meets_precision = sigma_hat > 0 and half_width <= 0.05 * sigma_hat

    return SigmaEstimate(
        sigma_hat=sigma_hat,
        sigma_lo=sigma_lo,
        sigma_hi=sigma_hi,
        n_val=n,
        meets_min_n=n >= N_VAL_MIN_SIGMA,
        meets_precision=meets_precision,
    )


def estimate_kappa(
    labels_a: Sequence[int],
    labels_b: Sequence[int],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
    fallback_threshold: float = 0.90,
) -> KappaEstimate:
    """Cohen's ``kappa`` (2 raters) on binary factuality labels (REDESIGN_v4 §4.6 B2).

    ``labels_a`` / ``labels_b`` are the hashed-evaluator and held-out-reference
    binary labels on the **validation** double-scored subset. Returns ``kappa_hat``
    with a bootstrap 95% CI; the attenuation correction uses the **lower CI**
    ``kappa_lo`` (conservative against overestimated agreement). Flags whether
    ``kappa_lo < fallback_threshold`` (0.90), which selects the claim-level
    aggregation fallback **at lock** (B2). Pure Python.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must align in length")
    n = len(labels_a)
    if n < 2:
        raise ValueError("need at least 2 double-scored items to estimate kappa")

    a = [int(v) for v in labels_a]
    b = [int(v) for v in labels_b]
    # Point estimate is computed gracefully: constant-label inputs (chance
    # agreement == 1) do NOT raise here (Opus minor: kappa must handle constant
    # labels gracefully). Perfect agreement on constant labels -> kappa 1.0;
    # mismatched-but-constant -> 0.0. The bootstrap loop below still uses the
    # raising ``_cohen_kappa`` so degenerate RESAMPLES are counted, not dropped.
    kappa_hat = _cohen_kappa_point(a, b)

    rng = random.Random(seed)
    boots: list[float] = []
    idx = list(range(n))
    n_degenerate = 0
    for _ in range(n_bootstrap):
        sample = [rng.choice(idx) for _ in range(n)]
        ka = [a[j] for j in sample]
        kb = [b[j] for j in sample]
        try:
            boots.append(_cohen_kappa(ka, kb))
        except ValueError:
            # Degenerate resample (chance agreement == 1, kappa undefined). COUNT
            # it rather than silently dropping it, so the caller can see how much
            # of the bootstrap mass was unusable.
            n_degenerate += 1
            continue
    all_degenerate = not boots
    if all_degenerate:
        # All resamples degenerate; the CI collapses to the point estimate. This
        # is surfaced via all_degenerate so apply_kappa_fallback BLOCKS the lock
        # rather than proceeding on a non-informative CI.
        boots = [kappa_hat]
    boots.sort()
    lo_q = (1.0 - ci) / 2.0
    kappa_lo = _quantile(boots, lo_q)
    kappa_hi = _quantile(boots, 1.0 - lo_q)

    return KappaEstimate(
        kappa_hat=kappa_hat,
        kappa_lo=kappa_lo,
        kappa_hi=kappa_hi,
        n_val=n,
        meets_min_n=n >= N_VAL_MIN_KAPPA,
        below_fallback_threshold=kappa_lo < fallback_threshold,
        n_bootstrap=n_bootstrap,
        n_degenerate_resamples=n_degenerate,
        all_degenerate=all_degenerate,
    )


def claim_level_aggregate(
    labels_a: Sequence[int],
    labels_b: Sequence[int],
    claim_ids: Sequence[object],
) -> tuple[list[int], list[int]]:
    """Aggregate item-level binary labels to **claim-level** factuality (§4.6 B2).

    The B2 fallback aggregates the per-item binary factuality labels to a
    per-claim majority/all-true outcome, which raises effective evaluator
    agreement (fewer, more separable units). ``claim_ids[i]`` groups item ``i``
    into its claim; within each claim, each rater's aggregated label is ``1`` iff
    **all** of that rater's item-labels for the claim are ``1`` (the conservative
    "claim is fully supported" aggregation), else ``0``.

    Returns ``(agg_a, agg_b)`` aligned per distinct claim (in first-seen order).
    Pure Python; validation-split-only by the same contract as ``estimate_kappa``.
    """
    if not (len(labels_a) == len(labels_b) == len(claim_ids)):
        raise ValueError("labels_a, labels_b, claim_ids must align in length")
    order: list[object] = []
    by_claim: dict[object, tuple[list[int], list[int]]] = {}
    for la, lb, cid in zip(labels_a, labels_b, claim_ids):
        if cid not in by_claim:
            by_claim[cid] = ([], [])
            order.append(cid)
        by_claim[cid][0].append(int(la))
        by_claim[cid][1].append(int(lb))
    agg_a = [1 if all(v == 1 for v in by_claim[c][0]) else 0 for c in order]
    agg_b = [1 if all(v == 1 for v in by_claim[c][1]) else 0 for c in order]
    return agg_a, agg_b


def apply_kappa_fallback(
    item_estimate: KappaEstimate,
    *,
    labels_a: Sequence[int] | None = None,
    labels_b: Sequence[int] | None = None,
    claim_ids: Sequence[object] | None = None,
    margin: float = 0.05,
    fallback_threshold: float = KAPPA_FALLBACK_THRESHOLD,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> KappaFallback:
    """Apply the §4.6 B2 deterministic kappa fallback at lock.

    Implements the pre-registered branch (NOT forward-marked — fully realised):

    * ``kappa_lo >= 0.90`` -> no fallback.
    * else, if ``claim_ids`` provided, aggregate to claim level
      (:func:`claim_level_aggregate`), re-measure ``kappa_lo``; if that recovers
      ``>= 0.90``, proceed on the aggregated ``kappa_lo``.
    * else widen the margin to ``margin / (2*kappa_lo - 1)``.

    A degenerate kappa (``all_degenerate`` on the item or aggregated estimate, or
    agreement at/below chance so the widened margin is undefined) sets
    ``blocks_lock = True``: the caller MUST assert on this and refuse to lock —
    degenerate kappa is **counted and surfaced, never silently dropped**.

    If ``claim_ids`` is ``None`` and item ``kappa_lo < 0.90``, the function does
    NOT silently proceed: it returns ``blocks_lock = True`` (claim-level data was
    required for the deterministic branch but not supplied).
    """
    # Degenerate item-level kappa must block a lock outright.
    if item_estimate.all_degenerate:
        return KappaFallback(
            used_claim_level=False,
            kappa_lo_used=item_estimate.kappa_lo,
            widened_margin=None,
            blocks_lock=True,
            reason=(
                "item-level kappa bootstrap fully degenerate "
                f"({item_estimate.n_degenerate_resamples}/{item_estimate.n_bootstrap} "
                "resamples); CI non-informative — lock BLOCKED"
            ),
        )

    if item_estimate.kappa_lo >= fallback_threshold:
        return KappaFallback(
            used_claim_level=False,
            kappa_lo_used=item_estimate.kappa_lo,
            widened_margin=None,
            blocks_lock=False,
            reason=f"kappa_lo={item_estimate.kappa_lo:.4f} >= {fallback_threshold}; no fallback",
        )

    # kappa_lo < threshold: the deterministic branch REQUIRES claim-level data.
    if claim_ids is None or labels_a is None or labels_b is None:
        return KappaFallback(
            used_claim_level=False,
            kappa_lo_used=item_estimate.kappa_lo,
            widened_margin=None,
            blocks_lock=True,
            reason=(
                f"kappa_lo={item_estimate.kappa_lo:.4f} < {fallback_threshold} but "
                "claim-level aggregation data not supplied — lock BLOCKED until B2 "
                "claim-level re-measurement is provided"
            ),
        )

    agg_a, agg_b = claim_level_aggregate(labels_a, labels_b, claim_ids)
    agg_estimate = estimate_kappa(
        agg_a, agg_b, n_bootstrap=n_bootstrap, seed=seed, fallback_threshold=fallback_threshold
    )
    if agg_estimate.all_degenerate:
        return KappaFallback(
            used_claim_level=True,
            kappa_lo_used=agg_estimate.kappa_lo,
            widened_margin=None,
            blocks_lock=True,
            reason=(
                "claim-level kappa bootstrap fully degenerate "
                f"({agg_estimate.n_degenerate_resamples}/{agg_estimate.n_bootstrap}); "
                "lock BLOCKED"
            ),
        )

    if agg_estimate.kappa_lo >= fallback_threshold:
        return KappaFallback(
            used_claim_level=True,
            kappa_lo_used=agg_estimate.kappa_lo,
            widened_margin=None,
            blocks_lock=False,
            reason=(
                f"claim-level aggregation recovered kappa_lo={agg_estimate.kappa_lo:.4f} "
                f">= {fallback_threshold}; proceed on aggregated kappa_lo"
            ),
        )

    # Still below threshold after aggregation: widen the margin (step 3). The
    # widened margin is only defined when agreement is above chance.
    denom = 2.0 * agg_estimate.kappa_lo - 1.0
    if denom <= 0.0:
        return KappaFallback(
            used_claim_level=True,
            kappa_lo_used=agg_estimate.kappa_lo,
            widened_margin=None,
            blocks_lock=True,
            reason=(
                f"claim-level kappa_lo={agg_estimate.kappa_lo:.4f} at/below chance "
                "(2*kappa_lo-1 <= 0); attenuation band undefined — lock BLOCKED"
            ),
        )
    widened = margin / denom
    return KappaFallback(
        used_claim_level=True,
        kappa_lo_used=agg_estimate.kappa_lo,
        widened_margin=widened,
        blocks_lock=False,
        reason=(
            f"claim-level kappa_lo={agg_estimate.kappa_lo:.4f} still < {fallback_threshold}; "
            f"margin widened to {widened:.4f} = {margin}/(2*{agg_estimate.kappa_lo:.4f}-1)"
        ),
    )


def pool_inflation(m_pool_per_example: Sequence[int]) -> float:
    """Proximity pool-shrinkage inflation ``infl = 1 + 1/mean(m_pool)`` (§4.6 B3).

    The proximity stratifier shrinks the per-example null pool; a finite pool of
    mean size ``bar m_pool`` sampled ``R_int`` times adds a finite-pool variance
    factor ``infl`` applied to ``R_power``. Raises if any pool is empty.
    """
    pools = [int(m) for m in m_pool_per_example]
    if not pools:
        raise ValueError("m_pool_per_example must be non-empty")
    if any(m <= 0 for m in pools):
        raise ValueError("each m_pool must be positive (empty pool -> coarsen bin, §4.6 B3)")
    mean_pool = sum(pools) / len(pools)
    return 1.0 + 1.0 / mean_pool


def r_power(sigma_hi: float, *, z: float, margin: float, infl: float = 1.0) -> int:
    """Required examples/cell ``R_power`` (REDESIGN_v4 §4.6 B1 / §4.7).

    ``R_power = ceil( (z * sigma_hi / margin)^2 * infl )``, using the **upper-CI**
    ``sigma_hi`` and the proximity inflation factor. Reproduces the §4.7 exhibited
    point: ``r_power(0.30, z=2.734, margin=0.03, infl=1.125) == 841`` (rounded to
    ``n = 850`` in the locked config). *Formula evaluation, not evidence.*
    """
    if sigma_hi < 0:
        raise ValueError("sigma_hi must be >= 0")
    if margin <= 0:
        raise ValueError("margin must be positive")
    if infl < 1.0:
        raise ValueError("inflation factor must be >= 1.0")
    base = (z * sigma_hi / margin) ** 2
    return math.ceil(base * infl)


def estimate_sigma_r(
    g_ij_by_pair: Sequence[tuple[object, object, float]],
    *,
    mc_var_per_pair: Sequence[float] | None = None,
    op_var_per_pair: Sequence[float] | None = None,
    r_null: int = 1,
    r_int: int = 1,
    ci_inflation: float = 1.0,
) -> SigmaREstimate:
    """``sigma_R`` from the U-statistic design effect (Eq. R-VAR, REDESIGN_v5 §4.6; MF-5).

    ``g_ij_by_pair`` is a sequence of ``(source_id, target_id, g_ij)`` tuples (the
    realised per-pair repair gains). The variance is assembled from Eq. R-VAR with
    the **ordered-pair two-way projection variance** using BOTH projections
    (findings 4, 10):

        Var(R_hat) ~= ( zeta_10 / n_source + zeta_01 / n_target )
                    + (1 / N_pair) * ( sigma_MC^2 / R_null + sigma_op^2 / R_int )

    * ``zeta_10`` is the **source** first-projection variance (var of the per-source
      mean gain) and ``zeta_01`` the **target-margin** projection variance (var of
      the per-target mean gain). The dominant pair-dependence term divides EACH
      projection by ITS OWN cluster-margin count — the same ordered two-way variance
      :func:`repair_transfer.hajek_projection_var` computes, so the power path and
      the analytic cross-check agree. The symmetric shorthand ``(4 / n_eff) * zeta_1``
      collapsed both margins to ``max(zeta_10, zeta_01)`` and **understated** the
      asymmetric-kernel variance; it is no longer used (findings 4, 10).
    * ``n_eff = min(n_source, n_target)`` is retained for the design-effect
      bookkeeping/reporting only; the variance uses the per-margin counts.
    * ``mc_var_per_pair`` / ``op_var_per_pair`` (already per-draw variances) are
      averaged and divided by ``R_null`` / ``R_int`` for the nested terms; when not
      supplied those terms are 0.

    ``sigma_r_hi = sqrt(var) * ci_inflation`` is the conservative upper SD power
    uses (the §4.6 "upper CI of sigma_R on V_sel"). **No number is fabricated**: all
    inputs are DATA_NEEDED, estimated on ``V_sel`` at lock; this is the frozen
    identity the locked config evaluates.
    """
    pairs = [(s, t, float(g)) for (s, t, g) in g_ij_by_pair]
    n_pair = len(pairs)
    if n_pair < 2:
        raise ValueError("need at least 2 pairs to estimate sigma_R")
    if r_null < 1 or r_int < 1:
        raise ValueError("r_null and r_int must be >= 1")
    if ci_inflation < 1.0:
        raise ValueError("ci_inflation must be >= 1.0")

    by_source: dict[object, list[float]] = {}
    by_target: dict[object, list[float]] = {}
    for s, t, g in pairs:
        by_source.setdefault(s, []).append(g)
        by_target.setdefault(t, []).append(g)
    source_means = [sum(v) / len(v) for v in by_source.values()]
    target_means = [sum(v) / len(v) for v in by_target.values()]
    zeta_10 = _sample_sd(source_means) ** 2
    zeta_01 = _sample_sd(target_means) ** 2
    # max(zeta_10, zeta_01): reporting/back-reference only, NOT the variance scale
    # (the variance uses both projections over their own margins; findings 4, 10).
    zeta_1_max_reporting_only = max(zeta_10, zeta_01)

    n_source = len(by_source)
    n_target = len(by_target)
    n_eff = float(min(n_source, n_target))
    if n_eff < 1.0:
        raise ValueError("n_eff < 1; degenerate cluster structure")

    if mc_var_per_pair is not None:
        mc_mean = sum(float(v) for v in mc_var_per_pair) / max(1, len(mc_var_per_pair))
        mc_term = mc_mean / r_null
    else:
        mc_term = 0.0
    if op_var_per_pair is not None:
        op_mean = sum(float(v) for v in op_var_per_pair) / max(1, len(op_var_per_pair))
        op_term = op_mean / r_int
    else:
        op_term = 0.0

    # ordered-pair two-way projection variance: BOTH zeta projections, each over its
    # own margin (findings 4, 10) — NOT (4/n_eff)*max(zeta_10, zeta_01).
    proj_var = zeta_10 / n_source + zeta_01 / n_target
    var_r_hat = proj_var + (1.0 / n_pair) * (mc_term + op_term)
    sigma_r = math.sqrt(var_r_hat)
    sigma_r_hi = sigma_r * ci_inflation
    return SigmaREstimate(
        zeta_10=zeta_10,
        zeta_01=zeta_01,
        zeta_1_max_reporting_only=zeta_1_max_reporting_only,
        n_source=n_source,
        n_target=n_target,
        n_eff=n_eff,
        n_pair=n_pair,
        mc_term=mc_term,
        op_term=op_term,
        proj_var=proj_var,
        var_r_hat=var_r_hat,
        sigma_r=sigma_r,
        sigma_r_hi=sigma_r_hi,
    )


def r_power_repair(
    sigma_r_hi: float,
    *,
    z: float,
    m_r: float,
    d_eff: float = 1.0,
    forward_surcharge: int = 0,
) -> tuple[int, int]:
    """Required examples/cell ``R_power`` for G9 (Eq. R-POWER, REDESIGN_v5 §4.6; MF-5).

    ``R_power = ceil( (z * sigma_r_hi / m_r)^2 * D_eff )``, using the conservative
    upper SD ``sigma_r_hi`` of the **cross-example U-statistic** ``sigma_R``
    (Eq. R-VAR) — **not** v4's ``sigma_u`` (the whole point of MF-5). ``D_eff >= 1``
    is the pre-registered class-imbalance design-effect factor.

    Returns ``(r_power, forwards)`` where ``forwards`` is the total forward count
    implied by ``r_power`` and the per-example ``forward_surcharge`` (a localized
    repair forward + ``R_null`` matched-null-repair forwards + ``R_int`` repair-op
    repeats per target, §6.3); the caller persists it as ``forwards_per_example``.
    **No number is fabricated**: ``sigma_r_hi``, ``z``, ``m_r``, ``D_eff`` are all
    DATA_NEEDED, estimated/frozen on ``V_sel`` at lock.
    """
    if sigma_r_hi < 0:
        raise ValueError("sigma_r_hi must be >= 0")
    if m_r <= 0:
        raise ValueError("m_r (repair margin) must be positive")
    if d_eff < 1.0:
        raise ValueError("d_eff must be >= 1.0")
    if forward_surcharge < 0:
        raise ValueError("forward_surcharge must be >= 0")
    base = (z * sigma_r_hi / m_r) ** 2
    r = math.ceil(base * d_eff)
    forwards = r * forward_surcharge
    return r, forwards


def u_target(margin: float, kappa_lo: float) -> float:
    """Attenuation-adjusted target ``U_target = margin / (2*kappa_lo - 1)`` (§4.6 B2).

    Uses the **lower-CI** ``kappa_lo``. Reproduces the §4.7 point:
    ``u_target(0.05, 0.92) = 0.05/0.84 = 0.0595``. Raises if ``2*kappa_lo - 1 <= 0``
    (agreement at/below chance gives no usable attenuation band).
    """
    denom = 2.0 * kappa_lo - 1.0
    if denom <= 0.0:
        raise ValueError("2*kappa_lo - 1 must be > 0 (agreement above chance) for attenuation")
    return margin / denom


# --- internal numerics ------------------------------------------------------


def _sample_sd(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def _cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa for two binary raters.

    Raises ``ValueError`` when chance agreement is 1.0 (e.g. constant labels),
    where kappa is mathematically undefined. The bootstrap loop relies on this
    raise to COUNT degenerate resamples; for the graceful point estimate use
    :func:`_cohen_kappa_point`.
    """
    n = len(a)
    if n == 0:
        raise ValueError("empty label arrays")
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    chance = pa1 * pb1 + (1.0 - pa1) * (1.0 - pb1)
    if chance >= 1.0:
        raise ValueError("chance agreement is 1.0; kappa undefined")
    return (agree - chance) / (1.0 - chance)


def _cohen_kappa_point(a: Sequence[int], b: Sequence[int]) -> float:
    """Graceful Cohen's kappa point estimate that does NOT raise on constant labels.

    When chance agreement is 1.0 (degenerate — e.g. one or both raters give a
    constant label across all items) kappa is mathematically undefined; by the
    standard convention we resolve it to the limiting value: ``1.0`` if the raters
    agree on every item (perfect agreement), else ``0.0`` (no agreement beyond the
    degenerate chance baseline). Otherwise defers to :func:`_cohen_kappa`.
    """
    n = len(a)
    if n == 0:
        raise ValueError("empty label arrays")
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    chance = pa1 * pb1 + (1.0 - pa1) * (1.0 - pb1)
    if chance >= 1.0:
        # Degenerate chance baseline: convention -> perfect agreement maps to 1.0,
        # any disagreement to 0.0.
        return 1.0 if agree >= 1.0 else 0.0
    return (agree - chance) / (1.0 - chance)


# ``_quantile`` is imported from ``._numerics`` (de-duplicated; Opus minor).
