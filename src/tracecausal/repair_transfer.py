"""Cross-example repair-transfer certification statistic ``R_hat`` (REDESIGN_v5 §4).

This is the v5 **headline** statistic. ``U_hat`` (v4) measures *necessity in-place*;
``R_hat`` measures *sufficiency across examples* — a source-derived repair policy
``rho`` applied to held-out in-class targets, scored against a **matched-null
repair** baseline, with valid dependent-pair inference.

Everything here is pure Python (no model, no GPU, no network, no run). The
``g_{ij}`` repair gains are supplied **already computed** by the (DATA_NEEDED)
repair pipeline; this module is the *inference* layer over them.

The MF-4 inference stack and the nine design-gate refinements honored:

* **Estimator** ``r_hat`` — the within-class, source != target U-statistic
  (Eq. R / §4.5). Class weights ``w_c`` are pre-registered (frozen on ``V_sel``).
* **Primary variance** ``two_way_cluster_bootstrap`` — resamples **source AND
  target** clusters independently (multiway / pigeonhole cluster bootstrap), since
  one example contaminates many pairs through *both* margins (MF-4; refinement 5).
* **Analytic cross-check** ``hajek_projection_var`` — the Hoeffding/Hajek variance
  for the **ordered/asymmetric** kernel, including **BOTH** the source projection
  ``zeta_10`` and the target-margin projection ``zeta_01`` (refinement 5), so the
  analytic floor is not understated (the cluster bootstrap must not undershoot it).
* **Null** ``class_block_permutation`` — a **diagnostic** (NOT the confirmatory test;
  G9-FIX). It is realised as source-block ``+-1`` sign flips, which probe **source-block
  sign-symmetry** under the no-transfer null (an A6-flavoured exchangeability check),
  NOT the composite sharp null ``R = 0``. Matched-null centring + A6 give *mean-zero +
  within-class exchangeability*, which do **not** imply the *distributional sign
  symmetry* a sign flip assumes, so the test is **not** exact for ``R = 0``; the
  returned object states that explicitly (``confirmatory = False``,
  ``exact_under_registered_sharp_null = False``). The **confirmatory** burden is carried
  by the two-way cluster-bootstrap CI lower bound (refinement 5 / §4.5).
* **Nested matched-null MC** ``repair_gain`` — the matched-null term is estimated
  by ``R_null`` draws; its MC variance is recorded per ``g_{ij}`` and propagated
  with **target-clustered MC covariance** (``target_clustered_mc_var``,
  refinement 6 / findings 9, 19), not naive per-pair quadrature, because one
  ``Pi_j`` null estimate is reused across every source pair sharing target ``j``, so
  its MC error is perfectly correlated **within** a target. ``two_way_cluster_bootstrap``
  realises this with a single shared per-target Gaussian draw per replicate;
  ``target_clustered_mc_var`` is the matching closed-form within-target-clustered
  variance contribution.
* **Common support / positivity** ``common_support_pairs`` — PROPOSED and the
  baselines are restricted to the **same positivity-conditioned pair set**
  (refinement 4) so they estimate the same estimand.
* **G9-NOV max baseline** ``g9_nov_margin_simultaneous`` — a **simultaneous
  correlated-bootstrap** lower CI for ``R_hat(PROPOSED) - max_b R_hat(B_b)``
  (refinement 8); Holm alone does not give a CI for a data-selected maximum.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from ._numerics import quantile as _quantile

__all__ = [
    "RepairGain",
    "Pair",
    "RHatEstimate",
    "PermutationResult",
    "HajekVariance",
    "G9NovMargin",
    "repair_gain",
    "r_hat",
    "two_way_cluster_bootstrap",
    "class_block_permutation",
    "hajek_projection_var",
    "target_clustered_mc_var",
    "common_support_pairs",
    "g9_nov_margin_simultaneous",
]


@dataclass(frozen=True)
class RepairGain:
    """A per-pair repair gain ``g_{ij}`` with its nested matched-null MC variance.

    Attributes
    ----------
    g:
        ``g_{ij}`` (Eq. g-ij): the localized-repair target gain minus the
        matched-null repair target gain.
    mc_var:
        The nested matched-null Monte-Carlo variance contribution
        ``sigma_MC^2 / R_null`` for this pair (REDESIGN_v5 §4.5).
    source_id / target_id:
        The example indices (clusters) the pair is built from.
    g3_class:
        The shared G3 class (source and target are in the same class).
    """

    g: float
    mc_var: float
    source_id: object
    target_id: object
    g3_class: str


# A lightweight alias: a Pair is just a RepairGain (carries its source/target ids).
Pair = RepairGain


@dataclass(frozen=True)
class RHatEstimate:
    """The within-class U-statistic ``R_hat`` (Eq. R, REDESIGN_v5 §4.5)."""

    r_hat: float
    per_class: dict[str, float]
    n_pairs: int
    class_weights: dict[str, float]
    n_source_clusters: int
    n_target_clusters: int


@dataclass(frozen=True)
class PermutationResult:
    """Class-block source-block sign-flip **diagnostic** outcome (REDESIGN_v5 §4.5; G9-FIX).

    ``tests`` states **exactly** what the procedure tests — and, just as importantly,
    what it does **not**. The class-block source-label permutation is realised as
    independent ``+-1`` sign flips on each (class, source) block. A sign flip is the
    correct relabeling **only if** each source block's contribution is *distributionally
    symmetric about 0* under the null.

    **Why that symmetry is NOT delivered by the design (the demotion).** Under the
    registered sharp null ``R = 0`` the per-pair gain ``g_{ij}`` (Eq. g-ij) is a
    matched-null-centred contrast, so it has **mean zero**, and A6 makes targets
    **within-class exchangeable**. But mean-zero + exchangeability is strictly weaker
    than the *sign-symmetry* a ``+-1`` flip assumes: a skewed but mean-zero source-block
    distribution is exchangeable yet not sign-symmetric. Matched-null centring therefore
    does **not** justify the sign-flip as an exact realisation of the source-label
    permutation, and on real AR-LLM repair gains the symmetry is unverifiable. Hence
    this object is a **diagnostic**, not the confirmatory test:

    * ``confirmatory = False`` — the G9 confirmatory burden is carried by the **two-way
      cluster-bootstrap CI** (which assumes neither symmetry nor a sharp null), per
      refinement 5 / §4.5. ``class_block_permutation`` corroborates it as a
      source-block sign-symmetry / A6-exchangeability **diagnostic**.
    * ``exact_under_registered_sharp_null = False`` — no exactness for ``R = 0`` is
      claimed (the prior ``True`` was an over-claim: centring gives mean-zero, not
      sign-symmetry).
    * ``exact_for_composite_null = False`` — likewise no arbitrary-composite-null
      exactness (refinement 7).

    A small ``p_value`` here is still *informative* (a strongly same-signed transfer
    signal makes the un-flipped statistic extreme under sign permutation), but it is
    read as a **diagnostic** corroborating the cluster-bootstrap CI, never as a
    standalone exact significance certificate.
    """

    p_value: float
    observed: float
    n_permutations: int
    tests: str = (
        "DIAGNOSTIC (G9-FIX, not confirmatory): source-block +-1 sign-flip probe of "
        "source-block sign-symmetry / A6 within-class exchangeability. This is NOT an "
        "exact test of R = 0: matched-null centring + A6 give mean-zero + "
        "exchangeability, which do NOT imply the distributional sign-symmetry a sign "
        "flip assumes. The G9 confirmatory burden is the two-way cluster-bootstrap CI "
        "lower bound (REDESIGN_v5 §4.5); this corroborates it as a diagnostic only."
    )
    # DIAGNOSTIC role: the confirmatory carrier is the two-way cluster-bootstrap CI.
    confirmatory: bool = False
    # NOT exact for the registered sharp null (centring gives mean-zero, NOT sign-symmetry).
    exact_under_registered_sharp_null: bool = False
    # NOT exact for an arbitrary asymmetric composite null either (refinement 7).
    exact_for_composite_null: bool = False


@dataclass(frozen=True)
class HajekVariance:
    """U-statistic Hajek/Hoeffding variance for the ordered kernel (refinement 5).

    The kernel ``g_{ij}`` is **asymmetric** (source role != target role), so the
    first-order projection has **two** parts:

    * ``zeta_10`` — variance of ``E[g_{ij} | i]`` (the *source* projection);
    * ``zeta_01`` — variance of ``E[g_{ij} | j]`` (the *target-margin* projection).

    The analytic variance floor used as the cluster-bootstrap cross-check is

        ``var_hat ~= zeta_10 / n_source + zeta_01 / n_target``

    which includes BOTH projections (refinement 5: not only ``4*zeta_1``), so the
    floor is not understated.
    """

    zeta_10: float
    zeta_01: float
    n_source: int
    n_target: int
    var_hat: float


@dataclass(frozen=True)
class G9NovMargin:
    """G9-NOV simultaneous margin ``R_hat(PROPOSED) - max_b R_hat(B_b)`` (refinement 8)."""

    margin_point: float
    margin_ci_low: float
    margin_ci_high: float
    argmax_baseline: str
    n_bootstrap: int
    clears_zero: bool


def repair_gain(
    y_localized: float,
    y_noop: float,
    matched_null_repair_samples: Sequence[float],
    *,
    source_id: object,
    target_id: object,
    g3_class: str,
) -> RepairGain:
    """Per-pair repair gain ``g_{ij}`` with nested matched-null MC variance (Eq. g-ij).

    ``g_{ij} = [Y_j(localized repair) - Y_j(no_op)]
              - mean_{S~Pi_j}[Y_j(matched-null repair) - Y_j(no_op)]``

    ``matched_null_repair_samples`` are the ``R_null`` matched-null **repair gains**
    ``Y_j(do(phi_rho^S)) - Y_j(no_op)`` already differenced against ``no_op`` (one
    per draw from ``Pi_j``). The nested matched-null MC variance recorded for this
    pair is the SE^2 of that mean, ``sample_var / R_null`` (REDESIGN_v5 §4.5), so
    the inner sampling noise can be propagated through the cluster bootstrap.

    Raises
    ------
    ValueError
        If ``matched_null_repair_samples`` is empty (the matched-null term is
        unestimated; the pair must be excluded or ``R_null`` raised).
    """
    samples = [float(v) for v in matched_null_repair_samples]
    r_null = len(samples)
    if r_null == 0:
        raise ValueError("matched_null_repair_samples is empty; cannot estimate the null term")
    localized_gain = float(y_localized) - float(y_noop)
    null_mean = sum(samples) / r_null
    g = localized_gain - null_mean
    if r_null >= 2:
        m = null_mean
        sample_var = sum((v - m) ** 2 for v in samples) / (r_null - 1)
        mc_var = sample_var / r_null
    else:
        mc_var = 0.0  # single draw: MC variance unestimable, recorded as 0 (caller raises R_null)
    return RepairGain(
        g=g, mc_var=mc_var, source_id=source_id, target_id=target_id, g3_class=g3_class
    )


def _per_class_means(pairs: Sequence[RepairGain]) -> dict[str, list[float]]:
    by_class: dict[str, list[float]] = {}
    for p in pairs:
        by_class.setdefault(p.g3_class, []).append(p.g)
    return by_class


def r_hat(
    pairs: Sequence[RepairGain],
    *,
    weights: Mapping[str, float] | None = None,
) -> RHatEstimate:
    """The within-class U-statistic ``R_hat`` (Eq. R, REDESIGN_v5 §4.5).

    Per class ``c``, ``R_hat_c = mean of g_{ij}`` over the ordered in-class pairs
    ``(i, j)``; overall ``R_hat = sum_c w_c R_hat_c`` with pre-registered weights
    ``w_c`` (default: **equal** across the classes present). Weights are normalized
    to sum to 1 over the classes actually present (so a frozen weight map with extra
    classes still yields a convex combination over observed classes).

    Pairs are expected to already be source != target, in-class, and positivity-OK
    (use :func:`common_support_pairs` to enforce the common-support restriction).
    """
    if not pairs:
        raise ValueError("no pairs supplied to r_hat")
    by_class = _per_class_means(pairs)
    per_class = {c: sum(gs) / len(gs) for c, gs in by_class.items()}

    classes = sorted(per_class)
    if weights is None:
        w = {c: 1.0 / len(classes) for c in classes}
    else:
        raw = {c: float(weights.get(c, 0.0)) for c in classes}
        total = sum(raw.values())
        if total <= 0.0:
            raise ValueError("class weights sum to <= 0 over observed classes")
        w = {c: raw[c] / total for c in classes}

    overall = sum(w[c] * per_class[c] for c in classes)
    source_clusters = {p.source_id for p in pairs}
    target_clusters = {p.target_id for p in pairs}
    return RHatEstimate(
        r_hat=overall,
        per_class=per_class,
        n_pairs=len(pairs),
        class_weights=w,
        n_source_clusters=len(source_clusters),
        n_target_clusters=len(target_clusters),
    )


def two_way_cluster_bootstrap(
    pairs: Sequence[RepairGain],
    *,
    weights: Mapping[str, float] | None = None,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
    propagate_mc: bool = True,
) -> tuple[float, float]:
    """Two-way (source x target) cluster bootstrap CI for ``R_hat`` (MF-4 primary).

    Resamples **source clusters and target clusters independently** (multiway /
    pigeonhole cluster bootstrap over the two example-index margins): each replicate
    draws a source-cluster multiset and a target-cluster multiset with replacement,
    then keeps every pair whose source AND target were drawn (with multiplicity =
    product of the two draw counts). This accounts for a single example
    contaminating many pairs through *both* margins (REDESIGN_v5 §4.5).

    When ``propagate_mc`` is ``True`` the nested matched-null MC noise
    (``RepairGain.mc_var``) is injected as a **target-clustered** Gaussian
    perturbation (refinement 6 / findings 9, 19): a **single** standard-normal draw
    ``z_t`` is taken **per target ``t`` per replicate** and each pair's ``g`` is
    perturbed by ``z_t * sqrt(mc_var)``. This is the correct propagation because one
    ``Pi_j`` matched-null estimate is **reused across every source pair sharing
    target ``j``**, so its MC error is *perfectly correlated within a target*, not
    independent per pair. An independent per-pair draw (the previous behaviour) would
    understate the within-target correlated MC variance and make the CI mildly
    anti-conservative; the shared ``z_t`` reproduces the within-target covariance
    (see :func:`target_clustered_mc_var` for the closed-form variance this matches).

    Returns the percentile interval ``(ci_lo, ci_hi)``; G9 gates on the
    Holm-corrected ``ci_lo`` (the caller applies the Holm fold).
    """
    if not pairs:
        raise ValueError("no pairs supplied to two_way_cluster_bootstrap")
    rng = random.Random(seed)

    sources = sorted({p.source_id for p in pairs}, key=repr)
    targets = sorted({p.target_id for p in pairs}, key=repr)
    n_s = len(sources)
    n_t = len(targets)

    # index pairs by (source, target) for fast multiplicity lookup
    by_st: dict[tuple[object, object], list[RepairGain]] = {}
    for p in pairs:
        by_st.setdefault((p.source_id, p.target_id), []).append(p)

    replicates: list[float] = []
    for _ in range(n_bootstrap):
        s_counts: dict[object, int] = {}
        for _k in range(n_s):
            s = sources[rng.randrange(n_s)]
            s_counts[s] = s_counts.get(s, 0) + 1
        t_counts: dict[object, int] = {}
        for _k in range(n_t):
            t = targets[rng.randrange(n_t)]
            t_counts[t] = t_counts.get(t, 0) + 1

        # target-clustered MC noise: one shared standard-normal z_t per DRAWN target
        # (refinement 6 / findings 9, 19). Every pair with target t is perturbed by
        # z_t * sqrt(mc_var), so the matched-null MC error is perfectly correlated
        # within a target (the reused Pi_j estimate), NOT independent per pair.
        if propagate_mc:
            z_by_target = {t: rng.gauss(0.0, 1.0) for t in t_counts}
        else:
            z_by_target = {}

        # accumulate weighted per-class sums under the two-way multiplicity
        class_sum: dict[str, float] = {}
        class_w: dict[str, float] = {}
        for s, sc in s_counts.items():
            for t, tc in t_counts.items():
                bucket = by_st.get((s, t))
                if not bucket:
                    continue
                mult = sc * tc
                z_t = z_by_target.get(t, 0.0)
                for p in bucket:
                    g = p.g
                    if propagate_mc and p.mc_var > 0.0:
                        g += z_t * math.sqrt(p.mc_var)
                    class_sum[p.g3_class] = class_sum.get(p.g3_class, 0.0) + mult * g
                    class_w[p.g3_class] = class_w.get(p.g3_class, 0.0) + mult
        if not class_w:
            continue  # degenerate replicate (no overlapping pair drawn)
        per_class = {c: class_sum[c] / class_w[c] for c in class_w}
        classes = sorted(per_class)
        if weights is None:
            ww = {c: 1.0 / len(classes) for c in classes}
        else:
            raw = {c: float(weights.get(c, 0.0)) for c in classes}
            tot = sum(raw.values())
            if tot <= 0.0:
                continue
            ww = {c: raw[c] / tot for c in classes}
        replicates.append(sum(ww[c] * per_class[c] for c in classes))

    if not replicates:
        raise ValueError("all two-way bootstrap replicates were degenerate")
    replicates.sort()
    lo_q = (1.0 - ci) / 2.0
    return _quantile(replicates, lo_q), _quantile(replicates, 1.0 - lo_q)


def _block_signflip_statistic(
    by_class: dict[str, list[RepairGain]],
    signs: Mapping[str, int],
    weights: Mapping[str, float] | None,
) -> float:
    """Class-weighted mean of source-block-sign-flipped gains (diagnostic statistic).

    Each *source block* (the gains sharing a source within a class) is multiplied by
    a +-1 sign; the statistic is the class-weighted mean of the flipped gains. This is
    the **diagnostic** statistic for the source-block sign-symmetry probe (G9-FIX): if
    source blocks were sign-symmetric about 0 under the null the sign assignment would
    be exchange-invariant, and a genuine same-signed transfer signal makes the
    un-flipped (all-+1) value extreme. Sign-symmetry is NOT guaranteed by matched-null
    centring + A6 (mean-zero + exchangeability is weaker), so the probe is read as a
    diagnostic, not an exact test -- see :func:`class_block_permutation`.
    """
    class_sum: dict[str, float] = {}
    class_n: dict[str, int] = {}
    for cls, group in by_class.items():
        for p in group:
            s = signs[(cls, p.source_id)]
            class_sum[cls] = class_sum.get(cls, 0.0) + s * p.g
            class_n[cls] = class_n.get(cls, 0) + 1
    per_class = {c: class_sum[c] / class_n[c] for c in class_n}
    classes = sorted(per_class)
    if weights is None:
        ww = {c: 1.0 / len(classes) for c in classes}
    else:
        raw = {c: float(weights.get(c, 0.0)) for c in classes}
        tot = sum(raw.values())
        if tot <= 0.0:
            ww = {c: 1.0 / len(classes) for c in classes}
        else:
            ww = {c: raw[c] / tot for c in classes}
    return sum(ww[c] * per_class[c] for c in classes)


def class_block_permutation(
    pairs: Sequence[RepairGain],
    *,
    weights: Mapping[str, float] | None = None,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> PermutationResult:
    """Class-block source-block sign-flip DIAGNOSTIC (REDESIGN_v5 sec 4.5; G9-FIX).

    **Role: diagnostic, NOT the confirmatory test.** This procedure flips an
    independent ``+-1`` sign on each (class, source) block of the matched-null-centred
    gains ``g_{ij}`` (Eq. g-ij) and compares the observed class-weighted mean to the
    sign-flip distribution. It is a probe of **source-block sign-symmetry / A6
    within-class exchangeability**, used to *corroborate* the G9 confirmatory test --
    the **two-way cluster-bootstrap CI** (:func:`two_way_cluster_bootstrap`), which
    carries the significance burden in :func:`tracecausal.ciu.g9_repair_gate`.

    **Why it is a diagnostic and not exact for ``R = 0`` (the demotion, G9-FIX).** A
    ``+-1`` sign flip realises the source-label permutation *exactly* only when each
    source block's contribution is **distributionally symmetric about 0** under the
    null. Under the registered sharp null ``g_{ij}`` is matched-null-centred (so it
    has **mean zero**) and A6 makes targets **within-class exchangeable** -- but
    mean-zero plus exchangeability is **strictly weaker** than sign-symmetry (a skewed
    mean-zero block is exchangeable yet not sign-symmetric). The earlier
    "exact-by-construction under A6" claim was therefore an over-claim: the centring
    does not license the sign flip as an exact permutation, and on real AR-LLM repair
    gains the symmetry is not testable. The returned :class:`PermutationResult` records
    ``confirmatory = False`` and ``exact_under_registered_sharp_null = False`` (and
    ``exact_for_composite_null = False``, refinement 7).

    Clustering the sign by (class, source) preserves the pair dependence (one source
    contaminates many pairs through the source margin), so the diagnostic stays powered
    against a strongly same-signed transfer signal (which makes the un-flipped
    statistic extreme). The two-sided p-value is the tail probability of
    ``|stat_perm| >= |stat_obs|`` with the +1 finite-sample correction. Read it as a
    diagnostic corroboration of the cluster-bootstrap CI, never as a standalone exact
    significance certificate.
    """
    if not pairs:
        raise ValueError("no pairs supplied to class_block_permutation")
    obs_r_hat = r_hat(pairs, weights=weights).r_hat

    by_class: dict[str, list[RepairGain]] = {}
    source_keys: list[tuple[str, object]] = []
    seen: set[tuple[str, object]] = set()
    for p in pairs:
        by_class.setdefault(p.g3_class, []).append(p)
        key = (p.g3_class, p.source_id)
        if key not in seen:
            seen.add(key)
            source_keys.append(key)

    # observed statistic uses all +1 signs (== the class-weighted mean)
    all_plus = {k: 1 for k in source_keys}
    obs_stat = _block_signflip_statistic(by_class, all_plus, weights)
    abs_obs = abs(obs_stat)

    rng = random.Random(seed)
    count_extreme = 0
    for _ in range(n_permutations):
        signs = {k: (1 if rng.random() < 0.5 else -1) for k in source_keys}
        stat = _block_signflip_statistic(by_class, signs, weights)
        if abs(stat) >= abs_obs - 1e-15:
            count_extreme += 1
    p_value = (count_extreme + 1) / (n_permutations + 1)
    return PermutationResult(
        p_value=p_value, observed=obs_r_hat, n_permutations=n_permutations
    )


def hajek_projection_var(
    pairs: Sequence[RepairGain],
    *,
    per_class: bool = False,
) -> HajekVariance:
    """Hajek/Hoeffding variance for the ordered kernel, BOTH projections (refinement 5).

    For the asymmetric two-sample U-statistic ``R_hat = mean g_{ij}`` the
    first-order Hajek projection has two margins:

    * source projection ``g_{i.} = E[g_{ij} | i]`` (mean over targets, for source i)
      with variance ``zeta_10``;
    * target projection ``g_{.j} = E[g_{ij} | j]`` (mean over sources, for target j)
      with variance ``zeta_01``.

    The analytic variance floor is ``zeta_10 / n_source + zeta_01 / n_target``,
    which **includes both** projections (refinement 5 — not only ``4*zeta_1``), so
    the two-way cluster bootstrap cross-checks against a floor that is not
    understated. ``per_class=True`` pools the projection variances across classes
    after centering within class (so class-mean differences do not inflate the
    floor).

    Returns a :class:`HajekVariance`. Requires at least 2 source and 2 target
    clusters for a defined sample variance.
    """
    if not pairs:
        raise ValueError("no pairs supplied to hajek_projection_var")

    # center within class when pooling across classes
    if per_class:
        class_mean = {c: m for c, m in _per_class_means_scalar(pairs).items()}
        items = [
            (p.source_id, p.target_id, p.g - class_mean[p.g3_class]) for p in pairs
        ]
    else:
        items = [(p.source_id, p.target_id, p.g) for p in pairs]

    by_source: dict[object, list[float]] = {}
    by_target: dict[object, list[float]] = {}
    for s, t, g in items:
        by_source.setdefault(s, []).append(g)
        by_target.setdefault(t, []).append(g)

    source_means = [sum(v) / len(v) for v in by_source.values()]
    target_means = [sum(v) / len(v) for v in by_target.values()]
    n_s = len(source_means)
    n_t = len(target_means)

    zeta_10 = _sample_var(source_means)
    zeta_01 = _sample_var(target_means)
    var_hat = (zeta_10 / n_s if n_s else 0.0) + (zeta_01 / n_t if n_t else 0.0)
    return HajekVariance(
        zeta_10=zeta_10, zeta_01=zeta_01, n_source=n_s, n_target=n_t, var_hat=var_hat
    )


def target_clustered_mc_var(
    pairs: Sequence[RepairGain],
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Within-target-clustered matched-null MC variance of ``R_hat`` (refinement 6).

    The nested matched-null estimate ``Pi_j`` is computed **once per target ``j``**
    and reused across **every** source pair that shares target ``j``. Its Monte-Carlo
    error is therefore **perfectly correlated within a target**, not independent per
    pair (findings 9, 19). The correct MC variance contribution to the (equal-weight,
    per class) mean ``R_hat_c = mean_{(i,j) in c} g_{ij}`` is the *target-clustered*

        ``Var_MC(R_hat_c) = sum_{targets t in c} ( n_{c,t} / N_c )^2 * s_{c,t}^2``

    where ``n_{c,t}`` is the number of pairs in class ``c`` with target ``t``, ``N_c``
    the class pair count, and ``s_{c,t}^2`` the shared per-target matched-null MC
    variance (the common value of ``RepairGain.mc_var`` across that target's pairs;
    the max is used if they differ). This is **larger** than the naive per-pair
    quadrature ``sum_pairs (1/N_c)^2 mc_var`` whenever a target appears in more than
    one pair, which is exactly the anti-conservative gap the independent-per-pair
    draw produced. The overall variance is the class-weighted combination
    ``sum_c w_c^2 Var_MC(R_hat_c)`` (independent matched-null draws across classes).

    This is the closed-form companion to the target-clustered Gaussian draw in
    :func:`two_way_cluster_bootstrap` (a single shared ``z_t`` per target).
    """
    if not pairs:
        raise ValueError("no pairs supplied to target_clustered_mc_var")

    by_class: dict[str, list[RepairGain]] = {}
    for p in pairs:
        by_class.setdefault(p.g3_class, []).append(p)

    classes = sorted(by_class)
    if weights is None:
        w = {c: 1.0 / len(classes) for c in classes}
    else:
        raw = {c: float(weights.get(c, 0.0)) for c in classes}
        total = sum(raw.values())
        if total <= 0.0:
            raise ValueError("class weights sum to <= 0 over observed classes")
        w = {c: raw[c] / total for c in classes}

    total_var = 0.0
    for c in classes:
        group = by_class[c]
        n_c = len(group)
        # shared per-target MC variance: pairs with the same target reuse one Pi_j
        # estimate, so take the (max) recorded mc_var for that target as the shared s^2.
        by_target_var: dict[object, float] = {}
        by_target_count: dict[object, int] = {}
        for p in group:
            by_target_var[p.target_id] = max(by_target_var.get(p.target_id, 0.0), p.mc_var)
            by_target_count[p.target_id] = by_target_count.get(p.target_id, 0) + 1
        var_c = 0.0
        for t, s2 in by_target_var.items():
            frac = by_target_count[t] / n_c
            var_c += (frac ** 2) * s2
        total_var += (w[c] ** 2) * var_c
    return total_var


def common_support_pairs(
    pairs_by_arm: Mapping[str, Sequence[RepairGain]],
) -> dict[str, list[RepairGain]]:
    """Restrict every arm to the SAME positivity-conditioned pair set (refinement 4).

    For G9-NOV (PROPOSED vs B1/B2/B3) to estimate the **same** estimand, all arms
    must be compared on the **common-support set**: the ordered ``(source, target)``
    index pairs for which *every* arm has a positivity-OK repair gain. This returns
    each arm's gains restricted to that intersection (keyed by
    ``(source_id, target_id, g3_class)``), so a pair present for PROPOSED but
    positivity-excluded for a baseline is dropped from **all** arms (else the arms
    estimate different estimands — refinement 4).
    """
    arms = list(pairs_by_arm)
    if not arms:
        return {}
    keysets: list[set] = []
    indexed: dict[str, dict[tuple, RepairGain]] = {}
    for arm in arms:
        idx: dict[tuple, RepairGain] = {}
        for p in pairs_by_arm[arm]:
            idx[(p.source_id, p.target_id, p.g3_class)] = p
        indexed[arm] = idx
        keysets.append(set(idx))
    common = set.intersection(*keysets) if keysets else set()
    return {arm: [indexed[arm][k] for k in sorted(common, key=repr)] for arm in arms}


def g9_nov_margin_simultaneous(
    proposed: Sequence[RepairGain],
    baselines: Mapping[str, Sequence[RepairGain]],
    *,
    weights: Mapping[str, float] | None = None,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
    enforce_common_support: bool = True,
) -> G9NovMargin:
    """Simultaneous correlated-bootstrap CI for the G9-NOV max-baseline margin (refinement 8).

    G9-NOV compares PROPOSED to ``max_b R_hat(B_b)``. A data-selected maximum has no
    valid CI from Holm alone (refinement 8); we form a **simultaneous
    correlated-bootstrap** distribution: in each two-way-cluster replicate we draw a
    single source-cluster and target-cluster multiset and apply it to **PROPOSED and
    every baseline jointly** (preserving their correlation), recompute each arm's
    ``R_hat``, and record ``R_hat(PROPOSED) - max_b R_hat(B_b)``. The CI is the
    percentile interval of that margin (which already accounts for the max-selection
    and the cross-arm correlation).

    **Identical pair keys enforced (findings 5, 17).** The IUT/common-support
    requirement is that every arm estimate the **same** estimand on the **same**
    ordered ``(source, target, class)`` pair set. With ``enforce_common_support``
    (default ``True``) this is checked **fail-closed**: if PROPOSED and any baseline
    do not carry *identical* pair keys, a :class:`ValueError` is raised naming the
    mismatch, rather than silently skipping the missing baseline pairs (which would
    let a baseline be scored on a smaller, easier pair set). Call
    :func:`common_support_pairs` first to align the arms; this guard then verifies the
    alignment held. Pass ``enforce_common_support=False`` only for a pre-aligned
    fixture you have already intersected.
    """
    if not proposed:
        raise ValueError("no PROPOSED pairs supplied")
    if not baselines:
        raise ValueError("no baseline arms supplied")

    if enforce_common_support:
        prop_keys = sorted(
            ((p.source_id, p.target_id, p.g3_class) for p in proposed), key=repr
        )
        prop_keyset = set(prop_keys)
        if len(prop_keyset) != len(prop_keys):
            raise ValueError(
                "PROPOSED carries duplicate (source, target, class) pair keys; the "
                "common-support pair set must be unique per arm (findings 5, 17)"
            )
        for b, ps in baselines.items():
            b_keys = [(p.source_id, p.target_id, p.g3_class) for p in ps]
            b_keyset = set(b_keys)
            if b_keyset != prop_keyset:
                missing = sorted(map(repr, prop_keyset - b_keyset))
                extra = sorted(map(repr, b_keyset - prop_keyset))
                raise ValueError(
                    f"baseline {b!r} pair keys differ from PROPOSED (common-support / "
                    f"IUT requires identical pair keys across arms; findings 5, 17). "
                    f"missing from {b!r}: {missing}; extra in {b!r}: {extra}. "
                    "Call common_support_pairs(...) to align the arms first."
                )

    # point margin
    r_prop = r_hat(proposed, weights=weights).r_hat
    base_points = {b: r_hat(ps, weights=weights).r_hat for b, ps in baselines.items()}
    argmax_b = max(base_points, key=lambda b: base_points[b])
    margin_point = r_prop - base_points[argmax_b]

    # shared cluster universe (common support => identical (s,t) keys across arms)
    sources = sorted({p.source_id for p in proposed}, key=repr)
    targets = sorted({p.target_id for p in proposed}, key=repr)
    n_s, n_t = len(sources), len(targets)

    def _index(ps: Sequence[RepairGain]) -> dict[tuple[object, object], list[RepairGain]]:
        out: dict[tuple[object, object], list[RepairGain]] = {}
        for p in ps:
            out.setdefault((p.source_id, p.target_id), []).append(p)
        return out

    prop_idx = _index(proposed)
    base_idx = {b: _index(ps) for b, ps in baselines.items()}

    def _rhat_under(counts_s, counts_t, idx, ww_weights):
        class_sum: dict[str, float] = {}
        class_w: dict[str, float] = {}
        for s, sc in counts_s.items():
            for t, tc in counts_t.items():
                bucket = idx.get((s, t))
                if not bucket:
                    continue
                mult = sc * tc
                for p in bucket:
                    class_sum[p.g3_class] = class_sum.get(p.g3_class, 0.0) + mult * p.g
                    class_w[p.g3_class] = class_w.get(p.g3_class, 0.0) + mult
        if not class_w:
            return None
        per_class = {c: class_sum[c] / class_w[c] for c in class_w}
        classes = sorted(per_class)
        if ww_weights is None:
            ww = {c: 1.0 / len(classes) for c in classes}
        else:
            raw = {c: float(ww_weights.get(c, 0.0)) for c in classes}
            tot = sum(raw.values())
            if tot <= 0.0:
                return None
            ww = {c: raw[c] / tot for c in classes}
        return sum(ww[c] * per_class[c] for c in classes)

    rng = random.Random(seed)
    margins: list[float] = []
    for _ in range(n_bootstrap):
        counts_s: dict[object, int] = {}
        for _k in range(n_s):
            s = sources[rng.randrange(n_s)]
            counts_s[s] = counts_s.get(s, 0) + 1
        counts_t: dict[object, int] = {}
        for _k in range(n_t):
            t = targets[rng.randrange(n_t)]
            counts_t[t] = counts_t.get(t, 0) + 1
        rp = _rhat_under(counts_s, counts_t, prop_idx, weights)
        if rp is None:
            continue
        bvals = []
        for b in baselines:
            rb = _rhat_under(counts_s, counts_t, base_idx[b], weights)
            if rb is None:
                bvals = None
                break
            bvals.append(rb)
        if not bvals:
            continue
        margins.append(rp - max(bvals))

    if not margins:
        raise ValueError("all simultaneous bootstrap replicates were degenerate")
    margins.sort()
    lo_q = (1.0 - ci) / 2.0
    ci_lo = _quantile(margins, lo_q)
    ci_hi = _quantile(margins, 1.0 - lo_q)
    return G9NovMargin(
        margin_point=margin_point,
        margin_ci_low=ci_lo,
        margin_ci_high=ci_hi,
        argmax_baseline=argmax_b,
        n_bootstrap=len(margins),
        clears_zero=ci_lo > 0.0,
    )


# --- internal numerics ------------------------------------------------------


def _per_class_means_scalar(pairs: Sequence[RepairGain]) -> dict[str, float]:
    by_class = _per_class_means(pairs)
    return {c: sum(gs) / len(gs) for c, gs in by_class.items()}


def _sample_var(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)
