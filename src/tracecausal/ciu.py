"""CIU (Counterfactual Intervention Utility) Phase-2 contract.

Implements the estimand record and gate machinery specified in
``docs/redesign/REDESIGN_v3.md`` §5 and extended additively by
``docs/redesign/REDESIGN_v4.md`` §5:

* ``CIURecord`` — the additive frozen dataclass persisting the **edit budget**,
  **null-pool hash**, **reference-run hash**, **evaluator hash**, the per-example
  ``tau_i`` and ``Pi_i`` provenance, and the v4 optional OOD/leakage fields.
* ``validate_ciu_record`` — contract checks (empty hashes, ``server_authorized``,
  budget mismatch, seed floor, MDE-required ``n``, and the v4 G7/G8 checks that
  fire only when the new fields are populated).
* ``ciu_gate`` — the revised **G5' novelty-margin gate**: the proposed selector
  must beat the best **adapted** detector on ``U_hat`` by the pre-registered
  margin ``0.03``. It **wraps** ``metrics.passes_intervention_gate`` (never edits
  it) and **never** returns ``"invalidated"`` for a detector with positive
  ``U_hat`` (the central v3 correction: that is a novelty question, not a
  causal-identification failure).
* ``baseline_readiness`` — preflight flag for baselines still marked
  ``pending_before_server_run`` / ``verify_before_run``.

v4 additive helpers (pure, validation-split-only, no model/GPU):
``leakage_slope_regression`` (yields ``beta_hat`` + paired-bootstrap CI for the
G7 upper-CI bound), and ``ood_deflation`` (subtracts the calibrated operator
footprint at matched ``displaced_mass``, §2.12).

**BUILD-NOW / RUN-LATER.** Nothing here runs an experiment, loads a model, or
authorises a server run. ``server_authorized`` defaults to ``False`` and a
``True`` value is a validation error.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

from ._numerics import quantile as _quantile
from .metrics import passes_intervention_gate

__all__ = [
    "CIUVerdict",
    "CIURecord",
    "validate_ciu_record",
    "ciu_gate",
    "g9_repair_gate",
    "g9_novelty_gate",
    "calibrate_m_r",
    "baseline_readiness",
    "leakage_slope_regression",
    "ood_deflation",
    "ood_slope_ci_excludes_manufacture",
    "NOVELTY_MARGIN",
    "LOCALIZATION_MARGIN",
    "NECESSITY_MARGIN",
    "OOD_INERT_MANUFACTURE_BOUND",
    "MIN_PAPER_SEEDS",
    "PROXIMITY_POOL_MIN",
    "REPAIR_UTILITY_BOUND",
    "POSITIVITY_EXCLUDED_MAX",
    "PENDING_MARKERS",
]

# Pre-registered margins (REDESIGN_v3 §4, REDESIGN_v4 §4); not weakened by v4.
NOVELTY_MARGIN: float = 0.03  # G5': selector must beat best adapted detector by this
LOCALIZATION_MARGIN: float = 0.03  # G6/G7: leakage bound beta_hi*Delta_pos must clear this
NECESSITY_MARGIN: float = 0.05  # G1: necessity threshold on the (deflated) lower CI
# G8 (REDESIGN_v4 §4.1 / §2.12): the calibrated OOD slope CI must EXCLUDE a slope
# large enough to manufacture an inert necessity signal U_inert >= this over the
# observed displaced_mass range. (Same 0.05 bar as G1's necessity margin.)
OOD_INERT_MANUFACTURE_BOUND: float = 0.05
MIN_PAPER_SEEDS: int = 20  # seed floor (REDESIGN_v4 preserves >= 20 floor)
# G9 (REDESIGN_v5 §5.1): bounded repair utility cost; the repair must not trade
# factuality for utility beyond this (mirrors G2's 0.02 utility bar).
REPAIR_UTILITY_BOUND: float = 0.02
# G9 (REDESIGN_v5 §4.4 A7 / §5.1): per-class excluded-pair fraction ceiling; above
# this the class is under-powered and routed to "insufficient positivity", not a null.
POSITIVITY_EXCLUDED_MAX: float = 0.5
# Mean in-bin proximity-pool floor (REDESIGN_v4 §4.6 B3); mirrors
# ``nuisance.POOL_MIN``. Below this the proximity stratifier shrank the pool too
# far and the bin must be coarsened before lock — a v4 lock requires it be met.
PROXIMITY_POOL_MIN: int = 8

# Registry markers that block a run at preflight (REDESIGN_v3 §5 baseline_readiness).
PENDING_MARKERS: tuple[str, ...] = ("pending_before_server_run", "verify_before_run")

# ``ciu_gate`` verdicts (REDESIGN_v3 §5). Note: there is deliberately NO
# "invalidated" verdict for a positive-U detector.
CIUVerdict = Literal["useful_candidate", "diagnostic", "not_novel"]


@dataclass(frozen=True)
class CIURecord:
    """The enforced CIU estimand record (REDESIGN_v3 §5; v4 §5 optional fields).

    The first block of fields is the v3 contract verbatim; the trailing block is
    the v4 additive, **defaulted** extension (back-compatible — v3 records still
    validate because every new field defaults to ``None``/``0``).
    """

    # --- which estimand --------------------------------------------------
    selector_id: str
    operator: str
    reference_type: str

    # --- matched-control identity (A1'/A2/A3, REDESIGN_v3 §2.3, §5) -------
    edit_budget: int  # k; must match between targeted and control provenance
    null_pool_hash: str  # serialised per-example Pi_i
    noop_run_hash: str  # shared no-op (A2)
    evaluator_hash: str  # proper-scored evaluator identity (A3)
    evaluator_kappa: float  # evaluator agreement, for (2k-1) attenuation (§4.5/§4.6)
    ref_hash: str  # reference-run identity (§2.7)

    # --- estimator + denominator -----------------------------------------
    n_examples: int
    r_int: int  # intervention-repeats sampling level
    b_boot: int  # bootstrap level
    s_seed: int  # seed level (>= 20 floor)
    u_hat: float
    ci_low: float
    ci_high: float
    d_util: float  # utility drop

    # --- per-example provenance (tau_i / Pi_i, REDESIGN_v3 §2.5, §5) ------
    # tau_per_example[i] is the targeted per-example effect tau_i(S*);
    # pi_mean_per_example[i] is the matched null-pool mean bar tau_i(Pi) estimate.
    tau_per_example: tuple[float, ...] = field(default_factory=tuple)
    pi_mean_per_example: tuple[float, ...] = field(default_factory=tuple)
    matched_control_provenance: tuple[str, ...] = field(default_factory=tuple)

    # --- novelty / localization optional context ------------------------
    adapter_hash: str | None = None  # baseline segment adapter (G5')
    best_detector_u: float | None = None  # best ADAPTED detector U_hat (G5')
    proximity_bin: int | None = None  # A4* proximity stratification (§2.10)
    displaced_mass: float = 0.0  # mask sanity (§2.2)
    invalid_count: int = 0

    # --- v4 additive, back-compatible (REDESIGN_v4 §5) -------------------
    u_deflated: float | None = None  # OOD-deflated U_hat (§2.12); G1 evaluated on this
    # The PROPAGATED OOD-deflated lower CI from ``ood_deflation`` (§2.12): when the
    # operator-footprint slope CI is propagated in quadrature the deflated interval
    # WIDENS, so the gated lower bound is NOT just ``ci_low`` shifted by
    # ``u_hat - u_deflated``. G1 must gate on this propagated lower bound when it is
    # recorded; persisting it closes the bypass where a record stores only
    # ``u_deflated`` and the necessity gate silently uses the un-widened interval.
    u_deflated_ci_low: float | None = None  # propagated deflated lower CI (§2.12); G1 gate
    ood_slope: float | None = None  # calibrated OOD dose-response slope s_OOD (§2.12)
    ood_slope_ci: tuple[float, float] | None = None  # 95% CI of s_OOD
    displaced_mass_range: float | None = None  # observed displaced_mass span D (§2.12); G8 kill-gate
    beta_hi: float | None = None  # upper-CI leakage slope (§2.11); G7 uses this
    proximity_bin_width: float | None = None  # Delta_pos (token units); G7 bound width
    m_pool_mean: float | None = None  # mean in-bin pool size (§4.6 B3)

    # --- v4 control OUTCOMES persisted for require_v4 enforcement (§4.1) -------
    # The design requires ALL preregistered controls — not only the leakage/OOD
    # slope fields — to be persisted and enforced before causal wording (review
    # fix 3). These persist the SHAM/no-op null results and the graded-oracle
    # outcomes so a record cannot omit a control to bypass a gate.
    sham_u_ci: tuple[float, float] | None = None  # SHAM-MASK U_hat CI (G8 null); must bracket 0
    noop_u_ci: tuple[float, float] | None = None  # no-op (A2) U_hat CI; must bracket 0
    oracle_pass: bool | None = None  # clean-oracle recovery pass (G7 conjunct 1a)
    graded_curve_pass: bool | None = None  # graded-family curve-match pass (G7 conjunct 1b)

    # --- v5 additive, back-compatible (REDESIGN_v5 §9) ------------------------
    # The cross-example repair-transfer certification fields. ALL default to None so
    # every v3/v4 record still validates; they are HARD-required only at a v5 lock
    # (validate_ciu_record(require_v5=True)).
    r_hat: float | None = None  # headline R_hat (Eq. R); G9 gates on its lower CI
    r_hat_ci: tuple[float, float] | None = None  # two-way cluster bootstrap CI (MF-4)
    r_hat_perm_p: float | None = None  # class-block permutation p-value (MF-4 null)
    d_util_repair: float | None = None  # repair utility cost; G9 bounds it <= 0.02
    matched_null_repair_ci: tuple[float, float] | None = None  # B4 within-g control CI
    positivity_excluded: float | None = None  # overall excluded-pair fraction (A7); G9 needs < 0.5
    # A7 is a PER-CLASS condition (§4.4: "excluded fraction < 0.5 per class"): an
    # overall scalar can hide a catastrophic class behind a good average (finding 12).
    # Maps g3_class -> excluded-pair fraction; the v5 lock requires EACH class < 0.5.
    positivity_excluded_by_class: dict[str, float] | None = None
    # baseline panel B0..B5 R_hats (REDESIGN_v5 §4.3); G9-NOV margins read from these.
    baseline_r_hats: tuple[float, ...] | None = None
    repair_policy_hash: str | None = None  # repair_ops.policy_hash (rho recipe)
    transport_map_hash: str | None = None  # repair_ops.transport_map_hash (anchor T)
    class_partition_hash: str | None = None  # frozen G3 taxonomy hash
    selection_event: str | None = None  # binning_selection.SelectionEvent (serialised)
    k_bin: int | None = None  # SI binning multiplicity (§6.2)
    k_op: int | None = None  # SI operator-grid multiplicity (§4.7 OS-2)
    xi_axis_x: float | None = None  # Axis X' confounding strength at the lock fixture
    axis_x_regime: str | None = None  # "detectable" | "blind"

    server_authorized: bool = False

    @property
    def gated_u(self) -> float:
        """The ``U_hat`` the necessity gate G1 is evaluated on.

        REDESIGN_v4 §2.13/§4: G1 is evaluated on ``u_deflated`` when present (the
        OOD-deflated estimator), otherwise on the raw ``u_hat`` (v3 behaviour).
        """
        return self.u_deflated if self.u_deflated is not None else self.u_hat

    @property
    def gated_ci_low(self) -> float:
        """The **lower CI bound** of the (deflated) estimator G1 is gated on.

        REDESIGN_v3 §4.5(E3) / §4 G1: the necessity gate clears on the **CI lower
        bound** (Holm-gated), not the point estimate.

        When the OOD-deflated estimator is present (§2.12/§2.13), the gated lower
        bound is the **propagated** deflated lower CI computed by
        :func:`ood_deflation`: the calibrated-footprint slope uncertainty is added
        in quadrature, so the deflated interval is **wider** than the raw interval
        merely shifted by ``u_hat - u_deflated``. G1 must read that propagated lower
        bound — if a record only shifted the raw ``ci_low`` it could clear necessity
        when the (correctly widened) deflated bound would not (review fix 2).

        Resolution order:

        * if the propagated deflated lower CI ``u_deflated_ci_low`` is recorded, use
          it directly (the only faithful G1 bound);
        * else if only ``u_deflated`` is recorded (no propagated CI), fall back to
          shifting ``ci_low`` by ``u_hat - u_deflated`` (the un-widened lower bound,
          which is an **upper** bound on the true deflated lower CI, hence the
          conservative-shift fallback only when no propagated CI exists);
        * else (no deflation) this is the raw ``ci_low``.
        """
        if self.u_deflated_ci_low is not None:
            return self.u_deflated_ci_low
        if self.u_deflated is None:
            return self.ci_low
        return self.ci_low - (self.u_hat - self.u_deflated)


def ood_slope_ci_excludes_manufacture(
    ood_slope_ci: tuple[float, float],
    displaced_mass_range: float,
    *,
    bound: float = OOD_INERT_MANUFACTURE_BOUND,
) -> bool:
    """G8 bounded-OOD-slope kill-gate (REDESIGN_v4 §4.1 G8 conjunct 2 / §2.12).

    The calibrated ``displaced_mass``-stratified OOD slope ``s_OOD`` has a CI
    ``(lo, hi)``. Over the observed ``displaced_mass`` range ``D``, a slope ``s``
    manufactures an inert necessity signal ``U_inert = s * D``. G8 requires the
    **whole CI** to exclude a slope large enough to manufacture
    ``U_inert >= bound`` (``0.05``): i.e. the worst-case manufactured signal
    ``max(|lo|, |hi|) * D`` must stay strictly **below** ``bound``.

    Returns ``True`` when the slope CI is bounded and the kill-gate is satisfied
    (no slope in the CI can manufacture an inert signal at the necessity bar),
    ``False`` otherwise (CI admits an artifact-manufacturing slope, or is
    non-finite / ill-ordered / the range is non-positive).
    """
    lo, hi = ood_slope_ci
    if math.isinf(lo) or math.isinf(hi) or lo != lo or hi != hi:
        return False
    if lo > hi:
        return False
    if displaced_mass_range <= 0.0:
        return False
    worst_case_manufactured = max(abs(lo), abs(hi)) * displaced_mass_range
    return worst_case_manufactured < bound


def validate_ciu_record(
    record: CIURecord,
    *,
    mde_min_n: int | None = None,
    require_v4: bool = False,
    require_v5: bool = False,
) -> list[str]:
    """Return contract violations for a ``CIURecord`` (REDESIGN_v3 §5; v4 §5).

    v3 checks: empty hashes, ``server_authorized`` True, budget mismatch (the
    edit budget must be a positive int consistent with the control provenance),
    ``s_seed < 20``, and ``n_examples`` below the MDE-required count
    (``mde_min_n`` when supplied).

    **v4 G7/G8 fields are HARD-required, not "when populated"** (aligns with
    ``redesign_v4_ar_lead.yaml:92`` ``leakage_slope_ci: required`` and the
    ``g8_requires:`` block). The enforcement is two-tier:

    * ``require_v4=True`` (a v4 lock) requires the **full** preregistered control
      set to be present and to PASS, not just the leakage/OOD-slope fields (review
      fix 3). Concretely it requires: the G7 leakage fields (``beta_hi``,
      ``proximity_bin_width``); the G8 OOD fields (``ood_slope_ci``,
      ``displaced_mass_range``); the SHAM-MASK null CI (``sham_u_ci``) and the
      no-op (A2) null CI (``noop_u_ci``), each of which must bracket zero; the
      clean-oracle and graded-family recovery outcomes (``oracle_pass`` /
      ``graded_curve_pass``), which must be truthy; and the mean in-bin pool size
      (``m_pool_mean``), which must meet the ``POOL_MIN`` (8) floor (§4.6 B3). A
      record therefore cannot omit any control to bypass a gate.
    * Even with ``require_v4=False`` (back-compatible v3 records still validate),
      **opting into the v4 path is all-or-nothing**: if *any* v4 OOD/leakage
      field is populated, the *complete* conjunct it belongs to is required and
      checked. A partially-populated record (e.g. ``ood_slope`` set but
      ``ood_slope_ci`` / ``displaced_mass_range`` missing) is a violation, so a
      gate can never be silently skipped by leaving its companion field ``None``.

    The substantive gate checks:

    * **G7** — require ``beta_hi * proximity_bin_width < LOCALIZATION_MARGIN``.
    * **G8** — require ``ood_slope_ci`` to be a finite, well-ordered ``(lo, hi)``
      interval **and** that this CI *excludes* a slope large enough to manufacture
      ``U_inert >= OOD_INERT_MANUFACTURE_BOUND`` (``0.05``) over the observed
      ``displaced_mass_range`` (REDESIGN_v4 §4.1 G8 conjunct 2 / §2.12).
    * if ``u_deflated`` is set, require it to lie at or below the raw ``u_hat``
      (deflation removes operator artifact, it cannot inflate the estimate).
    """
    errors: list[str] = []

    if record.server_authorized:
        errors.append("server_authorized must remain false (no run authorized)")

    for name in ("null_pool_hash", "noop_run_hash", "evaluator_hash", "ref_hash"):
        value = getattr(record, name)
        if not value or not str(value).strip():
            errors.append(f"{name} must be a non-empty provenance hash")

    if record.edit_budget <= 0:
        errors.append("edit_budget (k) must be a positive integer")

    if record.s_seed < MIN_PAPER_SEEDS:
        errors.append(f"s_seed must be >= {MIN_PAPER_SEEDS} (paper seed floor)")

    if record.n_examples <= 0:
        errors.append("n_examples must be positive")
    if mde_min_n is not None and record.n_examples < mde_min_n:
        errors.append(f"n_examples below MDE-required count ({record.n_examples} < {mde_min_n})")

    if not 0.0 <= record.evaluator_kappa <= 1.0:
        errors.append("evaluator_kappa must be in [0, 1]")
    # Cross-check evaluator_kappa against the attenuation requirement (Opus minor):
    # the (2*kappa - 1) attenuation band the record's U_target depends on is only
    # defined for kappa > 0.5 (agreement above chance). A kappa at/below chance
    # gives no usable attenuation band — the same condition nuisance.u_target /
    # apply_kappa_fallback block a lock on — so flag it here too.
    elif record.evaluator_kappa <= 0.5:
        errors.append(
            f"evaluator_kappa ({record.evaluator_kappa}) must be > 0.5 (agreement "
            "above chance) for the (2*kappa-1) attenuation band to be defined"
        )

    if record.ci_low > record.ci_high:
        errors.append("ci_low must not exceed ci_high")

    if not record.ci_low <= record.u_hat <= record.ci_high:
        errors.append("u_hat must lie within [ci_low, ci_high]")

    # Per-example provenance length agreement (when populated). The three paired
    # provenance vectors (tau_i, bar tau_i(Pi), matched-control identity) must align
    # whenever any is present, so a partially-filled record cannot mismatch silently.
    provenance_vectors = {
        "tau_per_example": record.tau_per_example,
        "pi_mean_per_example": record.pi_mean_per_example,
        "matched_control_provenance": record.matched_control_provenance,
    }
    populated_lengths = {n: len(v) for n, v in provenance_vectors.items() if v}
    if len(set(populated_lengths.values())) > 1:
        errors.append(
            "per-example provenance vectors must align in length "
            f"(tau/pi_mean/matched_control): {populated_lengths}"
        )

    # --- v4 field-presence policy (HARD-required vs all-or-nothing) -----
    # G7 leakage conjunct fields and G8 OOD conjunct fields.
    g7_fields = {"beta_hi": record.beta_hi, "proximity_bin_width": record.proximity_bin_width}
    g8_fields = {
        "ood_slope_ci": record.ood_slope_ci,
        "displaced_mass_range": record.displaced_mass_range,
    }

    # Preregistered control OUTCOMES (review fix 3): SHAM/no-op null CIs and the
    # graded-oracle pass flags must be persisted and enforced under a v4 lock, not
    # only the leakage/OOD-slope fields.
    control_fields = {
        "sham_u_ci": record.sham_u_ci,
        "noop_u_ci": record.noop_u_ci,
        "oracle_pass": record.oracle_pass,
        "graded_curve_pass": record.graded_curve_pass,
        "m_pool_mean": record.m_pool_mean,
    }

    if require_v4:
        for name, value in {**g7_fields, **g8_fields, **control_fields}.items():
            if value is None:
                errors.append(
                    f"v4 lock requires {name} to be populated "
                    "(G7/G8 + preregistered controls hard-required)"
                )
        # Paper-tier / lockable provenance is FAIL-CLOSED (review fix): the per-example
        # tau_i / bar tau_i(Pi) vectors and the matched-control identity vector must be
        # NON-EMPTY. A lockable CIURecord that omits provenance cannot be certified — the
        # default-empty tuples are a v3 back-compat convenience, never acceptable at lock.
        for name in (
            "tau_per_example",
            "pi_mean_per_example",
            "matched_control_provenance",
        ):
            if not getattr(record, name):
                errors.append(
                    f"v4 lock requires non-empty {name} "
                    "(paper-tier provenance is fail-closed; a lockable record may not "
                    "omit per-example provenance)"
                )
        # Enforce that the persisted controls actually PASS (not merely present).
        if record.sham_u_ci is not None:
            lo, hi = record.sham_u_ci
            if not (lo <= 0.0 <= hi):
                errors.append(
                    "v4 lock: SHAM-MASK null CI must bracket zero "
                    f"({record.sham_u_ci}); a CI off zero is an operator artifact"
                )
        if record.noop_u_ci is not None:
            lo, hi = record.noop_u_ci
            if not (lo <= 0.0 <= hi):
                errors.append(
                    "v4 lock: no-op (A2) null CI must bracket zero "
                    f"({record.noop_u_ci}); a CI off zero is an operator artifact"
                )
        if record.oracle_pass is False:
            errors.append("v4 lock: clean-oracle recovery (oracle_pass) must pass")
        if record.graded_curve_pass is False:
            errors.append("v4 lock: graded-oracle curve-match (graded_curve_pass) must pass")
        if record.m_pool_mean is not None and record.m_pool_mean < PROXIMITY_POOL_MIN:
            errors.append(
                f"v4 lock: mean in-bin pool size m_pool_mean ({record.m_pool_mean}) "
                f"must meet the proximity-pool floor {PROXIMITY_POOL_MIN} (§4.6 B3); "
                "coarsen the proximity bin before lock"
            )
    else:
        # All-or-nothing: opting into either conjunct requires the whole conjunct
        # so a gate cannot be skipped by leaving a companion field None.
        if any(v is not None for v in g7_fields.values()) and any(
            v is None for v in g7_fields.values()
        ):
            missing = [n for n, v in g7_fields.items() if v is None]
            errors.append(
                "G7 leakage conjunct partially populated; missing "
                f"{missing} (v4 fields are all-or-nothing, no silent gate skip)"
            )
        # The OOD conjunct is anchored by any of ood_slope / ood_slope_ci /
        # displaced_mass_range / u_deflated being present.
        ood_opted_in = (
            record.ood_slope is not None
            or record.ood_slope_ci is not None
            or record.displaced_mass_range is not None
        )
        if ood_opted_in and any(v is None for v in g8_fields.values()):
            missing = [n for n, v in g8_fields.items() if v is None]
            errors.append(
                "G8 OOD conjunct partially populated; missing "
                f"{missing} (v4 fields are all-or-nothing, no silent gate skip)"
            )

    # --- v4 G7 (upper-CI leakage bound) ---------------------------------
    if record.beta_hi is not None and record.proximity_bin_width is not None:
        bound = record.beta_hi * record.proximity_bin_width
        if not bound < LOCALIZATION_MARGIN:
            errors.append(
                "G7 fail: upper-CI leakage bound beta_hi*Delta_pos "
                f"({bound:.4f}) must be < localization margin {LOCALIZATION_MARGIN}"
            )

    # --- v4 G8 (bounded calibrated OOD slope CI + kill-gate) ------------
    if record.ood_slope_ci is not None:
        lo, hi = record.ood_slope_ci
        if lo > hi:
            errors.append("ood_slope_ci must be a (lo, hi) interval with lo <= hi")
        if math.isinf(lo) or math.isinf(hi) or lo != lo or hi != hi:
            errors.append("ood_slope_ci must be finite (G8 bounded-slope requirement)")
        elif lo <= hi:
            # Bounded-OOD-slope kill-gate: the CI must EXCLUDE a slope able to
            # manufacture U_inert >= 0.05 over the observed displaced_mass range.
            if record.displaced_mass_range is not None:
                if not ood_slope_ci_excludes_manufacture(
                    record.ood_slope_ci, record.displaced_mass_range
                ):
                    worst = max(abs(lo), abs(hi)) * record.displaced_mass_range
                    errors.append(
                        "G8 fail: OOD slope CI admits a slope manufacturing "
                        f"U_inert={worst:.4f} >= {OOD_INERT_MANUFACTURE_BOUND} over "
                        f"displaced_mass range {record.displaced_mass_range}; "
                        "kill-gate requires the CI to exclude such a slope"
                    )

    # --- v4 deflation sanity --------------------------------------------
    if record.u_deflated is not None and record.u_deflated > record.u_hat + 1e-12:
        errors.append("u_deflated must not exceed u_hat (deflation removes artifact)")

    # The propagated deflated lower CI (review fix 2) must be well-ordered: it is a
    # *lower* bound so it cannot exceed the deflated point estimate, and the slope
    # uncertainty only WIDENS the interval, so it cannot exceed the raw ci_low.
    if record.u_deflated_ci_low is not None:
        if record.u_deflated is None:
            errors.append(
                "u_deflated_ci_low recorded without u_deflated (propagated deflated "
                "CI requires the deflated point estimate)"
            )
        elif record.u_deflated_ci_low > record.u_deflated + 1e-12:
            errors.append("u_deflated_ci_low must not exceed u_deflated (it is a lower bound)")
        if record.u_deflated_ci_low > record.ci_low + 1e-12:
            errors.append(
                "u_deflated_ci_low must not exceed the raw ci_low (slope-propagation "
                "widens, never narrows, the deflated interval)"
            )

    # SHAM / no-op null CIs (when populated) must be well-ordered (lo <= hi),
    # regardless of the v4 lock, so a malformed control CI is never silently stored.
    for name in ("sham_u_ci", "noop_u_ci"):
        ci = getattr(record, name)
        if ci is not None:
            lo, hi = ci
            if lo > hi:
                errors.append(f"{name} must be a (lo, hi) interval with lo <= hi")

    # --- v5 field policy (HARD-required at a v5 lock; well-ordering always) -----
    # The R_hat CI and the matched-null repair CI must be well-ordered whenever
    # present (a malformed CI is never silently stored), independent of require_v5.
    for name in ("r_hat_ci", "matched_null_repair_ci"):
        ci = getattr(record, name)
        if ci is not None:
            lo, hi = ci
            if lo > hi:
                errors.append(f"{name} must be a (lo, hi) interval with lo <= hi")
    if record.r_hat is not None and record.r_hat_ci is not None:
        lo, hi = record.r_hat_ci
        if not lo <= record.r_hat <= hi:
            errors.append("r_hat must lie within r_hat_ci")
    if record.positivity_excluded is not None and not 0.0 <= record.positivity_excluded <= 1.0:
        errors.append("positivity_excluded must be a fraction in [0, 1]")
    if record.positivity_excluded_by_class is not None:
        for cls, frac in record.positivity_excluded_by_class.items():
            if not 0.0 <= frac <= 1.0:
                errors.append(
                    f"positivity_excluded_by_class[{cls!r}] ({frac}) must be a fraction in [0, 1]"
                )
    if record.r_hat_perm_p is not None and not 0.0 <= record.r_hat_perm_p <= 1.0:
        errors.append("r_hat_perm_p must be a p-value in [0, 1]")
    if record.d_util_repair is not None and record.d_util_repair < 0.0:
        errors.append("d_util_repair must be >= 0 (a utility cost)")
    for name in ("k_bin", "k_op"):
        v = getattr(record, name)
        if v is not None and v < 1:
            errors.append(f"{name} must be >= 1 (a selection-grid cardinality)")
    if record.axis_x_regime is not None and record.axis_x_regime not in {"detectable", "blind"}:
        errors.append("axis_x_regime must be 'detectable' or 'blind'")

    if require_v5:
        errors.extend(_v5_lock_errors(record))

    return errors


# The v5-lock hard-required fields (REDESIGN_v5 §9: ciu.py CIURecord optional v5
# fields become required at a v5 lock, mirroring require_v4). Provenance hashes and
# the SI multiplicity factors and the Axis X' stress fixture must all be present.
_V5_REQUIRED_FIELDS: tuple[str, ...] = (
    "r_hat",
    "r_hat_ci",
    "r_hat_perm_p",
    "d_util_repair",
    "matched_null_repair_ci",
    "positivity_excluded",
    "positivity_excluded_by_class",
    "baseline_r_hats",
    "repair_policy_hash",
    "transport_map_hash",
    "class_partition_hash",
    "selection_event",
    "k_bin",
    "k_op",
    "xi_axis_x",
    "axis_x_regime",
)


def _v5_lock_errors(record: CIURecord) -> list[str]:
    """Hard-require the G9/G9-NOV + SI + transport/positivity fields at a v5 lock.

    Mirrors the require_v4 fail-closed enforcement (REDESIGN_v5 §9): a v5-lockable
    record may not omit any v5 field, and the persisted G9 controls must actually
    PASS (not merely be present):

    * ``positivity_excluded < POSITIVITY_EXCLUDED_MAX`` (A7; else "insufficient
      positivity", not a null);
    * ``d_util_repair <= REPAIR_UTILITY_BOUND`` (else "trades factuality for
      utility", reframe as abstention);
    * the provenance hashes must be non-empty;
    * the v5 lock additionally REQUIRES the v4 lock to hold (G9 sits on top of the
      preserved G1-G8 machinery), so the v4 hard-field set is enforced too.
    """
    errors: list[str] = []
    # v5 sits on top of v4: the preserved G1-G8 controls remain hard-required.
    errors.extend(validate_ciu_record(record, require_v4=True))

    for name in _V5_REQUIRED_FIELDS:
        if getattr(record, name) is None:
            errors.append(
                f"v5 lock requires {name} to be populated "
                "(G9/G9-NOV + SI + transport/positivity hard-required)"
            )

    for name in ("repair_policy_hash", "transport_map_hash", "class_partition_hash"):
        value = getattr(record, name)
        if value is not None and (not value or not str(value).strip()):
            errors.append(f"v5 lock: {name} must be a non-empty provenance hash")

    if record.positivity_excluded is not None and record.positivity_excluded >= POSITIVITY_EXCLUDED_MAX:
        errors.append(
            f"v5 lock: positivity_excluded ({record.positivity_excluded}) must be "
            f"< {POSITIVITY_EXCLUDED_MAX} (A7); else route to 'insufficient positivity', "
            "not a null"
        )
    # A7 is PER CLASS: EACH class must clear the bound, so a catastrophic class cannot
    # hide behind a good overall average (finding 12).
    if record.positivity_excluded_by_class is not None:
        for cls, frac in record.positivity_excluded_by_class.items():
            if frac >= POSITIVITY_EXCLUDED_MAX:
                errors.append(
                    f"v5 lock: positivity_excluded_by_class[{cls!r}] ({frac}) must be "
                    f"< {POSITIVITY_EXCLUDED_MAX} (A7 is per-class); class {cls!r} is "
                    "under-powered -> route to 'insufficient positivity', not a null"
                )
    if record.d_util_repair is not None and record.d_util_repair > REPAIR_UTILITY_BOUND:
        errors.append(
            f"v5 lock: d_util_repair ({record.d_util_repair}) must be "
            f"<= {REPAIR_UTILITY_BOUND} (G9 bounded utility cost); else reframe as abstention"
        )
    return errors


def ciu_gate(
    record: CIURecord,
    *,
    scalar_gate: bool | None = None,
    best_detector_u: float | None = None,
    sham_u: float | None = None,
    sham_u_ci: tuple[float, float] | None = None,
    oracle_pass: bool | None = None,
    graded_curve_pass: bool | None = None,
    u_deflated: float | None = None,
    u_deflated_ci_low: float | None = None,
    beta_hi: float | None = None,
    s_ood_ci: tuple[float, float] | None = None,
    displaced_mass_range: float | None = None,
    require_controls: bool = True,
) -> CIUVerdict:
    """The revised G5' novelty-margin gate (REDESIGN_v3 §5; REDESIGN_v4 §5).

    Decision logic (returns one of ``useful_candidate``, ``diagnostic``,
    ``not_novel``; **never** ``invalidated``):

    1. **Scalar gate G1+G2 (necessity + utility).** G1 necessity is cleared on the
       **CI lower bound** of the (OOD-deflated) estimator — *not* the point
       estimate (REDESIGN_v3 §4.5(E3): "CI lower bound > 0.05"; REDESIGN_v4 §2.13:
       evaluated on ``u_deflated``). The lower bound must reach ``NECESSITY_MARGIN``
       (``0.05``) — **inclusive** at the margin (``>=``, the pre-registered
       direction; see :func:`metrics.passes_intervention_gate`) — and the utility
       drop must clear G2. Utility is still checked via
       the wrapped :func:`metrics.passes_intervention_gate`. ``scalar_gate`` may be
       supplied pre-computed; otherwise it is derived from the record's
       ``gated_ci_low`` and ``d_util``. If the scalar gate fails, the verdict is
       ``diagnostic``.
    2. **Novelty gate G5'.** The proposed selector must beat the best **adapted**
       detector on ``U_hat`` by ``NOVELTY_MARGIN`` (``0.03``). If a detector
       matches or beats the selector within that margin, the verdict is
       ``not_novel`` — a *novelty downgrade*, **not** an identification failure.
       A detector having positive ``U_hat`` is expected (its covariates track
       ``tau``) and is **never** treated as invalidating.
    3. **Controls (SHAM null G8, OOD slope G8, leakage G7, graded-oracle G7).**
       These four controls are **hard-required** for a ``useful_candidate`` verdict
       when ``require_controls`` is ``True`` (the default): a *missing* required
       control withholds the useful verdict (-> ``diagnostic``), not just a failing
       one (review fix 1). A failing or absent control downgrades a would-be
       ``useful_candidate`` to ``diagnostic`` (necessity reading withheld), again
       never ``invalidated``. Set ``require_controls=False`` only to isolate a
       single control path in a focused unit test:

       * **SHAM null** — tested on the SHAM CI: the SHAM-MASK ``U_hat`` CI must
         **bracket zero** (``lo <= 0 <= hi``). A SHAM CI lying entirely above 0
         (e.g. an operator artifact in ``(0, 0.03)``) fails — it cannot pass
         silently just because the *point* is below ``0.03`` (review fix d).
       * **G8 bounded OOD slope** — ``s_ood_ci`` must exclude a slope able to
         manufacture ``U_inert >= 0.05`` over ``displaced_mass_range``.
       * **G7 leakage upper-CI bound** — ``beta_hi * proximity_bin_width`` must be
         ``< LOCALIZATION_MARGIN``.
       * **G7 graded oracle** — ``oracle_pass`` / ``graded_curve_pass``.

    Parameters
    ----------
    scalar_gate:
        Optional pre-computed G1+G2 pass; if ``None`` it is derived from the
        record's ``gated_ci_low`` (lower CI of the deflated estimator) and
        ``d_util``.
    best_detector_u:
        The best adapted detector's ``U_hat``; falls back to
        ``record.best_detector_u``.
    sham_u:
        SHAM-MASK ``U_hat`` point estimate (legacy fallback, only used when
        ``sham_u_ci`` is not given).
    sham_u_ci:
        SHAM-MASK ``U_hat`` 95% CI ``(lo, hi)`` (G8 null). Must bracket zero
        (``lo <= 0 <= hi``); otherwise the necessity reading is an operator
        artifact -> ``diagnostic``.
    oracle_pass:
        Clean-oracle recovery pass (G7 conjunct 1a).
    graded_curve_pass:
        Graded-family curve-match pass on all three axes (G7 conjunct 1b,
        REDESIGN_v4 §2.10G). ``False`` withholds the useful verdict.
    u_deflated:
        OOD-deflated ``U_hat`` (§2.12); overrides ``record.u_deflated`` when
        supplied, so G1 is evaluated on the deflated lower CI at runtime. This is
        the un-widened shift fallback — prefer ``u_deflated_ci_low``.
    u_deflated_ci_low:
        The **propagated** OOD-deflated lower CI from :func:`ood_deflation` (slope
        uncertainty added in quadrature, §2.12). When supplied this is the lower
        bound G1 is gated on directly (review fix 2): it overrides the un-widened
        ``u_deflated`` shift so a record cannot clear necessity on an artificially
        narrow interval.
    require_controls:
        When ``True`` (default), the four v4 controls (SHAM/no-op null,
        bounded-OOD-slope CI, leakage-slope CI, graded-oracle pass) are
        hard-required for ``useful_candidate``; a missing control yields
        ``diagnostic``. ``False`` disables the presence requirement (focused tests
        only).
    beta_hi:
        Upper-CI answer-adjacent leakage slope (§2.11); G7 bound
        ``beta_hi * proximity_bin_width`` is applied at runtime. Falls back to
        ``record.beta_hi``.
    s_ood_ci:
        Calibrated OOD slope CI ``(lo, hi)`` (§2.12); the G8 kill-gate is applied
        at runtime. Falls back to ``record.ood_slope_ci``.
    displaced_mass_range:
        Observed ``displaced_mass`` span for the G8 kill-gate; falls back to
        ``record.displaced_mass_range``.
    """
    # --- resolve the matched-null arm and the scalar (G1+G2) gate -------
    if record.pi_mean_per_example:
        random_delta = sum(record.pi_mean_per_example) / len(record.pi_mean_per_example)
    else:
        # Fall back to (gated_u - matched_null) == u_hat decomposition: with no
        # per-example pool means recorded, treat the null arm as 0 so the gate
        # reduces to "gated_ci_low >= min_margin".
        random_delta = 0.0

    # Allow a runtime-supplied deflated estimate to set the gated lower bound. A
    # runtime PROPAGATED deflated lower CI (``u_deflated_ci_low``) is preferred: it
    # is the slope-uncertainty-widened bound from ``ood_deflation`` and is the only
    # faithful G1 bound (review fix 2). A bare ``u_deflated`` shift is the un-widened
    # fallback used only when no propagated CI is supplied.
    if u_deflated_ci_low is not None:
        gated_ci_low = u_deflated_ci_low
    elif u_deflated is not None:
        gated_ci_low = record.ci_low - (record.u_hat - u_deflated)
    else:
        gated_ci_low = record.gated_ci_low

    if scalar_gate is None:
        # G1 necessity is cleared on the CI LOWER BOUND of the (deflated)
        # estimator, not the point estimate. Utility (G2) is checked by the
        # wrapped gate; we pass the lower bound as the "targeted_delta" so the
        # necessity arm of passes_intervention_gate tests the lower CI.
        scalar_gate = passes_intervention_gate(
            gated_ci_low + random_delta,  # lower-CI necessity arm
            random_delta,
            record.d_util,
            min_margin=NECESSITY_MARGIN,
        )

    if not scalar_gate:
        return "diagnostic"

    # --- novelty gate G5' ------------------------------------------------
    detector_u = best_detector_u if best_detector_u is not None else record.best_detector_u
    if detector_u is not None:
        if (record.u_hat - detector_u) < NOVELTY_MARGIN:
            # A detector matches/beats us within the margin: novelty downgrade.
            # This is NEVER an identification failure; positive detector U is
            # expected (its covariates track tau).
            return "not_novel"

    # --- controls (G8 SHAM null, G8 OOD slope, G7 leakage/oracle) -------
    # The four v4 causal-identification controls are HARD-required for a
    # ``useful_candidate`` verdict (review fix 1): a missing required control is an
    # absent-evidence failure, NOT a silent pass. Previously the gate only
    # downgraded when a control arg was explicitly supplied and failing, so a public
    # call that simply omitted the controls could reach ``useful_candidate`` — the
    # exact bypass the design forbids (REDESIGN_v4 §4.1: both G7 conjuncts and both
    # G8 conjuncts are *required* before causal wording). With
    # ``require_controls=True`` (default) the absence of any required control
    # withholds the useful verdict (-> ``diagnostic``, never ``invalidated``).
    #
    # Resolve each control from its runtime arg first, then the record. The four
    # required controls are: SHAM/no-op null, bounded-OOD-slope CI (+ range),
    # answer-adjacent leakage-slope bound, and the graded-oracle pass.
    eff_s_ood_ci = s_ood_ci if s_ood_ci is not None else record.ood_slope_ci
    eff_dm_range = (
        displaced_mass_range
        if displaced_mass_range is not None
        else record.displaced_mass_range
    )
    eff_beta_hi = beta_hi if beta_hi is not None else record.beta_hi
    # SHAM null + graded-oracle outcomes fall back to the record's persisted control
    # fields (review fix 3: the record persists these so a run-packet record carries
    # its own control evidence and the gate need not be re-handed them).
    eff_sham_u_ci = sham_u_ci if sham_u_ci is not None else record.sham_u_ci
    eff_oracle_pass = oracle_pass if oracle_pass is not None else record.oracle_pass
    eff_graded_pass = (
        graded_curve_pass if graded_curve_pass is not None else record.graded_curve_pass
    )

    # SHAM/no-op null is "present" if either a CI or a legacy point estimate exists.
    sham_present = eff_sham_u_ci is not None or sham_u is not None
    # G8 OOD slope control is present only with BOTH the slope CI and the range.
    ood_present = eff_s_ood_ci is not None and eff_dm_range is not None
    # G7 leakage control is present only with BOTH the upper-CI slope and the bin
    # width (Delta_pos) the bound is computed over.
    leakage_present = eff_beta_hi is not None and record.proximity_bin_width is not None
    # Graded-oracle pass is present only when an explicit pass flag is supplied. The
    # clean-oracle conjunct (``oracle_pass``) and the graded-family conjunct
    # (``graded_curve_pass``) are both part of the G7 graded-oracle control; require
    # an explicit, truthy graded-curve pass (the binding conjunct, §2.10G).
    oracle_present = eff_graded_pass is not None

    if require_controls:
        missing: list[str] = []
        if not sham_present:
            missing.append("SHAM/no-op null")
        if not ood_present:
            missing.append("bounded-OOD-slope CI")
        if not leakage_present:
            missing.append("answer-adjacent leakage-slope CI")
        if not oracle_present:
            missing.append("graded-oracle pass")
        if missing:
            # A required control is absent: cannot certify useful_candidate.
            return "diagnostic"

    # SHAM null tested on the CI: it must bracket zero. An artifact CI sitting
    # entirely above 0 (even in (0, 0.03)) fails -> diagnostic (review fix d).
    if eff_sham_u_ci is not None:
        sham_lo, sham_hi = eff_sham_u_ci
        if not (sham_lo <= 0.0 <= sham_hi):
            return "diagnostic"
    elif sham_u is not None:
        # Legacy point fallback: any materially positive SHAM signal is an
        # operator artifact. Use a strict >0 (with float tolerance) so an
        # artifact in (0, 0.03) cannot pass silently.
        if sham_u > 1e-9:
            return "diagnostic"

    # No-op (A2) null persisted on the record: same bracket-zero requirement.
    if record.noop_u_ci is not None:
        noop_lo, noop_hi = record.noop_u_ci
        if not (noop_lo <= 0.0 <= noop_hi):
            return "diagnostic"

    # G8 bounded OOD slope kill-gate (runtime arg or record).
    if ood_present:
        if not ood_slope_ci_excludes_manufacture(eff_s_ood_ci, eff_dm_range):
            return "diagnostic"

    # G7 leakage upper-CI bound (runtime arg or record).
    if leakage_present:
        if not (eff_beta_hi * record.proximity_bin_width) < LOCALIZATION_MARGIN:
            return "diagnostic"

    if eff_oracle_pass is False:
        return "diagnostic"
    if eff_graded_pass is False:
        return "diagnostic"

    return "useful_candidate"


def calibrate_m_r(m_r0: float, kappa_lo_repair: float) -> float:
    """Calibrate the **repair** margin ``m_R`` on ``V_sel`` (Eq. m-R, REDESIGN_v5 §4.8).

    ``m_R := m_R0 / (2 * kappa_lo_repair - 1)`` — the design-repair margin ``m_R0``
    (pinned on ``V_sel``) attenuation-adjusted by the **cross-example repair
    evaluator agreement** ``kappa^{repair}`` (re-estimated on target labels), NOT
    inherited from the v4 necessity margin ``0.05`` (finding 11). Whether ``m_R0``
    numerically equals ``0.05`` is a ``V_sel`` calibration decision the locked config
    records — it is never silently defaulted. Mirrors :func:`nuisance.u_target` but
    with the repair kappa, so the two attenuation paths share one formula.

    Raises if ``2 * kappa_lo_repair - 1 <= 0`` (repair-evaluator agreement at/below
    chance gives no usable attenuation band) or ``m_r0 <= 0``.
    """
    if m_r0 <= 0.0:
        raise ValueError("m_r0 (design repair margin) must be positive")
    denom = 2.0 * kappa_lo_repair - 1.0
    if denom <= 0.0:
        raise ValueError(
            "2*kappa_lo_repair - 1 must be > 0 (repair-evaluator agreement above chance)"
        )
    return m_r0 / denom


def g9_repair_gate(
    r_hat_estimate: float,
    r_hat_ci: tuple[float, float],
    perm_p: float,
    d_util_repair: float,
    positivity_excluded_frac: float,
    class_leakage_ok: bool,
    matched_null_repair_ci: tuple[float, float],
    *,
    alpha_1_prime: float,
    m_r: float,
    positivity_excluded_by_class: Mapping[str, float] | None = None,
) -> CIUVerdict:
    """The headline G9 repair-transfer certification gate (REDESIGN_v5 §5.1).

    Reuses the fail-closed, **never-``invalidated``** ``ciu_gate`` template: it
    returns one of ``useful_candidate`` / ``diagnostic`` / ``not_novel`` and never
    an identification-failure verdict.

    ``m_r`` is a **required** argument (finding 11): the REDESIGN v5 repair margin
    must be **calibrated on ``V_sel`` with the repair kappa** (:func:`calibrate_m_r`,
    Eq. m-R), never silently inherited from the v4 necessity margin ``0.05``. The
    gate therefore refuses to supply a default — the caller passes the realised,
    recorded ``m_R``.

    Certification (``useful_candidate``) requires ALL of:

    1. **Holm/SI-corrected two-way-cluster-bootstrap CI lower bound > ``m_r``** —
       the **confirmatory** certification test (G9-FIX). The two-way cluster bootstrap
       assumes neither a sharp null nor source-block sign-symmetry, so it carries the
       significance burden; the Holm fold is applied by the caller via
       ``alpha_1_prime``; the gate verifies the CI is the level-correct one and checks
       the lower bound against ``m_r``;
    2. **class-block sign-flip diagnostic ``p < alpha_1_prime``** — a **required
       corroborating diagnostic** (G9-FIX), NOT the confirmatory test. It probes
       source-block sign-symmetry / A6 within-class exchangeability
       (:func:`repair_transfer.class_block_permutation`, which sets
       ``confirmatory=False`` and ``exact_under_registered_sharp_null=False`` because
       matched-null centring + A6 give mean-zero + exchangeability, NOT the sign
       symmetry an exact permutation would need). Certification requires the diagnostic
       to *agree* with the confirmatory CI at the level-correct
       :func:`selective_inference.holm_alpha`; the confirmatory burden rests on (1);
    3. **bounded utility cost** ``d_util_repair <= REPAIR_UTILITY_BOUND`` (0.02) —
       else the repair trades factuality for utility -> ``diagnostic`` (reframe as
       abstention, §5.1);
    4. **positivity, per class** — A7 requires the excluded-pair fraction
       ``< POSITIVITY_EXCLUDED_MAX`` (0.5) **per class** (§4.4 A7: "G9 requires
       excluded fraction < 0.5 *per class*"). A single overall scalar can hide a
       catastrophic class behind a good average (finding 12): when
       ``positivity_excluded_by_class`` is supplied the gate checks the **worst**
       class (``max`` over classes) against the bound; the overall
       ``positivity_excluded_frac`` is still checked as a backstop. Either failing
       routes to "insufficient positivity" -> ``diagnostic``;
    5. **no class/self leakage** (``class_leakage_ok``) — source != target, an
       example is never repaired by a policy localized on itself.

    The B4 within-``g`` matched-null repair control routes a *generic* repair: if
    ``matched_null_repair_ci`` brackets 0 the repair is *localized* (the B4 baseline
    transfers no signal); if it sits **off zero above 0** the repair is **generic,
    not localized** -> ``diagnostic`` (never ``invalidated``; REDESIGN_v5 §5.1 "B4
    also passes" route).

    All thresholds are inclusive/strict exactly as the design pins them: the CI
    lower bound is **strictly >** ``m_r`` (the G9 "> m_R" wording), the permutation
    p is **strictly <** ``alpha_1_prime``, and the utility/positivity bounds are
    inclusive/strict per §5.1.
    """
    lo, hi = r_hat_ci
    if lo > hi:
        return "diagnostic"  # malformed CI -> withhold
    # (3) bounded utility cost
    if d_util_repair > REPAIR_UTILITY_BOUND:
        return "diagnostic"
    # (4) positivity -- PER CLASS (A7), so a catastrophic class cannot hide behind a
    # good overall average (finding 12). Check the worst class AND the overall scalar.
    if positivity_excluded_frac >= POSITIVITY_EXCLUDED_MAX:
        return "diagnostic"
    if positivity_excluded_by_class is not None and positivity_excluded_by_class:
        worst = max(positivity_excluded_by_class.values())
        if worst >= POSITIVITY_EXCLUDED_MAX:
            return "diagnostic"
    # (5) no class/self leakage
    if not class_leakage_ok:
        return "diagnostic"
    # (1) CONFIRMATORY: Holm/SI-corrected two-way-cluster-bootstrap CI lower bound > m_r
    # (the cluster bootstrap carries the significance burden; G9-FIX).
    if not lo > m_r:
        return "diagnostic"
    # (2) REQUIRED DIAGNOSTIC corroboration: class-block sign-flip p < alpha_1'. This is
    # the source-block sign-symmetry / A6 diagnostic, NOT the confirmatory test; it must
    # agree with (1) for certification but does not itself certify (G9-FIX).
    if not perm_p < alpha_1_prime:
        return "diagnostic"
    # B4 within-g control: if the matched-null repair CI sits off-zero ABOVE 0, the
    # repair is generic (any matched span transfers) -> diagnostic, not localized.
    b4_lo, b4_hi = matched_null_repair_ci
    if b4_lo > 0.0:
        return "diagnostic"
    return "useful_candidate"


def g9_novelty_gate(
    r_hat_proposed: float,
    r_hat_baselines: Mapping[str, float],
    *,
    margin_ci_low: float | None = None,
) -> CIUVerdict:
    """G9-NOV baseline-conditional novelty gate (REDESIGN_v5 §5.2; MF-3/MF-9).

    Tests ``R_hat(PROPOSED) - max_b{ R_hat(B1), R_hat(B2), R_hat(B3) } > 0`` on the
    Holm/SI-corrected **simultaneous** lower CI (refinement 8): the CIU localization
    must out-transfer the detector-selected localizations (B1 TraceDet, B2 entropy,
    B3 probe) through the **identical** ``repair_ops`` pipeline.

    * ``r_hat_baselines`` should contain the detector baselines B1/B2/B3 (B0 no-op
      floor, B4 matched-null, B5 oracle are NOT part of the novelty max — they are
      the floor / within-``g`` control / ceiling, §4.3); the caller passes the
      contrast set.
    * When ``margin_ci_low`` (the :func:`repair_transfer.g9_nov_margin_simultaneous`
      lower CI) is supplied, the gate certifies novelty (``useful_candidate``) iff
      that lower CI clears 0 — the only valid test for a *data-selected maximum*
      (Holm alone gives no CI for a max, refinement 8). Without it, the gate uses
      the point margin and returns ``not_novel`` if PROPOSED does not strictly beat
      the max baseline (a conservative point check).

    **Never** returns ``invalidated``: a baseline matching PROPOSED is a *novelty
    downgrade* (the certification protocol is the contribution, §3.4), not an
    identification failure.
    """
    if not r_hat_baselines:
        # no detector baselines to beat: novelty is trivially the protocol's; treat
        # as a downgrade (not_novel) rather than asserting causal-beats-correlational.
        return "not_novel"
    max_baseline = max(r_hat_baselines.values())
    if margin_ci_low is not None:
        return "useful_candidate" if margin_ci_low > 0.0 else "not_novel"
    return "useful_candidate" if r_hat_proposed > max_baseline else "not_novel"


def baseline_readiness(registry: Mapping[str, object]) -> list[str]:
    """Flag baselines not yet run-ready at preflight (REDESIGN_v3 §5).

    Scans a parsed ``baseline_registry.yaml`` mapping (or its ``baselines:``
    sub-mapping) and returns a list of human-readable readiness violations: any
    baseline whose ``implementation_commit`` / ``license`` still carries a
    ``PENDING_MARKERS`` value blocks the run. This is the preflight that keeps a
    run blocked while baselines are pending (REDESIGN_v3 §5: "blocks the run
    while 7/9 baselines are pending").

    The function does **not** authorise anything; it only reports readiness.
    """
    baselines = registry.get("baselines", registry) if isinstance(registry, Mapping) else {}
    if not isinstance(baselines, Mapping):
        return ["baseline registry has no 'baselines' mapping"]

    violations: list[str] = []
    for name, spec in baselines.items():
        if not isinstance(spec, Mapping):
            violations.append(f"baseline {name!r} spec is not a mapping")
            continue
        for key in ("implementation_commit", "license"):
            value = spec.get(key)
            if isinstance(value, str) and value in PENDING_MARKERS:
                violations.append(f"baseline {name!r} {key} is pending: {value!r}")
    return violations


# ---------------------------------------------------------------------------
# v4 additive helpers (pure, validation-split-only; no model, no GPU, no run).
# ---------------------------------------------------------------------------


def leakage_slope_regression(
    delta_inert: Sequence[float],
    proximity: Sequence[float],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Fit the answer-adjacent leakage regression (REDESIGN_v4 §2.11).

    Model: ``delta_inert_i = beta * proximity_i + gamma + eps_i`` by ordinary
    least squares, returning ``(beta_hat, beta_lo, beta_hi)`` with a
    **paired-bootstrap** 95% CI. G7 gates on the **upper CI** ``beta_hi``: the
    A4* bias bound is ``B_UCI = beta_hi * Delta_pos`` and must clear ``0.03``.

    Validation-split-only by contract: the caller passes frozen validation
    arrays; there is no test-split access path. Pure Python (no numpy).

    Raises
    ------
    ValueError
        If the inputs are misaligned, too short, or ``proximity`` has zero
        variance (slope undefined).
    """
    x = [float(v) for v in proximity]
    y = [float(v) for v in delta_inert]
    n = len(x)
    if n != len(y):
        raise ValueError("delta_inert and proximity must align in length")
    if n < 3:
        raise ValueError("need at least 3 points to fit the leakage slope")

    beta_hat = _ols_slope(x, y)

    rng = random.Random(seed)
    slopes: list[float] = []
    idx = list(range(n))
    for _ in range(n_bootstrap):
        sample = [rng.choice(idx) for _ in range(n)]
        xs = [x[j] for j in sample]
        ys = [y[j] for j in sample]
        try:
            slopes.append(_ols_slope(xs, ys))
        except ValueError:
            continue  # degenerate resample (zero x-variance); skip
    if not slopes:
        raise ValueError("all bootstrap resamples were degenerate (zero x-variance)")
    slopes.sort()
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    beta_lo = _quantile(slopes, lo_q)
    beta_hi = _quantile(slopes, hi_q)
    return beta_hat, beta_lo, beta_hi


def ood_deflation(
    u_hat: float,
    displaced_mass: float,
    slope: float,
    intercept: float,
    *,
    u_hat_ci: tuple[float, float] | None = None,
    slope_ci: tuple[float, float] | None = None,
) -> tuple[float, tuple[float, float] | None]:
    """Deflate a real-span ``U_hat`` by the calibrated operator footprint (§2.12).

    The calibrated OOD artifact at the span's own ``displaced_mass = d*`` is
    ``art(d*) = slope * d* + intercept`` (the inert-span calibration line). The
    deflated estimate is ``U_deflated = U_hat - art(d*)``; G1 is evaluated on it.
    When both the estimator CI and the slope CI are supplied, the footprint's
    uncertainty is propagated into the deflated CI **in quadrature** (REDESIGN_v4
    §2.12).

    Returns
    -------
    (u_deflated, u_deflated_ci)
        ``u_deflated_ci`` is ``None`` when CIs are not supplied.
    """
    art = slope * displaced_mass + intercept
    u_deflated = u_hat - art

    if u_hat_ci is None:
        return u_deflated, None

    lo, hi = u_hat_ci
    half = (hi - lo) / 2.0
    if slope_ci is not None:
        s_lo, s_hi = slope_ci
        # Footprint half-width from the slope CI at this displaced_mass.
        art_half = abs((s_hi - s_lo) / 2.0) * abs(displaced_mass)
        half = math.sqrt(half * half + art_half * art_half)
    return u_deflated, (u_deflated - half, u_deflated + half)


def _ols_slope(x: Sequence[float], y: Sequence[float]) -> float:
    """OLS slope of ``y`` on ``x``; raises if ``x`` has zero variance."""
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    if sxx <= 0.0:
        raise ValueError("proximity has zero variance; slope undefined")
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    return sxy / sxx


# ``_quantile`` is imported from ``._numerics`` (de-duplicated; Opus minor).
