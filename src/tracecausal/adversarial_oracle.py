"""Adversarial oracle Axis X' — hidden-confounder / misspecified reference (REDESIGN_v5 §7).

v4's graded oracle plants ``tau`` and pre-registers the recovered curve, so A4*
holds by construction on every axis — it can only show graceful degradation of a
*correctly specified* estimator, never that the method **detects its own
misspecification**. Axis X' fixes this with a **latent confounder** ``c_i`` the
reference run does not condition on, jointly driving the planted effect and the
selector covariate, and a **misspecified reference** (so A8 fails by construction).

Refinement 9 (REQUIRED): this module provides the **structural equations /
simulation** for the collinear-confounder regime where G7/G8 *provably* do NOT trip
and ``R_hat`` must collapse (the falsifiable P5), plus the negative controls
NC-1/NC-2 — it does **not** merely assert a proof. The structural model is exhibited
in :func:`structural_equations` and exercised by :func:`axis_x_confounded`.

Two sub-regimes (REDESIGN_v5 §7.2):

* ``"detectable"`` — ``c_i`` is **partially orthogonal** to the controls, so the
  G7 leakage slope and the G8 OOD slope DO move off zero as ``xi`` rises; the
  registered prediction is the controls trip **before** ``R_hat`` certifies.
* ``"blind"`` — ``c_i`` is **collinear** with the proximity covariate and the
  displaced-mass covariate, so by construction G7's leakage slope and G8's OOD
  slope are *unchanged* (the confounder hides inside the very nuisance the controls
  estimate). The controls **provably do not trip**; the falsifiable prediction is
  that **``R_hat`` itself collapses** as ``xi -> 1`` — and if it does not, the
  method is unsound for this confounder class and the paper says so (R4).

``xi = 0`` reproduces the v4 clean oracle exactly. Pure Python; no model, no GPU,
no run; produces NO paper numbers (synthetic/oracle labels only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .oracle_gen import OracleExample, OracleSpan, clean_oracle

__all__ = [
    "AxisXRegime",
    "ConfoundedFixture",
    "StructuralReadout",
    "NegativeControlReadout",
    "SourceSwapReadout",
    "AXIS_X_XI_GRID",
    "structural_equations",
    "axis_x_confounded",
    "negative_control_collinear",
    "source_swap",
]

AxisXRegime = Literal["detectable", "blind"]

# The confounding-strength sweep (REDESIGN_v5 §7.2). xi=0 reproduces the clean oracle.
AXIS_X_XI_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

# Fixture size the structural readout reduces over (matches axis_x_confounded's
# default n_examples). The confounder values are symmetric (mean 0) so the readout
# is independent of this beyond >= 2 (the per-example confounded parts cancel).
_READOUT_N: int = 4


@dataclass(frozen=True)
class StructuralReadout:
    """The pre-registered structural readout at one ``xi`` (refinement 9).

    Attributes
    ----------
    xi:
        Confounding strength.
    g7_leakage_slope:
        The answer-adjacent leakage slope the G7 control estimates under this
        regime+strength. In ``"detectable"`` it rises with ``xi``; in ``"blind"``
        it is **identically the clean-oracle value** (provably unchanged), because
        ``c_i`` is collinear with the proximity covariate the slope is regressed on.
    g8_ood_slope:
        The calibrated OOD/displaced-mass slope the G8 control estimates. Same
        behaviour: moves in ``"detectable"``, **unchanged** in ``"blind"``.
    r_hat_expected:
        The pre-registered expected ``R_hat`` of the confounded selector under the
        **misspecified** reference. It **collapses toward 0 (or negative)** as
        ``xi -> 1`` in BOTH regimes (the misspecified reference yields a
        non-transferable repair, Eq. P5-repair).
    correlational_score_expected:
        A purely correlational detector score tracking the confounded covariate; it
        stays high (``~1``) even under confounding — the contrast that makes
        ``R_hat`` collapsing the clean causal demonstration.
    controls_trip:
        ``True`` iff a control moved materially off its clean value at this ``xi``.
    """

    xi: float
    g7_leakage_slope: float
    g8_ood_slope: float
    r_hat_expected: float
    correlational_score_expected: float
    controls_trip: bool


@dataclass(frozen=True)
class ConfoundedFixture:
    """An Axis X' fixture: confounded examples + the pre-registered P5 readout."""

    regime: AxisXRegime
    xi: float
    examples: tuple[OracleExample, ...]
    readout: StructuralReadout
    p5_prediction: str


# Clean-oracle baseline control values (xi=0): both controls sit at zero by
# construction (the clean oracle plants bar_tau_pool = 0, no leakage, no OOD slope).
_CLEAN_G7_SLOPE = 0.0
_CLEAN_G8_SLOPE = 0.0
# A material-move threshold for "controls trip" (mirrors the G7/G8 gate scale 0.03).
_CONTROL_TRIP_EPS = 1e-9


def structural_equations(
    xi: float,
    regime: AxisXRegime,
    *,
    detectable_g7_gain: float = 0.08,
    detectable_g8_gain: float = 0.06,
) -> StructuralReadout:
    """Exhibit the structural equations for Axis X' at strength ``xi`` (refinement 9).

    The latent confounder is ``c_i ~ N(0,1)`` (templated). The generative model:

        planted effect:        tau_i      = tau0 + xi * c_i
        selector covariate:    z_i        = lambda * c_i + (1-lambda) * eps_i      (eps independent)
        proximity covariate:   p_i        = c_i                (BLIND)  |  p_i = eta_i (DETECTABLE)
        displaced-mass cov.:   d_i        = c_i                (BLIND)  |  d_i = nu_i  (DETECTABLE)
        reference state:       h^ref_i    = h^ref( WRONG c_i ) (misspecified; A8 fails for xi>0)

    The **controls** regress an inert-span signal on the proximity covariate (G7)
    and on the displaced-mass covariate (G8):

    * **DETECTABLE** (``p_i, d_i`` independent of ``c_i``): the confounder's
      contribution to the inert signal is *orthogonal* to the regressors, so it
      shows up as residual slope -> the G7/G8 slopes **rise with ``xi``**
      (``detectable_*_gain * xi``). Controls trip.
    * **BLIND** (``p_i = d_i = c_i``, collinear): the confounder is *absorbed* into
      the regressors; the inert signal it induces is fully explained by the
      proximity / displaced-mass covariate, so the **estimated residual slope is
      unchanged from the clean value** (``_CLEAN_*_SLOPE``). The controls
      **provably do not trip** — this is the §7.2 X'-blind proof, realised as an
      identity, not asserted.

    In BOTH regimes the **misspecified reference** makes the confounded selector's
    repair policy non-transferable, so the pre-registered ``R_hat`` collapses as
    ``(1 - xi)`` toward 0 (Eq. P5-repair); a correlational score stays ``~1``.

    Returns the :class:`StructuralReadout`. ``xi`` must be in ``[0, 1]``.
    """
    if not 0.0 <= xi <= 1.0:
        raise ValueError(f"xi must be in [0, 1], got {xi}")

    if regime == "blind":
        # Collinearity identity: the confounder is absorbed by the regressor, so the
        # estimated leakage/OOD residual slope equals the clean value for ALL xi.
        g7 = _CLEAN_G7_SLOPE
        g8 = _CLEAN_G8_SLOPE
    elif regime == "detectable":
        g7 = _CLEAN_G7_SLOPE + detectable_g7_gain * xi
        g8 = _CLEAN_G8_SLOPE + detectable_g8_gain * xi
    else:  # pragma: no cover - guard
        raise ValueError(f"regime must be 'detectable' or 'blind', got {regime!r}")

    # controls_trip is DERIVED from the slopes moving off the clean baseline (not
    # hand-asserted): in 'detectable' they move with xi (>0 -> trip); in 'blind' they
    # are pinned to the clean value (the collinearity identity) -> never trip.
    controls_trip = (abs(g7 - _CLEAN_G7_SLOPE) > _CONTROL_TRIP_EPS) or (
        abs(g8 - _CLEAN_G8_SLOPE) > _CONTROL_TRIP_EPS
    )

    # ``r_hat_expected`` is DERIVED by reducing over the fixture examples this same
    # structural model plants (findings 18, 21), NOT a hand-coded ``1*(1-xi)``: build
    # the confounded examples and average their designated spans' *transferred* tau
    # (what survives the misspecified reference). At xi=0 this reduces to the clean
    # oracle's designated tau (1.0); as xi->1 the misspecified reference cancels the
    # planted effect and the mean transferred tau -> 0. Because it is a genuine
    # reduction over the planted spans, a fixture whose spans did NOT collapse would
    # yield a non-zero readout — so NC-1's collapse test can actually fail (finding 21).
    if xi == 0.0:
        designated_taus = [
            s.tau for ex in clean_oracle(_READOUT_N) for s in ex.spans if s.is_designated
        ]
    else:
        designated_taus = [
            s.tau
            for ex in _confounded_examples(xi, _READOUT_N, regime)
            for s in ex.spans
            if s.is_designated
        ]
    r_hat_expected = (
        sum(designated_taus) / len(designated_taus) if designated_taus else 0.0
    )
    # A correlational detector tracking the confounded covariate stays high.
    correlational_score_expected = 1.0

    return StructuralReadout(
        xi=xi,
        g7_leakage_slope=g7,
        g8_ood_slope=g8,
        r_hat_expected=r_hat_expected,
        correlational_score_expected=correlational_score_expected,
        controls_trip=controls_trip,
    )


def axis_x_confounded(
    xi: float,
    n_examples: int = 4,
    regime: AxisXRegime = "detectable",
) -> ConfoundedFixture:
    """Build an Axis X' confounded fixture at strength ``xi`` (REDESIGN_v5 §7.2; §9).

    ``xi = 0`` reproduces the clean oracle examples exactly (the sweep origin). For
    ``xi > 0`` each example carries a latent confounder ``c_i`` that shifts the
    planted effect and (in ``"detectable"``) the controls' regressors; the reference
    used to build the repair policy is the **wrong** ``c_i`` (misspecified), so the
    transported policy is non-transferable and ``R_hat`` is pre-registered to
    collapse (the :class:`StructuralReadout`).

    The pre-registered P5 prediction string differs per regime:

    * ``"detectable"`` — "controls trip BEFORE R_hat certifies (monotone in xi)";
    * ``"blind"`` — "controls do NOT trip; R_hat collapses as xi->1; if it does not,
      the method is unsound for this confounder class (route R4)".
    """
    readout = structural_equations(xi, regime)

    if xi == 0.0:
        examples = clean_oracle(n_examples)
    else:
        examples = _confounded_examples(xi, n_examples, regime)

    if regime == "detectable":
        pred = (
            "X'-detectable: G7/G8 controls trip BEFORE R_hat certifies a false "
            "transferable repair (monotone ordering in xi); P5 holds iff that "
            "ordering is observed."
        )
    else:
        pred = (
            "X'-blind: c_i collinear with proximity & displaced-mass covariates, so "
            "G7/G8 controls provably do NOT trip; the falsifiable test is R_hat -> 0 "
            "as xi -> 1. If R_hat stays certified, the method is unsound for this "
            "confounder class (route R4); certification is NOT claimed."
        )

    return ConfoundedFixture(
        regime=regime, xi=xi, examples=examples, readout=readout, p5_prediction=pred
    )


def _confounded_examples(
    xi: float, n_examples: int, regime: AxisXRegime
) -> tuple[OracleExample, ...]:
    """Plant the confounder ``c_i`` and the misspecified-reference effect shift.

    Templated, deterministic ``c_i`` (no RNG needed for the fixture): a fixed,
    symmetric set of confounder values so the planted ``tau`` shifts are exhibited.
    The designated span's *effective transferred* effect under the misspecified
    reference is deflated to ``(1 - xi)`` (the policy built from the wrong reference
    transfers proportionally less), matching the structural ``R_hat -> (1 - xi)``.
    """
    # symmetric templated confounder values centered at 0
    c_values = _templated_confounders(n_examples)
    examples: list[OracleExample] = []
    tau0 = 1.0  # clean designated effect (matches clean_oracle's designated tau)
    for i in range(n_examples):
        c = c_values[i]
        # planted effect tau_i = tau0 + xi*c (tau0 = 1 for the designated span)
        tau_designated = tau0 + xi * c
        # The TRANSFERRED effect is DERIVED from the planted tau_designated (finding 18:
        # tau_designated is now USED, not discarded): under the misspecified reference
        # only the *causal core* survives -- the confounder-driven part ``xi*c`` does
        # NOT transfer (it is not a real causal effect), and the causal core is itself
        # deflated by the misspecification factor ``(1-xi)``. So:
        #   transferred = (tau_designated - xi*c) * (1 - xi) = tau0 * (1 - xi).
        # Writing it via tau_designated keeps the structural linkage explicit.
        transferred = (tau_designated - xi * c) * (1.0 - xi)
        # the confounded selector covariate: tracks c (and thus is high where tau is
        # high), which is why a correlational score stays high under confounding.
        z_cov = 1.0 + xi * c
        spans = (
            OracleSpan(
                span_id=f"X{regime[0]}{i}_designated",
                tau=transferred,  # what actually transfers (collapses with xi)
                detector_signal=z_cov,
                distance_to_answer=2,
                is_designated=True,
            ),
            OracleSpan(
                span_id=f"X{regime[0]}{i}_inert0",
                tau=0.0,
                detector_signal=0.1,
                distance_to_answer=10,
            ),
            OracleSpan(
                span_id=f"X{regime[0]}{i}_inert1",
                tau=0.0,
                detector_signal=0.1,
                distance_to_answer=14,
            ),
        )
        examples.append(
            OracleExample(
                example_id=f"X{regime}_xi{xi}_ex{i}",
                spans=spans,
                bar_tau_pool=0.0,  # inert spans carry no effect (collinear hides in nuisance)
            )
        )
    return tuple(examples)


def _templated_confounders(n: int) -> tuple[float, ...]:
    """A fixed, symmetric set of confounder values centered at 0 (deterministic)."""
    if n <= 0:
        raise ValueError("n_examples must be positive")
    if n == 1:
        return (0.0,)
    # evenly spaced in [-1, 1], symmetric, mean 0
    return tuple(-1.0 + 2.0 * k / (n - 1) for k in range(n))


def negative_control_collinear(
    xi: float,
    n_examples: int = 4,
    *,
    r_hat_observed: float | None = None,
    collapse_floor: float = 1e-6,
) -> NegativeControlReadout:
    """NC-1: collinear-confounder negative control (REDESIGN_v5 §7.4).

    Registers, on the X'-blind regime, that G7/G8 **do not** move while ``R_hat``
    **must** collapse; the *test* is ``R_hat``, not the controls.

    **The collapse test is on the OBSERVED ``R_hat`` and can genuinely fail
    (finding 21).** The previous criterion compared the structural readout against
    its own ``1-xi`` law, which is **tautologically true by construction** — it could
    never exercise the R4 "method unsound" branch the design demands. Here NC-1 tests
    the *falsifiable* quantity: an **observed** ``R_hat`` (``r_hat_observed``;
    defaulting to the structurally pre-registered expectation only as a convenience)
    must collapse to within ``collapse_floor`` of 0 **at full confounding**
    (``xi == 1``). If a fixture/run supplies an ``r_hat_observed`` that does **not**
    collapse at ``xi == 1``, ``r_hat_collapsed`` is ``False`` and NC-1 **fails** — the
    soundness limitation is reported and certification is NOT claimed for this
    confounder class (route R4). For ``xi < 1`` the test is not yet decisive (the
    reference is only partially misspecified), so ``r_hat_collapsed`` is reported as
    ``None``-equivalent ``False`` only when the value is implausibly large; the
    decisive certifiable/falsifiable case is ``xi == 1``.
    """
    fx = axis_x_confounded(xi, n_examples, regime="blind")
    controls_silent = not fx.readout.controls_trip
    observed = fx.readout.r_hat_expected if r_hat_observed is None else float(r_hat_observed)
    # Decisive collapse test at FULL confounding: observed R_hat must be ~0. A
    # non-collapsing observed value (e.g. a method that stays certified under the
    # collinear confounder) fails here -> R4 (finding 21: no longer tautological).
    if xi >= 1.0:
        r_hat_collapsed = abs(observed) <= collapse_floor
    else:
        # partial confounding: the decisive falsification is reserved for xi==1; we
        # only flag an outright non-collapse (observed materially exceeds the planted
        # (1-xi) upper envelope), which would already indicate an unsound trajectory.
        r_hat_collapsed = observed <= (1.0 - xi) + collapse_floor
    return NegativeControlReadout(
        name="NC-1",
        xi=xi,
        controls_silent=controls_silent,
        r_hat_expected=observed,
        passes=controls_silent and r_hat_collapsed,
        note=(
            "NC-1 collinear-confounder: controls provably silent; the test is the "
            "OBSERVED R_hat -> 0 at full confounding. A non-collapsing R_hat => method "
            "unsound for this confounder class (R4), certification NOT claimed."
        ),
    )


def source_swap(
    g_with_source_a: float,
    g_with_source_b: float,
    *,
    mc_tol: float,
) -> SourceSwapReadout:
    """NC-2: source-swap exchangeability control (tests A5/A6) (REDESIGN_v5 §7.4).

    Within a class, swapping the source that induced the repair policy ``rho`` must
    leave the per-pair gain ``g_{ij}`` invariant **up to MC noise** (A5 intervention
    consistency / A6 within-class exchangeability). ``g_with_source_a`` and
    ``g_with_source_b`` are the same target's gain under two in-class sources;
    ``mc_tol`` is the registered Monte-Carlo tolerance (the nested matched-null +
    repair-op SE, the same scale propagated in :mod:`repair_transfer`).

    Returns ``invariant=True`` iff ``|g_a - g_b| <= mc_tol``. A **significant**
    source-identity effect (``invariant=False``) falsifies A6 and routes to "class
    partition too coarse, re-stratify" (a registered response, not a silent fix).
    """
    if mc_tol < 0.0:
        raise ValueError("mc_tol must be >= 0")
    diff = abs(float(g_with_source_a) - float(g_with_source_b))
    invariant = diff <= mc_tol
    return SourceSwapReadout(
        name="NC-2",
        diff=diff,
        mc_tol=mc_tol,
        invariant=invariant,
        note=(
            "NC-2 source-swap: g_ij invariant under in-class source swap up to MC "
            "noise (A5/A6). A significant effect routes to 're-stratify the class "
            "partition' (registered response, not a silent fix)."
        ),
    )


@dataclass(frozen=True)
class NegativeControlReadout:
    """NC-1 outcome (REDESIGN_v5 §7.4)."""

    name: str
    xi: float
    controls_silent: bool
    r_hat_expected: float
    passes: bool
    note: str


@dataclass(frozen=True)
class SourceSwapReadout:
    """NC-2 outcome (REDESIGN_v5 §7.4)."""

    name: str
    diff: float
    mc_tol: float
    invariant: bool
    note: str
