"""Graded synthetic oracle family for A4* positive identification (REDESIGN_v4 §2.10G).

REDESIGN_v4 replaces v3's single *clean* oracle with a **graded family** that must
demonstrate the CIU estimator ``U_hat`` **degrades correctly** as identification
erodes — not merely passes the easy case. Three controlled one-parameter stress
axes sweep around the clean oracle, each with a **pre-registered expected
``U_hat`` curve** and a **planted crossing point** at the localization margin
``0.03`` (gate floor; see ``ciu.py::ciu_gate`` / G7).

* **Axis P (partial-leakage).** An inert span at proximity gap ``Delta_pos`` is
  given a planted effect ``tau_inert = beta * Delta_pos``, sweeping
  ``beta*Delta_pos in {0, 0.01, 0.02, 0.03, 0.04, 0.06}`` (up to and *past* the
  ``0.03`` margin). The matched-null arm is biased upward, so a *correct*
  selector's ``U_hat`` is deflated by exactly ``tau_inert``:
  ``U_hat(oracle) = 1 - tau_inert`` to first order, crossing the ``0.03`` gate
  floor at the planted point.
* **Axis M (multi-cause).** The planted effect is split across ``m_c in
  {1,2,3,4}`` co-equal spans, each carrying ``tau = 1/m_c``. A single-span
  selector recovers only ``U_hat ~= 1/m_c`` (honest under-recovery); a top-``m_c``
  selector recovers ``-> 1``.
* **Axis D (distractor-span).** A causally **inert** span (``tau = 0``) carries a
  **high detector signal** (sweep ``d in {0.5,1,2,4}x`` the true span's). A naive
  detector selects it but its ``U_hat -> 0``; the causal selector ignores it; the
  G5' selector-minus-detector difference stays ``>= 0.03``.

These are **Phase-2 unit fixtures** on a frozen templated generator: they produce
**no paper numbers**, use synthetic/oracle labels, and **never** trigger a server
run. Everything here is pure Python (no model, no GPU).

The fixtures expose the *ground truth* ``tau`` structure and the analytically
predicted ``U_hat`` so a test (``tests/test_ciu_nulldata.py``) can assert the
estimator tracks the planted curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "OracleSpan",
    "OracleExample",
    "GradedOracleFixture",
    "AXIS_P_LEAKAGE_GRID",
    "AXIS_M_CAUSE_GRID",
    "AXIS_D_DISTRACTOR_GRID",
    "LOCALIZATION_MARGIN",
    "clean_oracle",
    "graded_oracle_family",
    "expected_u_hat",
]

OracleAxis = Literal[
    "clean", "partial_leakage", "multi_cause", "distractor", "adversarial"
]

# Pre-registered sweep grids (REDESIGN_v4 §2.10G).
AXIS_P_LEAKAGE_GRID: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.04, 0.06)
AXIS_M_CAUSE_GRID: tuple[int, ...] = (1, 2, 3, 4)
AXIS_D_DISTRACTOR_GRID: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)

# The localization / gate-floor margin (REDESIGN_v4 §2.10G, §4 G7; equals the
# novelty margin 0.03). Axis P's planted crossing is at this value.
LOCALIZATION_MARGIN: float = 0.03


@dataclass(frozen=True)
class OracleSpan:
    """A single span in a templated oracle example, with planted ground truth.

    Attributes
    ----------
    span_id:
        Stable identifier within the example.
    tau:
        The **true** per-example causal effect ``tau_i(S)`` of editing this span
        (planted by construction; this is what makes the oracle an oracle).
    detector_signal:
        A planted detector covariate (entropy / claim-bearing-ness / RACE-style
        consistency). High ``detector_signal`` with ``tau = 0`` is the Axis D
        distractor; correlated with ``tau`` otherwise.
    distance_to_answer:
        Proximity gap used by Axis P's leakage term and by proximity
        stratification.
    is_designated:
        ``True`` for the designated true-cause span(s) ``S_designated``.
    """

    span_id: str
    tau: float
    detector_signal: float
    distance_to_answer: int
    is_designated: bool = False


@dataclass(frozen=True)
class OracleExample:
    """One templated oracle example ``x_i`` with planted spans.

    ``bar_tau_pool`` is the **pool mean** ``bar tau_i(Pi)`` over the inert
    (non-designated, matched) spans — i.e. what an unbiased matched-null arm
    estimates (Prop. 2.5a). In the clean oracle it is ``0`` exactly; on Axis P it
    picks up the planted leakage term, which is the whole point of the sweep.
    """

    example_id: str
    spans: tuple[OracleSpan, ...]
    bar_tau_pool: float

    def designated(self) -> tuple[OracleSpan, ...]:
        return tuple(s for s in self.spans if s.is_designated)

    def inert(self) -> tuple[OracleSpan, ...]:
        return tuple(s for s in self.spans if not s.is_designated)


@dataclass(frozen=True)
class GradedOracleFixture:
    """A pre-registered axis fixture: examples + the expected ``U_hat`` curve.

    Attributes
    ----------
    axis:
        Which stress axis this fixture sweeps.
    param:
        The sweep parameter value (``beta*Delta_pos`` for P, ``m_c`` for M, the
        distractor strength ``d`` for D).
    examples:
        The templated examples at this sweep point.
    expected_u_hat_selector:
        Pre-registered expected ``U_hat`` of the **correct** causal selector.
    expected_u_hat_detector:
        Pre-registered expected ``U_hat`` of the naive **detector** selector
        (Axis D: the detector-on-distractor; ``None`` when not applicable).
    crosses_gate_floor:
        **Disambiguated (REDESIGN_v4 §2.10G).** ``True`` iff the **leakage / bias**
        term crossing the ``0.03`` gate floor — i.e. the answer-adjacent leakage
        ``tau_inert >= 0.03`` (Axis P planted crossing of the localization margin
        ``B = beta*Delta_pos``, the quantity G7 bounds). This is the *bias*
        crossing, **not** the selector-curve crossing. For axes with no leakage
        term (M, D, clean) the bias is ``0`` and this is ``False``.
    crosses_selector_floor:
        ``True`` iff the **selector curve** ``U_hat`` itself has fallen to / below
        the ``0.03`` margin at this sweep point (Axis P: ``U_hat = 1 - tau_inert``;
        Axis M: ``U_hat = 1/m_c``). This answers the distinct question "does the
        recovered effect itself drop below the margin", which is *not* the same as
        the leakage-bias crossing on Axis P.
    """

    axis: OracleAxis
    param: float
    examples: tuple[OracleExample, ...]
    expected_u_hat_selector: float
    expected_u_hat_detector: float | None = None
    crosses_gate_floor: bool = False  # leakage/bias crossing (tau_inert >= 0.03)
    crosses_selector_floor: bool = False  # selector curve U_hat <= 0.03
    notes: str = ""


def expected_u_hat(tau_selected: float, bar_tau_pool: float) -> float:
    """The CIU contrast ``U = tau_i(S*) - bar tau_i(Pi)`` (REDESIGN_v3 §2.5/§2.6).

    Closed-form expectation of the estimator on a planted fixture; the test
    asserts the realised estimator tracks this.
    """
    return tau_selected - bar_tau_pool


def clean_oracle(n_examples: int = 4) -> tuple[OracleExample, ...]:
    """The v3 clean oracle, retained as the sweep origin (REDESIGN_v4 §2.10G).

    Each example has one designated span with ``tau = 1`` and several inert spans
    with ``tau = 0``; ``bar tau_i(Pi) = 0`` exactly, so A4* holds by construction
    and ``U_hat(oracle_selector) -> 1``, ``U_hat(random/inert) -> 0``.
    """
    examples: list[OracleExample] = []
    for i in range(n_examples):
        spans = (
            OracleSpan(
                span_id=f"ex{i}_designated",
                tau=1.0,
                detector_signal=1.0,
                distance_to_answer=2,
                is_designated=True,
            ),
            OracleSpan(span_id=f"ex{i}_inert0", tau=0.0, detector_signal=0.1, distance_to_answer=10),
            OracleSpan(span_id=f"ex{i}_inert1", tau=0.0, detector_signal=0.1, distance_to_answer=14),
            OracleSpan(span_id=f"ex{i}_inert2", tau=0.0, detector_signal=0.1, distance_to_answer=18),
        )
        examples.append(OracleExample(example_id=f"clean_ex{i}", spans=spans, bar_tau_pool=0.0))
    return tuple(examples)


def _axis_partial_leakage(tau_inert: float, n_examples: int) -> GradedOracleFixture:
    """Axis P: inert spans carry ``tau_inert = beta*Delta_pos`` (REDESIGN_v4 §2.10G).

    The matched-null pool mean becomes ``bar tau_i(Pi) = tau_inert`` (the inert
    spans now carry the planted leakage), so a correct selector's expected
    ``U_hat = 1 - tau_inert`` to first order. The pre-registered crossing of the
    ``0.03`` gate floor occurs when ``1 - tau_inert`` would drop the residual
    leakage past the margin — operationally, when ``tau_inert >= 0.03``.
    """
    examples: list[OracleExample] = []
    for i in range(n_examples):
        spans = (
            OracleSpan(
                span_id=f"P{i}_designated",
                tau=1.0,
                detector_signal=1.0,
                distance_to_answer=2,
                is_designated=True,
            ),
            OracleSpan(
                span_id=f"P{i}_leak0",
                tau=tau_inert,
                detector_signal=0.1,
                distance_to_answer=10,
            ),
            OracleSpan(
                span_id=f"P{i}_leak1",
                tau=tau_inert,
                detector_signal=0.1,
                distance_to_answer=12,
            ),
        )
        examples.append(
            OracleExample(example_id=f"P_ex{i}", spans=spans, bar_tau_pool=tau_inert)
        )
    selector_u = expected_u_hat(1.0, tau_inert)  # = 1 - tau_inert
    return GradedOracleFixture(
        axis="partial_leakage",
        param=tau_inert,
        examples=tuple(examples),
        expected_u_hat_selector=selector_u,
        expected_u_hat_detector=None,
        # Leakage/bias crossing: the A4* bias term tau_inert crosses 0.03 (what G7
        # bounds). This is the gate-floor crossing the design pre-registers on P.
        crosses_gate_floor=tau_inert >= LOCALIZATION_MARGIN,
        # Selector-curve crossing: U_hat = 1 - tau_inert dipping to <= 0.03 (a
        # distinct question; only happens at extreme leakage tau_inert >= 0.97).
        crosses_selector_floor=selector_u <= LOCALIZATION_MARGIN,
        notes=(
            "U_hat(oracle) = 1 - tau_inert; matched-null biased up by leakage. "
            "crosses_gate_floor = leakage/bias crossing (tau_inert >= 0.03)."
        ),
    )


def _axis_multi_cause(m_c: int, n_examples: int) -> GradedOracleFixture:
    """Axis M: effect split across ``m_c`` co-equal spans (REDESIGN_v4 §2.10G).

    Each of the ``m_c`` designated spans carries ``tau = 1/m_c``. A single-span
    selector recovers ``U_hat ~= 1/m_c`` (honest under-recovery); a top-``m_c``
    selector recovers ``-> 1``. ``bar tau_i(Pi) = 0`` (the remaining inert spans
    carry no effect).
    """
    if m_c < 1:
        raise ValueError(f"m_c must be >= 1, got {m_c}")
    per = 1.0 / m_c
    examples: list[OracleExample] = []
    for i in range(n_examples):
        cause_spans = tuple(
            OracleSpan(
                span_id=f"M{i}_cause{j}",
                tau=per,
                detector_signal=1.0,
                distance_to_answer=2 + j,
                is_designated=True,
            )
            for j in range(m_c)
        )
        inert_spans = (
            OracleSpan(span_id=f"M{i}_inert0", tau=0.0, detector_signal=0.1, distance_to_answer=20),
            OracleSpan(span_id=f"M{i}_inert1", tau=0.0, detector_signal=0.1, distance_to_answer=24),
        )
        examples.append(
            OracleExample(
                example_id=f"M_ex{i}", spans=cause_spans + inert_spans, bar_tau_pool=0.0
            )
        )
    return GradedOracleFixture(
        axis="multi_cause",
        param=float(m_c),
        examples=tuple(examples),
        expected_u_hat_selector=per,  # single-span selector under-recovers to 1/m_c
        expected_u_hat_detector=None,
        # No leakage term on Axis M: bar_tau_pool = 0, so the bias/gate-floor
        # crossing is False. The earlier code mislabelled the *selector-curve*
        # crossing (1/m_c <= 0.03) as crosses_gate_floor; that is now its own field.
        crosses_gate_floor=False,
        crosses_selector_floor=per <= LOCALIZATION_MARGIN,
        notes=(
            "single-span selector U_hat ~= 1/m_c; top-k selector -> 1. "
            "No leakage bias (crosses_gate_floor=False); selector-floor crossing "
            "tracked by crosses_selector_floor (1/m_c <= 0.03)."
        ),
    )


def _axis_distractor(strength: float, n_examples: int) -> GradedOracleFixture:
    """Axis D: inert span with high detector signal (REDESIGN_v4 §2.10G).

    The distractor has ``tau = 0`` but ``detector_signal = strength x`` the true
    span's. The naive detector selects it -> ``U_hat -> 0`` (inert); the causal
    selector keeps the designated span -> ``U_hat -> 1``; G5' difference >= 0.03.
    """
    examples: list[OracleExample] = []
    for i in range(n_examples):
        spans = (
            OracleSpan(
                span_id=f"D{i}_designated",
                tau=1.0,
                detector_signal=1.0,
                distance_to_answer=2,
                is_designated=True,
            ),
            OracleSpan(
                span_id=f"D{i}_distractor",
                tau=0.0,
                detector_signal=strength,  # high detector signal, causally inert
                distance_to_answer=8,
            ),
            OracleSpan(span_id=f"D{i}_inert", tau=0.0, detector_signal=0.1, distance_to_answer=16),
        )
        examples.append(OracleExample(example_id=f"D_ex{i}", spans=spans, bar_tau_pool=0.0))
    return GradedOracleFixture(
        axis="distractor",
        param=strength,
        examples=tuple(examples),
        expected_u_hat_selector=1.0,  # causal selector unaffected by distractor
        expected_u_hat_detector=0.0,  # detector-on-distractor is inert
        crosses_gate_floor=False,
        notes="detector-on-distractor U_hat stays 0; selector stays 1; G5' diff >= 0.03.",
    )


def graded_oracle_family(
    axis: OracleAxis,
    param: float,
    *,
    n_examples: int = 4,
    regime: str = "detectable",
) -> GradedOracleFixture:
    """Build one graded-oracle fixture for ``axis`` at sweep value ``param``.

    Parameters
    ----------
    axis:
        ``"clean"`` | ``"partial_leakage"`` (P) | ``"multi_cause"`` (M) |
        ``"distractor"`` (D) | ``"adversarial"`` (Axis X', REDESIGN_v5 §7).
    param:
        ``beta*Delta_pos`` for P (use the ``AXIS_P_LEAKAGE_GRID``), ``m_c`` for M
        (cast to int, use ``AXIS_M_CAUSE_GRID``), distractor strength ``d`` for D
        (use ``AXIS_D_DISTRACTOR_GRID``), confounding strength ``xi`` for the
        ``"adversarial"`` axis (use ``adversarial_oracle.AXIS_X_XI_GRID``). Ignored
        for ``"clean"``.
    regime:
        Only used by ``"adversarial"``: ``"detectable"`` | ``"blind"`` (§7.2).

    Returns
    -------
    GradedOracleFixture
        Examples plus the pre-registered expected ``U_hat`` curve point, so a
        test can assert the estimator tracks the planted ground truth. For the
        ``"adversarial"`` axis the ``expected_u_hat_selector`` is the confounded
        selector's pre-registered effective effect under the **misspecified
        reference** (it collapses with ``xi``), and ``notes`` carries the P5
        prediction; the full structural readout lives in
        :mod:`adversarial_oracle`.
    """
    if axis == "clean":
        examples = clean_oracle(n_examples)
        return GradedOracleFixture(
            axis="clean",
            param=0.0,
            examples=examples,
            expected_u_hat_selector=1.0,
            expected_u_hat_detector=0.0,
            crosses_gate_floor=False,
            notes="clean oracle sweep origin; U_hat(oracle)->1, U_hat(inert)->0.",
        )
    if axis == "partial_leakage":
        return _axis_partial_leakage(float(param), n_examples)
    if axis == "multi_cause":
        return _axis_multi_cause(int(param), n_examples)
    if axis == "distractor":
        return _axis_distractor(float(param), n_examples)
    if axis == "adversarial":
        # Lazy import to avoid a circular dependency (adversarial_oracle imports
        # OracleExample/OracleSpan/clean_oracle from this module).
        from .adversarial_oracle import axis_x_confounded

        fx = axis_x_confounded(float(param), n_examples, regime=regime)  # type: ignore[arg-type]
        return GradedOracleFixture(
            axis="adversarial",
            param=float(param),
            examples=fx.examples,
            # The confounded selector's effective transferred effect under the
            # misspecified reference collapses as xi -> 1 (Eq. P5-repair).
            expected_u_hat_selector=fx.readout.r_hat_expected,
            # A correlational detector tracking the confounded covariate stays high.
            expected_u_hat_detector=fx.readout.correlational_score_expected,
            crosses_gate_floor=False,
            notes=f"Axis X' ({fx.regime}); {fx.p5_prediction}",
        )
    raise ValueError(f"unknown oracle axis: {axis!r}")
