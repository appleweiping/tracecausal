from __future__ import annotations


def intervention_effect(before_factuality: float, after_factuality: float) -> float:
    """Positive values mean the intervention improved factuality."""
    return after_factuality - before_factuality


def passes_intervention_gate(
    targeted_delta: float,
    random_delta: float,
    utility_drop: float,
    *,
    min_margin: float = 0.05,
    max_utility_drop: float = 0.02,
) -> bool:
    """Check the early TraceCausal intervention gate."""
    return (targeted_delta - random_delta) >= min_margin and utility_drop <= max_utility_drop

