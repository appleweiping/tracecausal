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
    """Check the early TraceCausal intervention gate.

    Pre-registered inequality direction (Opus minor: pick ge vs strict gt at the
    margin and document it): the necessity margin is **inclusive** — the gate
    passes when ``targeted_delta - random_delta >= min_margin`` (``>=``, NOT strict
    ``>``). A contrast landing exactly on the 0.05 margin clears it. This matches
    the config marker ``random_intervention_margin_abs: 0.05`` and the utility
    conjunct ``utility_drop <= max_utility_drop`` (also inclusive), so both arms use
    the same closed-boundary convention. (Prior design prose informally wrote
    "> 0.05"; the locked code convention is ``>=`` at the margin.)
    """
    return (targeted_delta - random_delta) >= min_margin and utility_drop <= max_utility_drop

