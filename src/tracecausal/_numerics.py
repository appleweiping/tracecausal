"""Shared internal numerics (pure Python; no numpy, no model, no GPU).

These helpers were previously duplicated verbatim in ``ciu.py`` and
``nuisance.py``. They are de-duplicated here (Opus minor: dedup the duplicated
quantile) so the bootstrap CI machinery has a single definition. Private by
convention (leading underscore); not part of the public package API.
"""

from __future__ import annotations

import math
from typing import Sequence

__all__ = ["quantile"]


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an **already-sorted** sequence.

    ``q`` in ``[0, 1]``. Mirrors the ``numpy.quantile`` linear (``'linear'``)
    interpolation on a sorted input. Raises ``ValueError`` on an empty sequence.
    """
    if not sorted_values:
        raise ValueError("cannot take quantile of empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo_i = int(math.floor(pos))
    hi_i = int(math.ceil(pos))
    if lo_i == hi_i:
        return sorted_values[lo_i]
    frac = pos - lo_i
    return sorted_values[lo_i] * (1.0 - frac) + sorted_values[hi_i] * frac
