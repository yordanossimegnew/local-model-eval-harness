"""Small statistics helpers (no SciPy)."""

from __future__ import annotations

import math


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion; fine for small eval sets."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def percentile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
