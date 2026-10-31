"""Small-sample proportion statistics, standard-library only.

Wilson score intervals (z = 1.96) for survival fractions, and paired
summaries for the stock-vs-ADR comparison. Runs are paired by
(seed, dt, operator_mode): the two arms differ only in the residual
function, so per-configuration deltas and discordant counts are the
primary comparison, not two independent proportions. No threshold was
tuned anywhere in this module.
"""

from __future__ import annotations

import math

Z_95 = 1.96


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion. Returns (low, high)."""
    if trials <= 0:
        raise ValueError("trials must be > 0")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be in [0, trials]")
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise_arm(outcomes: list[bool]) -> dict:
    """Survival summary for one arm: fraction + Wilson interval."""
    n = len(outcomes)
    k = sum(1 for o in outcomes if o)
    low, high = wilson_interval(k, n)
    return {
        "n": n,
        "survived": k,
        "survival_fraction": k / n if n else 0.0,
        "wilson_low": low,
        "wilson_high": high,
    }


def paired_summary(stock: list[bool], adr: list[bool]) -> dict:
    """Paired comparison of two arms over the same configurations."""
    if len(stock) != len(adr):
        raise ValueError("paired arms must have the same length")
    both_survive = sum(1 for s, a in zip(stock, adr) if s and a)
    both_fail = sum(1 for s, a in zip(stock, adr) if not s and not a)
    adr_only = sum(1 for s, a in zip(stock, adr) if not s and a)
    stock_only = sum(1 for s, a in zip(stock, adr) if s and not a)
    return {
        "n": len(stock),
        "both_survive": both_survive,
        "both_fail": both_fail,
        "adr_only_survives": adr_only,
        "stock_only_survives": stock_only,
        "discordant": adr_only + stock_only,
        "survival_delta_adr_minus_stock": (
            (sum(adr) - sum(stock)) / len(stock) if stock else 0.0
        ),
    }
