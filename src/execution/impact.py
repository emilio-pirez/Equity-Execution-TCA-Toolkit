"""Square-root market impact model.

Pre-trade estimate for a parent order of Q shares executed over (roughly) one day:

    cost_bps = spread_bps / 2  +  k * sigma_d_bps * sqrt(Q / ADV)

* spread_bps / 2 : paying half the spread on every share (aggressive/crossing fills)
* sigma_d_bps    : daily volatility in bps (std of daily log returns x 10,000)
* Q / ADV        : order size as a fraction of average daily volume
* k              : impact coefficient, O(1). Empirical estimates range ~0.5-1
                   (Almgren et al. 2005; Toth et al. 2011; Bouchaud et al. 2018).
                   We use k = 0.7 and report sensitivity at 0.5 and 1.0.

In the simulator the same law is applied *locally in time*, as a function of the
participation rate rho_t (our share of volume over a short rolling window):

    impact_t (bps) = k * sigma_d_bps * sqrt(rho_t)

If an order is executed at a constant participation rho = Q / V_day over the full day,
both expressions coincide (Q / ADV ~ rho). Trading faster (higher rho) costs more per
share; trading unevenly costs more than trading evenly (Jensen: E[rho^1.5] >= E[rho]^1.5).
"""

from __future__ import annotations

import numpy as np

from src.config import IMPACT_K


def sqrt_impact_bps(qty: float, adv: float, sigma_daily_bps: float, k: float = IMPACT_K) -> float:
    """Square-root impact in bps: k * sigma_d_bps * sqrt(qty / adv)."""
    if adv <= 0:
        raise ValueError("ADV must be positive")
    return float(k * sigma_daily_bps * np.sqrt(max(qty, 0.0) / adv))


def pretrade_cost_bps(
    qty: float, adv: float, sigma_daily_bps: float, spread_bps: float, k: float = IMPACT_K
) -> dict[str, float]:
    """Pre-trade expected cost: spread/2 + k * sigma * sqrt(Q/ADV), with its components."""
    half_spread = spread_bps / 2.0
    impact = sqrt_impact_bps(qty, adv, sigma_daily_bps, k)
    return {"half_spread_bps": half_spread, "impact_bps": impact, "total_bps": half_spread + impact}


def participation_impact_bps(participation, sigma_daily_bps: float, k: float = IMPACT_K):
    """Instantaneous impact for participation rate(s) rho in [0, 1]: k * sigma_d_bps * sqrt(rho)."""
    rho = np.clip(np.asarray(participation, dtype=float), 0.0, 1.0)
    return k * sigma_daily_bps * np.sqrt(rho)
