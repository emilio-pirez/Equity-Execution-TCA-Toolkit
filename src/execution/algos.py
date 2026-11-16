"""Execution schedules: TWAP, VWAP, POV and Implementation Shortfall (Almgren-Chriss).

Each scheduler returns integer child quantities (numpy array), one per bar of the
execution window, for a parent order of `qty` shares. TWAP/VWAP/IS always sum exactly
to `qty`; POV may leave a remainder (reported as unfilled by the simulator).
"""

from __future__ import annotations

import numpy as np


def round_preserving_sum(x: np.ndarray, total: int) -> np.ndarray:
    """Round non-negative weights*total to integers that sum exactly to `total`.

    Largest-remainder (Hamilton) method: floor everything, then give the leftover units
    to the bars with the largest fractional parts.
    """
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    if total <= 0 or x.sum() <= 0:
        return np.zeros(len(x), dtype=np.int64)
    raw = x / x.sum() * total
    base = np.floor(raw).astype(np.int64)
    rem = int(total - base.sum())
    if rem > 0:
        idx = np.argsort(-(raw - base), kind="stable")[:rem]
        base[idx] += 1
    return base


def twap(n_bars: int, qty: int) -> np.ndarray:
    """Time-weighted: q_t = Q / N in every bar (integer shares, exact sum)."""
    if n_bars <= 0:
        raise ValueError("n_bars must be positive")
    return round_preserving_sum(np.ones(n_bars), int(qty))


def vwap(expected_volume: np.ndarray, qty: int) -> np.ndarray:
    """Volume-weighted: q_t = Q * E[V_t] / sum_s E[V_s], using a historical volume curve.

    The curve must be estimated without the execution day (no look-ahead); a zero curve
    falls back to TWAP.
    """
    ev = np.nan_to_num(np.asarray(expected_volume, dtype=float), nan=0.0)
    if ev.sum() <= 0:
        return twap(len(ev), qty)
    return round_preserving_sum(ev, int(qty))


def pov(market_volume: np.ndarray, rate: float, qty: int) -> np.ndarray:
    """Percent-of-volume: trade so that our share of total volume equals `rate`.

    q_t = rate / (1 - rate) * V_t   (V_t = other participants' volume in bar t),
    capped by the remaining quantity. Reacts to realized volume; may not complete.
    """
    if not 0 < rate < 1:
        raise ValueError("rate must be in (0, 1)")
    v = np.nan_to_num(np.asarray(market_volume, dtype=float), nan=0.0)
    target = np.floor(rate / (1.0 - rate) * v).astype(np.int64)
    cum = np.minimum(np.cumsum(target), int(qty))
    return np.diff(cum, prepend=0)


def ac_kappa(lam: float, sigma_bar_bps: float, eta_bps: float) -> float:
    """Almgren-Chriss decay rate per bar from risk aversion.

    Discrete AC (2000) with tau = 1 bar and no permanent impact:
        2 (cosh(kappa) - 1) = lam * sigma^2 / eta
    lam in 1/bps, sigma_bar_bps = per-bar volatility, eta_bps = linear temporary impact
    coefficient (bps of notional per unit of order fraction traded per bar).
    """
    if lam <= 0:
        return 0.0
    return float(np.arccosh(1.0 + lam * sigma_bar_bps**2 / (2.0 * eta_bps)))


def ac_holdings(n_bars: int, kappa: float) -> np.ndarray:
    """Remaining fraction x_j after j bars, j = 0..N:  x_j = sinh(kappa (N - j)) / sinh(kappa N).

    kappa -> 0 gives the straight line (TWAP); larger kappa front-loads execution.
    """
    j = np.arange(n_bars + 1)
    if kappa < 1e-10:
        return 1.0 - j / n_bars
    # numerically stable for large kappa*N: sinh(a)/sinh(b) = exp(a-b) (1-exp(-2a))/(1-exp(-2b))
    a, b = kappa * (n_bars - j), kappa * n_bars
    return np.exp(a - b) * (-np.expm1(-2 * a)) / (-np.expm1(-2 * b))


def implementation_shortfall(n_bars: int, qty: int, urgency: float = 3.0) -> np.ndarray:
    """IS / Almgren-Chriss schedule with urgency kappa*T (dimensionless).

    urgency 0 = TWAP; 3 ~ half the order done in the first ~20% of the window.
    """
    x = ac_holdings(n_bars, urgency / n_bars)
    return round_preserving_sum(-np.diff(x), int(qty))


def ac_cost_risk(n_bars: int, kappa: float, sigma_bar_bps: float, eta_bps: float) -> tuple[float, float]:
    """Analytic expected cost and risk (bps of order notional) of an AC trajectory.

    E = eta * sum_j n_j^2          (temporary impact, n_j = fraction traded in bar j)
    V = sigma^2 * sum_{j>=1} x_j^2 (price risk on the remaining position)
    Returns (E, sqrt(V)).
    """
    x = ac_holdings(n_bars, kappa)
    n = -np.diff(x)
    e = eta_bps * np.sum(n**2)
    v = sigma_bar_bps**2 * np.sum(x[1:] ** 2)
    return float(e), float(np.sqrt(v))


def ac_frontier(
    n_bars: int, sigma_bar_bps: float, eta_bps: float, lambdas: np.ndarray | None = None
) -> list[dict[str, float]]:
    """Efficient frontier: (risk, cost) for a grid of risk aversions lambda (1/bps)."""
    if lambdas is None:
        lambdas = np.r_[0.0, np.logspace(-6, 1, 40)]
    out = []
    for lam in lambdas:
        k = ac_kappa(lam, sigma_bar_bps, eta_bps)
        e, s = ac_cost_risk(n_bars, k, sigma_bar_bps, eta_bps)
        out.append({"lambda": float(lam), "kappa": k, "urgency": k * n_bars, "cost_bps": e, "risk_bps": s})
    return out


def schedule(
    algo: str,
    qty: int,
    n_bars: int,
    expected_volume: np.ndarray | None = None,
    market_volume: np.ndarray | None = None,
    **params,
) -> np.ndarray:
    """Dispatch by name: 'twap', 'vwap', 'pov' (rate=), 'is' (urgency=)."""
    algo = algo.lower()
    if algo == "twap":
        return twap(n_bars, qty)
    if algo == "vwap":
        return vwap(expected_volume, qty)
    if algo == "pov":
        return pov(market_volume, params.get("rate", 0.10), qty)
    if algo == "is":
        return implementation_shortfall(n_bars, qty, params.get("urgency", 3.0))
    raise ValueError(f"unknown algo {algo!r}")
