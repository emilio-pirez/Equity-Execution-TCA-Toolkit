"""Transaction cost analysis: slippage vs benchmarks and cost decomposition.

Sign convention: positive bps = cost to the trader.
    slippage_bps = side * (avg_exec_px - benchmark) / benchmark * 10,000,  side = +1 buy, -1 sell

Implementation shortfall decomposition (Perold 1988), all vs the arrival price A,
in bps of the *parent order* notional Q * A:

    fill_t   = ref_t * (1 + side * (h_t + I_t) / 10,000)       (simulator fill model)
    timing   = side * sum_t q_t (ref_t - A)            / (Q A) * 1e4   market drift while working
    spread   =        sum_t q_t ref_t h_t              / (Q A)          half-spread paid
    impact   =        sum_t q_t ref_t I_t              / (Q A)          our own price pressure
    opport.  = [side * (P_close - A) / A * 1e4 + C_cleanup] * (Q - Q_filled) / Q
               unfilled shares: drift to the close plus C_cleanup, the pre-trade cost of
               completing the remainder in the next session (src.execution.impact)
    total IS = timing + spread + impact + opportunity       (exact identity)

and `is_filled_bps` = (timing + spread + impact) * Q / Q_filled = side * (avg_px - A) / A * 1e4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BPS


def slippage_bps(avg_px: float, benchmark: float, side: int) -> float:
    """side * (avg_px - benchmark) / benchmark * 10,000 (positive = cost)."""
    if not np.isfinite(avg_px) or not np.isfinite(benchmark) or benchmark <= 0:
        return np.nan
    return float(side * (avg_px - benchmark) / benchmark * BPS)


def interval_vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Market VWAP over an interval: sum p_t V_t / sum V_t (NaN if no volume)."""
    v = np.asarray(volumes, dtype=float)
    return float(np.sum(np.asarray(prices) * v) / v.sum()) if v.sum() > 0 else np.nan


def decompose(
    q: np.ndarray,
    ref: np.ndarray,
    half_spread_bps: np.ndarray,
    impact_bps: np.ndarray,
    side: int,
    arrival: float,
    parent_qty: int,
    close_px: float,
    cleanup_cost_bps: float = 0.0,
) -> dict[str, float]:
    """Cost decomposition (bps of parent notional) for child fills q_t at reference prices ref_t."""
    q = np.asarray(q, dtype=float)
    ref = np.asarray(ref, dtype=float)
    notional = parent_qty * arrival
    filled = q.sum()
    timing = side * np.sum(q * (ref - arrival)) / notional * BPS
    spread = np.sum(q * ref * half_spread_bps) / notional
    impact = np.sum(q * ref * impact_bps) / notional
    unfilled = parent_qty - filled
    opp = 0.0
    if unfilled > 0:
        opp = (side * (close_px - arrival) / arrival * BPS + cleanup_cost_bps) * unfilled / parent_qty
    total = timing + spread + impact + opp
    return {
        "timing_bps": float(timing),
        "spread_bps": float(spread),
        "impact_bps": float(impact),
        "opportunity_bps": float(opp),
        "is_total_bps": float(total),
        "is_filled_bps": float((timing + spread + impact) * parent_qty / filled) if filled > 0 else np.nan,
        "fill_rate": float(filled / parent_qty) if parent_qty else np.nan,
    }


def summarize(orders: pd.DataFrame, by: list[str], col: str = "is_total_bps") -> pd.DataFrame:
    """Distribution of a cost column by group: mean, std, median, p10, p90, n."""
    g = orders.groupby(by)[col]
    return pd.DataFrame(
        {
            "mean": g.mean(),
            "std": g.std(),
            "median": g.median(),
            "p10": g.quantile(0.10),
            "p90": g.quantile(0.90),
            "n": g.size(),
        }
    ).reset_index()


def cost_table(orders: pd.DataFrame) -> pd.DataFrame:
    """Mean cost components and benchmark slippages by ticker x algo x size."""
    cols = [
        "is_total_bps", "spread_bps", "impact_bps", "timing_bps", "opportunity_bps",
        "vs_vwap_bps", "vs_close_bps", "fill_rate", "duration_min", "avg_participation",
    ]
    out = orders.groupby(["ticker", "algo", "pct_adv"])[cols].mean()
    out["is_std_bps"] = orders.groupby(["ticker", "algo", "pct_adv"])["is_total_bps"].std()
    return out.reset_index()
