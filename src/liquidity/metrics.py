"""Liquidity metrics. Costs in bps; volumes in shares unless suffixed _usd.

All functions take either daily bars (date, open, high, low, close, volume) or
enriched 1-minute bars (see src.data.loaders.enrich_bars).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ADV_WINDOW_DAYS, BPS, BY_TICKER, PROFILE_BUCKET_MIN, VOL_WINDOW_DAYS
from src.liquidity.spread_estimators import corwin_schultz_bps, quoted_spread_bps, roll_bps

TRADING_DAYS = 252

__all__ = [
    "quoted_spread_bps",
    "to_usd",
    "adv",
    "daily_volatility",
    "intraday_realized_vol",
    "amihud_illiquidity",
    "usd_per_bp",
    "volume_profile",
    "expected_volume_curve",
    "volume_coverage",
    "liquidity_summary",
]


def to_usd(daily: pd.DataFrame, market: str, fx: pd.DataFrame | None) -> pd.Series:
    """Daily traded value in USD: close * volume / USDMXN for MX names (fx forward-filled).

    value_usd_d = close_d * volume_d / fx_d,  fx_d = MXN per USD (yfinance 'MXN=X' close).
    """
    value = daily["close"] * daily["volume"]
    if market != "MX":
        return value
    if fx is None or fx.empty:
        raise ValueError("USD/MXN rates required for MX names")
    rate = fx.set_index("date")["close"].reindex(daily["date"]).ffill().bfill().to_numpy()
    return value / rate


def adv(daily: pd.DataFrame, window: int = ADV_WINDOW_DAYS, asof=None) -> float:
    """Average daily volume (shares) over the last `window` sessions up to `asof` (inclusive).

    ADV = (1/N) * sum_{d=1..N} volume_d. Uses official daily volume (incl. auctions/crosses).
    """
    d = daily[daily["volume"] > 0]
    if asof is not None:
        d = d[d["date"] <= pd.Timestamp(asof)]
    return float(d["volume"].tail(window).mean())


def daily_volatility(daily: pd.DataFrame, window: int = VOL_WINDOW_DAYS, asof=None) -> float:
    """Daily volatility (decimal): std of log close-to-close returns over `window` days.

    sigma_d = stdev( ln(C_d / C_{d-1}) ),  ddof = 1.
    """
    d = daily[daily["volume"] > 0]
    if asof is not None:
        d = d[d["date"] <= pd.Timestamp(asof)]
    r = np.log(d["close"]).diff().dropna().tail(window)
    return float(r.std(ddof=1))


def intraday_realized_vol(bars: pd.DataFrame) -> pd.Series:
    """Per-day realized volatility from continuous-session 1m log returns (decimal, not annualized).

    RV_d = sqrt( sum_t r_t^2 ),  r_t = ln(C_t / C_{t-1}) within day d.
    Note: excludes the overnight return and, for BMV in yfinance, the missing last hour.
    """
    b = bars[bars["is_continuous"]]
    r = np.log(b["close"]).groupby(b["date"]).diff()
    return np.sqrt((r**2).groupby(b["date"]).sum()).rename("rv")


def amihud_illiquidity(daily: pd.DataFrame, value_usd: pd.Series, window: int = ADV_WINDOW_DAYS) -> float:
    """Amihud (2002) illiquidity in bps of price move per USD 1m traded.

    ILLIQ = mean_d( |r_d| * 10,000 / (value_usd_d / 1e6) ),  r_d = daily log return.
    """
    r = np.log(daily["close"]).diff().abs() * BPS
    x = (r / (value_usd / 1e6)).replace([np.inf, -np.inf], np.nan)
    x = x[daily["volume"] > 0]
    return float(x.tail(window).mean())


def usd_per_bp(bars: pd.DataFrame, usd_rate: float = 1.0) -> float:
    """Depth proxy: USD of executable 1m volume per 1 bp of absolute 1m price move.

    For each day: D_d = sum_t (exec_volume_t * C_t / fx) / sum_t |r_t| * 10,000;
    the reported value is the median of D_d across days (an inverse intraday Amihud).
    """
    b = bars[bars["is_continuous"]].copy()
    b["r_bps"] = np.log(b["close"]).groupby(b["date"]).diff().abs() * BPS
    b["usd"] = b["exec_volume"] * b["close"] / usd_rate
    g = b.groupby("date")
    d = g["usd"].sum() / g["r_bps"].sum()
    return float(d.replace([np.inf, -np.inf], np.nan).median())


def volume_profile(bars: pd.DataFrame, bucket_min: int = PROFILE_BUCKET_MIN, col: str = "exec_volume") -> pd.Series:
    """Intraday volume profile: share of continuous-session volume by bucket since the open.

    profile_b = mean_d(V_{d,b}) / sum_b mean_d(V_{d,b}), averaging each bucket only over the
    days where it is observed (robust to yfinance truncating the last BMV hour on some days).
    """
    b = bars[bars["is_continuous"]]
    bucket = (b["minute_of_session"] // bucket_min) * bucket_min
    per_day = b.groupby(["date", bucket])[col].sum().rename("v").reset_index()
    level = per_day.groupby("minute_of_session")["v"].mean()
    return (level / level.sum()).rename("volume_share")


def expected_volume_curve(bars: pd.DataFrame, exclude_date=None, col: str = "exec_volume") -> pd.Series:
    """Expected executable volume per minute_of_session (shares), leave-one-day-out.

    E[V_m] = mean over days d != exclude_date of V_{d,m}, with missing minutes counted as 0
    (no trade in that minute). Smoothed with a centered 15-minute rolling mean.
    """
    b = bars[bars["is_continuous"]]
    if exclude_date is not None:
        b = b[b["date"] != pd.Timestamp(exclude_date)]
    grid = b.pivot_table(index="date", columns="minute_of_session", values=col, aggfunc="sum")
    # A minute is "observed" on a day if that day's data extends past it.
    last_min = b.groupby("date")["minute_of_session"].max()
    obs = pd.DataFrame(
        {m: (last_min >= m).to_numpy() for m in grid.columns}, index=last_min.index
    ).reindex(grid.index)
    grid = grid.fillna(0.0).where(obs)
    curve = grid.mean(axis=0)
    return curve.rolling(15, center=True, min_periods=1).mean().rename("expected_volume")


def volume_coverage(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Share of official daily volume visible in 1-minute bars, per day.

    coverage_all  = sum_t V_t / V_daily       (all 1m prints)
    coverage_exec = sum_t exec_V_t / V_daily  (excluding auctions and block/bad prints)
    """
    g = bars.groupby("date")
    out = pd.DataFrame({"v_1m": g["volume"].sum(), "v_exec": g["exec_volume"].sum()})
    dv = daily.set_index("date")["volume"]
    out["v_daily"] = dv.reindex(out.index)
    out["coverage_all"] = out["v_1m"] / out["v_daily"]
    out["coverage_exec"] = out["v_exec"] / out["v_daily"]
    return out


def liquidity_summary(
    ticker: str, bars: pd.DataFrame, daily: pd.DataFrame, fx: pd.DataFrame | None
) -> dict[str, float | str]:
    """One-row liquidity profile for a ticker over the 1m sample window."""
    inst = BY_TICKER[ticker]
    end = bars["date"].max()
    d = daily[daily["date"] <= end].reset_index(drop=True)
    value_usd = to_usd(d, inst.market, fx)
    fx_last = 1.0
    if inst.market == "MX":
        fx_last = float(fx.set_index("date")["close"].reindex([end]).ffill().iloc[0]) if fx is not None else np.nan
        if not np.isfinite(fx_last):
            fx_last = float(fx[fx["date"] <= end]["close"].iloc[-1])
    adv_sh = adv(d)
    px = float(d["close"].iloc[-1])
    cs_daily = corwin_schultz_bps(bars, by="date")
    roll_daily = roll_bps(bars, by="date")
    cov = volume_coverage(bars, d)
    cont = bars[bars["is_continuous"]]
    n_days = cont["date"].nunique()
    minutes_with_trade = (cont["volume"] > 0).groupby(cont["date"]).sum()
    minutes_observed = cont.groupby("date")["minute_of_session"].max()
    return {
        "ticker": ticker,
        "name": inst.name,
        "market": inst.market,
        "asset_type": inst.asset_type,
        "days_1m": int(n_days),
        "price_local": px,
        "adv_shares": adv_sh,
        "adv_usd_m": float(value_usd.tail(ADV_WINDOW_DAYS).mean() / 1e6),
        "daily_vol_bps": daily_volatility(d) * BPS,
        "cs_spread_bps": float(cs_daily.median()),
        "roll_spread_bps": float(roll_daily.median()),
        "amihud_bps_per_usd_m": amihud_illiquidity(d, value_usd),
        "usd_per_bp_1m": usd_per_bp(bars, fx_last),
        "pct_minutes_traded": float((minutes_with_trade / minutes_observed).median()),
        "coverage_1m_vs_daily": float(cov["coverage_all"].median()),
        "coverage_exec_vs_daily": float(cov["coverage_exec"].median()),
        "outlier_volume_share": float(bars.loc[bars["vol_outlier"], "volume"].sum() / bars["volume"].sum()),
    }
