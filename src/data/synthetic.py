"""Synthetic 1-minute bars with a KNOWN true spread.

Data-generating process (one trading day = `minutes_per_day` bars):

* Efficient (mid) log-price: Brownian motion with per-minute volatility
  sigma_1m = daily_vol / sqrt(minutes_per_day), plus an overnight gap.
* Trades arrive as a Poisson process with intensity trades_per_minute * u_vol(t),
  where u_vol is U-shaped (more activity at the open and close). Each trade is a buy
  or sell with probability 1/2 and prints at mid * exp(+-S_t / 2).
* True spread S_t (fraction of price) is mean-reverting around a U-shaped intraday
  level:  ln S_t = ln(S_bar * u_spr(t)) + z_t,   z_{t+1} = (1 - kappa) z_t + eta * eps.
* Bars aggregate trades: open/high/low/close/volume; bid/ask are the quotes around
  the mid at the last trade of the minute.

This is exactly the world Corwin-Schultz (2012) and Roll (1984) assume (plus discrete
trading), so the estimators should recover `true_spread_bps` up to sampling noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BPS, DEFAULT_SEED


def u_shape(n: int, amplitude: float) -> np.ndarray:
    """U-shaped curve with mean 1: 1 + 3*amp*((2x-1)^2 - 1/3). Edges 1+2amp, centre 1-amp."""
    x = np.linspace(0.0, 1.0, n)
    return 1.0 + 3.0 * amplitude * ((2.0 * x - 1.0) ** 2 - 1.0 / 3.0)


def generate_bars(
    n_days: int = 20,
    minutes_per_day: int = 390,
    start_price: float = 100.0,
    daily_vol: float = 0.02,
    mean_spread_bps: float = 10.0,
    spread_u_amplitude: float = 0.3,
    spread_kappa: float = 0.05,
    spread_eta: float = 0.05,
    trades_per_minute: float = 30.0,
    volume_u_amplitude: float = 0.4,
    adv_shares: float = 1_000_000.0,
    overnight_vol: float = 0.01,
    start_date: str = "2024-01-02",
    open_time: str = "09:30",
    tz: str = "America/New_York",
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate synthetic bars. Returns the common schema plus `mid` and `true_spread_bps`."""
    rng = np.random.default_rng(seed)
    m = minutes_per_day
    n_bars = n_days * m

    # --- true spread path (fraction of price) ---
    level = mean_spread_bps / BPS * np.tile(u_shape(m, spread_u_amplitude), n_days)
    z = np.empty(n_bars)
    z[0] = 0.0
    eps = rng.standard_normal(n_bars)
    for t in range(1, n_bars):
        z[t] = (1.0 - spread_kappa) * z[t - 1] + spread_eta * eps[t]
    spread = level * np.exp(z - 0.5 * spread_eta**2 / (1 - (1 - spread_kappa) ** 2))

    # --- trades ---
    intensity = trades_per_minute * np.tile(u_shape(m, volume_u_amplitude), n_days)
    n_trades = np.maximum(rng.poisson(intensity), 1)
    bar_of_trade = np.repeat(np.arange(n_bars), n_trades)
    t_in_bar = rng.random(bar_of_trade.size)
    order = np.lexsort((t_in_bar, bar_of_trade))
    bar_of_trade, t_in_bar = bar_of_trade[order], t_in_bar[order]
    clock = bar_of_trade + t_in_bar  # minutes since first open (per-day clocks are contiguous)

    sigma_1m = daily_vol / np.sqrt(m)
    dt = np.diff(clock, prepend=0.0)
    log_mid_inc = sigma_1m * np.sqrt(dt) * rng.standard_normal(clock.size)
    day_of_trade = bar_of_trade // m
    new_day = np.r_[False, np.diff(day_of_trade) > 0]
    log_mid_inc[new_day] += overnight_vol * rng.standard_normal(new_day.sum())
    log_mid = np.log(start_price) + np.cumsum(log_mid_inc)

    side = rng.choice([-1.0, 1.0], size=clock.size)
    s_trade = spread[bar_of_trade]
    price = np.exp(log_mid + side * s_trade / 2.0)
    size = rng.lognormal(mean=0.0, sigma=1.0, size=clock.size)
    size *= adv_shares * n_days / size.sum()

    trades = pd.DataFrame({"bar": bar_of_trade, "price": price, "size": size, "log_mid": log_mid})
    g = trades.groupby("bar", sort=True)
    bars = pd.DataFrame(
        {
            "open": g["price"].first(),
            "high": g["price"].max(),
            "low": g["price"].min(),
            "close": g["price"].last(),
            "volume": g["size"].sum().round(),
            "mid": np.exp(g["log_mid"].last()),
        }
    ).reindex(np.arange(n_bars))
    bars["bid"] = bars["mid"] * np.exp(-spread / 2.0)
    bars["ask"] = bars["mid"] * np.exp(spread / 2.0)
    bars["true_spread_bps"] = spread * BPS

    days = pd.bdate_range(start_date, periods=n_days)
    oh, om = (int(x) for x in open_time.split(":"))
    ts = [
        (d + pd.Timedelta(hours=oh, minutes=om + k)).tz_localize(tz) for d in days for k in range(m)
    ]
    bars.insert(0, "timestamp", pd.DatetimeIndex(ts))
    return bars.reset_index(drop=True)
