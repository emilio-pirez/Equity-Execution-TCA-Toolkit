"""Execute a parent order against historical 1-minute bars with a square-root impact model.

Fill model for a child order of q_t shares in bar t (side = +1 buy, -1 sell):

    ref_t  = typical price (H + L + C) / 3 of bar t (a proxy for the minute's VWAP);
             last close if the minute had no trade
    h_t    = half the estimated spread for that time of day (bps)
    rho_t  = sum_{w} q / (sum_{w} q + sum_{w} exec_V)  over the trailing `window` bars
    I_t    = k * sigma_d_bps * sqrt(rho_t)              (src.execution.impact)
    fill_t = ref_t * (1 + side * (h_t + I_t) / 10,000)

Assumptions (stated in the README): our trades do not alter the historical path
(no permanent impact feedback); all child orders cross the spread; the arrival price
is the last price when the order arrives (close of the opening-auction minute).
Every run is deterministic: no randomness is involved for a given input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import BPS, IMPACT_K, PARTICIPATION_WINDOW_MIN, PROFILE_BUCKET_MIN
from src.execution import algos
from src.execution.impact import participation_impact_bps, pretrade_cost_bps
from src.liquidity.metrics import adv, daily_volatility, expected_volume_curve
from src.liquidity.spread_estimators import corwin_schultz_bps, roll_bps
from src.tca.analysis import decompose, interval_vwap, slippage_bps

TICK = 0.01  # minimum price increment (MXN on BMV, USD in the US) used as a spread floor
SPARSE_TRADING = 0.5  # if < 50% of minutes trade, CS is unreliable -> use Roll


def build_day_grid(bars: pd.DataFrame, date) -> pd.DataFrame:
    """Full minute grid for one day's continuous session (missing minutes = no trade).

    Rows run from minute_of_session 1 to the last observed minute. Missing bars get
    volume 0 and OHLC equal to the previous close. `arrival_px` is stored in attrs.
    """
    day = bars[bars["date"] == pd.Timestamp(date)].sort_values("timestamp")
    if day.empty:
        raise ValueError(f"no bars on {date}")
    auction = day[day["is_open_auction"]]
    cont = day[day["is_continuous"]].set_index("minute_of_session")
    last = int(cont.index.max())
    first = int(day["minute_of_session"].min()) + 1
    grid = cont.reindex(range(first, last + 1))
    grid.index.name = "minute_of_session"
    prev_close = grid["close"].ffill()
    arrival = float(auction["close"].iloc[0]) if not auction.empty else float(cont["open"].iloc[0])
    prev_close = prev_close.fillna(arrival)
    for c in ("open", "high", "low", "close"):
        grid[c] = grid[c].fillna(prev_close)
    grid[["volume", "exec_volume"]] = grid[["volume", "exec_volume"]].fillna(0.0)
    open_ts = day["timestamp"].iloc[0] - pd.Timedelta(minutes=int(day["minute_of_session"].iloc[0]))
    grid["timestamp"] = [open_ts + pd.Timedelta(minutes=int(m)) for m in grid.index]
    grid["ref_px"] = np.where(grid["volume"] > 0, (grid["high"] + grid["low"] + grid["close"]) / 3.0, grid["close"])
    grid = grid.reset_index()[["timestamp", "minute_of_session", "open", "high", "low", "close",
                               "volume", "exec_volume", "ref_px"]]
    grid.attrs["arrival_px"] = arrival
    return grid


def spread_curve_bps(bars: pd.DataFrame, pct_minutes_traded: float, bucket_min: int = PROFILE_BUCKET_MIN) -> pd.Series:
    """Spread (bps) used by the simulator, by time-of-day bucket.

    Liquid names: Corwin-Schultz by 30-min bucket. Sparse names (< 50% of minutes with a
    trade): flat Roll estimate (CS collapses toward 0 when bars hold few trades - see
    synthetic validation). In both cases floored at one tick: 0.01 / median price.
    """
    floor = TICK / float(bars["close"].median()) * BPS
    buckets = (bars["minute_of_session"] // bucket_min) * bucket_min
    b = bars.assign(tod_bucket=buckets)
    if pct_minutes_traded >= SPARSE_TRADING:
        s = corwin_schultz_bps(b, by="tod_bucket")
    else:
        level = roll_bps(b, by=None)
        s = pd.Series(level, index=sorted(b.loc[b["is_continuous"], "tod_bucket"].unique()))
    return s.clip(lower=floor).rename("spread_bps")


@dataclass
class TickerContext:
    """Everything the simulator needs for one ticker (built once, reused per order)."""

    ticker: str
    bars: pd.DataFrame
    daily: pd.DataFrame
    spread_tod: pd.Series
    bucket_min: int = PROFILE_BUCKET_MIN
    k: float = IMPACT_K
    window: int = PARTICIPATION_WINDOW_MIN
    _grids: dict = field(default_factory=dict, repr=False)

    @property
    def dates(self) -> list[pd.Timestamp]:
        return sorted(self.bars["date"].unique())

    def grid(self, date) -> pd.DataFrame:
        key = pd.Timestamp(date)
        if key not in self._grids:
            g = build_day_grid(self.bars, key)
            g["expected_volume"] = (
                expected_volume_curve(self.bars, exclude_date=key).reindex(g["minute_of_session"]).fillna(0.0).to_numpy()
            )
            bucket = (g["minute_of_session"] // self.bucket_min) * self.bucket_min
            sp = self.spread_tod.reindex(bucket.unique()).ffill().bfill()
            g["spread_bps"] = bucket.map(sp).fillna(self.spread_tod.median()).to_numpy()
            self._grids[key] = g
        return self._grids[key]

    def prior_stats(self, date) -> dict[str, float]:
        """ADV and daily vol known *before* `date` (no look-ahead); official close of `date`."""
        d = pd.Timestamp(date)
        prior = self.daily[self.daily["date"] < d]
        row = self.daily[self.daily["date"] == d]
        close = float(row["close"].iloc[0]) if not row.empty else float(self.grid(d)["close"].iloc[-1])
        return {"adv": adv(prior), "sigma_bps": daily_volatility(prior) * BPS, "close_px": close}


def simulate_fills(
    grid: pd.DataFrame, q: np.ndarray, side: int, sigma_bps: float, k: float = IMPACT_K,
    window: int = PARTICIPATION_WINDOW_MIN,
) -> pd.DataFrame:
    """Apply the fill model to a schedule `q` aligned with `grid` rows. Returns per-bar fills."""
    q = np.asarray(q, dtype=float)
    qs = pd.Series(q).rolling(window, min_periods=1).sum().to_numpy()
    vs = pd.Series(grid["exec_volume"].to_numpy(float)).rolling(window, min_periods=1).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(qs > 0, qs / (qs + vs), 0.0)
    impact = np.where(q > 0, participation_impact_bps(rho, sigma_bps, k), 0.0)
    half_spread = np.where(q > 0, grid["spread_bps"].to_numpy() / 2.0, 0.0)
    ref = grid["ref_px"].to_numpy(float)
    fill = ref * (1.0 + side * (half_spread + impact) / BPS)
    return pd.DataFrame({
        "timestamp": grid["timestamp"].to_numpy(),
        "minute_of_session": grid["minute_of_session"].to_numpy(),
        "qty": q,
        "ref_px": ref,
        "fill_px": fill,
        "half_spread_bps": half_spread,
        "impact_bps": impact,
        "participation": rho,
        "market_volume": grid["exec_volume"].to_numpy(float),
    })


def run_order(
    ctx: TickerContext, date, algo: str, pct_adv: float, side: int = 1, **algo_params
) -> tuple[dict, pd.DataFrame]:
    """Simulate one parent order of `pct_adv` x ADV on `date`. Returns (summary, fills)."""
    g = ctx.grid(date)
    st = ctx.prior_stats(date)
    qty = int(round(pct_adv * st["adv"]))
    if qty <= 0:
        raise ValueError("order quantity rounds to zero")
    n = len(g)
    q = algos.schedule(
        algo, qty, n,
        expected_volume=g["expected_volume"].to_numpy(),
        market_volume=g["exec_volume"].to_numpy(),
        **algo_params,
    ).astype(float)
    fills = simulate_fills(g, q, side, st["sigma_bps"], ctx.k, ctx.window)
    arrival = g.attrs["arrival_px"]
    traded = fills[fills["qty"] > 0]
    filled = traded["qty"].sum()
    avg_px = float((traded["qty"] * traded["fill_px"]).sum() / filled) if filled else np.nan
    if filled:
        last_i = int(traded.index.max())
        first_i = int(traded.index.min())
        window = g.iloc[first_i : last_i + 1]
        vwap_px = interval_vwap(window["ref_px"].to_numpy(), window["exec_volume"].to_numpy())
        duration = int(g["minute_of_session"].iloc[last_i] - g["minute_of_session"].iloc[first_i] + 1)
        part = float(filled / (filled + window["exec_volume"].sum()))
    else:
        vwap_px, duration, part = np.nan, 0, 0.0
    unfilled = qty - filled
    cleanup = (
        pretrade_cost_bps(unfilled, st["adv"], st["sigma_bps"], float(g["spread_bps"].median()), ctx.k)["total_bps"]
        if unfilled > 0 else 0.0
    )
    dec = decompose(
        fills["qty"].to_numpy(), fills["ref_px"].to_numpy(), fills["half_spread_bps"].to_numpy(),
        fills["impact_bps"].to_numpy(), side, arrival, qty, st["close_px"], cleanup,
    )
    label = algo.upper() + (f"{int(algo_params['rate'] * 100)}" if algo.lower() == "pov" else "")
    summary = {
        "ticker": ctx.ticker,
        "date": pd.Timestamp(date).date().isoformat(),
        "algo": label,
        "pct_adv": pct_adv,
        "side": side,
        "qty": qty,
        "adv": st["adv"],
        "sigma_bps": st["sigma_bps"],
        "arrival_px": arrival,
        "avg_px": avg_px,
        "vwap_px": vwap_px,
        "close_px": st["close_px"],
        "vs_arrival_bps": slippage_bps(avg_px, arrival, side),
        "vs_vwap_bps": slippage_bps(avg_px, vwap_px, side),
        "vs_close_bps": slippage_bps(avg_px, st["close_px"], side),
        "duration_min": duration,
        "avg_participation": part,
        **dec,
    }
    return summary, fills


ALGO_SET: list[tuple[str, dict]] = [
    ("twap", {}),
    ("vwap", {}),
    ("pov", {"rate": 0.10}),
    ("pov", {"rate": 0.20}),
    ("is", {"urgency": 3.0}),
]


def run_grid(
    ctx: TickerContext,
    sizes=(0.01, 0.05, 0.10, 0.20),
    algo_set=ALGO_SET,
    sides=(1, -1),
    dates=None,
    keep_fills: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Run every (date, size, algo, side) combination. Deterministic."""
    rows, fills_all = [], []
    for d in dates or ctx.dates:
        for pct in sizes:
            for algo, params in algo_set:
                for side in sides:
                    s, f = run_order(ctx, d, algo, pct, side, **params)
                    rows.append(s)
                    if keep_fills:
                        fills_all.append(f[f["qty"] > 0].assign(
                            ticker=ctx.ticker, date=s["date"], algo=s["algo"], pct_adv=pct, side=side))
    orders = pd.DataFrame(rows)
    return orders, (pd.concat(fills_all, ignore_index=True) if keep_fills else None)
