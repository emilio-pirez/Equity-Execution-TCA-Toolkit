"""Common data interface: `load_bars(ticker, source)` and session enrichment.

Every downstream module consumes the schema
    timestamp (tz-aware, exchange local), open, high, low, close, volume, bid, ask
optionally enriched by `enrich_bars` with session/auction/outlier flags.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from src.config import BY_TICKER, session_times
from src.data.bbg_loader import load_bbg_1m
from src.data.synthetic import generate_bars
from src.data.yf_loader import BAR_COLUMNS, load_yf_1m

log = logging.getLogger(__name__)

SOURCES = ("yf", "bbg", "synthetic")
OUTLIER_MULT = 25.0  # bar volume > 25x the day's median 1m volume -> non-executable print


def load_bars(ticker: str, source: str = "yf", **kwargs) -> pd.DataFrame:
    """Load 1-minute bars for `ticker` from `source` in the common schema.

    source: "yf" (cached yfinance), "bbg" (Bloomberg CSV, raises FileNotFoundError if
    absent) or "synthetic" (ticker is ignored; kwargs go to `generate_bars`).
    bid/ask are NaN when the source does not provide quotes.
    """
    if source == "yf":
        df = load_yf_1m(ticker, **kwargs)
    elif source == "bbg":
        df = load_bbg_1m(ticker, **kwargs)
    elif source == "synthetic":
        df = generate_bars(**kwargs)
    else:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    extra = [c for c in df.columns if c not in BAR_COLUMNS]
    return df[BAR_COLUMNS + extra]


def _time_to_min(t) -> int:
    return t.hour * 60 + t.minute


def enrich_bars(df: pd.DataFrame, market: str | None = None, ticker: str | None = None) -> pd.DataFrame:
    """Add session-aware columns.

    Added columns
    -------------
    date               local trading date
    session_open       continuous-session open for that date (local time; BMV shifts with US DST)
    minute_of_session  minutes since the continuous open (0 = first minute)
    is_open_auction    first bar of the day (contains the opening auction print)
    is_close_auction   bars at/after the official continuous close (closing auction prints)
    is_continuous      not an auction bar
    vol_outlier        volume > 25x the day's median positive 1m volume (likely block cross or bad print)
    exec_volume        volume an algorithm could plausibly participate in: 0 for outliers
                       and auction bars, else volume
    """
    if market is None:
        market = BY_TICKER[ticker].market if ticker else "US"
    out = df.copy()
    ts = out["timestamp"]
    out["date"] = ts.dt.tz_localize(None).dt.normalize()
    sess = {d: session_times(market, d) for d in out["date"].unique()}
    open_min = out["date"].map(lambda d: _time_to_min(sess[d][0]))
    close_min = out["date"].map(lambda d: _time_to_min(sess[d][1]))
    clock_min = ts.dt.hour * 60 + ts.dt.minute
    out["session_open"] = out["date"].map(lambda d: sess[d][0].strftime("%H:%M"))
    out["minute_of_session"] = (clock_min - open_min).astype(int)
    first = out.groupby("date")["timestamp"].transform("min") == ts
    out["is_open_auction"] = first.to_numpy()
    out["is_close_auction"] = (clock_min >= close_min).to_numpy()
    out["is_continuous"] = ~(out["is_open_auction"] | out["is_close_auction"])
    pos_med = out["volume"].where(out["volume"] > 0).groupby(out["date"]).transform("median")
    out["vol_outlier"] = (out["volume"] > OUTLIER_MULT * pos_med).fillna(False).to_numpy()
    out["exec_volume"] = np.where(out["vol_outlier"] | ~out["is_continuous"], 0.0, out["volume"])
    return out


def load_enriched(ticker: str, source: str = "yf", **kwargs) -> pd.DataFrame:
    """`load_bars` + `enrich_bars` (market looked up from the universe)."""
    return enrich_bars(load_bars(ticker, source, **kwargs), ticker=ticker)


def available_tickers(tickers: list[str], source: str = "yf") -> list[str]:
    """Tickers that load with at least one bar; failures are logged and skipped."""
    ok = []
    for t in tickers:
        try:
            if len(load_bars(t, source)) > 0:
                ok.append(t)
            else:
                log.warning("%s: no bars from %s, skipped", t, source)
        except Exception as exc:
            log.warning("%s: %s load failed (%s), skipped", t, source, exc)
    return ok

