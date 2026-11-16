"""Yahoo Finance loader: chunked 1-minute download with a parquet cache.

yfinance serves 1-minute bars for at most ~7 days per request and only for the
last ~30 days. We therefore request 7-day windows, concatenate, de-duplicate,
sort and convert to exchange local time. The cache is *merged* on refresh, so
re-running the downloader every few weeks accumulates history beyond 30 days.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, FX_TICKER, UNIVERSE, instrument

log = logging.getLogger(__name__)

BAR_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "bid", "ask"]
MAX_LOOKBACK_DAYS = 29  # stay inside Yahoo's 30-day limit for 1m data
CHUNK_DAYS = 7


def cache_path(ticker: str, interval: str = "1m", cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{ticker}_{interval}.parquet"


def _normalize(raw: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Map a yfinance history frame to the common bar schema."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    df.index = idx.tz_convert(tz)
    df.index.name = "timestamp"
    df = df.reset_index()
    df["volume"] = df["volume"].astype("float64")
    df["bid"] = np.nan
    df["ask"] = np.nan
    return df[BAR_COLUMNS]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
    return df.reset_index(drop=True)


def download_1m(
    ticker: str,
    lookback_days: int = MAX_LOOKBACK_DAYS,
    end: datetime | None = None,
    chunk_days: int = CHUNK_DAYS,
) -> pd.DataFrame:
    """Download 1-minute bars in `chunk_days` windows covering `lookback_days`.

    Returns the common bar schema in exchange local time; empty frame on failure.
    """
    import yfinance as yf

    tz = instrument(ticker).tz if ticker != FX_TICKER else "UTC"
    end = end or datetime.now(timezone.utc)
    start = end - timedelta(days=min(lookback_days, MAX_LOOKBACK_DAYS))
    frames = []
    t0 = start
    while t0 < end:
        t1 = min(t0 + timedelta(days=chunk_days), end)
        try:
            raw = yf.Ticker(ticker).history(
                start=t0, end=t1, interval="1m", auto_adjust=False, prepost=False, actions=False
            )
            frames.append(_normalize(raw, tz))
        except Exception as exc:  # network / parsing: log and continue
            log.warning("%s: chunk %s-%s failed: %s", ticker, t0.date(), t1.date(), exc)
        t0 = t1
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=BAR_COLUMNS)
    return _clean(pd.concat(frames, ignore_index=True))


def download_daily(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Daily OHLCV (official close and full-day volume incl. auctions)."""
    import yfinance as yf

    raw = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False, actions=False)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df.reset_index()


def _merge_cache(path: Path, new: pd.DataFrame, key: str) -> pd.DataFrame:
    if path.exists():
        old = pd.read_parquet(path)
        if key == "timestamp" and not new.empty:
            old["timestamp"] = old["timestamp"].dt.tz_convert(new["timestamp"].dt.tz)
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(subset=key, keep="last").sort_values(key).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    return new


def load_yf_1m(ticker: str, refresh: bool = False, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Cached 1-minute bars. Downloads only if the cache is missing or `refresh`."""
    path = cache_path(ticker, "1m", cache_dir)
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
    else:
        new = download_1m(ticker)
        if new.empty:
            log.warning("%s: no 1m data returned by yfinance", ticker)
            if not path.exists():
                return pd.DataFrame(columns=BAR_COLUMNS)
        df = _merge_cache(path, new, "timestamp")
    tz = instrument(ticker).tz
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(tz)
    return df


def load_yf_daily(ticker: str, refresh: bool = False, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Cached daily bars (also used for USD/MXN via `FX_TICKER`)."""
    path = cache_path(ticker, "1d", cache_dir)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    new = download_daily(ticker)
    if new.empty and not path.exists():
        log.warning("%s: no daily data returned by yfinance", ticker)
        return new
    return _merge_cache(path, new, "date")


def refresh_universe(tickers: list[str] | None = None) -> dict[str, int]:
    """Download/refresh every ticker; failures are logged and skipped.

    Returns {ticker: number of 1m bars in cache} (0 = failed).
    """
    status: dict[str, int] = {}
    for t in tickers or [i.yf for i in UNIVERSE]:
        try:
            bars = load_yf_1m(t, refresh=True)
            load_yf_daily(t, refresh=True)
            status[t] = len(bars)
            log.info("%s: %d 1m bars cached", t, len(bars))
        except Exception as exc:
            log.error("%s: download failed, skipping: %s", t, exc)
            status[t] = 0
    try:
        load_yf_daily(FX_TICKER, refresh=True)
    except Exception as exc:
        log.error("FX download failed: %s", exc)
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(refresh_universe())
