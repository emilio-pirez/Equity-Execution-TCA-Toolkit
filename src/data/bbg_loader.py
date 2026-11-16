"""Bloomberg CSV loader (validation data only; never committed - see .gitignore).

Expected file: data/raw/bbg/<STEM>_1min.csv, where STEM is the first token of the
Bloomberg ticker without '*' (e.g. WALMEX, GMEXICOB, AAPL). Columns (case-insensitive):

    timestamp, open, high, low, close, volume, bid, ask

`timestamp` is the bar start in exchange local time (no offset) or ISO with offset.
`bid`/`ask` are the last bid/ask quote of the minute (close of BarTp=Bid / BarTp=Ask
bars in the Excel add-in). See docs/bloomberg_extraction.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BBG_DIR, instrument
from src.data.yf_loader import BAR_COLUMNS

ALIASES = {
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "last_price": "close",
    "px_last": "close",
    "px_bid": "bid",
    "px_ask": "ask",
}


def bbg_path(ticker: str, bbg_dir: Path = BBG_DIR) -> Path:
    return bbg_dir / f"{instrument(ticker).bbg_stem}_1min.csv"


def has_bbg(ticker: str, bbg_dir: Path = BBG_DIR) -> bool:
    return bbg_path(ticker, bbg_dir).exists()


def load_bbg_1m(ticker: str, bbg_dir: Path = BBG_DIR) -> pd.DataFrame:
    """Read a Bloomberg 1-minute CSV into the common bar schema (local exchange time)."""
    path = bbg_path(ticker, bbg_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No Bloomberg file for {ticker} at {path}. Bloomberg validation is optional; "
            "see docs/bloomberg_extraction.md."
        )
    df = pd.read_csv(path)
    df.columns = [ALIASES.get(c.strip().lower(), c.strip().lower()) for c in df.columns]
    missing = {"timestamp", "open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    for col in ("bid", "ask"):
        if col not in df:
            df[col] = np.nan
    tz = instrument(ticker).tz
    ts = pd.to_datetime(df["timestamp"])
    df["timestamp"] = ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)
    num = ["open", "high", "low", "close", "volume", "bid", "ask"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    return df[BAR_COLUMNS].reset_index(drop=True)
