import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import enrich_bars  # noqa: E402
from src.data.synthetic import generate_bars  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_bars() -> pd.DataFrame:
    """20 days of liquid synthetic bars with a known 10 bps spread."""
    return enrich_bars(generate_bars(n_days=20, trades_per_minute=100, mean_spread_bps=10.0, seed=7), market="US")


def make_flat_day(price: float = 100.0, minutes: int = 391, volume: float = 1000.0, date: str = "2024-03-05") -> pd.DataFrame:
    """One US session of bars with a constant price (bar 0 = opening auction)."""
    ts = pd.date_range(f"{date} 09:30", periods=minutes, freq="1min", tz="America/New_York")
    df = pd.DataFrame({
        "timestamp": ts, "open": price, "high": price, "low": price, "close": price,
        "volume": volume, "bid": np.nan, "ask": np.nan,
    })
    return enrich_bars(df, market="US")


def make_daily(n: int = 80, price: float = 100.0, volume: float = 1_000_000.0, end: str = "2024-03-05", seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=n)
    close = price * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    close[-1] = price
    return pd.DataFrame({"date": dates, "open": close, "high": close, "low": close, "close": close, "volume": volume})
