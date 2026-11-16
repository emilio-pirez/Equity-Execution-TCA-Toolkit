"""Universe, market sessions and project-wide constants.

Every other module reads tickers, time zones and session times from here so that
there is exactly one place to change them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
BBG_DIR = DATA_DIR / "raw" / "bbg"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "reports" / "figures"
DB_PATH = DATA_DIR / "tca.db"

BPS = 1e4  # 1 = 10,000 bps


@dataclass(frozen=True)
class Market:
    """Exchange session definition (all times in exchange local time)."""

    code: str
    tz: str
    open: time  # start of continuous trading
    close: time  # end of continuous trading (official)
    currency: str


MARKETS: dict[str, Market] = {
    # BMV: continuous session 08:30-15:00 CDMX while the US is on standard time
    # (Nov-Mar); 07:30-14:00 while the US is on daylight time. Mexico has no DST.
    "MX": Market("MX", "America/Mexico_City", time(8, 30), time(15, 0), "MXN"),
    "US": Market("US", "America/New_York", time(9, 30), time(16, 0), "USD"),
}


@dataclass(frozen=True)
class Instrument:
    name: str
    yf: str
    bbg: str
    market: str
    asset_type: str  # "equity" | "etf"

    @property
    def tz(self) -> str:
        return MARKETS[self.market].tz

    @property
    def bbg_stem(self) -> str:
        """File stem used for Bloomberg CSV exports: data/raw/bbg/<stem>_1min.csv."""
        return self.bbg.split()[0].replace("*", "")


UNIVERSE: list[Instrument] = [
    Instrument("Walmex", "WALMEX.MX", "WALMEX* MF Equity", "MX", "equity"),
    Instrument("Grupo México", "GMEXICOB.MX", "GMEXICOB MF Equity", "MX", "equity"),
    Instrument("América Móvil", "AMXB.MX", "AMXB MF Equity", "MX", "equity"),
    Instrument("Banorte", "GFNORTEO.MX", "GFNORTEO MF Equity", "MX", "equity"),
    Instrument("NAFTRAC", "NAFTRACISHRS.MX", "NAFTRACISHRS MF Equity", "MX", "etf"),
    Instrument("IVV", "IVV", "IVV US Equity", "US", "etf"),
    Instrument("Apple", "AAPL", "AAPL US Equity", "US", "equity"),
]

BY_TICKER: dict[str, Instrument] = {i.yf: i for i in UNIVERSE}
FX_TICKER = "MXN=X"  # USD/MXN, MXN per 1 USD

# Analysis parameters (documented in README "Methodology")
ADV_WINDOW_DAYS = 20
VOL_WINDOW_DAYS = 60
PROFILE_BUCKET_MIN = 30
ORDER_SIZES_PCT_ADV = (0.01, 0.05, 0.10, 0.20)
IMPACT_K = 0.7  # square-root impact coefficient (sensitivity: 0.5 / 1.0)
PARTICIPATION_WINDOW_MIN = 5  # rolling window for participation in the impact model
DEFAULT_SEED = 42


def session_times(market: str, day) -> tuple[time, time]:
    """Continuous-session (open, close) in local time for a given date.

    BMV aligns with NYSE: when New York is on daylight time (UTC-4) the BMV session
    is 07:30-14:00 CDMX, otherwise 08:30-15:00 (Mexico abolished DST in 2022).
    """
    import pandas as pd

    m = MARKETS[market]
    if market != "MX":
        return m.open, m.close
    ny_noon = pd.Timestamp(day).tz_localize(None).replace(hour=12).tz_localize("America/New_York")
    if ny_noon.utcoffset() == pd.Timedelta(hours=-4):
        return time(7, 30), time(14, 0)
    return m.open, m.close


def instrument(ticker: str) -> Instrument:
    """Look up an instrument by yfinance ticker (raises KeyError with a clear message)."""
    try:
        return BY_TICKER[ticker]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"{ticker!r} not in universe: {sorted(BY_TICKER)}") from exc
