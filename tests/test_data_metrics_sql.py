from datetime import time

import numpy as np
import pandas as pd
import pytest

from src.config import session_times
from src.data import db
from src.data.bbg_loader import load_bbg_1m
from src.data.loaders import enrich_bars, load_bars
from src.data.synthetic import generate_bars, u_shape
from src.liquidity.metrics import adv, amihud_illiquidity, daily_volatility, volume_profile
from src.tca.analysis import cost_table
from tests.conftest import make_daily, make_flat_day


def test_synthetic_schema_and_determinism():
    a = generate_bars(n_days=2, seed=3)
    b = generate_bars(n_days=2, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert {"timestamp", "open", "high", "low", "close", "volume", "bid", "ask", "true_spread_bps"} <= set(a.columns)
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a["ask"] > a["bid"]).all()
    assert u_shape(391, 0.4).mean() == pytest.approx(1.0, abs=5e-3)  # exact mean 1 in the continuum


def test_load_bars_synthetic_and_bad_source():
    df = load_bars("X", "synthetic", n_days=1)
    assert list(df.columns[:8]) == ["timestamp", "open", "high", "low", "close", "volume", "bid", "ask"]
    with pytest.raises(ValueError):
        load_bars("X", "nope")


def test_bmv_session_follows_us_dst():
    assert session_times("MX", "2026-10-20") == (time(7, 30), time(14, 0))  # US on daylight time
    assert session_times("MX", "2026-11-10") == (time(8, 30), time(15, 0))  # US on standard time
    assert session_times("US", "2026-11-10") == (time(9, 30), time(16, 0))


def test_enrich_flags_auction_and_outliers():
    bars = make_flat_day(minutes=60)
    bars = bars[["timestamp", "open", "high", "low", "close", "volume", "bid", "ask"]].copy()
    bars.loc[30, "volume"] = 1000.0 * 100  # a block print
    e = enrich_bars(bars, market="US")
    assert e["is_open_auction"].sum() == 1 and e["is_open_auction"].iloc[0]
    assert e.loc[30, "vol_outlier"] and e.loc[30, "exec_volume"] == 0
    assert e["exec_volume"].iloc[0] == 0  # auction bar not executable
    assert e["minute_of_session"].iloc[0] == 0


def test_bbg_loader_missing_and_present(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bbg_1m("AAPL", bbg_dir=tmp_path)
    pd.DataFrame({
        "Date": ["2026-11-10 09:30", "2026-11-10 09:31"], "Open": [1, 2], "High": [1, 2], "Low": [1, 2],
        "LAST_PRICE": [1, 2], "Volume": [10, 20], "Bid": [0.99, 1.99], "Ask": [1.01, 2.01],
    }).to_csv(tmp_path / "AAPL_1min.csv", index=False)
    df = load_bbg_1m("AAPL", bbg_dir=tmp_path)
    assert str(df["timestamp"].dt.tz) == "America/New_York"
    assert df["close"].tolist() == [1, 2] and df["ask"].iloc[1] == 2.01


def test_adv_and_vol_hand_computed():
    d = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=4), "close": [100, 110, 99, 99.0],
                      "volume": [10, 20, 30, 40.0]})
    assert adv(d, window=2) == 35.0
    assert adv(d, window=2, asof="2024-01-03") == 25.0
    r = np.log([110 / 100, 99 / 110, 1.0])
    assert daily_volatility(d) == pytest.approx(np.std(r, ddof=1))
    amihud = amihud_illiquidity(d, d["close"] * d["volume"], window=3)
    expected = np.mean(np.abs(r) * 1e4 / (d["close"].iloc[1:] * d["volume"].iloc[1:] / 1e6))
    assert amihud == pytest.approx(expected)


def test_volume_profile_sums_to_one(synthetic_bars):
    p = volume_profile(synthetic_bars)
    assert p.sum() == pytest.approx(1.0)
    assert p.iloc[0] > p.iloc[len(p) // 2]  # U-shape: open busier than midday


def test_sql_queries_run(tmp_path, synthetic_bars):
    """Every query in sql/queries runs against a DB built from synthetic data."""
    con = db.connect(tmp_path / "t.db")
    db.init_db(con)
    db.insert_df(con, "bars_1m", db.bars_to_db_frame(synthetic_bars.head(2000), "AAPL", "synthetic"))
    orders = pd.DataFrame({
        "ticker": "AAPL", "date": "2024-01-02", "algo": ["VWAP", "TWAP"] * 4, "pct_adv": [0.05] * 4 + [0.10] * 4,
        "side": [1, 1, -1, -1] * 2, "qty": 100, "is_total_bps": np.arange(8.0), "spread_bps": 1.0,
        "impact_bps": np.arange(8.0) - 1.0, "timing_bps": 0.0, "opportunity_bps": 0.0, "fill_rate": 1.0,
        "vs_arrival_bps": 1.0, "vs_vwap_bps": 0.5, "vs_close_bps": 2.0, "duration_min": 390, "avg_participation": 0.05,
    })
    db.insert_df(con, "sim_orders", orders.rename_axis("order_id").reset_index())
    db.insert_df(con, "liquidity_summary", pd.DataFrame([{"ticker": "AAPL", "market": "US", "adv_usd_m": 1.0}]))
    for q in sorted(db.QUERIES.glob("*.sql")):
        out = db.run_query(con, q.stem)
        assert isinstance(out, pd.DataFrame), q.name
    dec = db.run_query(con, "05_cost_decomposition")
    assert dec["identity_check"].abs().max() < 1e-6
    assert len(cost_table(orders)) == 4
