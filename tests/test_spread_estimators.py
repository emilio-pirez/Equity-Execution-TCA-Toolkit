import numpy as np
import pandas as pd
import pytest

from src.liquidity.spread_estimators import (
    compare, corwin_schultz_bps, corwin_schultz_pairs, quoted_spread_bps, roll_bps, roll_from_returns,
    validate_estimators,
)


def test_quoted_spread_hand_computed():
    bid, ask = pd.Series([99.95, 10.0]), pd.Series([100.05, 9.0])
    s = quoted_spread_bps(bid, ask)
    assert s.iloc[0] == pytest.approx(0.10 / 100.0 * 1e4)  # 10 bps
    assert np.isnan(s.iloc[1])  # crossed quote -> NaN


def _bars(h, l, c, date="2024-01-02"):
    n = len(h)
    ts = pd.date_range(f"{date} 10:00", periods=n, freq="1min", tz="America/New_York")
    return pd.DataFrame({"timestamp": ts, "open": c, "high": h, "low": l, "close": c, "volume": 1.0,
                         "date": pd.Timestamp(date)})


def test_corwin_schultz_hand_computed():
    # Two bars: H/L = 101/99 and 101.5/99.5, close of bar 1 = 100 (no gap).
    b = _bars([101.0, 101.5], [99.0, 99.5], [100.0, 100.5])
    beta = np.log(101 / 99) ** 2 + np.log(101.5 / 99.5) ** 2
    gamma = np.log(101.5 / 99) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    expected = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    got = corwin_schultz_pairs(b)
    assert len(got) == 1
    assert got.iloc[0] == pytest.approx(expected)


def test_corwin_schultz_constant_range_no_vol_gives_range():
    # No price movement, bid/ask bounce only: H = ask, L = bid in every bar -> S = (H-L)/mid approx.
    ask, bid = 100.05, 99.95
    b = _bars([ask] * 50, [bid] * 50, [ask, bid] * 25)
    s = corwin_schultz_bps(b, by=None)
    assert s == pytest.approx(10.0, rel=0.01)


def test_pairs_do_not_cross_days():
    b1 = _bars([101, 101], [99, 99], [100, 100], "2024-01-02")
    b2 = _bars([101, 101], [99, 99], [100, 100], "2024-01-03")
    s = corwin_schultz_pairs(pd.concat([b1, b2], ignore_index=True))
    assert len(s) == 2  # one pair per day, none across the overnight


def test_negative_estimates_clipped():
    # Large trend + tiny ranges makes alpha < 0 -> clipped to 0 under mean_of_clipped.
    b = _bars([100.01, 110.01], [100.0, 110.0], [100.0, 110.0])
    assert corwin_schultz_pairs(b, adjust_gaps=False).iloc[0] < 0
    assert corwin_schultz_bps(b, by=None, adjust_gaps=False) == 0.0


def test_roll_known_covariance():
    # Pure bid-ask bounce: returns alternate +s, -s -> cov = -s^2 -> spread = 2s.
    r = np.array([0.001, -0.001] * 100)
    assert roll_from_returns(r[1:], r[:-1]) == pytest.approx(0.002, rel=0.02)
    assert roll_from_returns(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.1, 0.2, 0.3, 0.4])) == 0.0


def test_estimators_recover_synthetic_spread(synthetic_bars):
    """With 100 trades/min and a 10 bps true spread both estimators are within 15% on average."""
    v = validate_estimators(synthetic_bars, ref_col="true_spread_bps")["summary"]
    daily = v[v["level"] == "daily"].set_index("estimator")
    assert daily.loc["cs_bps", "ratio"] == pytest.approx(1.0, abs=0.15)
    assert daily.loc["roll_bps", "ratio"] == pytest.approx(1.0, abs=0.15)
    tod = v[v["level"] == "time_of_day"].set_index("estimator")
    assert tod.loc["cs_bps", "corr"] > 0.8  # tracks the U-shaped intraday spread


def test_roll_by_day_matches_pooled_sign(synthetic_bars):
    assert (roll_bps(synthetic_bars, by="date") >= 0).all()


def test_compare_metrics():
    est, ref = pd.Series([2.0, 4.0, 6.0]), pd.Series([1.0, 3.0, 5.0])
    c = compare(est, ref)
    assert c["bias_bps"] == pytest.approx(1.0)
    assert c["mae_bps"] == pytest.approx(1.0)
    assert c["corr"] == pytest.approx(1.0)
    assert c["ratio"] == pytest.approx(4.0 / 3.0)
