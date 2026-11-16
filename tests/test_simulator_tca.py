import numpy as np
import pandas as pd
import pytest

from src.execution.simulator import TickerContext, build_day_grid, run_grid, run_order
from src.tca.analysis import decompose, interval_vwap, slippage_bps
from tests.conftest import make_daily, make_flat_day


def _ctx(bars, spread=0.0, k=0.7):
    daily = make_daily(end=str(bars["date"].max().date()))
    sp = pd.Series(spread, index=range(0, 400, 30))
    return TickerContext("AAPL", bars, daily, sp, k=k)


def test_slippage_sign_convention():
    assert slippage_bps(101.0, 100.0, +1) == pytest.approx(100.0)  # buy above benchmark = cost
    assert slippage_bps(101.0, 100.0, -1) == pytest.approx(-100.0)  # sell above benchmark = gain
    assert interval_vwap(np.array([10.0, 20.0]), np.array([1.0, 3.0])) == pytest.approx(17.5)


def test_decomposition_identity():
    rng = np.random.default_rng(0)
    q = rng.integers(0, 100, 50).astype(float)
    ref = 100 + rng.normal(0, 0.5, 50)
    hs, imp = rng.uniform(0, 5, 50), rng.uniform(0, 20, 50)
    for side in (1, -1):
        parent = int(q.sum() + 300)
        d = decompose(q, ref, hs, imp, side, 100.0, parent, 101.0, cleanup_cost_bps=12.0)
        parts = d["timing_bps"] + d["spread_bps"] + d["impact_bps"] + d["opportunity_bps"]
        assert parts == pytest.approx(d["is_total_bps"])
        fill = ref * (1 + side * (hs + imp) / 1e4)
        avg = (q * fill).sum() / q.sum()
        assert d["is_filled_bps"] == pytest.approx(slippage_bps(avg, 100.0, side))


def test_vwap_on_flat_price_has_zero_slippage_without_costs():
    """Flat prices, zero spread, k = 0 -> every benchmark slippage is exactly 0."""
    bars = make_flat_day()
    ctx = _ctx(bars, spread=0.0, k=0.0)
    for algo, p in (("vwap", {}), ("twap", {}), ("is", {"urgency": 3.0}), ("pov", {"rate": 0.1})):
        s, _ = run_order(ctx, bars["date"].iloc[0], algo, 0.01, +1, **p)
        assert s["vs_arrival_bps"] == pytest.approx(0.0, abs=1e-9)
        assert s["vs_vwap_bps"] == pytest.approx(0.0, abs=1e-9)
        assert s["is_total_bps"] == pytest.approx(0.0, abs=1e-9)


def test_flat_price_cost_equals_half_spread_plus_impact():
    bars = make_flat_day(volume=1000.0)
    ctx = _ctx(bars, spread=4.0, k=0.7)
    s, fills = run_order(ctx, bars["date"].iloc[0], "vwap", 0.01, +1)
    traded = fills[fills["qty"] > 0]
    expected_impact = np.average(traded["impact_bps"], weights=traded["qty"])
    assert s["spread_bps"] == pytest.approx(2.0)
    assert s["impact_bps"] == pytest.approx(expected_impact)
    assert s["timing_bps"] == pytest.approx(0.0, abs=1e-9)
    assert s["is_total_bps"] == pytest.approx(2.0 + expected_impact)


def test_faster_execution_costs_more_impact():
    bars = make_flat_day(volume=5000.0)
    ctx = _ctx(bars, spread=0.0)
    d = bars["date"].iloc[0]
    slow, _ = run_order(ctx, d, "pov", 0.05, +1, rate=0.05)
    fast, _ = run_order(ctx, d, "pov", 0.05, +1, rate=0.20)
    assert fast["impact_bps"] > slow["impact_bps"]
    assert fast["duration_min"] < slow["duration_min"]


def test_grid_fills_missing_minutes():
    bars = make_flat_day()  # 09:30 auction bar + 09:31..15:59 continuous + 16:00 closing-auction bar
    bars = bars.drop(index=[10, 11, 12]).reset_index(drop=True)
    g = build_day_grid(bars, bars["date"].iloc[0])
    assert len(g) == 389 and g["minute_of_session"].tolist() == list(range(1, 390))
    assert g.loc[g["minute_of_session"] == 11, "volume"].iloc[0] == 0


def test_simulation_deterministic(synthetic_bars):
    daily = make_daily(end=str(synthetic_bars["date"].max().date()))
    ctx = TickerContext("AAPL", synthetic_bars, daily, pd.Series(10.0, index=range(0, 400, 30)))
    dates = ctx.dates[:2]
    a, _ = run_grid(ctx, sizes=(0.05,), dates=dates)
    b, _ = run_grid(ctx, sizes=(0.05,), dates=dates)
    pd.testing.assert_frame_equal(a, b)
    assert (a["fill_rate"] <= 1.0 + 1e-12).all()
