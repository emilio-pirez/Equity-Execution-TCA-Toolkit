import numpy as np
import pytest

from src.execution import algos
from src.execution.impact import participation_impact_bps, pretrade_cost_bps, sqrt_impact_bps


@pytest.mark.parametrize("qty,n", [(1000, 390), (7, 390), (123_457, 331), (390, 390)])
def test_twap_sums_to_parent_and_is_even(qty, n):
    q = algos.twap(n, qty)
    assert q.sum() == qty
    assert q.max() - q.min() <= 1
    assert (q >= 0).all()


def test_vwap_sums_and_follows_curve():
    curve = np.array([5.0, 1.0, 1.0, 1.0, 2.0])
    q = algos.vwap(curve, 10_000)
    assert q.sum() == 10_000
    assert np.allclose(q / q.sum(), curve / curve.sum(), atol=1e-3)


def test_vwap_zero_curve_falls_back_to_twap():
    q = algos.vwap(np.zeros(4), 100)
    assert q.tolist() == [25, 25, 25, 25]


def test_pov_respects_rate_and_cap():
    v = np.array([1000.0, 0.0, 2000.0, 500.0])
    q = algos.pov(v, 0.20, 10_000)
    # q = 0.2/0.8 * V -> participation q/(q+V) = 20%
    assert q.tolist() == [250, 0, 500, 125]
    capped = algos.pov(v, 0.20, 600)
    assert capped.sum() == 600 and capped.tolist() == [250, 0, 350, 0]


def test_ac_holdings_limits():
    x0 = algos.ac_holdings(10, 0.0)
    assert np.allclose(x0, np.linspace(1, 0, 11))  # TWAP straight line
    x = algos.ac_holdings(10, 0.5)
    assert x[0] == pytest.approx(1.0) and x[-1] == pytest.approx(0.0)
    assert np.all(np.diff(x) < 0)
    assert x[5] < x0[5]  # front-loaded
    assert np.isfinite(algos.ac_holdings(400, 5.0)).all()  # stable for large kappa*N


def test_is_schedule_front_loaded_and_sums():
    q = algos.implementation_shortfall(100, 50_000, urgency=3.0)
    assert q.sum() == 50_000
    assert q[:20].sum() > 0.4 * 50_000


def test_ac_frontier_monotone():
    fr = algos.ac_frontier(100, sigma_bar_bps=5.0, eta_bps=500.0)
    cost = np.array([f["cost_bps"] for f in fr])
    risk = np.array([f["risk_bps"] for f in fr])
    assert np.all(np.diff(cost) >= -1e-9)  # more risk-averse -> more cost
    assert np.all(np.diff(risk) <= 1e-9)  # ... and less risk
    assert cost[0] == pytest.approx(500.0 / 100)  # TWAP: eta * N * (1/N)^2


def test_sqrt_impact_hand_computed():
    # k=0.7, sigma=150 bps, Q/ADV = 4% -> 0.7 * 150 * 0.2 = 21 bps
    assert sqrt_impact_bps(40_000, 1_000_000, 150.0, 0.7) == pytest.approx(21.0)
    c = pretrade_cost_bps(40_000, 1_000_000, 150.0, 6.0, 0.7)
    assert c["total_bps"] == pytest.approx(24.0)


def test_participation_impact_matches_pretrade_at_constant_rate():
    rho = 0.04
    assert participation_impact_bps(rho, 150.0, 0.7) == pytest.approx(sqrt_impact_bps(4, 100, 150.0, 0.7))
    assert participation_impact_bps(0.0, 150.0) == 0.0
