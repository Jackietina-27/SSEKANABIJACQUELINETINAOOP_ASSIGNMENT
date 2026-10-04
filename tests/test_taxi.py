import numpy as np
import pytest

from src.taxi import (Backtester, FleetPlanner, LinearTrend, MarketModel,
                      MovingAverageForecaster, Route, SeasonalNaiveForecaster,
                      SESForecaster, load_routes, simulate_weekly_demand)


def test_route_revenue_and_stats():
    ntinda = load_routes()[0]
    assert ntinda.total_revenue() == 474 * 2000
    assert ntinda.daily_revenue()[0] == 70_000
    assert np.isclose(ntinda.stats()["mean"], 47.4)


@pytest.mark.parametrize("pax,fare", [([], 2000), ([10, -1], 2000), ([10, 20], 0)])
def test_route_rejects_invalid(pax, fare):
    with pytest.raises(ValueError):
        Route("X", pax, fare)


def test_equilibrium_matches_hand_solution():
    q, p = MarketModel().equilibrium()
    assert np.isclose(p, 2200) and np.isclose(q, 76)


def test_parallel_curves_have_no_equilibrium():
    with pytest.raises(np.linalg.LinAlgError):
        MarketModel(b=0.03, d=-0.03).equilibrium()


def test_moving_average():
    assert MovingAverageForecaster(3).fit([1, 2, 3, 4, 5, 6]).predict(2).tolist() == [5, 5]
    with pytest.raises(ValueError):
        MovingAverageForecaster(3).fit([1, 2])     # too few observations


def test_ses_hand_calculation_and_limits():
    # l0=10; l1=0.5*20+0.5*10=15; l2=0.5*30+0.5*15=22.5
    assert np.isclose(SESForecaster(0.5).fit([10, 20, 30]).predict(1)[0], 22.5)
    assert SESForecaster(1.0).fit([10, 20, 30]).predict(1)[0] == 30   # naive
    with pytest.raises(ValueError):
        SESForecaster(0)


def test_seasonal_naive_repeats_last_week():
    y = np.arange(14)
    assert SeasonalNaiveForecaster(7).fit(y).predict(9).tolist() == [7, 8, 9, 10, 11, 12, 13, 7, 8]


def test_backtest_uses_only_past_data():
    y = [10, 10, 10, 100]                      # jump on the last day
    f = Backtester(y, first_day=4).forecasts(lambda: MovingAverageForecaster(3))
    assert f.loc[4, "forecast"] == 10          # did not peek at day 4


def test_tune_alpha_returns_grid_value():
    alpha, curve = Backtester(load_routes()[0].passengers).tune_alpha()
    assert alpha in curve.index and curve[alpha] == curve.min()


def test_fleet_planner():
    fp = FleetPlanner()
    assert fp.capacity_per_vehicle == 112
    assert fp.vehicles(45) == 1
    assert fp.vehicles(540) == 6               # 540*1.15/112 = 5.54 -> 6
    assert fp.vehicles(0) == 0
    with pytest.raises(ValueError):
        fp.vehicles(-1)


def test_seasonal_beats_moving_average_on_weekly_data():
    y = simulate_weekly_demand(60, seed=11)
    bt = Backtester(y, first_day=15)
    assert bt.mae(SeasonalNaiveForecaster) < bt.mae(lambda: MovingAverageForecaster(3))


def test_linear_trend_reused():
    assert np.allclose(LinearTrend().fit([1, 2, 3]).predict(1), [4])
