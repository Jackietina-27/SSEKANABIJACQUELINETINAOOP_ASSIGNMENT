import math
import statistics

import numpy as np
import pytest

from src.common.forecasting import Forecaster, mae, mape, rmse
from src.population import (CAGRForecaster, ClassroomPlanner, DistrictPopulation,
                            FibonacciRatioForecaster, LinearTrendForecaster,
                            ModelSelector, bootstrap_intervals, fibonacci, load_districts)


@pytest.fixture
def kampala() -> DistrictPopulation:
    return load_districts(["Kampala"])[0]


def test_repr_and_len(kampala):
    assert len(kampala) == 10
    assert "Kampala" in repr(kampala)


@pytest.mark.parametrize("years,pops", [
    ([2020, 2021], [1, 2, 3]),        # unequal lengths
    ([2020, 2021], [100, -1]),        # negative value
    ([], []),                         # empty
    ([2021, 2020], [1, 2]),           # not increasing
])
def test_invalid_input_rejected(years, pops):
    with pytest.raises(ValueError):
        DistrictPopulation("X", years, pops)


def test_variance_ddof_relationship(kampala):
    s = kampala.stats_statistics()
    assert np.isclose(s["variance"], kampala.stats_numpy(ddof=1)["variance"])
    assert np.isclose(kampala.stats_numpy(ddof=0)["variance"], s["variance"] * 9 / 10)
    assert s["mean"] == statistics.mean(kampala.populations.tolist())


def test_cagr_matches_hand_calculation(kampala):
    assert math.isclose(kampala.cagr(), 1.5 ** (1 / 9) - 1)
    assert np.allclose(kampala.growth_rates()[0], 1250 / 1200 - 1)


def test_linear_forecaster_recovers_exact_line():
    y = 3 + 2 * np.arange(6)
    pred = LinearTrendForecaster().fit(y).predict(3)
    assert np.allclose(pred, [15, 17, 19])


def test_cagr_forecaster_on_geometric_series():
    y = 100 * 1.1 ** np.arange(5)
    model = CAGRForecaster().fit(y)
    assert np.isclose(model.growth_, 0.1)
    assert np.allclose(model.predict(2), y[-1] * 1.1 ** np.array([1, 2]))


def test_fibonacci_forecaster():
    assert fibonacci(6) == [1, 1, 2, 3, 5, 8]
    assert np.allclose(FibonacciRatioForecaster().fit([10]).predict(4), [10, 20, 30, 50])


def test_predict_before_fit_and_bad_horizon():
    with pytest.raises(RuntimeError):
        LinearTrendForecaster().predict(1)
    with pytest.raises(ValueError):
        LinearTrendForecaster().fit([1, 2, 3]).predict(0)
    with pytest.raises(TypeError):
        Forecaster()  # abstract


def test_metrics():
    assert mae([1, 2], [2, 4]) == 1.5
    assert np.isclose(rmse([0, 0], [3, 4]), math.sqrt(12.5))
    with pytest.raises(ValueError):
        mape([0, 1], [1, 1])


def test_model_selector_picks_cagr_for_kampala(kampala):
    assert ModelSelector().best_model_name(kampala) == "Exponential (CAGR)"


def test_classroom_planner():
    p = ClassroomPlanner()
    # 10k extra people -> 1800 pupils -> 1800/53 = 33.96 -> 34 classrooms
    assert p.additional_classrooms(100, 110) == 34
    assert p.additional_classrooms(100, 90) == 0     # shrinking: none needed
    with pytest.raises(ValueError):
        ClassroomPlanner(pupils_per_classroom=0)


def test_bootstrap_interval_contains_point_forecast(kampala):
    t = kampala.years - kampala.years[0]
    lo, hi = bootstrap_intervals(CAGRForecaster, kampala.populations, t, 3,
                                 n_boot=200, rng=np.random.default_rng(1))
    point = CAGRForecaster().fit(kampala.populations, t).predict(3)
    assert np.all(lo <= point) and np.all(point <= hi)
