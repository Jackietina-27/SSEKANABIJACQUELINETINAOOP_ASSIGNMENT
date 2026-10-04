import numpy as np
import pytest

from src.fishery import (FishStock, FisheryScenario, PriceModel, RiskAssessor,
                         closed_season_mask)


def test_first_step_matches_hand_calculation():
    fs = FishStock(r=0.4, K=10_000, N0=4_000, h=0.1)
    assert np.isclose(fs.simulate(1).stock[1], 4000 + 0.4 * 4000 * 0.6 - 400)


@pytest.mark.parametrize("h", [0.05, 0.10, 0.20, 0.30])
def test_converges_to_theoretical_equilibrium(h):
    fs = FishStock(h=h)
    assert np.isclose(fs.simulate(300).final_stock, fs.equilibrium_stock, rtol=1e-3)


def test_msy_at_half_r():
    fs = FishStock(h=0.2)
    assert fs.msy == 1000 and np.isclose(fs.equilibrium_yield, fs.msy)


def test_zero_initial_stock_stays_zero():
    res = FishStock(N0=0).simulate(52)
    assert res.stock.max() == 0 and res.total_harvest == 0


def test_overharvest_collapses_and_never_negative():
    res = FishStock(h=0.9).simulate(200)
    assert res.stock.min() >= 0 and res.final_stock < 1e-6


@pytest.mark.parametrize("kwargs", [dict(r=0), dict(K=-1), dict(N0=-5), dict(h=1.5)])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        FishStock(**kwargs)


def test_price_path_bounded_and_reproducible():
    a = PriceModel(seed=3).simulate_paths(100, 50)
    b = PriceModel(seed=3).simulate_paths(100, 50)
    assert np.array_equal(a, b)
    assert a.min() >= 9000 and a.max() <= 16000 and np.all(a[:, 0] == 12000)


def test_cv_is_unit_free_and_classification():
    ra = RiskAssessor()
    rev = np.array([100.0, 120, 80, 110, 90])
    assert np.isclose(ra.describe(rev)["cv"], ra.describe(rev / 3700)["cv"])
    assert ra.classify_cv(0.05) == "Low"
    assert ra.classify_cv(0.15) == "Moderate"
    assert ra.classify_cv(0.30) == "High"
    with pytest.raises(ValueError):
        ra.describe([1.0])      # too few values


def test_value_at_risk():
    v = RiskAssessor.value_at_risk(np.arange(1, 101))
    assert np.isclose(v["quantile"], np.percentile(np.arange(1, 101), 5))
    assert np.isclose(v["VaR"], 50.5 - v["quantile"])
    with pytest.raises(ValueError):
        RiskAssessor.value_at_risk([])


def test_revenue_is_harvest_times_price():
    sc = FisheryScenario(FishStock(h=0.1), PriceModel(sigma=0), weeks=10)
    out = sc.run()
    assert np.allclose(out["revenue"], out["sim"].harvest * 1000 * 12000)


def test_closed_season_mask():
    m = closed_season_mask(2, closed_weeks=8, start_week=13)
    assert m.size == 104 and (~m).sum() == 16 and not m[13] and m[12]
