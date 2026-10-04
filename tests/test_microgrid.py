import numpy as np
import pandas as pd
import pytest

from src.microgrid import (CostModel, DemandInput, HybridMicroGrid, MicroGrid,
                           monte_carlo_sensitivity)


@pytest.fixture
def grid() -> MicroGrid:
    return MicroGrid()


def test_determinant_and_condition(grid):
    assert np.isclose(grid.determinant(), -5)
    assert grid.is_well_posed()
    assert 5 < grid.condition_number() < 6


def test_solve_day_matches_hand_formula(grid):
    d1, d2 = 130.0, 160.0
    x, y = grid.solve_day(d1, d2)
    assert np.isclose(x, (2 * d2 - d1) / 5)
    assert np.isclose(y, (4 * d1 - 3 * d2) / 5)


def test_zero_demand_gives_zero_dispatch(grid):
    assert np.allclose(grid.solve_day(0, 0), [0, 0])


@pytest.mark.parametrize("d", [(-1, 10), (np.nan, 10), (10,)])
def test_invalid_demand_rejected(grid, d):
    with pytest.raises(ValueError):
        grid.solve_day(*d)


def test_loop_equals_vectorised(grid):
    D = np.random.default_rng(0).uniform(80, 160, size=(2, 30))
    assert np.allclose(grid.solve_loop(D), grid.solve_many(D))


def test_infeasible_day_repaired_non_negative(grid):
    D = np.array([[100.0, 130.0], [200.0, 160.0]])   # day 0: D2 > 4/3 D1
    plan = grid.dispatch(D, strategy="nnls")
    assert plan["infeasible"].tolist() == [True, False]
    assert (plan[["solar", "battery"]] >= 0).all().all()
    clip = grid.dispatch(D, strategy="clip")
    assert clip.loc[0, "battery"] == 0


def test_singular_system_raises():
    dep = HybridMicroGrid([[3, 2, 1], [4, 1, 0], [7, 3, 1]])
    assert dep.rank() == 2
    with pytest.raises(np.linalg.LinAlgError):
        dep.solve_day(100, 130, 230)


def test_hybrid_recovers_known_solution():
    h = HybridMicroGrid()
    true = np.array([30.0, 10.0, 5.0])
    assert np.allclose(h.solve_day(*(h.A @ true)), true)


@pytest.mark.parametrize("text", ["", "   ", "abc", "-5", "inf", "nan"])
def test_parse_demand_rejects_bad_text(text):
    with pytest.raises(ValueError):
        DemandInput.parse_demand(text)


def test_prompt_reprompts_until_valid():
    answers = iter(["", "x", "-3", "42"])
    assert DemandInput.prompt_demand("?", input_fn=lambda _: next(answers),
                                     echo=lambda _: None) == 42.0


def test_csv_round_trip_is_reproducible(tmp_path):
    a = DemandInput.generate_csv(tmp_path / "a.csv", days=30, seed=1)
    b = DemandInput.generate_csv(tmp_path / "b.csv", days=30, seed=1)
    pd.testing.assert_frame_equal(a, b)
    loaded = DemandInput.load_csv(tmp_path / "a.csv")
    assert len(loaded) == 30 and (loaded[["D1", "D2"]] >= 0).all().all()


def test_cost_model():
    plan = pd.DataFrame({"solar": [10.0, 0.0], "battery": [2.0, 0.0]})
    cost = CostModel({"solar": 150, "battery": 450}).daily_cost(plan)
    assert cost.tolist() == [2400.0, 0.0]


def test_monte_carlo_shape(grid):
    sims = monte_carlo_sensitivity(grid, [130, 160], draws=50,
                                   rng=np.random.default_rng(0))
    assert sims.shape == (50, 2)
