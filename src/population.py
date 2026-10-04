"""Mini-Project 1: UBOS District Population Forecaster.

Domain classes:

* :class:`DistrictPopulation` - validated container for one district's series.
* :class:`LinearTrendForecaster`, :class:`CAGRForecaster`,
  :class:`FibonacciRatioForecaster` - forecasting models (subclasses of
  :class:`src.common.forecasting.Forecaster`).
* :class:`ModelSelector` - train/test validation and model selection.
* :class:`ClassroomPlanner` - converts population growth into classrooms.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from src.common.forecasting import Forecaster, mae, mape, rmse

# Illustrative data (thousands of people), 2015-2024.
YEARS = list(range(2015, 2025))
DISTRICT_DATA: dict[str, list[float]] = {
    "Kampala": [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800],
    "Wakiso": [950, 1000, 1070, 1150, 1220, 1300, 1390, 1480, 1570, 1670],
    "Gulu": [320, 330, 345, 360, 375, 390, 410, 430, 455, 480],
    # Two additional districts (illustrative, chosen by the author).
    "Mbarara": [470, 482, 495, 509, 523, 538, 553, 569, 586, 603],
    "Mukono": [600, 625, 652, 680, 710, 741, 773, 807, 842, 879],
}


class DistrictPopulation:
    """A district's population time series (values in thousands of people).

    Parameters
    ----------
    name : str
        District name.
    years : array-like of int
        Calendar years, strictly increasing.
    populations : array-like of float
        Population estimates in thousands; must be non-negative.
    """

    def __init__(self, name: str, years: ArrayLike, populations: ArrayLike) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")
        yrs = np.asarray(years, dtype=int)
        pops = np.asarray(populations, dtype=float)
        if yrs.ndim != 1 or pops.ndim != 1:
            raise ValueError("years and populations must be 1-D")
        if yrs.size == 0:
            raise ValueError("series must not be empty")
        if yrs.size != pops.size:
            raise ValueError(
                f"years ({yrs.size}) and populations ({pops.size}) differ in length"
            )
        if np.any(pops < 0) or not np.all(np.isfinite(pops)):
            raise ValueError("populations must be finite and non-negative")
        if yrs.size > 1 and np.any(np.diff(yrs) <= 0):
            raise ValueError("years must be strictly increasing")
        self.name = name
        self.years = yrs
        self.populations = pops

    # ----- dunder methods ---------------------------------------------
    def __repr__(self) -> str:
        return (f"DistrictPopulation(name={self.name!r}, years={self.years[0]}-"
                f"{self.years[-1]}, n={len(self)}, latest={self.populations[-1]:,.0f}k)")

    def __len__(self) -> int:
        return int(self.years.size)

    # ----- descriptive statistics --------------------------------------
    def stats_statistics(self) -> dict[str, float]:
        """Mean, median, (sample) variance and std using the ``statistics`` module."""
        data = self.populations.tolist()
        if len(data) < 2:
            raise ValueError("need at least two observations for variance")
        return {
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "variance": statistics.variance(data),   # divides by n-1
            "stdev": statistics.stdev(data),
        }

    def stats_numpy(self, ddof: int = 0) -> dict[str, float]:
        """Same statistics with NumPy. ``ddof=0`` (default) is population variance."""
        p = self.populations
        return {
            "mean": float(np.mean(p)),
            "median": float(np.median(p)),
            "variance": float(np.var(p, ddof=ddof)),  # divides by n-ddof
            "stdev": float(np.std(p, ddof=ddof)),
        }

    # ----- growth -----------------------------------------------------
    def growth_rates(self) -> np.ndarray:
        """Year-on-year growth rates (fractions), length ``len(self) - 1``."""
        p = self.populations
        if np.any(p[:-1] == 0):
            raise ValueError("growth rate undefined when a population is zero")
        return p[1:] / p[:-1] - 1

    def cagr(self) -> float:
        """Compound annual growth rate between the first and last year."""
        n_periods = self.years[-1] - self.years[0]
        if n_periods <= 0:
            raise ValueError("CAGR needs at least two years")
        if self.populations[0] == 0:
            raise ValueError("CAGR undefined when the first value is zero")
        return float((self.populations[-1] / self.populations[0]) ** (1 / n_periods) - 1)

    def split(self, last_train_year: int) -> tuple["DistrictPopulation", "DistrictPopulation"]:
        """Split into (train, test) at ``last_train_year`` (inclusive in train)."""
        mask = self.years <= last_train_year
        if mask.all() or not mask.any():
            raise ValueError("split year leaves train or test empty")
        return (DistrictPopulation(self.name, self.years[mask], self.populations[mask]),
                DistrictPopulation(self.name, self.years[~mask], self.populations[~mask]))


# ----------------------------------------------------------------------
# Forecasting models
# ----------------------------------------------------------------------
class LinearTrendForecaster(Forecaster):
    """Straight-line trend ``y = a + b t`` fitted by least squares (``np.polyfit``)."""

    min_obs = 2

    def _fit(self) -> None:
        self.slope_, self.intercept_ = np.polyfit(self.t_, self.y_, deg=1)

    def _future_t(self, horizon: int) -> np.ndarray:
        step = self.t_[-1] - self.t_[-2]
        return self.t_[-1] + step * np.arange(1, horizon + 1)

    def _predict(self, horizon: int) -> np.ndarray:
        return self.intercept_ + self.slope_ * self._future_t(horizon)

    def _fitted(self) -> np.ndarray:
        return self.intercept_ + self.slope_ * self.t_

    @property
    def name(self) -> str:
        return "Linear trend"


class CAGRForecaster(Forecaster):
    """Exponential growth at the compound annual growth rate of the training data.

    ``y_hat(t) = y_0 (1 + g)^t`` in-sample and ``y_last (1 + g)^h`` out-of-sample,
    so forecasts continue from the latest observation.
    """

    min_obs = 2

    def _fit(self) -> None:
        if self.y_[0] <= 0:
            raise ValueError("CAGR model needs a positive first value")
        n_periods = self.t_[-1] - self.t_[0]
        self.growth_ = (self.y_[-1] / self.y_[0]) ** (1 / n_periods) - 1

    def _predict(self, horizon: int) -> np.ndarray:
        return self.y_[-1] * (1 + self.growth_) ** np.arange(1, horizon + 1)

    def _fitted(self) -> np.ndarray:
        return self.y_[0] * (1 + self.growth_) ** (self.t_ - self.t_[0])

    @property
    def name(self) -> str:
        return "Exponential (CAGR)"


def fibonacci(n: int) -> list[int]:
    """Return the first ``n`` Fibonacci numbers starting 1, 1, 2, 3, ..."""
    if n < 0:
        raise ValueError("n must be non-negative")
    seq: list[int] = []
    a, b = 1, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


class FibonacciRatioForecaster(Forecaster):
    """Previous cohort's method: scale the last value by successive Fibonacci ratios.

    Forecast ``k`` is ``y_last * prod_{i=1..k} F(s+i) / F(s+i-1)``. With the default
    ``start=1`` the ratios are 1/1, 2/1, 3/2, 5/3, ... which converge to the golden
    ratio (~1.618), i.e. ~62% growth per step - see the notebook's critique.
    """

    min_obs = 1

    def __init__(self, start: int = 1) -> None:
        super().__init__()
        if start < 1:
            raise ValueError("start must be >= 1")
        self.start = start

    def ratios(self, horizon: int) -> np.ndarray:
        """The ``horizon`` successive Fibonacci ratios used for scaling."""
        fib = fibonacci(self.start + horizon + 1)
        return np.array([fib[self.start + i] / fib[self.start + i - 1]
                         for i in range(horizon)], dtype=float)

    def _fit(self) -> None:  # nothing to estimate
        pass

    def _predict(self, horizon: int) -> np.ndarray:
        return self.y_[-1] * np.cumprod(self.ratios(horizon))

    def _fitted(self) -> np.ndarray:
        # One-step-ahead in-sample: y[t-1] scaled by the first ratio.
        out = np.full_like(self.y_, np.nan)
        out[1:] = self.y_[:-1] * self.ratios(1)[0]
        return out

    @property
    def name(self) -> str:
        return "Fibonacci ratio"


MODEL_FACTORIES: dict[str, Callable[[], Forecaster]] = {
    "Linear trend": LinearTrendForecaster,
    "Exponential (CAGR)": CAGRForecaster,
    "Fibonacci ratio": FibonacciRatioForecaster,
}


# ----------------------------------------------------------------------
# Validation and model selection
# ----------------------------------------------------------------------
@dataclass
class ModelSelector:
    """Train on years <= ``last_train_year``, test on the rest, pick lowest ``metric``."""

    last_train_year: int = 2021
    metric: str = "RMSE"
    factories: dict[str, Callable[[], Forecaster]] | None = None

    def __post_init__(self) -> None:
        if self.metric not in {"MAE", "RMSE", "MAPE"}:
            raise ValueError("metric must be MAE, RMSE or MAPE")
        if self.factories is None:
            self.factories = dict(MODEL_FACTORIES)

    def evaluate(self, district: DistrictPopulation) -> pd.DataFrame:
        """Return a table of MAE/RMSE/MAPE per model on the test period."""
        train, test = district.split(self.last_train_year)
        rows = []
        for name, factory in self.factories.items():
            model = factory().fit(train.populations, train.years - train.years[0])
            pred = model.predict(len(test))
            rows.append({"model": name,
                         "MAE": mae(test.populations, pred),
                         "RMSE": rmse(test.populations, pred),
                         "MAPE": mape(test.populations, pred)})
        return pd.DataFrame(rows).set_index("model").sort_values(self.metric)

    def best_model_name(self, district: DistrictPopulation) -> str:
        """Name of the model with the lowest chosen metric on the test set."""
        return str(self.evaluate(district).index[0])

    def fit_best(self, district: DistrictPopulation) -> Forecaster:
        """Refit the selected model on the full series (ready to forecast)."""
        name = self.best_model_name(district)
        return self.factories[name]().fit(district.populations,
                                          district.years - district.years[0])


def bootstrap_intervals(factory: Callable[[], Forecaster], y: ArrayLike,
                        t: ArrayLike, horizon: int, n_boot: int = 1000,
                        level: float = 0.95,
                        rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Residual-bootstrap prediction intervals.

    1. Fit the model, compute residuals ``e = y - fitted``.
    2. For each resample, build ``y* = fitted + e*`` (residuals drawn with
       replacement), refit, forecast, and add a further resampled residual to each
       forecast step (so the interval covers new noise, not just parameter
       uncertainty).
    3. Take the ``(1-level)/2`` and ``(1+level)/2`` percentiles.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = rng if rng is not None else np.random.default_rng(0)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    base = factory().fit(y, t)
    fitted = base.fitted()
    resid = (y - fitted)[~np.isnan(fitted)]
    resid = resid - resid.mean()
    sims = np.empty((n_boot, horizon))
    for b in range(n_boot):
        y_star = fitted + rng.choice(resid, size=y.size, replace=True)
        pred = factory().fit(y_star, t).predict(horizon)
        sims[b] = pred + rng.choice(resid, size=horizon, replace=True)
    alpha = (1 - level) / 2
    return np.percentile(sims, 100 * alpha, axis=0), np.percentile(sims, 100 * (1 - alpha), axis=0)


# ----------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ClassroomPlanner:
    """Turns population change (thousands) into additional primary classrooms."""

    school_age_share: float = 0.18
    pupils_per_classroom: int = 53

    def __post_init__(self) -> None:
        if not 0 < self.school_age_share <= 1:
            raise ValueError("school_age_share must be in (0, 1]")
        if self.pupils_per_classroom <= 0:
            raise ValueError("pupils_per_classroom must be positive")

    def pupils(self, population_thousands: float) -> float:
        """Primary-school-age children for a population given in thousands."""
        return population_thousands * 1000 * self.school_age_share

    def additional_classrooms(self, current_thousands: float,
                              future_thousands: float) -> int:
        """Extra classrooms (rounded up) to seat the growth in pupils; 0 if shrinking."""
        extra = self.pupils(future_thousands) - self.pupils(current_thousands)
        return max(0, math.ceil(extra / self.pupils_per_classroom))


def load_districts(names: Sequence[str] | None = None) -> list[DistrictPopulation]:
    """Build :class:`DistrictPopulation` objects from the bundled data."""
    names = list(DISTRICT_DATA) if names is None else list(names)
    return [DistrictPopulation(n, YEARS, DISTRICT_DATA[n]) for n in names]
