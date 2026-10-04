"""Mini-Project 5: Taxi (matatu) Route Revenue, Pricing & Fleet Planner.

* :class:`Route` - passenger counts, fare and revenue statistics.
* :class:`MarketModel` - linear supply/demand equilibrium via ``scipy.linalg.solve``.
* Forecasters: :class:`MovingAverageForecaster`, :class:`SESForecaster`,
  :class:`SeasonalNaiveForecaster` (+ ``LinearTrendForecaster`` reused from P1).
* :class:`Backtester` - rolling-origin evaluation and alpha grid search.
* :class:`FleetPlanner` - vehicles needed from a passenger forecast.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import linalg

from src.common.forecasting import Forecaster, mae
from src.population import LinearTrendForecaster

ROUTE_DATA = {
    "Kampala-Ntinda": ([35, 40, 42, 50, 55, 60, 48, 52, 47, 45], 2_000),
    "Kampala-Entebbe": ([60, 58, 65, 70, 72, 80, 75, 68, 66, 64], 5_000),
    "Kampala-Mukono": ([45, 47, 50, 49, 55, 62, 58, 53, 51, 50], 3_000),
}


class Route:
    """A taxi route: daily passenger counts and a flat fare (UGX)."""

    def __init__(self, name: str, passengers: ArrayLike, fare: float) -> None:
        p = np.asarray(passengers, dtype=float)
        if not name:
            raise ValueError("name must be non-empty")
        if p.ndim != 1 or p.size == 0:
            raise ValueError("passengers must be a non-empty 1-D sequence")
        if np.any(p < 0) or not np.all(np.isfinite(p)):
            raise ValueError("passenger counts must be finite and non-negative")
        if fare <= 0:
            raise ValueError("fare must be positive")
        self.name, self.passengers, self.fare = name, p, float(fare)

    def __repr__(self) -> str:
        return f"Route({self.name!r}, days={len(self)}, fare=UGX {self.fare:,.0f})"

    def __len__(self) -> int:
        return int(self.passengers.size)

    def daily_revenue(self) -> np.ndarray:
        """UGX per day (every passenger pays the full fare)."""
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        return float(self.daily_revenue().sum())

    def stats(self) -> dict[str, float]:
        """Mean, variance and sd of passengers (``statistics`` module, sample)."""
        data = self.passengers.tolist()
        if len(data) < 2:
            raise ValueError("need at least two days for variance")
        return {"mean": statistics.mean(data), "variance": statistics.variance(data),
                "stdev": statistics.stdev(data)}


@dataclass(frozen=True)
class MarketModel:
    """Linear market: Qd = a - b P, Qs = c + d P."""

    a: float = 120
    b: float = 0.02
    c: float = 10
    d: float = 0.03

    def system(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (A, rhs) for unknowns [Q, P]:  Q + bP = a;  Q - dP = c."""
        return np.array([[1.0, self.b], [1.0, -self.d]]), np.array([self.a, self.c], float)

    def equilibrium(self) -> tuple[float, float]:
        """(Q*, P*) solving the 2x2 system with ``scipy.linalg.solve``."""
        A, rhs = self.system()
        if np.isclose(np.linalg.det(A), 0):
            raise np.linalg.LinAlgError("parallel supply and demand: no unique equilibrium")
        q, p = linalg.solve(A, rhs)
        return float(q), float(p)

    def quantity_demanded(self, price: float) -> float:
        return self.a - self.b * price

    def quantity_supplied(self, price: float) -> float:
        return self.c + self.d * price


# ----------------------------------------------------------------------
# Forecasters
# ----------------------------------------------------------------------
class MovingAverageForecaster(Forecaster):
    """Mean of the last ``window`` observations (flat multi-step forecast)."""

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.min_obs = window

    def _fit(self) -> None:
        self.level_ = float(self.y_[-self.window:].mean())

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level_)

    @property
    def name(self) -> str:
        return f"{self.window}-day moving average"


class SESForecaster(Forecaster):
    """Simple exponential smoothing: ``l_t = alpha y_t + (1-alpha) l_{t-1}``, l_0 = y_0."""

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha

    def _fit(self) -> None:
        level = self.y_[0]
        fitted = [np.nan]
        for y in self.y_[1:]:
            fitted.append(level)                 # one-step-ahead forecast
            level = self.alpha * y + (1 - self.alpha) * level
        self.level_ = float(level)
        self._fitted_vals = np.array(fitted)

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level_)

    def _fitted(self) -> np.ndarray:
        return self._fitted_vals

    @property
    def name(self) -> str:
        return f"SES (alpha={self.alpha:.2f})"


class SeasonalNaiveForecaster(Forecaster):
    """Forecast = value one season (``period`` days) earlier."""

    def __init__(self, period: int = 7) -> None:
        super().__init__()
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.min_obs = period

    def _fit(self) -> None:
        self.last_season_ = self.y_[-self.period:].copy()

    def _predict(self, horizon: int) -> np.ndarray:
        reps = math.ceil(horizon / self.period)
        return np.tile(self.last_season_, reps)[:horizon]

    @property
    def name(self) -> str:
        return f"Seasonal naive (m={self.period})"


class LinearTrend(LinearTrendForecaster):
    """Linear trend reused from Mini-Project 1 (inherits all behaviour)."""


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
class Backtester:
    """Rolling-origin (walk-forward) one-step-ahead evaluation.

    For each target day ``t`` (1-based) from ``first_day`` to the last day, fit on
    days ``1 .. t-1`` and forecast day ``t``.
    """

    def __init__(self, y: ArrayLike, first_day: int = 4) -> None:
        self.y = np.asarray(y, dtype=float)
        if self.y.size < 2:
            raise ValueError("need at least two observations")
        if not 2 <= first_day <= self.y.size:
            raise ValueError("first_day must be between 2 and len(y)")
        self.first_day = first_day

    def forecasts(self, factory: Callable[[], Forecaster]) -> pd.DataFrame:
        """Table of day, actual and one-step forecast."""
        rows = []
        for t in range(self.first_day, self.y.size + 1):
            model = factory().fit(self.y[: t - 1])
            rows.append({"day": t, "actual": self.y[t - 1], "forecast": model.predict(1)[0]})
        return pd.DataFrame(rows).set_index("day")

    def mae(self, factory: Callable[[], Forecaster]) -> float:
        f = self.forecasts(factory)
        return mae(f["actual"], f["forecast"])

    def compare(self, factories: dict[str, Callable[[], Forecaster]]) -> pd.Series:
        """MAE for each named model, sorted ascending."""
        return pd.Series({n: self.mae(f) for n, f in factories.items()}, name="MAE").sort_values()

    def tune_alpha(self, grid: ArrayLike | None = None) -> tuple[float, pd.Series]:
        """Grid-search SES alpha by backtest MAE. Returns (best alpha, MAE curve)."""
        grid = np.round(np.arange(0.05, 1.0001, 0.05), 2) if grid is None else np.asarray(grid)
        curve = pd.Series({float(a): self.mae(lambda a=a: SESForecaster(a)) for a in grid},
                          name="MAE")
        return float(curve.idxmin()), curve


@dataclass(frozen=True)
class FleetPlanner:
    """Vehicles needed = ceil(passengers x (1 + buffer) / (trips x seats))."""

    trips_per_vehicle: int = 8
    seats: int = 14
    buffer: float = 0.15

    def __post_init__(self) -> None:
        if self.trips_per_vehicle <= 0 or self.seats <= 0:
            raise ValueError("trips and seats must be positive")
        if self.buffer < 0:
            raise ValueError("buffer must be non-negative")

    @property
    def capacity_per_vehicle(self) -> int:
        return self.trips_per_vehicle * self.seats

    def vehicles(self, passengers: float) -> int:
        """Whole vehicles needed (at least 1 if there is any demand)."""
        if passengers < 0:
            raise ValueError("passengers must be non-negative")
        if passengers == 0:
            return 0
        return max(1, math.ceil(passengers * (1 + self.buffer) / self.capacity_per_vehicle))


def load_routes() -> list[Route]:
    return [Route(n, p, f) for n, (p, f) in ROUTE_DATA.items()]


def simulate_weekly_demand(days: int = 60, base: float = 50, seed: int = 11,
                           start_dow: int = 0, noise_sd: float = 3.0) -> np.ndarray:
    """Synthetic daily passengers with a weekly pattern (busy Friday, quiet Sunday).

    ``start_dow`` is the weekday of day 1 (0 = Monday).
    """
    if days < 1:
        raise ValueError("days must be positive")
    profile = np.array([1.00, 0.95, 0.97, 1.05, 1.35, 0.90, 0.60])   # Mon..Sun
    rng = np.random.default_rng(seed)
    dow = (start_dow + np.arange(days)) % 7
    y = base * profile[dow] + rng.normal(0, noise_sd, days)
    return np.round(np.clip(y, 0, None))
