"""Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model (Jinja cooperative).

* :class:`FishStock` - discrete logistic growth with proportional harvesting.
* :class:`PriceModel` - bounded, seeded random walk for the UGX/kg price.
* :class:`RiskAssessor` - CV-based risk classes and Monte Carlo Value-at-Risk.
* :class:`FisheryScenario` - composes the three to produce revenue.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

KG_PER_TONNE = 1000


@dataclass
class SimulationResult:
    """Output of :meth:`FishStock.simulate` (tonnes)."""

    stock: np.ndarray      # length weeks + 1, N(0) ... N(T)
    harvest: np.ndarray    # length weeks, catch during each week

    @property
    def final_stock(self) -> float:
        return float(self.stock[-1])

    @property
    def total_harvest(self) -> float:
        return float(self.harvest.sum())


class FishStock:
    """Discrete logistic model with proportional harvesting (units: tonnes, weeks).

    ``N(t+1) = N(t) + r N(t) (1 - N(t)/K) - h N(t)``, floored at zero.
    """

    def __init__(self, r: float = 0.4, K: float = 10_000, N0: float = 4_000,
                 h: float = 0.10) -> None:
        if r <= 0:
            raise ValueError("growth rate r must be positive")
        if K <= 0:
            raise ValueError("carrying capacity K must be positive")
        if N0 < 0:
            raise ValueError("initial stock N0 must be non-negative")
        if not 0 <= h <= 1:
            raise ValueError("harvest rate h must be in [0, 1]")
        self.r, self.K, self.N0, self.h = float(r), float(K), float(N0), float(h)

    def __repr__(self) -> str:
        return f"FishStock(r={self.r}, K={self.K:,.0f}, N0={self.N0:,.0f}, h={self.h})"

    # ----- theory -----------------------------------------------------
    @property
    def msy(self) -> float:
        """Maximum sustainable yield rK/4 (tonnes per week)."""
        return self.r * self.K / 4

    @property
    def equilibrium_stock(self) -> float:
        """Non-zero fixed point K(1 - h/r); 0 if h >= r (collapse)."""
        return max(0.0, self.K * (1 - self.h / self.r))

    @property
    def equilibrium_yield(self) -> float:
        """Sustained catch h * N* (tonnes per week)."""
        return self.h * self.equilibrium_stock

    def step(self, N: float, h: float | None = None) -> float:
        """Advance one week from stock ``N`` using harvest rate ``h``."""
        h = self.h if h is None else h
        return max(0.0, N + self.r * N * (1 - N / self.K) - h * N)

    def simulate(self, weeks: int = 52, open_weeks: ArrayLike | None = None) -> SimulationResult:
        """Simulate ``weeks`` steps.

        ``open_weeks`` is an optional boolean array; where False, the season is
        closed and ``h`` is set to 0 for that week.
        """
        if weeks < 1:
            raise ValueError("weeks must be >= 1")
        mask = np.ones(weeks, bool) if open_weeks is None else np.asarray(open_weeks, bool)
        if mask.size != weeks:
            raise ValueError("open_weeks must have one entry per week")
        stock = np.empty(weeks + 1)
        harvest = np.empty(weeks)
        stock[0] = self.N0
        for t in range(weeks):
            h_t = self.h if mask[t] else 0.0
            harvest[t] = h_t * stock[t]
            stock[t + 1] = self.step(stock[t], h_t)
        return SimulationResult(stock, harvest)


class PriceModel:
    """Weekly fish price (UGX/kg) as a bounded Gaussian random walk.

    ``P(t+1) = clip(P(t) + e_t, lower, upper)``, ``e_t ~ N(0, sigma)``.
    """

    def __init__(self, start: float = 12_000, lower: float = 9_000, upper: float = 16_000,
                 sigma: float = 400, seed: int | None = 0) -> None:
        if not lower < upper:
            raise ValueError("lower must be below upper")
        if not lower <= start <= upper:
            raise ValueError("start must lie within the bounds")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.start, self.lower, self.upper, self.sigma = start, lower, upper, sigma
        self.rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return (f"PriceModel(start={self.start:,}, bounds=[{self.lower:,}, {self.upper:,}], "
                f"sigma={self.sigma})")

    def simulate_paths(self, weeks: int, n_paths: int = 1) -> np.ndarray:
        """Return an array (n_paths, weeks) of prices; week 0 equals ``start``."""
        if weeks < 1 or n_paths < 1:
            raise ValueError("weeks and n_paths must be >= 1")
        shocks = self.rng.normal(0, self.sigma, size=(n_paths, weeks))
        prices = np.empty((n_paths, weeks))
        prices[:, 0] = self.start
        for t in range(1, weeks):
            prices[:, t] = np.clip(prices[:, t - 1] + shocks[:, t], self.lower, self.upper)
        return prices

    def simulate(self, weeks: int = 52) -> np.ndarray:
        """A single price path of length ``weeks``."""
        return self.simulate_paths(weeks, 1)[0]


@dataclass(frozen=True)
class RiskAssessor:
    """Classifies revenue risk by the coefficient of variation (CV = sd / mean).

    Default thresholds: CV < 0.10 -> "Low", 0.10-0.25 -> "Moderate", >= 0.25 -> "High".
    """

    low: float = 0.10
    high: float = 0.25

    def __post_init__(self) -> None:
        if not 0 < self.low < self.high:
            raise ValueError("need 0 < low < high")

    @staticmethod
    def describe(revenue: ArrayLike) -> dict[str, float]:
        """Mean, median, variance, sd and CV using the ``statistics`` module."""
        data = [float(v) for v in np.asarray(revenue).ravel()]
        if len(data) < 2:
            raise ValueError("need at least two revenue values")
        mean = statistics.mean(data)
        sd = statistics.stdev(data)
        return {"mean": mean, "median": statistics.median(data),
                "variance": statistics.variance(data), "stdev": sd,
                "cv": sd / mean if mean else float("inf")}

    def classify_cv(self, cv: float) -> str:
        """Map a CV to a risk class."""
        if cv < 0 or np.isnan(cv):
            raise ValueError("CV must be a non-negative number")
        if cv < self.low:
            return "Low"
        return "Moderate" if cv < self.high else "High"

    def classify(self, revenue: ArrayLike) -> str:
        """Risk class of a revenue series."""
        return self.classify_cv(self.describe(revenue)["cv"])

    @staticmethod
    def value_at_risk(outcomes: ArrayLike, alpha: float = 0.05) -> dict[str, float]:
        """Historical-simulation VaR of a distribution of outcomes (e.g. annual revenue).

        Returns the ``alpha`` quantile (the revenue level only beaten on the downside
        in ``alpha`` of cases) and VaR = mean - quantile (shortfall versus expected).
        """
        x = np.asarray(outcomes, dtype=float)
        if x.size == 0:
            raise ValueError("outcomes must not be empty")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        q = float(np.percentile(x, 100 * alpha))
        return {"quantile": q, "mean": float(x.mean()), "VaR": float(x.mean() - q)}


@dataclass
class FisheryScenario:
    """Couples a :class:`FishStock` with a :class:`PriceModel` to produce revenue."""

    stock: FishStock
    prices: PriceModel
    weeks: int = 52
    open_weeks: np.ndarray | None = field(default=None)

    def run(self, price_path: np.ndarray | None = None) -> dict[str, np.ndarray | SimulationResult]:
        """One simulation: stock path, harvest, price and weekly revenue (UGX)."""
        sim = self.stock.simulate(self.weeks, self.open_weeks)
        price = self.prices.simulate(self.weeks) if price_path is None else np.asarray(price_path)
        if price.size != self.weeks:
            raise ValueError("price path length must equal weeks")
        revenue = sim.harvest * KG_PER_TONNE * price
        return {"sim": sim, "price": price, "revenue": revenue}

    def monte_carlo_annual_revenue(self, n_paths: int = 1000) -> np.ndarray:
        """Total revenue over the horizon for ``n_paths`` price paths.

        The biological model is deterministic, so the harvest is computed once and
        combined with each simulated price path (vectorised).
        """
        sim = self.stock.simulate(self.weeks, self.open_weeks)
        paths = self.prices.simulate_paths(self.weeks, n_paths)
        return (paths * sim.harvest * KG_PER_TONNE).sum(axis=1)


def closed_season_mask(years: int, closed_weeks: int = 8, start_week: int = 13,
                       weeks_per_year: int = 52) -> np.ndarray:
    """Boolean 'open' mask with ``closed_weeks`` closed each year from ``start_week``."""
    if not 0 <= closed_weeks < weeks_per_year:
        raise ValueError("closed_weeks must be in [0, weeks_per_year)")
    year = np.ones(weeks_per_year, bool)
    idx = (start_week + np.arange(closed_weeks)) % weeks_per_year
    year[idx] = False
    return np.tile(year, years)
