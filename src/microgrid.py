"""Mini-Project 2: Solar Micro-Grid Dispatch Planner (Kasese health centre).

The daily dispatch problem is a square linear system ``A @ s = d`` where ``s`` are
the energy amounts drawn from each source (kWh) and ``d`` the demand constraints.

* :class:`MicroGrid` - 2x2 solar/battery system (base class, works for any n x n).
* :class:`HybridMicroGrid` - adds a diesel generator and a third constraint.
* :class:`DemandInput` - interactive (``input()``) and CSV demand sources.
* :class:`CostModel` - converts dispatch into UGX.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import linalg
from scipy.optimize import nnls


class MicroGrid:
    """Linear dispatch model ``A s = d``.

    Default system (from the brief)::

        3x + 2y = D1   (daytime load)
        4x +  y = D2   (critical-equipment load)

    Parameters
    ----------
    coefficients : array-like, shape (n, n)
        Coefficient matrix ``A``.
    sources : sequence of str
        Names of the n energy sources (columns of ``A``).
    constraints : sequence of str
        Names of the n demand constraints (rows of ``A``).
    """

    DEFAULT_A = [[3.0, 2.0], [4.0, 1.0]]

    def __init__(self, coefficients: ArrayLike | None = None,
                 sources: Sequence[str] = ("solar", "battery"),
                 constraints: Sequence[str] = ("D1", "D2")) -> None:
        A = np.asarray(self.DEFAULT_A if coefficients is None else coefficients, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("coefficient matrix must be square")
        if len(sources) != A.shape[1] or len(constraints) != A.shape[0]:
            raise ValueError("number of names must match matrix size")
        self.A = A
        self.sources = tuple(sources)
        self.constraints = tuple(constraints)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(n={self.n}, sources={self.sources}, "
                f"det={self.determinant():.3g}, cond={self.condition_number():.3g})")

    @property
    def n(self) -> int:
        """Number of sources / constraints."""
        return self.A.shape[0]

    # ----- well-posedness ---------------------------------------------
    def determinant(self) -> float:
        """det(A); zero means no unique solution exists."""
        return float(np.linalg.det(self.A))

    def condition_number(self) -> float:
        """2-norm condition number; relative error in d is amplified by up to this."""
        return float(np.linalg.cond(self.A))

    def rank(self) -> int:
        """Matrix rank (``n`` for a well-posed system)."""
        return int(np.linalg.matrix_rank(self.A))

    def is_well_posed(self, max_cond: float = 1e8) -> bool:
        """True if A is non-singular and not badly conditioned."""
        return self.rank() == self.n and self.condition_number() < max_cond

    def _check(self) -> None:
        if not self.is_well_posed():
            raise np.linalg.LinAlgError(
                f"system is singular or ill-conditioned (rank={self.rank()}, "
                f"cond={self.condition_number():.3g})")

    # ----- solving ----------------------------------------------------
    @staticmethod
    def _validate_demand(values: np.ndarray) -> None:
        if values.size == 0:
            raise ValueError("demand must not be empty")
        if not np.all(np.isfinite(values)):
            raise ValueError("demand must be finite numbers")
        if np.any(values < 0):
            raise ValueError("demand must be non-negative")

    def solve_day(self, *demands: float) -> np.ndarray:
        """Solve one day, e.g. ``grid.solve_day(d1, d2)``. Returns source amounts."""
        d = np.asarray(demands, dtype=float)
        if d.size != self.n:
            raise ValueError(f"expected {self.n} demand values, got {d.size}")
        self._validate_demand(d)
        self._check()
        return linalg.solve(self.A, d)

    def solve_loop(self, D: ArrayLike) -> np.ndarray:
        """Solve many days one at a time. ``D`` has shape (n, days)."""
        D = self._as_rhs(D)
        self._check()
        out = np.empty_like(D)
        for j in range(D.shape[1]):
            out[:, j] = linalg.solve(self.A, D[:, j])
        return out

    def solve_many(self, D: ArrayLike) -> np.ndarray:
        """Vectorised solve of all days at once. ``D`` has shape (n, days)."""
        D = self._as_rhs(D)
        self._check()
        return linalg.solve(self.A, D)

    def _as_rhs(self, D: ArrayLike) -> np.ndarray:
        D = np.asarray(D, dtype=float)
        if D.ndim != 2 or D.shape[0] != self.n:
            raise ValueError(f"D must have shape ({self.n}, days)")
        self._validate_demand(D)
        return D

    # ----- feasibility ------------------------------------------------
    @staticmethod
    def infeasible_mask(S: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """Boolean mask of days (columns) where any source is negative."""
        return np.any(np.asarray(S) < -tol, axis=0)

    def dispatch(self, D: ArrayLike, strategy: str = "nnls") -> pd.DataFrame:
        """Solve all days and repair physically infeasible ones.

        strategy
            ``"nnls"`` - re-solve infeasible days with non-negative least squares
            (``scipy.optimize.nnls``), i.e. the closest achievable dispatch;
            ``"clip"`` - set negative amounts to zero.
        The returned table reports the raw solution, the repaired dispatch, the
        infeasible flag and the resulting shortfall/excess on each constraint.
        """
        if strategy not in {"nnls", "clip"}:
            raise ValueError("strategy must be 'nnls' or 'clip'")
        D = self._as_rhs(D)
        raw = self.solve_many(D)
        bad = self.infeasible_mask(raw)
        fixed = raw.copy()
        for j in np.where(bad)[0]:
            fixed[:, j] = nnls(self.A, D[:, j])[0] if strategy == "nnls" \
                else np.clip(raw[:, j], 0, None)
        gap = self.A @ fixed - D   # + = over-supply, - = unmet demand
        data: dict[str, np.ndarray] = {}
        for i, s in enumerate(self.sources):
            data[f"{s}_raw"] = raw[i]
        for i, s in enumerate(self.sources):
            data[s] = fixed[i]
        data["infeasible"] = bad
        for i, c in enumerate(self.constraints):
            data[f"gap_{c}"] = gap[i]
        return pd.DataFrame(data)


class HybridMicroGrid(MicroGrid):
    """Solar + battery + diesel generator with a third constraint.

    Default system::

        3x + 2y +  z = D1   (daytime load; diesel now also serves it)
        4x +  y      = D2   (critical-equipment load, as before)
         x +  y +  z = D3   (total energy drawn over the day)
    """

    DEFAULT_A = [[3.0, 2.0, 1.0], [4.0, 1.0, 0.0], [1.0, 1.0, 1.0]]

    def __init__(self, coefficients: ArrayLike | None = None,
                 sources: Sequence[str] = ("solar", "battery", "diesel"),
                 constraints: Sequence[str] = ("D1", "D2", "D3")) -> None:
        super().__init__(coefficients, sources, constraints)
        if self.n != 3:
            raise ValueError("HybridMicroGrid needs a 3x3 system")


@dataclass(frozen=True)
class CostModel:
    """Tariffs in UGX per kWh for each source (illustrative)."""

    prices: dict[str, float]

    def __post_init__(self) -> None:
        if any(p < 0 for p in self.prices.values()):
            raise ValueError("prices must be non-negative")

    def daily_cost(self, dispatch: pd.DataFrame) -> pd.Series:
        """UGX cost for each day of a dispatch table."""
        missing = set(self.prices) - set(dispatch.columns)
        if missing:
            raise KeyError(f"dispatch lacks columns {missing}")
        return sum(dispatch[s] * p for s, p in self.prices.items())

    def total_cost(self, dispatch: pd.DataFrame) -> float:
        """UGX cost for the whole period."""
        return float(self.daily_cost(dispatch).sum())


class DemandInput:
    """Demand sources: interactive keyboard entry and CSV files."""

    @staticmethod
    def parse_demand(text: str) -> float:
        """Convert text to a non-negative finite float or raise ``ValueError``."""
        if text is None or not str(text).strip():
            raise ValueError("value is empty")
        value = float(str(text).strip())          # raises ValueError if non-numeric
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        if value < 0:
            raise ValueError("value must be non-negative")
        return value

    @classmethod
    def prompt_demand(cls, prompt: str, input_fn: Callable[[str], str] = input,
                      max_attempts: int = 5, echo: Callable[[str], None] = print) -> float:
        """Ask repeatedly until a valid demand is entered (``input_fn`` is injectable
        so the loop can be tested and demonstrated without a keyboard)."""
        for _ in range(max_attempts):
            raw = input_fn(prompt)
            try:
                return cls.parse_demand(raw)
            except ValueError as err:
                echo(f"  invalid input {raw!r}: {err}. Please try again.")
        raise ValueError(f"no valid input after {max_attempts} attempts")

    @classmethod
    def interactive(cls, names: Sequence[str] = ("D1", "D2"),
                    input_fn: Callable[[str], str] = input) -> list[float]:
        """Read one day's demands from the keyboard."""
        return [cls.prompt_demand(f"Enter {n} (kWh): ", input_fn) for n in names]

    @staticmethod
    def generate_csv(path: str | Path, days: int = 30, seed: int = 42,
                     start: str = "2026-09-01") -> pd.DataFrame:
        """Create a synthetic month of demand with a weekly pattern and noise.

        * D1 (daytime load) is higher on weekdays (outpatient clinics, lab work)
          and lower at weekends.
        * D2 (critical equipment: vaccine fridges, oxygen concentrators,
          theatre lights) tracks D1 (about 1.2 x D1) with an extra bump on the
          Tuesday/Thursday theatre days and its own random variation.
        The system only has a non-negative solution when D1/2 <= D2 <= 4*D1/3,
        so noise occasionally produces physically infeasible days.
        """
        if days <= 0:
            raise ValueError("days must be positive")
        rng = np.random.default_rng(seed)
        dates = pd.date_range(start, periods=days, freq="D")
        dow = dates.dayofweek.to_numpy()
        weekday = (dow < 5).astype(float)
        theatre = np.isin(dow, [1, 3]).astype(float)
        d1 = 110 + 25 * weekday + rng.normal(0, 6, days)
        d2 = 1.2 * d1 + 10 * theatre + rng.normal(0, 10, days)
        df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"),
                           "D1": np.round(np.clip(d1, 0, None), 1),
                           "D2": np.round(np.clip(d2, 0, None), 1)})
        df.to_csv(path, index=False)
        return df

    @classmethod
    def load_csv(cls, path: str | Path, columns: Sequence[str] = ("D1", "D2")) -> pd.DataFrame:
        """Load and validate a demand CSV (every value must be a valid demand)."""
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError("CSV has no data rows")
        out = {"date": [r.get("date", str(i)) for i, r in enumerate(rows)]}
        for c in columns:
            try:
                out[c] = [cls.parse_demand(r[c]) for r in rows]
            except KeyError as err:
                raise ValueError(f"CSV lacks column {c}") from err
        return pd.DataFrame(out)


def monte_carlo_sensitivity(grid: MicroGrid, demand: ArrayLike, rel: float = 0.05,
                            draws: int = 1000,
                            rng: np.random.Generator | None = None) -> np.ndarray:
    """Perturb every demand by U(-rel, +rel) independently and re-solve.

    Returns an array of shape (draws, n) of source amounts.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    d = np.asarray(demand, dtype=float)
    factors = 1 + rng.uniform(-rel, rel, size=(d.size, draws))
    return grid.solve_many(d[:, None] * factors).T
