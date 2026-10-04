"""Shared forecasting infrastructure used by Mini-Projects 1 and 5.

Contains the abstract :class:`Forecaster` base class and the error metrics
(MAE, RMSE, MAPE) so that no project duplicates this logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


def _as_float_array(values: ArrayLike, name: str = "values") -> np.ndarray:
    """Convert ``values`` to a 1-D float array and reject empty/non-finite input."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numbers")
    return arr


class Forecaster(ABC):
    """Abstract base class for univariate time-series forecasters.

    Subclasses implement :meth:`_fit` (learn parameters from ``self.y_``) and
    :meth:`_predict` (produce ``horizon`` future values). The public
    :meth:`fit` / :meth:`predict` methods handle validation and state.
    """

    #: minimum number of observations a model needs to be fitted
    min_obs: int = 1

    def __init__(self) -> None:
        self.y_: np.ndarray | None = None
        self.t_: np.ndarray | None = None

    # ----- public API -------------------------------------------------
    def fit(self, y: ArrayLike, t: ArrayLike | None = None) -> "Forecaster":
        """Fit the model to observations ``y`` (optionally at times ``t``).

        Returns ``self`` so calls can be chained: ``model.fit(y).predict(3)``.
        """
        y_arr = _as_float_array(y, "y")
        if y_arr.size < self.min_obs:
            raise ValueError(
                f"{type(self).__name__} needs at least {self.min_obs} observations, "
                f"got {y_arr.size}"
            )
        self.y_ = y_arr
        self.t_ = (np.arange(y_arr.size, dtype=float) if t is None
                   else _as_float_array(t, "t"))
        if self.t_.size != self.y_.size:
            raise ValueError("t and y must have the same length")
        self._fit()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the next ``horizon`` values after the training data."""
        if self.y_ is None:
            raise RuntimeError("call fit() before predict()")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        return np.asarray(self._predict(int(horizon)), dtype=float)

    def fitted(self) -> np.ndarray:
        """In-sample fitted values (default: NaN where undefined)."""
        if self.y_ is None:
            raise RuntimeError("call fit() before fitted()")
        return np.asarray(self._fitted(), dtype=float)

    # ----- hooks for subclasses ----------------------------------------
    @abstractmethod
    def _fit(self) -> None:
        """Estimate model parameters from ``self.y_`` and ``self.t_``."""

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray:
        """Return ``horizon`` forecasts."""

    def _fitted(self) -> np.ndarray:
        return np.full_like(self.y_, np.nan)

    @property
    def name(self) -> str:
        """Human-readable model name used in tables and legends."""
        return type(self).__name__

    def __repr__(self) -> str:
        state = "fitted" if self.y_ is not None else "unfitted"
        return f"{self.name}({state})"


# ----------------------------------------------------------------------
# Error metrics
# ----------------------------------------------------------------------
def _pair(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a = _as_float_array(actual, "actual")
    p = _as_float_array(predicted, "predicted")
    if a.shape != p.shape:
        raise ValueError("actual and predicted must have the same length")
    return a, p


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean absolute error."""
    a, p = _pair(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Root mean squared error."""
    a, p = _pair(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean absolute percentage error, in percent. Undefined if any actual is 0."""
    a, p = _pair(actual, predicted)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when an actual value is zero")
    return float(np.mean(np.abs((a - p) / a)) * 100)
