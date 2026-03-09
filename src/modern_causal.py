"""Modern causal estimators and wrappers for DR-style uplift modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from .baselines import _default_classifier


def _to_numpy_2d(X: Any) -> np.ndarray:
    """Convert features to a 2D NumPy array."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("X must be 2-dimensional.")
    return arr


def _to_numpy_1d(values: Any) -> np.ndarray:
    """Convert vector-like input to a flattened NumPy array."""
    return np.asarray(values).reshape(-1)


def _predict_class_1(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict positive class probability from a binary estimator."""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)
    return np.asarray(model.predict(X)).reshape(-1)


@dataclass
class DRLearnerWrapper:
    """Practical DR-Learner wrapper with optional EconML backend.

    This class supports two paths:
    1. EconML DRLearner (if installed).
    2. A lightweight manual doubly-robust pseudo-outcome implementation.
    """

    model_regression: Optional[Any] = None
    model_propensity: Optional[Any] = None
    model_final: Optional[Any] = None
    random_state: int = 42
    prefer_econml: bool = True

    def __post_init__(self) -> None:
        """Initialize internal model references."""
        self.backend_: str = "manual"
        self.model_: Optional[Any] = None

        self.mu1_: Optional[Any] = None
        self.mu0_: Optional[Any] = None
        self.e_: Optional[Any] = None
        self.tau_model_: Optional[Any] = None

    def fit(self, X: Any, treatment: Any, y: Any) -> "DRLearnerWrapper":
        """Fit DR-Learner on binary treatment data.

        Parameters
        ----------
        X, treatment, y:
            Features, treatment indicator, and observed outcomes.

        Returns
        -------
        DRLearnerWrapper
            Fitted wrapper.
        """
        X_arr = _to_numpy_2d(X)
        t_arr = _to_numpy_1d(treatment).astype(int)
        y_arr = _to_numpy_1d(y).astype(float)

        if self.prefer_econml:
            try:
                from econml.dr import DRLearner

                self.model_ = DRLearner(
                    model_regression=self.model_regression,
                    model_propensity=self.model_propensity,
                    model_final=self.model_final,
                    random_state=self.random_state,
                )
                self.model_.fit(y_arr, t_arr, X=X_arr)
                self.backend_ = "econml"
                return self
            except Exception:
                self.backend_ = "manual"

        reg_model = self.model_regression or _default_classifier(model_family="xgboost", random_state=self.random_state)
        prop_model = self.model_propensity or LogisticRegression(max_iter=1000)
        final_model = self.model_final or GradientBoostingRegressor(random_state=self.random_state)

        idx_t = t_arr == 1
        idx_c = t_arr == 0
        if idx_t.sum() == 0 or idx_c.sum() == 0:
            raise ValueError("Both treatment and control rows are required.")

        self.mu1_ = clone(reg_model)
        self.mu0_ = clone(reg_model)
        self.mu1_.fit(X_arr[idx_t], y_arr[idx_t])
        self.mu0_.fit(X_arr[idx_c], y_arr[idx_c])

        self.e_ = clone(prop_model)
        self.e_.fit(X_arr, t_arr)

        mu1 = _predict_class_1(self.mu1_, X_arr)
        mu0 = _predict_class_1(self.mu0_, X_arr)
        e = np.clip(_predict_class_1(self.e_, X_arr), 1e-3, 1 - 1e-3)

        # Doubly robust pseudo-outcome for binary treatment.
        dr_tau = (mu1 - mu0) + (t_arr * (y_arr - mu1) / e) - ((1 - t_arr) * (y_arr - mu0) / (1 - e))

        self.tau_model_ = clone(final_model)
        self.tau_model_.fit(X_arr, dr_tau)
        self.backend_ = "manual"
        return self

    def predict_uplift(self, X: Any) -> np.ndarray:
        """Predict CATE/uplift estimates."""
        X_arr = _to_numpy_2d(X)

        if self.backend_ == "econml":
            if self.model_ is None:
                raise RuntimeError("DRLearnerWrapper is not fitted.")
            return np.asarray(self.model_.effect(X_arr)).reshape(-1)

        if self.tau_model_ is None:
            raise RuntimeError("DRLearnerWrapper is not fitted.")
        return np.asarray(self.tau_model_.predict(X_arr)).reshape(-1)


@dataclass
class OrthogonalForestWrapper:
    """Optional orthogonal/causal forest wrapper.

    Uses EconML CausalForestDML when available; otherwise raises ImportError.
    """

    model_y: Optional[Any] = None
    model_t: Optional[Any] = None
    random_state: int = 42

    def __post_init__(self) -> None:
        """Initialize internal model storage."""
        self.model_: Optional[Any] = None

    def fit(self, X: Any, treatment: Any, y: Any) -> "OrthogonalForestWrapper":
        """Fit orthogonal/causal forest model if EconML is installed."""
        try:
            from econml.dml import CausalForestDML
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "EconML is required for OrthogonalForestWrapper. Install via `pip install econml`."
            ) from exc

        X_arr = _to_numpy_2d(X)
        t_arr = _to_numpy_1d(treatment).astype(int)
        y_arr = _to_numpy_1d(y).astype(float)

        self.model_ = CausalForestDML(
            model_y=self.model_y,
            model_t=self.model_t,
            random_state=self.random_state,
        )
        self.model_.fit(y_arr, t_arr, X=X_arr)
        return self

    def predict_uplift(self, X: Any) -> np.ndarray:
        """Predict uplift from fitted orthogonal forest."""
        if self.model_ is None:
            raise RuntimeError("OrthogonalForestWrapper is not fitted.")
        X_arr = _to_numpy_2d(X)
        return np.asarray(self.model_.effect(X_arr)).reshape(-1)
