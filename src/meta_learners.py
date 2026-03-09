"""Classical meta-learners for heterogeneous treatment effect estimation.

These learners are foundational for uplift modeling because they estimate
counterfactual outcome surfaces and derive CATE/ITE rankings for policy.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from .baselines import _default_classifier


def _to_numpy_2d(X: Any) -> np.ndarray:
    """Convert features to 2D NumPy array."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("X must be 2-dimensional.")
    return arr


def _to_numpy_1d(values: Any) -> np.ndarray:
    """Convert vector-like input to a flattened NumPy array."""
    return np.asarray(values).reshape(-1)


def _predict_class_1(model: Any, X: Any) -> np.ndarray:
    """Predict positive class probability from binary classifier."""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)
    return np.asarray(model.predict(X)).reshape(-1)


class SLearner:
    """S-Learner for uplift.

    Intuition
    ---------
    Train a single supervised model `mu(x, t)` with treatment as a feature.
    Uplift is estimated by toggling treatment:
    `tau_hat(x) = mu(x, 1) - mu(x, 0)`.
    """

    def __init__(self, base_estimator: Optional[Any] = None, random_state: int = 42) -> None:
        """Initialize S-Learner.

        Parameters
        ----------
        base_estimator:
            Optional model with `fit` and `predict_proba`.
        random_state:
            Seed used when creating default estimator.
        """
        self.random_state = random_state
        self.base_estimator = base_estimator or _default_classifier(model_family="xgboost", random_state=random_state)
        self.model_: Optional[Any] = None

    def fit(self, X: Any, treatment: Any, y: Any) -> "SLearner":
        """Fit S-Learner on observed outcomes."""
        X_arr = _to_numpy_2d(X)
        t_arr = _to_numpy_1d(treatment).astype(int)
        y_arr = _to_numpy_1d(y)

        X_aug = np.column_stack([X_arr, t_arr])
        self.model_ = clone(self.base_estimator)
        self.model_.fit(X_aug, y_arr)
        return self

    def predict_uplift(self, X: Any) -> np.ndarray:
        """Predict uplift scores for each row."""
        if self.model_ is None:
            raise RuntimeError("SLearner must be fitted before prediction.")

        X_arr = _to_numpy_2d(X)
        x_t = np.column_stack([X_arr, np.ones(len(X_arr))])
        x_c = np.column_stack([X_arr, np.zeros(len(X_arr))])
        return _predict_class_1(self.model_, x_t) - _predict_class_1(self.model_, x_c)

    def predict(self, X: Any) -> np.ndarray:
        """Alias for `predict_uplift` to mimic sklearn usage."""
        return self.predict_uplift(X)


class TLearner:
    """T-Learner for uplift.

    Intuition
    ---------
    Train separate models for treatment and control outcomes:
    `mu1(x)` and `mu0(x)`, then estimate `tau_hat(x)=mu1(x)-mu0(x)`.
    """

    def __init__(self, base_estimator: Optional[Any] = None, random_state: int = 42) -> None:
        """Initialize T-Learner with a shared base estimator family."""
        self.random_state = random_state
        self.base_estimator = base_estimator or _default_classifier(model_family="xgboost", random_state=random_state)
        self.model_t_: Optional[Any] = None
        self.model_c_: Optional[Any] = None

    def fit(self, X: Any, treatment: Any, y: Any) -> "TLearner":
        """Fit treatment and control outcome models."""
        X_arr = _to_numpy_2d(X)
        t_arr = _to_numpy_1d(treatment).astype(int)
        y_arr = _to_numpy_1d(y)

        idx_t = t_arr == 1
        idx_c = t_arr == 0
        if idx_t.sum() == 0 or idx_c.sum() == 0:
            raise ValueError("Both treatment and control examples are required.")

        self.model_t_ = clone(self.base_estimator)
        self.model_c_ = clone(self.base_estimator)
        self.model_t_.fit(X_arr[idx_t], y_arr[idx_t])
        self.model_c_.fit(X_arr[idx_c], y_arr[idx_c])
        return self

    def predict_uplift(self, X: Any) -> np.ndarray:
        """Predict uplift from fitted treatment/control models."""
        if self.model_t_ is None or self.model_c_ is None:
            raise RuntimeError("TLearner must be fitted before prediction.")

        X_arr = _to_numpy_2d(X)
        return _predict_class_1(self.model_t_, X_arr) - _predict_class_1(self.model_c_, X_arr)

    def predict(self, X: Any) -> np.ndarray:
        """Alias for `predict_uplift` to mimic sklearn usage."""
        return self.predict_uplift(X)


class XLearner:
    """X-Learner for uplift.

    Intuition
    ---------
    1. Fit treatment/control outcome models.
    2. Construct imputed treatment effects for each arm.
    3. Fit second-stage effect models on imputed effects.
    4. Combine arm-specific effects with propensity weighting.

    X-Learner is often helpful when treatment/control groups are imbalanced.
    """

    def __init__(
        self,
        base_outcome_estimator: Optional[Any] = None,
        effect_estimator: Optional[Any] = None,
        propensity_estimator: Optional[Any] = None,
        random_state: int = 42,
    ) -> None:
        """Initialize XLearner components."""
        self.random_state = random_state
        self.base_outcome_estimator = base_outcome_estimator or _default_classifier(model_family="xgboost", random_state=random_state)
        self.effect_estimator = effect_estimator or GradientBoostingRegressor(random_state=random_state)
        self.propensity_estimator = propensity_estimator or LogisticRegression(max_iter=1000)

        self.mu1_: Optional[Any] = None
        self.mu0_: Optional[Any] = None
        self.tau1_: Optional[Any] = None
        self.tau0_: Optional[Any] = None
        self.e_: Optional[Any] = None

    def fit(self, X: Any, treatment: Any, y: Any) -> "XLearner":
        """Fit XLearner with imputed effects and propensity weighting."""
        X_arr = _to_numpy_2d(X)
        t_arr = _to_numpy_1d(treatment).astype(int)
        y_arr = _to_numpy_1d(y)

        idx_t = t_arr == 1
        idx_c = t_arr == 0
        if idx_t.sum() == 0 or idx_c.sum() == 0:
            raise ValueError("Both treatment and control examples are required.")

        X_t, X_c = X_arr[idx_t], X_arr[idx_c]
        y_t, y_c = y_arr[idx_t], y_arr[idx_c]

        self.mu1_ = clone(self.base_outcome_estimator)
        self.mu0_ = clone(self.base_outcome_estimator)
        self.mu1_.fit(X_t, y_t)
        self.mu0_.fit(X_c, y_c)

        d1 = y_t - _predict_class_1(self.mu0_, X_t)
        d0 = _predict_class_1(self.mu1_, X_c) - y_c

        self.tau1_ = clone(self.effect_estimator)
        self.tau0_ = clone(self.effect_estimator)
        self.tau1_.fit(X_t, d1)
        self.tau0_.fit(X_c, d0)

        self.e_ = clone(self.propensity_estimator)
        self.e_.fit(X_arr, t_arr)
        return self

    def predict_uplift(self, X: Any) -> np.ndarray:
        """Predict CATE/uplift scores."""
        if any(m is None for m in [self.tau1_, self.tau0_, self.e_]):
            raise RuntimeError("XLearner must be fitted before prediction.")

        X_arr = _to_numpy_2d(X)
        tau1 = np.asarray(self.tau1_.predict(X_arr)).reshape(-1)
        tau0 = np.asarray(self.tau0_.predict(X_arr)).reshape(-1)
        prop = np.clip(_predict_class_1(self.e_, X_arr), 1e-3, 1 - 1e-3)
        return prop * tau0 + (1.0 - prop) * tau1

    def predict(self, X: Any) -> np.ndarray:
        """Alias for `predict_uplift` to mimic sklearn usage."""
        return self.predict_uplift(X)
