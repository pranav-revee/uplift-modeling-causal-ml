"""Baseline models for uplift benchmarking (naive and two-model)."""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier


try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


def _default_classifier(
    model_family: str = "xgboost",
    random_state: int = 42,
    n_estimators: int = 300,
) -> Any:
    """Build a strong tabular classifier for baseline response modeling.

    Parameters
    ----------
    model_family:
        Preferred model family: `xgboost` or `lightgbm`.
    random_state:
        Seed for reproducibility.
    n_estimators:
        Number of boosting trees.

    Returns
    -------
    Any
        A classifier instance with `fit` and `predict_proba` support.
    """
    n_threads = max(1, os.cpu_count() or 1)
    family = model_family.lower()

    if family == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cpu",
            nthread=n_threads,
            random_state=random_state,
        )

    if family == "lightgbm" and LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary",
            n_jobs=n_threads,
            random_state=random_state,
        )

    # Fallback for maximal portability.
    return RandomForestClassifier(
        n_estimators=max(200, n_estimators),
        min_samples_leaf=20,
        n_jobs=n_threads,
        random_state=random_state,
    )


def _to_numpy_2d(X: Any) -> np.ndarray:
    """Convert input features to a 2D NumPy array."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("Features must be 2-dimensional.")
    return arr


def _to_numpy_1d(values: Any) -> np.ndarray:
    """Convert input vector to 1D NumPy array."""
    return np.asarray(values).reshape(-1)


def _positive_class_probability(model: Any, X: Any) -> np.ndarray:
    """Extract positive-class probability from a fitted classifier."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if np.asarray(proba).ndim == 2 and np.asarray(proba).shape[1] >= 2:
            return np.asarray(proba)[:, 1]
        return np.asarray(proba).reshape(-1)
    return np.asarray(model.predict(X)).reshape(-1)


def fit_naive_treated_response_model(
    X: Any,
    y: Any,
    treatment: Any,
    model: Optional[Any] = None,
    model_family: str = "xgboost",
    random_state: int = 42,
) -> Any:
    """Fit naive treated-only response model.

    This model is intentionally misspecified for causal targeting because it
    predicts treated conversion propensity rather than incremental effect.
    """
    estimator = clone(model) if model is not None else _default_classifier(model_family=model_family, random_state=random_state)

    X_arr = _to_numpy_2d(X)
    y_arr = _to_numpy_1d(y)
    t_arr = _to_numpy_1d(treatment).astype(int)

    treated_idx = t_arr == 1
    if treated_idx.sum() == 0:
        raise ValueError("No treated rows found; cannot fit treated-only model.")

    estimator.fit(X_arr[treated_idx], y_arr[treated_idx])
    return estimator


def predict_naive_treated_response_model(model: Any, X: Any) -> np.ndarray:
    """Predict treated response propensity scores for ranking."""
    X_arr = _to_numpy_2d(X)
    return _positive_class_probability(model, X_arr)


def fit_two_model_uplift(
    X: Any,
    y: Any,
    treatment: Any,
    model_t: Optional[Any] = None,
    model_c: Optional[Any] = None,
    model_family: str = "xgboost",
    random_state: int = 42,
) -> Tuple[Any, Any]:
    """Fit separate response models for treated and control arms."""
    est_t = clone(model_t) if model_t is not None else _default_classifier(model_family=model_family, random_state=random_state)
    est_c = clone(model_c) if model_c is not None else _default_classifier(model_family=model_family, random_state=random_state)

    X_arr = _to_numpy_2d(X)
    y_arr = _to_numpy_1d(y)
    t_arr = _to_numpy_1d(treatment).astype(int)

    treated_idx = t_arr == 1
    control_idx = t_arr == 0

    if treated_idx.sum() == 0 or control_idx.sum() == 0:
        raise ValueError("Both treatment and control rows are required.")

    est_t.fit(X_arr[treated_idx], y_arr[treated_idx])
    est_c.fit(X_arr[control_idx], y_arr[control_idx])
    return est_t, est_c


def predict_two_model_uplift(model_t: Any, model_c: Any, X: Any) -> np.ndarray:
    """Predict uplift as treated probability minus control probability."""
    X_arr = _to_numpy_2d(X)
    p_t = _positive_class_probability(model_t, X_arr)
    p_c = _positive_class_probability(model_c, X_arr)
    return p_t - p_c
