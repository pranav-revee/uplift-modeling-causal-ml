"""Preprocessing helpers for Hillstrom uplift modeling experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_OUTCOMES = ["visit", "conversion", "spend"]
DEFAULT_TREATMENT_COL = "treatment"
DEFAULT_SEGMENT_COL = "segment"


def load_hillstrom_data(path: str) -> pd.DataFrame:
    """Load Hillstrom data from CSV.

    Parameters
    ----------
    path:
        Path to `hillstrom.csv`.

    Returns
    -------
    pd.DataFrame
        Raw dataframe loaded from disk.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Hillstrom CSV not found at: {csv_path}")
    return pd.read_csv(csv_path)


def _find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    """Resolve the first matching column among candidate names."""
    lower_to_actual = {c.lower(): c for c in df.columns}
    for name in candidates:
        col = lower_to_actual.get(name.lower())
        if col is not None:
            return col
    if required:
        raise ValueError(f"None of the expected columns were found: {list(candidates)}")
    return None


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known Hillstrom-like column names to canonical names."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    alias_map = {
        "segment": ["segment", "treatment_segment", "treatment_group", "offer_segment"],
        "visit": ["visit", "visited", "is_visit"],
        "conversion": ["conversion", "converted", "is_conversion", "purchase"],
        "spend": ["spend", "amount", "revenue"],
    }

    rename_map: Dict[str, str] = {}
    for target, aliases in alias_map.items():
        found = _find_column(out, aliases, required=(target in {"segment", "visit", "conversion", "spend"}))
        if found and found != target:
            rename_map[found] = target

    if rename_map:
        out = out.rename(columns=rename_map)

    return out


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate Hillstrom-like data.

    Steps:
    - normalize schema names for core columns
    - ensure required columns exist
    - normalize segment labels
    - construct binary treatment indicator where:
      * treatment=1 means any email arm (`Mens E-Mail` or `Womens E-Mail`)
      * treatment=0 means `No E-Mail`

    Notes
    -----
    This project intentionally collapses the original 3-arm setup into a binary
    treatment simplification for uplift modeling.
    """
    cleaned = _normalize_schema(df).copy()

    required = [DEFAULT_SEGMENT_COL, "visit", "conversion", "spend"]
    missing = [c for c in required if c not in cleaned.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    cleaned[DEFAULT_SEGMENT_COL] = cleaned[DEFAULT_SEGMENT_COL].astype(str).str.strip()

    # Normalize common segment typos/variants.
    cleaned[DEFAULT_SEGMENT_COL] = cleaned[DEFAULT_SEGMENT_COL].replace(
        {
            "No Email": "No E-Mail",
            "No_E-Mail": "No E-Mail",
            "Mens Email": "Mens E-Mail",
            "Womens Email": "Womens E-Mail",
        }
    )

    allowed_segments = {"No E-Mail", "Mens E-Mail", "Womens E-Mail"}
    if not set(cleaned[DEFAULT_SEGMENT_COL].unique()).issubset(allowed_segments):
        # Keep working for Hillstrom-like variants, but warn by error message format.
        unexpected = sorted(set(cleaned[DEFAULT_SEGMENT_COL].unique()) - allowed_segments)
        raise ValueError(
            "Unexpected segment labels found. Expected Hillstrom-like labels "
            f"{sorted(allowed_segments)} but got extras: {unexpected}"
        )

    cleaned[DEFAULT_TREATMENT_COL] = (cleaned[DEFAULT_SEGMENT_COL] != "No E-Mail").astype(int)

    # Cast outcomes to numeric if needed.
    for col in ["visit", "conversion", "spend"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Conservative missing handling: fill binary outcomes with 0, spend with 0.
    cleaned["visit"] = cleaned["visit"].fillna(0.0)
    cleaned["conversion"] = cleaned["conversion"].fillna(0.0)
    cleaned["spend"] = cleaned["spend"].fillna(0.0)

    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def _infer_pre_treatment_features(df: pd.DataFrame, treatment_col: str) -> list[str]:
    """Infer pre-treatment covariates by dropping outcomes/treatment indicators."""
    excluded = {
        "visit",
        "conversion",
        "spend",
        "segment",
        treatment_col,
        "treatment_group",
        "treatment_label",
    }
    return [c for c in df.columns if c not in excluded]


def prepare_features(
    df: pd.DataFrame,
    treatment_col: str = DEFAULT_TREATMENT_COL,
    feature_cols: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare one-hot encoded pre-treatment feature matrix.

    Parameters
    ----------
    df:
        Input dataframe.
    treatment_col:
        Treatment indicator column.
    feature_cols:
        Optional explicit pre-treatment features. If omitted, features are inferred
        by excluding outcomes (`visit`, `conversion`, `spend`), segment label,
        and treatment indicator.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Encoded feature matrix and list of encoded feature names.
    """
    if treatment_col not in df.columns:
        raise ValueError(f"Treatment column `{treatment_col}` not found.")

    pre_treatment_cols = list(feature_cols) if feature_cols is not None else _infer_pre_treatment_features(df, treatment_col)
    if not pre_treatment_cols:
        raise ValueError("No pre-treatment covariates available.")

    X_raw = df[pre_treatment_cols].copy()

    # Fill missing values conservatively before encoding.
    for col in X_raw.columns:
        if pd.api.types.is_numeric_dtype(X_raw[col]):
            X_raw[col] = X_raw[col].fillna(X_raw[col].median())
        else:
            X_raw[col] = X_raw[col].astype(str).fillna("missing")

    categorical_cols = X_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)
    return X, X.columns.tolist()


def split_features_outcomes_treatment(
    df: pd.DataFrame,
    outcome_col: str = "conversion",
    treatment_col: str = DEFAULT_TREATMENT_COL,
    feature_cols: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into features, outcome, and treatment vectors."""
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column `{outcome_col}` not found.")
    if treatment_col not in df.columns:
        raise ValueError(f"Treatment column `{treatment_col}` not found.")

    X, _ = prepare_features(df, treatment_col=treatment_col, feature_cols=feature_cols)
    y = pd.to_numeric(df[outcome_col], errors="coerce").fillna(0.0).astype(float)
    t = pd.to_numeric(df[treatment_col], errors="coerce").fillna(0).astype(int)
    return X, y, t


def create_train_valid_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    treatment: pd.Series,
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Dict[str, pd.DataFrame | pd.Series]:
    """Create reproducible train/validation/test splits with shared logic.

    Parameters
    ----------
    X, y, treatment:
        Modeling inputs with aligned indices.
    test_size:
        Fraction reserved for final holdout test set.
    valid_size:
        Fraction of remaining train_val set used for validation.
    random_state:
        Global seed for deterministic splits.
    stratify:
        If True, stratify on combined treatment+outcome label.

    Returns
    -------
    dict
        Keys include `X_train`, `X_valid`, `X_test`, `y_train`, `y_valid`,
        `y_test`, `t_train`, `t_valid`, `t_test`.
    """
    if not (0 < test_size < 1):
        raise ValueError("test_size must be in (0, 1).")
    if not (0 < valid_size < 1):
        raise ValueError("valid_size must be in (0, 1).")

    X_df = X.reset_index(drop=True)
    y_s = y.reset_index(drop=True)
    t_s = treatment.reset_index(drop=True)

    if len(X_df) != len(y_s) or len(y_s) != len(t_s):
        raise ValueError("X, y, and treatment must have the same length.")

    strat_train_test = (t_s.astype(str) + "_" + y_s.astype(int).astype(str)) if stratify else None

    X_train_val, X_test, y_train_val, y_test, t_train_val, t_test = train_test_split(
        X_df,
        y_s,
        t_s,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_train_test,
    )

    strat_train_valid = (
        t_train_val.astype(str) + "_" + y_train_val.astype(int).astype(str)
        if stratify
        else None
    )

    X_train, X_valid, y_train, y_valid, t_train, t_valid = train_test_split(
        X_train_val,
        y_train_val,
        t_train_val,
        test_size=valid_size,
        random_state=random_state,
        stratify=strat_train_valid,
    )

    return {
        "X_train": X_train.reset_index(drop=True),
        "X_valid": X_valid.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_valid": y_valid.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "t_train": t_train.reset_index(drop=True),
        "t_valid": t_valid.reset_index(drop=True),
        "t_test": t_test.reset_index(drop=True),
    }
