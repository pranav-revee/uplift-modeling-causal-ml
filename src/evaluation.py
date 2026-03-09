"""Uplift evaluation metrics and policy-impact estimation helpers."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _to_1d(values) -> np.ndarray:
    """Convert array-like input to 1D NumPy array."""
    return np.asarray(values).reshape(-1)


def _validate_inputs(y_true, treatment, uplift_score) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate uplift metric inputs and return aligned NumPy arrays."""
    y = _to_1d(y_true).astype(float)
    t = _to_1d(treatment).astype(int)
    s = _to_1d(uplift_score).astype(float)

    if not (len(y) == len(t) == len(s)):
        raise ValueError("y_true, treatment, and uplift_score must have equal lengths.")

    if not set(np.unique(t)).issubset({0, 1}):
        raise ValueError("treatment must be binary in {0, 1}.")

    if t.sum() == 0 or (1 - t).sum() == 0:
        raise ValueError("Both treatment and control observations are required.")

    return y, t, s


def _rank_frame(y_true, treatment, uplift_score) -> pd.DataFrame:
    """Build a sorted dataframe by descending uplift score."""
    y, t, s = _validate_inputs(y_true, treatment, uplift_score)
    rank_df = pd.DataFrame({"y": y, "t": t, "score": s}).sort_values("score", ascending=False).reset_index(drop=True)
    rank_df["population"] = np.arange(1, len(rank_df) + 1)
    rank_df["fraction_targeted"] = rank_df["population"] / len(rank_df)
    rank_df["cum_treated_n"] = rank_df["t"].cumsum()
    rank_df["cum_control_n"] = (1 - rank_df["t"]).cumsum()
    rank_df["cum_treated_y"] = (rank_df["y"] * rank_df["t"]).cumsum()
    rank_df["cum_control_y"] = (rank_df["y"] * (1 - rank_df["t"])).cumsum()
    return rank_df


def qini_curve(y_true, treatment, uplift_score, n_points: Optional[int] = None) -> pd.DataFrame:
    """Compute Qini curve values across ranked population.

    Returns cumulative incremental outcomes over ranking depth:
    `cum_y_treated - cum_y_control * (cum_n_treated / cum_n_control)`.
    """
    ranked = _rank_frame(y_true, treatment, uplift_score)
    expected_control_for_treated = np.divide(
        ranked["cum_control_y"] * ranked["cum_treated_n"],
        ranked["cum_control_n"],
        out=np.zeros(len(ranked), dtype=float),
        where=ranked["cum_control_n"] > 0,
    )
    ranked["incremental_outcomes"] = ranked["cum_treated_y"] - expected_control_for_treated

    out = ranked[["population", "fraction_targeted", "incremental_outcomes"]].copy()
    if n_points is not None and 1 < n_points < len(out):
        idx = np.linspace(0, len(out) - 1, n_points).astype(int)
        out = out.iloc[idx].reset_index(drop=True)
    return out


def uplift_curve(y_true, treatment, uplift_score, n_points: Optional[int] = None) -> pd.DataFrame:
    """Compute uplift curve (incremental outcomes per targeted individual)."""
    qini = qini_curve(y_true, treatment, uplift_score, n_points=n_points)
    out = qini.copy()
    out["uplift"] = np.divide(
        out["incremental_outcomes"],
        out["population"],
        out=np.zeros(len(out), dtype=float),
        where=out["population"] > 0,
    )
    return out[["population", "fraction_targeted", "uplift", "incremental_outcomes"]]


def qini_coefficient(y_true, treatment, uplift_score) -> float:
    """Compute Qini coefficient as area above random baseline."""
    qini = qini_curve(y_true, treatment, uplift_score)
    y, t, _ = _validate_inputs(y_true, treatment, uplift_score)

    n_t = t.sum()
    n_c = len(t) - n_t
    total_incremental = y[t == 1].sum() - y[t == 0].sum() * (n_t / n_c)

    baseline = qini["fraction_targeted"].to_numpy() * total_incremental
    area = np.trapz(qini["incremental_outcomes"].to_numpy() - baseline, qini["fraction_targeted"].to_numpy())
    return float(area)


def auuc_score(y_true, treatment, uplift_score) -> float:
    """Compute AUUC (area under uplift curve in incremental outcomes space)."""
    qini = qini_curve(y_true, treatment, uplift_score)
    area = np.trapz(qini["incremental_outcomes"].to_numpy(), qini["fraction_targeted"].to_numpy())
    return float(area)


def cumulative_gain_curve(y_true, treatment, uplift_score, n_points: Optional[int] = None) -> pd.DataFrame:
    """Compute cumulative gain curve from uplift ranking."""
    qini = qini_curve(y_true, treatment, uplift_score, n_points=n_points)
    out = qini.copy()
    out["cumulative_gain"] = np.divide(
        out["incremental_outcomes"],
        out["population"],
        out=np.zeros(len(out), dtype=float),
        where=out["population"] > 0,
    )
    return out[["population", "fraction_targeted", "cumulative_gain", "incremental_outcomes"]]


def treatment_control_counts_by_decile(y_true, treatment, uplift_score, n_deciles: int = 10) -> pd.DataFrame:
    """Count treated and control observations by uplift decile."""
    ranked = _rank_frame(y_true, treatment, uplift_score)
    ranked["decile"] = pd.qcut(ranked.index + 1, q=n_deciles, labels=False) + 1

    counts = (
        ranked.groupby("decile")["t"]
        .agg(n="count", treated_n="sum")
        .reset_index()
    )
    counts["control_n"] = counts["n"] - counts["treated_n"]
    counts["treated_share"] = counts["treated_n"] / counts["n"]
    return counts


def uplift_by_decile(y_true, treatment, uplift_score, n_deciles: int = 10) -> pd.DataFrame:
    """Compute uplift statistics by decile of predicted uplift score."""
    ranked = _rank_frame(y_true, treatment, uplift_score)
    ranked["decile"] = pd.qcut(ranked.index + 1, q=n_deciles, labels=False) + 1

    rows = []
    for decile, grp in ranked.groupby("decile", sort=True):
        treated = grp[grp["t"] == 1]
        control = grp[grp["t"] == 0]
        tr_rate = float(treated["y"].mean()) if len(treated) else np.nan
        cr_rate = float(control["y"].mean()) if len(control) else np.nan
        uplift = tr_rate - cr_rate if np.isfinite(tr_rate) and np.isfinite(cr_rate) else np.nan
        rows.append(
            {
                "decile": int(decile),
                "n": int(len(grp)),
                "treated_n": int(len(treated)),
                "control_n": int(len(control)),
                "treated_rate": tr_rate,
                "control_rate": cr_rate,
                "uplift": uplift,
            }
        )

    return pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)


def _top_k_mask(scores: np.ndarray, budget: float) -> np.ndarray:
    """Create a binary selection mask for top-k fraction by score."""
    if not 0 < budget <= 1:
        raise ValueError("budget must be in (0, 1].")
    n = len(scores)
    k = max(1, int(np.ceil(n * budget)))
    order = np.argsort(-scores)
    mask = np.zeros(n, dtype=int)
    mask[order[:k]] = 1
    return mask


def incremental_conversions_at_budget(y_true, treatment, uplift_score, budget: float) -> dict:
    """Estimate incremental conversions for top-k targeting.

    Estimation approach:
    - rank users by score
    - select top `budget` fraction
    - compute conversion rate difference between treated and control in that
      selected bucket
    - multiply rate difference by number targeted

    This is an offline policy estimate, not direct production counterfactual truth.
    """
    y, t, s = _validate_inputs(y_true, treatment, uplift_score)
    mask = _top_k_mask(s, budget)

    selected_y = y[mask == 1]
    selected_t = t[mask == 1]

    treated = selected_y[selected_t == 1]
    control = selected_y[selected_t == 0]

    treated_rate = float(np.mean(treated)) if treated.size else np.nan
    control_rate = float(np.mean(control)) if control.size else np.nan

    if not (np.isfinite(treated_rate) and np.isfinite(control_rate)):
        incremental_rate = np.nan
        incremental_conversions = np.nan
    else:
        incremental_rate = treated_rate - control_rate
        incremental_conversions = incremental_rate * float(mask.sum())

    return {
        "budget": float(budget),
        "targeted_count": int(mask.sum()),
        "targeted_rate": float(mask.mean()),
        "treated_rate_in_target": treated_rate,
        "control_rate_in_target": control_rate,
        "incremental_rate": incremental_rate,
        "incremental_conversions": incremental_conversions,
    }


def policy_value_at_budget(y_true, treatment, uplift_score, budget: float) -> dict:
    """Estimate policy value at a budget in outcome-units-per-customer.

    Policy value is estimated as incremental conversions divided by full
    population size.
    """
    y = _to_1d(y_true).astype(float)
    res = incremental_conversions_at_budget(y_true, treatment, uplift_score, budget)
    if np.isnan(res["incremental_conversions"]):
        policy_value = np.nan
    else:
        policy_value = float(res["incremental_conversions"] / len(y))

    return {
        **res,
        "policy_value": policy_value,
    }


def incremental_revenue_at_budget(
    y_true,
    treatment,
    uplift_score,
    budget: float,
    average_order_value: float,
) -> dict:
    """Estimate incremental revenue using incremental conversions and AOV assumption."""
    if average_order_value < 0:
        raise ValueError("average_order_value must be non-negative.")

    res = incremental_conversions_at_budget(y_true, treatment, uplift_score, budget)
    if np.isnan(res["incremental_conversions"]):
        incremental_revenue = np.nan
    else:
        incremental_revenue = float(res["incremental_conversions"] * average_order_value)

    return {
        **res,
        "average_order_value": float(average_order_value),
        "incremental_revenue": incremental_revenue,
    }


def budget_sensitivity_table(
    y_true,
    treatment,
    uplift_score,
    budgets: Iterable[float],
    average_order_value: Optional[float] = None,
) -> pd.DataFrame:
    """Evaluate policy outcomes across a sequence of budgets."""
    rows = []
    for b in sorted(set(float(x) for x in budgets)):
        if average_order_value is None:
            row = policy_value_at_budget(y_true, treatment, uplift_score, b)
            row["average_order_value"] = np.nan
            row["incremental_revenue"] = np.nan
        else:
            row = incremental_revenue_at_budget(
                y_true,
                treatment,
                uplift_score,
                b,
                average_order_value=average_order_value,
            )
            y_arr = _to_1d(y_true)
            row["policy_value"] = (
                row["incremental_conversions"] / len(y_arr) if np.isfinite(row["incremental_conversions"]) else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("budget").reset_index(drop=True)
