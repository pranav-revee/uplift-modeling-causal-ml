"""Policy simulation helpers for budget-constrained uplift targeting."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .evaluation import incremental_revenue_at_budget, policy_value_at_budget


def rank_customers_for_targeting(uplift_score, customer_ids=None) -> pd.DataFrame:
    """Return customers ranked by predicted uplift (descending)."""
    scores = np.asarray(uplift_score).reshape(-1)
    n = len(scores)
    ids = np.arange(n) if customer_ids is None else np.asarray(customer_ids).reshape(-1)
    if len(ids) != n:
        raise ValueError("customer_ids length must match uplift_score length.")

    ranked = pd.DataFrame({"customer_id": ids, "uplift_score": scores}).sort_values(
        "uplift_score", ascending=False
    )
    ranked = ranked.reset_index(drop=True)
    ranked["rank"] = np.arange(1, n + 1)
    return ranked


def select_top_k_percent(score, k_percent: float) -> np.ndarray:
    """Select top-k percent of rows by score and return binary mask."""
    if not 0 < k_percent <= 1:
        raise ValueError("k_percent must be in (0, 1].")

    s = np.asarray(score).reshape(-1)
    n = len(s)
    k = max(1, int(np.ceil(n * k_percent)))
    order = np.argsort(-s)
    mask = np.zeros(n, dtype=int)
    mask[order[:k]] = 1
    return mask


def simulate_top_k_targeting(
    y_true,
    treatment,
    uplift_score,
    k_percent: float,
    average_order_value: Optional[float] = None,
) -> Dict[str, float]:
    """Simulate top-k targeting policy for a model ranking."""
    if average_order_value is None:
        return policy_value_at_budget(y_true, treatment, uplift_score, budget=k_percent)

    return incremental_revenue_at_budget(
        y_true,
        treatment,
        uplift_score,
        budget=k_percent,
        average_order_value=average_order_value,
    )


def simulate_budget_constrained_policy(
    y_true,
    treatment,
    score,
    budget: float,
    average_order_value: Optional[float] = None,
) -> Dict[str, float]:
    """Simulate a budget-constrained policy for any ranking score."""
    return simulate_top_k_targeting(
        y_true=y_true,
        treatment=treatment,
        uplift_score=score,
        k_percent=budget,
        average_order_value=average_order_value,
    )


def compare_targeting_policies(
    y_true,
    treatment,
    budget: float,
    uplift_score,
    response_score=None,
    random_state: int = 42,
    average_order_value: Optional[float] = None,
) -> pd.DataFrame:
    """Compare core targeting policies at a fixed budget.

    Policies included:
    - treat-none
    - treat-all
    - random targeting
    - propensity/response-only targeting (if response_score provided)
    - uplift-based targeting
    """
    y = np.asarray(y_true).reshape(-1)
    t = np.asarray(treatment).reshape(-1).astype(int)
    u = np.asarray(uplift_score).reshape(-1)

    if len(y) != len(t) or len(t) != len(u):
        raise ValueError("y_true, treatment, and uplift_score must have equal lengths.")

    rng = np.random.default_rng(random_state)

    # Treat-none and treat-all are represented as extreme scores.
    treat_none_score = np.full(len(y), -1.0)
    treat_all_score = np.full(len(y), 1.0)
    random_score = rng.uniform(0, 1, size=len(y))

    policy_scores = {
        "treat_none": treat_none_score,
        "treat_all": treat_all_score,
        "random_targeting": random_score,
        "uplift_targeting": u,
    }
    if response_score is not None:
        policy_scores["propensity_targeting"] = np.asarray(response_score).reshape(-1)

    rows = []
    for name, score in policy_scores.items():
        use_budget = 0.0 if name == "treat_none" else (1.0 if name == "treat_all" else budget)
        if use_budget == 0.0:
            # no targeting action
            row = {
                "policy": name,
                "budget": 0.0,
                "targeted_count": 0,
                "targeted_rate": 0.0,
                "incremental_conversions": 0.0,
                "policy_value": 0.0,
                "average_order_value": average_order_value if average_order_value is not None else np.nan,
                "incremental_revenue": 0.0 if average_order_value is not None else np.nan,
            }
        else:
            if average_order_value is None:
                row = policy_value_at_budget(y, t, score, use_budget)
                row["average_order_value"] = np.nan
                row["incremental_revenue"] = np.nan
            else:
                row = incremental_revenue_at_budget(y, t, score, use_budget, average_order_value)
                row["policy_value"] = row["incremental_conversions"] / len(y)
            row["policy"] = name
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_col = "incremental_revenue" if average_order_value is not None else "policy_value"
    return out.sort_values(sort_col, ascending=False).reset_index(drop=True)
