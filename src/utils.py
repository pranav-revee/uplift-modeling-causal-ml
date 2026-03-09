"""Utility helpers for reproducibility, plotting, and formatting."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Iterable, Tuple

import numpy as np


def set_seed(seed: int = 42, include_torch: bool = False) -> None:
    """Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed:
        Seed value for Python and NumPy.
    include_torch:
        If True, also seed PyTorch RNGs. Defaults to False to keep lightweight
        environments stable when Torch is unavailable.
    """
    random.seed(seed)
    np.random.seed(seed)

    if not include_torch:
        return

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # No dedicated MPS seed call; manual_seed covers shared RNG path.
            pass
    except Exception:
        # Optional dependency: keep silent for portable notebooks.
        pass


def ensure_output_dirs(project_root: str | Path) -> dict[str, Path]:
    """Ensure standard output directories exist and return their paths."""
    root = Path(project_root)
    out_root = root / "outputs"
    fig_dir = out_root / "figures"
    table_dir = out_root / "tables"
    model_dir = out_root / "models"

    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (out_root / ".gitkeep").touch(exist_ok=True)

    return {
        "outputs": out_root,
        "figures": fig_dir,
        "tables": table_dir,
        "models": model_dir,
    }


def save_plot(fig, path: str | Path, dpi: int = 300) -> None:
    """Save a Matplotlib figure with consistent high-resolution defaults."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")


def standardized_mean_difference(treated_values: Iterable[float], control_values: Iterable[float]) -> float:
    """Compute standardized mean difference between treated and control samples."""
    x_t = np.asarray(list(treated_values), dtype=float)
    x_c = np.asarray(list(control_values), dtype=float)

    if x_t.size == 0 or x_c.size == 0:
        raise ValueError("Both treated_values and control_values must be non-empty.")

    mean_diff = x_t.mean() - x_c.mean()
    pooled_var = 0.5 * (x_t.var(ddof=1) + x_c.var(ddof=1))
    if pooled_var <= 1e-12:
        return 0.0
    return float(mean_diff / np.sqrt(pooled_var))


def bootstrap_confidence_interval(
    values: Iterable[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Compute percentile bootstrap confidence interval for a statistic."""
    sample = np.asarray(list(values), dtype=float)
    if sample.size == 0:
        raise ValueError("values must be non-empty.")

    rng = np.random.default_rng(random_state)
    stats = []
    for _ in range(n_bootstrap):
        draw = rng.choice(sample, size=sample.size, replace=True)
        stats.append(float(statistic(draw)))

    lower = float(np.quantile(stats, alpha / 2))
    upper = float(np.quantile(stats, 1 - alpha / 2))
    return lower, upper


def format_percent(value: float, decimals: int = 2) -> str:
    """Format decimal value as percentage string."""
    return f"{100 * value:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """Format numeric value as USD-style currency string."""
    return f"${value:,.{decimals}f}"
