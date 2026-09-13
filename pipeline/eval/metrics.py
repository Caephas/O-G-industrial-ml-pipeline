"""Metric implementations for the evaluation harness (FR-04).

Metric keys follow the canonical names in ``pipeline.schemas.metrics`` so
records, manifests, and thresholds stay consistent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from pipeline.config import PROJECT_ROOT


def load_thresholds(path: Path | None = None) -> dict[str, dict[str, float]]:
    """Load the committed acceptance thresholds (FR-04)."""
    threshold_path = path or PROJECT_ROOT / "metrics" / "thresholds.yaml"
    with threshold_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric PHM08 scoring function (lower is better)."""
    true = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    error = predicted - true
    penalties = np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)
    return float(penalties.sum())


def evaluate_failure(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """AUC, average precision, Brier, and log loss for a binary classifier."""
    true = np.asarray(y_true)
    proba = np.asarray(y_proba, dtype=float)
    classes = np.unique(true)
    if len(classes) < 2:
        raise ValueError("evaluation requires both classes in the ground truth")
    return {
        "auc": float(roc_auc_score(true, proba)),
        "average_precision": float(average_precision_score(true, proba)),
        "brier": float(brier_score_loss(true, proba)),
        "log_loss": float(log_loss(true, proba)),
    }


def evaluate_rul(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE and NASA score for RUL regression."""
    true = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((true - predicted) ** 2)))
    return {"rmse": rmse, "nasa_score": nasa_score(true, predicted)}


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAPE and RMSE for per-sensor forecasts."""
    true = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    mape = float(np.mean(np.abs((true - predicted) / true)))
    rmse = float(np.sqrt(np.mean((true - predicted) ** 2)))
    return {"mape": mape, "rmse": rmse}
