"""Baselines every model must beat (FR-04)."""

from __future__ import annotations

import numpy as np

from pipeline.eval.metrics import evaluate_failure, evaluate_forecast, evaluate_rul


def fit_failure_baseline(train_failure: np.ndarray) -> float:
    """Class prior; the majority-class probability baseline."""
    return float(np.mean(np.asarray(train_failure, dtype=float)))


def evaluate_failure_baseline(
    prior: float,
    validation_failure: np.ndarray,
) -> dict[str, float]:
    probs = np.full(len(validation_failure), prior)
    return evaluate_failure(validation_failure, probs)


def fit_rul_baseline(train_rul: np.ndarray) -> float:
    """Median training RUL as the constant predictor."""
    return float(np.median(np.asarray(train_rul, dtype=float)))


def evaluate_rul_baseline(
    constant: float,
    validation_rul: np.ndarray,
) -> dict[str, float]:
    return evaluate_rul(validation_rul, np.full(len(validation_rul), constant))


def persistence_forecast(series: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Naive forecast: repeat the last observed value ``horizon`` steps."""
    values = np.asarray(series, dtype=float)
    if len(values) <= horizon:
        raise ValueError("series must be longer than the forecast horizon")
    context_end = len(values) - horizon
    predictions = np.full(horizon, values[context_end - 1])
    actual = values[context_end:]
    return predictions, actual


def persistence_mape(series: np.ndarray, horizon: int) -> float:
    predictions, actual = persistence_forecast(series, horizon)
    return evaluate_forecast(actual, predictions)["mape"]
