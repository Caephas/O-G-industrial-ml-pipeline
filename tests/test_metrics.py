"""Metric implementation tests."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.eval.metrics import (
    evaluate_failure,
    evaluate_forecast,
    evaluate_rul,
    load_thresholds,
    nasa_score,
)


def test_failure_metrics_perfect_ranking() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate_failure(y_true, y_proba)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["brier"] < 0.1
    assert metrics["log_loss"] < 0.5


def test_failure_metrics_random_probs() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.full(4, 0.5)
    metrics = evaluate_failure(y_true, y_proba)
    assert metrics["auc"] == pytest.approx(0.5)
    assert metrics["brier"] == pytest.approx(0.25)


def test_failure_metrics_require_both_classes() -> None:
    with pytest.raises(ValueError):
        evaluate_failure(np.array([1, 1, 1]), np.array([0.9, 0.9, 0.9]))


def test_nasa_score_asymmetry() -> None:
    assert nasa_score([10.0], [0.0]) == pytest.approx(np.exp(10.0 / 13.0) - 1.0)
    assert nasa_score([10.0], [20.0]) == pytest.approx(np.exp(1.0) - 1.0)


def test_rul_metrics_perfect() -> None:
    metrics = evaluate_rul(np.array([10, 20, 30]), np.array([10, 20, 30]))
    assert metrics["rmse"] == 0.0
    assert metrics["nasa_score"] == 0.0


def test_forecast_metrics() -> None:
    metrics = evaluate_forecast(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert metrics["mape"] == pytest.approx(0.10)
    assert metrics["rmse"] == pytest.approx(np.sqrt(250.0))


def test_committed_thresholds_are_sane() -> None:
    thresholds = load_thresholds()
    assert thresholds["failure"]["min_auc"] > 0.5
    assert thresholds["rul"]["relative_rmse_gain_vs_baseline"] > 0
    assert thresholds["forecast"]["mape_ratio_vs_persistence_max"] < 1.0
