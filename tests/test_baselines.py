"""Baseline model tests."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.eval.baselines import (
    evaluate_failure_baseline,
    evaluate_rul_baseline,
    fit_failure_baseline,
    fit_rul_baseline,
    persistence_forecast,
    persistence_mape,
)


def test_failure_baseline_prior() -> None:
    train = np.array([0, 0, 1, 1, 1, 1])
    prior = fit_failure_baseline(train)
    assert prior == pytest.approx(4 / 6)
    metrics = evaluate_failure_baseline(prior, np.array([0, 1, 0, 1, 1]))
    assert set(metrics) == {"auc", "average_precision", "brier", "log_loss"}


def test_rul_baseline_constant() -> None:
    constant = fit_rul_baseline(np.array([10, 20, 30, 40]))
    assert constant == pytest.approx(25.0)
    metrics = evaluate_rul_baseline(constant, np.array([10, 40]))
    assert metrics["rmse"] > 0


def test_persistence_forecast_and_mape() -> None:
    series = np.arange(1, 13, dtype=float)
    predictions, actual = persistence_forecast(series, horizon=3)
    np.testing.assert_array_equal(predictions, [9.0, 9.0, 9.0])
    np.testing.assert_array_equal(actual, [10.0, 11.0, 12.0])
    expected = np.mean([1.0 / 10, 2.0 / 11, 3.0 / 12])
    assert persistence_mape(series, horizon=3) == pytest.approx(expected)
