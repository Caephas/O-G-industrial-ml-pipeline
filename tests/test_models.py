"""Model family tests: fit/predict behavior on synthetic data."""

from __future__ import annotations

import numpy as np

from pipeline.models.anomaly import AnomalyModel
from pipeline.models.failure import FailureModel
from pipeline.models.forecasting import ForecastingModel
from pipeline.models.performance import RulModel
from tests.fixtures.cmapss_synthetic import generate_synthetic_cmapss


def _split_rows(X: np.ndarray, y: np.ndarray, count: int):
    return X[:count], y[:count], X[count:], y[count:]


def test_anomaly_model_scores_outliers_higher() -> None:
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(0, 1, size=(90, 5)), rng.normal(8, 1, size=(10, 5))])
    rul = np.concatenate([np.full(90, 100.0), np.full(10, 5.0)])
    model = AnomalyModel(seed=3).fit(X, rul, horizon_cycles=30)
    assert 0.0 < model.contamination <= 0.5
    scores, flags = model.predict(X)
    assert scores.shape == (100,)
    assert flags.shape == (100,)
    assert scores[-10:].mean() > scores[:90].mean()


def test_failure_model_probabilities_separate_classes() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, size=(400, 6))
    y = (X[:, 0] > 0.0).astype(int)
    fit_end = 280
    calib_end = 360
    model = FailureModel(seed=5).fit(
        X[:fit_end],
        y[:fit_end],
        X[fit_end:calib_end],
        y[fit_end:calib_end],
    )
    proba = model.predict_proba(X[calib_end:])
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert proba[y[calib_end:] == 1].mean() > proba[y[calib_end:] == 0].mean()


def test_rul_model_interval_contains_truth() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, size=(1400, 8))
    y = np.clip(
        50.0 + 6.0 * X[:, 0] - 3.0 * X[:, 1] + rng.normal(0, 4.0, size=1400),
        1.0,
        125.0,
    )
    model = RulModel(seed=7, n_estimators=50).fit(
        X[:900],
        y[:900],
        X[900:1100],
        y[900:1100],
    )
    assert model.scale_factor > 1.0
    result = model.predict(X[1100:])
    coverage = float(np.mean((result["lower"] <= y[1100:]) & (y[1100:] <= result["upper"])))
    assert coverage >= 0.8
    assert np.all(result["lower"] <= result["point"])
    assert np.all(result["point"] <= result["upper"])


def test_forecasting_selects_trending_sensors_and_evaluates() -> None:
    data = generate_synthetic_cmapss(
        subset="FD002",
        n_train_engines=6,
        n_test_engines=1,
        seed=11,
        min_lifecycle=40,
        max_lifecycle=60,
    )
    trending = data.train.copy()
    for sensor in trending.columns:
        if sensor not in {"unit", "cycle", "op1", "op2", "op3"}:
            trending[sensor] = 100.0
    trending["s2"] = 100.0 + trending["cycle"].astype(float)
    from pipeline.data.features import prepare_features

    features, _, _ = prepare_features(trending, window_size=5)
    selected = ForecastingModel.select_sensors(features, count=1)
    assert selected == ["s2"]

    model = ForecastingModel(sensors=["s2"], horizon=3, seed=11)
    result = model.evaluate(data.train, features, engine_sample=2)
    assert result["engine_count"] == 2
    assert "mape" in result
    assert "mape_ratio_vs_persistence" in result
