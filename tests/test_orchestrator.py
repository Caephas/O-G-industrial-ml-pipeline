"""Orchestrator and DAG tests: ordering, determinism, artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.models.orchestrator import run_dag, train_all
from pipeline.schemas.artifact import ArtifactManifest
from tests.fixtures.cmapss_synthetic import generate_synthetic_cmapss


class _FakeAnomaly:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def predict(self, _: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.order.append("anomaly")
        return np.array([1.0]), np.array([True])


class _FakeFailure:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def predict_proba(self, _: np.ndarray) -> np.ndarray:
        self.order.append("failure")
        return np.array([0.8])


class _FakeRul:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def predict(self, _: np.ndarray) -> dict[str, np.ndarray]:
        self.order.append("rul")
        return {
            "point": np.array([25.0]),
            "lower": np.array([10.0]),
            "upper": np.array([40.0]),
            "interval_level": np.array([0.9]),
        }


def test_run_dag_executes_models_in_order() -> None:
    order: list[str] = []
    models = {
        "anomaly": _FakeAnomaly(order),
        "failure": _FakeFailure(order),
        "rul": _FakeRul(order),
    }
    result = run_dag(models, np.zeros(5))
    assert order == ["anomaly", "failure", "rul"]
    assert result["dag_order"] == ["anomaly", "failure", "rul"]
    assert result["anomaly"]["flag"] is True
    assert result["rul"]["point"] == 25.0


def _synthetic_inputs(seed: int = 9):
    data = generate_synthetic_cmapss(
        subset="FD002",
        n_train_engines=10,
        n_test_engines=1,
        seed=seed,
        min_lifecycle=60,
        max_lifecycle=80,
    )
    from pipeline.data.features import prepare_features

    features, _, _ = prepare_features(data.train, window_size=5)
    return data.train, features


def _assert_metrics_close(first: dict, second: dict) -> None:
    for family in ("anomaly", "failure", "performance", "forecasting"):
        left = first[family]
        right = second[family]
        assert set(left) == set(right)
        for key in left:
            if left[key] is None or right[key] is None:
                assert left[key] is None and right[key] is None
            else:
                assert left[key] == pytest.approx(right[key], rel=1e-6)


def test_train_all_is_deterministic_and_packages_artifacts(tmp_path: Path) -> None:
    raw, features = _synthetic_inputs()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = train_all(
        seed=9,
        run_dir=first_dir,
        train_raw=raw,
        train_features=features,
        forecast_engine_sample=2,
    )
    second = train_all(
        seed=9,
        run_dir=second_dir,
        train_raw=raw,
        train_features=features,
        forecast_engine_sample=2,
    )
    _assert_metrics_close(first["metrics"], second["metrics"])
    assert first["git_sha"] == second["git_sha"]

    for family in ("anomaly", "failure", "performance", "forecasting"):
        manifest_path = first_dir / family / "manifest.json"
        assert manifest_path.exists()
        assert (first_dir / family / "model.joblib").exists()
        manifest = ArtifactManifest.model_validate_json(manifest_path.read_text())
        assert manifest.family == family
        assert manifest.version >= 1
        assert manifest.stage == "staging"
        assert manifest.thresholds
        assert "model.joblib" in manifest.files
    assert (first_dir / "run.json").exists()
