"""Serving tests: schema-valid predictions, errors, health, dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from pipeline.api.server import create_app
from pipeline.config import Settings
from pipeline.data.features import prepare_features, save_features
from pipeline.lifecycle.registry import SqliteRegistry
from pipeline.models.orchestrator import train_all
from pipeline.schemas.artifact import ArtifactManifest
from tests.fixtures.cmapss_synthetic import generate_synthetic_cmapss


def _boot_app(tmp_path: Path):
    data = generate_synthetic_cmapss(
        subset="FD002",
        n_train_engines=10,
        n_test_engines=1,
        seed=21,
        min_lifecycle=60,
        max_lifecycle=80,
    )
    features, _, encoder = prepare_features(data.train, window_size=5)
    processed_dir = tmp_path / "data" / "processed"
    save_features(
        features,
        encoder,
        processed_dir=processed_dir,
        window_size=5,
    )

    run_dir = tmp_path / "run"
    train_all(
        seed=21,
        run_dir=run_dir,
        train_raw=data.train,
        train_features=features,
        forecast_engine_sample=1,
    )
    registry = SqliteRegistry(tmp_path / "registry.db")
    for family in ("anomaly", "failure", "performance", "forecasting"):
        manifest = ArtifactManifest.model_validate_json(
            (run_dir / family / "manifest.json").read_text(encoding="utf-8")
        )
        registry.register(manifest)
        registry.promote(manifest.artifact_id, reason="test production")

    settings = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    app = create_app(settings=settings, registry=registry)
    return app, data


def _readings(data, unit: int = 1, count: int = 6) -> list[dict]:
    engine = data.train[data.train["unit"] == unit].sort_values("cycle").tail(count)
    readings = []
    for _, row in engine.iterrows():
        readings.append(
            {
                "cycle": int(row["cycle"]),
                "operating_settings": (
                    float(row["op1"]),
                    float(row["op2"]),
                    float(row["op3"]),
                ),
                "sensors": tuple(float(row[f"s{i}"]) for i in range(1, 22)),
            }
        )
    return readings


def test_predict_endpoint_returns_schema_valid_response(tmp_path: Path) -> None:
    app, data = _boot_app(tmp_path)
    client = TestClient(app)
    response = client.post(
        "/v1/predict",
        json={
            "subset": "FD002",
            "engine_id": "unit-1",
            "readings": _readings(data),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "request_id" in payload
    assert payload["engine_id"] == "unit-1"
    assert 0.0 <= payload["failure"]["probability"] <= 1.0
    assert (
        payload["rul"]["lower"] <= payload["rul"]["point"] <= payload["rul"]["upper"]
    )
    assert len(payload["model_versions"]) == 4
    assert all(item["stage"] == "production" for item in payload["model_versions"])


def test_invalid_input_rejected_with_problem_details(tmp_path: Path) -> None:
    app, _ = _boot_app(tmp_path)
    client = TestClient(app)
    response = client.post(
        "/v1/predict",
        json={
            "subset": "FD002",
            "engine_id": "unit-1",
            "readings": [{"cycle": 1, "operating_settings": [1, 2, 3], "sensors": []}],
        },
    )
    assert response.status_code == 422
    payload = response.json()
    assert payload["title"] == "Validation Error"
    assert payload["status"] == 422


def test_health_and_models_endpoints(tmp_path: Path) -> None:
    app, data = _boot_app(tmp_path)
    client = TestClient(app)
    client.post(
        "/v1/predict",
        json={"subset": "FD002", "engine_id": "e", "readings": _readings(data)},
    )
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["p50_latency_ms"] is not None
    assert set(health["model_versions"]) == {
        "anomaly",
        "failure",
        "performance",
        "forecasting",
    }
    models = client.get("/v1/models").json()
    assert len(models) == 4


def test_dashboard_serves_and_partials_render(tmp_path: Path) -> None:
    app, _ = _boot_app(tmp_path)
    client = TestClient(app)
    for path in (
        "/dashboard",
        "/dashboard/partials/registry",
        "/dashboard/partials/drift",
        "/dashboard/partials/gates",
        "/dashboard/partials/predictions",
    ):
        response = client.get(path)
        assert response.status_code == 200
    assert "Model registry" in client.get("/dashboard").text
