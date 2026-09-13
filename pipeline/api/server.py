"""FastAPI inference service: /v1/predict, /v1/models, /health (FR-13)."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline.config import Settings, get_settings
from pipeline.data.features import RegimeEncoder, build_feature_frame, feature_columns
from pipeline.data.loader import COLUMNS
from pipeline.lifecycle.events import EventLog
from pipeline.lifecycle.registry import SqliteRegistry
from pipeline.models.forecasting import ForecastingModel
from pipeline.models.orchestrator import run_dag
from pipeline.monitoring.dashboard import build_dashboard_router
from pipeline.monitoring.health import HealthTracker
from pipeline.schemas.api import (
    AnomalyResult,
    FailureResult,
    ForecastPoint,
    ModelVersionInfo,
    PredictRequest,
    PredictResponse,
    RulResult,
)
from pipeline.schemas.artifact import ArtifactManifest

_MONITORING_DIR = Path(__file__).resolve().parents[1] / "monitoring"


def _load_production_models(
    registry: SqliteRegistry,
) -> tuple[dict[str, Any], dict[str, ArtifactManifest]]:
    models: dict[str, Any] = {}
    manifests: dict[str, ArtifactManifest] = {}
    for family in ("anomaly", "failure", "performance", "forecasting"):
        entry = registry.get_production(family)
        if entry is None:
            raise RuntimeError(
                f"no production {family} artifact in the registry; run the lifecycle demo first"
            )
        # The registry entry is authoritative for version/stage metadata; the
        # artifact path points at the serialized model.
        models[family] = joblib.load(Path(entry.manifest.artifact_path))
        manifests[family] = entry.manifest
    return models, manifests


def _readings_to_raw_frame(request: PredictRequest) -> pd.DataFrame:
    rows = []
    for reading in request.readings:
        settings_values = list(reading.operating_settings)
        rows.append(
            {
                "unit": 1,
                "cycle": reading.cycle,
                **dict(zip(["op1", "op2", "op3"], settings_values, strict=True)),
                **dict(zip([f"s{i}" for i in range(1, 22)], reading.sensors, strict=True)),
            }
        )
    frame = pd.DataFrame(rows, columns=COLUMNS)
    return frame.astype({"unit": "int64", "cycle": "int64"})


def _sensor_history(
    request: PredictRequest,
    sensor_index: int,
) -> np.ndarray:
    return np.asarray(
        [reading.sensors[sensor_index - 1] for reading in request.readings],
        dtype=float,
    )


def create_app(
    *,
    settings: Settings | None = None,
    registry: SqliteRegistry | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    registry = registry or SqliteRegistry()
    models, manifests = _load_production_models(registry)
    tracker = HealthTracker()
    encoder = RegimeEncoder.load(settings.processed_dir)

    app = FastAPI(
        title="industrial-ml-pipeline API",
        version="0.1.0",
        description="Predictive maintenance inference over the 4-model DAG.",
    )
    app.mount(
        "/static",
        StaticFiles(directory=_MONITORING_DIR / "static"),
        name="static",
    )
    app.include_router(
        build_dashboard_router(
            registry=registry,
            tracker=tracker,
            event_log=EventLog(settings.lifecycle_dir / "sim_events.jsonl"),
        )
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        tracker.record_error()
        return JSONResponse(
            status_code=422,
            content={
                "type": "about:blank",
                "title": "Validation Error",
                "status": 422,
                "detail": jsonable_encoder(
                    exc.errors(),
                    custom_encoder={ValueError: lambda error: str(error)},
                ),
                "instance": str(request.url.path),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        tracker.record_error()
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "type": "about:blank",
                "title": exc.detail,
                "status": exc.status_code,
                "detail": exc.detail,
                "instance": str(request.url.path),
            },
        )

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "model_versions": {
                family: {
                    "artifact_id": manifests[family].artifact_id,
                    "version": manifests[family].version,
                }
                for family in manifests
            },
            **tracker.summary(),
        }

    @app.get("/v1/models", tags=["registry"])
    def models_list() -> list[dict[str, object]]:
        return [
            {
                "family": entry.manifest.family,
                "artifact_id": entry.manifest.artifact_id,
                "version": entry.manifest.version,
                "stage": entry.manifest.stage,
                "metrics": entry.manifest.metrics,
                "registered_at": entry.registered_at.isoformat(),
            }
            for entry in registry.list()
        ]

    @app.post("/v1/predict", response_model=PredictResponse, tags=["inference"])
    def predict(request: PredictRequest) -> PredictResponse:
        started = time.perf_counter()
        raw = _readings_to_raw_frame(request)
        features = build_feature_frame(raw, encoder, window_size=settings.window_size)
        last_row = features.iloc[[-1]]
        feature_row = last_row[feature_columns()].to_numpy(dtype=float).ravel()

        dag = run_dag(
            {
                "anomaly": models["anomaly"],
                "failure": models["failure"],
                "rul": models["performance"],
            },
            feature_row,
        )
        anomaly = AnomalyResult(
            score=max(0.0, float(dag["anomaly"]["score"])),
            flag=bool(dag["anomaly"]["flag"]),
        )
        failure = FailureResult(
            probability=float(dag["failure"]["probability"]),
            horizon_cycles=settings.failure_horizon_cycles,
            calibrated=True,
        )
        rul = RulResult(
            point=float(dag["rul"]["point"]),
            lower=float(dag["rul"]["lower"]),
            upper=float(dag["rul"]["upper"]),
            interval_level=float(dag["rul"]["interval_level"]),
        )

        forecasts: list[ForecastPoint] = []
        if request.forecast_horizon_cycles:
            forecasting: ForecastingModel = models["forecasting"]
            for sensor in forecasting.sensors:
                history = _sensor_history(request, int(sensor[1:]))
                if len(history) > forecasting.horizon + 2:
                    predicted = forecasting.fit_forecast(
                        history,
                        request.forecast_horizon_cycles,
                    )
                    for offset, value in enumerate(predicted, start=1):
                        forecasts.append(
                            ForecastPoint(
                                sensor_index=int(sensor[1:]),
                                cycle_offset=offset,
                                value=float(value),
                            )
                        )

        latency_ms = (time.perf_counter() - started) * 1000.0
        response = PredictResponse(
            request_id=uuid.uuid4().hex,
            subset=request.subset,
            engine_id=request.engine_id,
            final_cycle=request.readings[-1].cycle,
            anomaly=anomaly,
            failure=failure,
            rul=rul,
            forecasts=forecasts,
            model_versions=[
                ModelVersionInfo(
                    family=family,
                    artifact_id=manifests[family].artifact_id,
                    version=manifests[family].version,
                    stage="production",
                )
                for family in ("anomaly", "failure", "performance", "forecasting")
            ],
            latency_ms=latency_ms,
            generated_at=pd.Timestamp.now(tz="UTC").to_pydatetime(),
        )
        tracker.record_prediction(
            {
                "subset": request.subset,
                "engine_id": str(request.engine_id),
                "final_cycle": request.readings[-1].cycle,
                "failure_probability": round(failure.probability, 4),
                "rul_point": round(rul.point, 1),
                "anomaly_flag": anomaly.flag,
            },
            latency_ms,
        )
        return response

    return app
