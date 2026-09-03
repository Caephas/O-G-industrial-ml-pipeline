"""Frozen contract tests: manifests, API, registry, events."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from pipeline.schemas import (
    ArtifactManifest,
    DataProvenance,
    DriftEvent,
    LifecycleEvent,
    MetricNames,
    MetricRecord,
    PredictRequest,
    RegistryEntry,
    RulResult,
    StageTransitionEvent,
    TrainingConfig,
)

lifecycle_adapter = TypeAdapter(LifecycleEvent)


def _sensor_window(count: int = 5) -> list[dict]:
    readings: list[dict] = []
    for cycle in range(1, count + 1):
        readings.append(
            {
                "cycle": cycle,
                "operating_settings": (1.0, 2.0, 3.0),
                "sensors": tuple(float(i) for i in range(1, 22)),
            }
        )
    return readings


def _manifest() -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id="failure-fd002-0001",
        family="failure",
        version=1,
        stage="staging",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        run_id="run-0001",
        subset="FD002",
        metrics={"auc": 0.9},
        features=["s2_mean", "s3_slope"],
        thresholds={"auc": 0.85},
        training_config=TrainingConfig(
            subset="FD002",
            seed=42,
            window_size=5,
            features_per_sensor=6,
            failure_horizon_cycles=30,
            rul_clip=125,
            forecast_horizon_cycles=10,
        ),
        data_provenance=DataProvenance(subset="FD002", source="synthetic"),
        artifact_path="artifacts/runs/run-0001/failure.joblib",
    )


def test_manifest_json_roundtrip() -> None:
    manifest = _manifest()
    restored = ArtifactManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest


def test_registry_entry_has_empty_history_by_default() -> None:
    entry = RegistryEntry(
        manifest=_manifest(),
        registered_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert entry.history == []


def test_predict_request_accepts_valid_window() -> None:
    request = PredictRequest(subset="FD002", engine_id="engine-1", readings=_sensor_window())
    assert len(request.readings) == 5
    assert request.readings[-1].cycle == 5


def test_predict_request_rejects_wrong_sensor_count() -> None:
    readings = _sensor_window(count=1)
    readings[0]["sensors"] = tuple(float(i) for i in range(1, 21))
    with pytest.raises(ValidationError):
        PredictRequest(subset="FD002", engine_id="engine-1", readings=readings)


def test_predict_request_rejects_non_chronological_cycles() -> None:
    readings = _sensor_window(count=2)
    readings[1]["cycle"] = 1
    with pytest.raises(ValidationError):
        PredictRequest(subset="FD002", engine_id="engine-1", readings=readings)


def test_rul_interval_ordering_enforced() -> None:
    with pytest.raises(ValidationError):
        RulResult(point=5.0, lower=6.0, upper=7.0, interval_level=0.9)


def test_lifecycle_events_discriminate() -> None:
    stage = StageTransitionEvent(
        artifact_id="failure-fd002-0001",
        from_stage="staging",
        to_stage="production",
        reason="shadow gate passed",
    )
    drift = DriftEvent(feature="s2_mean", psi=0.25, threshold=0.20, triggered=True)
    for event in (stage, drift):
        parsed = lifecycle_adapter.validate_python(event.model_dump())
        assert parsed.event_type == event.event_type


def test_unknown_event_type_rejected() -> None:
    with pytest.raises(ValidationError):
        lifecycle_adapter.validate_python({"event_type": "mystery"})


def test_metric_record_and_names() -> None:
    record = MetricRecord(
        run_id="run-0001",
        family="failure",
        split="validation",
        metrics={MetricNames.AUC: 0.9},
    )
    assert record.metrics["auc"] == 0.9
