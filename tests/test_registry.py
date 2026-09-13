"""Registry lifecycle tests (FR-10)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from pipeline.lifecycle.registry import SqliteRegistry
from pipeline.schemas.artifact import ArtifactManifest, DataProvenance, TrainingConfig
from pipeline.schemas.registry import RegistryError


def _manifest(run_id: str, family: str = "failure", version: int = 1) -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id=f"{family}-{run_id}",
        family=family,
        version=version,
        stage="staging",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        run_id=run_id,
        subset="FD002",
        metrics={"auc": 0.9},
        features=["s2_mean"],
        thresholds={"min_auc": 0.85},
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
        artifact_path="artifacts/model.joblib",
    )


def test_register_assigns_sequential_versions(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.db")
    first = registry.register(_manifest("run-1"))
    second = registry.register(_manifest("run-2"))
    assert first.manifest.version == 1
    assert second.manifest.version == 2
    assert first.manifest.stage == "staging"


def test_promotion_archives_prior_production(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.db")
    first = registry.register(_manifest("run-1"))
    registry.promote(first.manifest.artifact_id)
    second = registry.register(_manifest("run-2"))
    registry.promote(second.manifest.artifact_id)

    assert registry.get(first.manifest.artifact_id).manifest.stage == "archived"
    assert registry.get(second.manifest.artifact_id).manifest.stage == "production"
    assert registry.get_production("failure").manifest.artifact_id == second.manifest.artifact_id
    assert len(registry.history(first.manifest.artifact_id)) == 3  # register, promote, archive


def test_illegal_transitions_are_rejected(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.db")
    entry = registry.register(_manifest("run-1"))
    registry.promote(entry.manifest.artifact_id)
    with pytest.raises(RegistryError):
        registry.promote(entry.manifest.artifact_id)
    registry.archive(entry.manifest.artifact_id, reason="superseded")
    with pytest.raises(RegistryError):
        registry.promote(entry.manifest.artifact_id)


def test_duplicate_registration_rejected(tmp_path: Path) -> None:
    registry = SqliteRegistry(tmp_path / "registry.db")
    registry.register(_manifest("run-1"))
    with pytest.raises(RegistryError):
        registry.register(_manifest("run-1"))
