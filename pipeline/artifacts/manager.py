"""Persist and reload self-describing model artifacts.

Each artifact directory contains the serialized model (joblib) plus a JSON
manifest following the frozen ``ArtifactManifest`` schema, so the registry can
be reconstructed without external state. FR-10, FR-16.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from pipeline.config import ModelFamily, Subset
from pipeline.schemas.artifact import ArtifactManifest, DataProvenance, TrainingConfig


def save_artifact(
    *,
    family: ModelFamily,
    run_id: str,
    subset: Subset,
    model: Any,
    metrics: dict[str, float],
    thresholds: dict[str, float],
    features: list[str],
    training_config: TrainingConfig,
    data_provenance: DataProvenance | None,
    git_sha: str | None,
    run_dir: Path,
    version: int = 1,
) -> ArtifactManifest:
    """Write ``model.joblib`` and ``manifest.json`` for one model family."""
    family_dir = run_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"{family}-{run_id}"
    model_path = family_dir / "model.joblib"
    joblib.dump(model, model_path, compress=3)

    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        family=family,
        version=version,
        stage="staging",
        created_at=datetime.now(UTC),
        run_id=run_id,
        subset=subset,
        metrics=metrics,
        features=features,
        thresholds=thresholds,
        training_config=training_config,
        data_provenance=data_provenance
        or DataProvenance(subset=subset, source="processed-parquet"),
        git_sha=git_sha,
        artifact_path=str(model_path),
        files=["model.joblib", "manifest.json"],
    )
    manifest_path = family_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def load_artifact(run_dir: Path, family: ModelFamily) -> tuple[Any, ArtifactManifest]:
    """Load a model and its manifest from an artifact directory."""
    family_dir = run_dir / family
    model = joblib.load(family_dir / "model.joblib")
    manifest = ArtifactManifest.model_validate_json(
        (family_dir / "manifest.json").read_text(encoding="utf-8")
    )
    return model, manifest


def write_run_json(run_dir: Path, summary: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
