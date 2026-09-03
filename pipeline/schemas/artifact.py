"""Self-describing model artifact manifest (frozen contract).

Every trained artifact persisted at runtime carries this manifest next to the
serialized estimator so the registry can be reconstructed without external
state. FR-10, FR-16.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline.config import ModelFamily, ModelStage, Subset


class TrainingConfig(BaseModel):
    """Hyper-parameter and labeling choices frozen into an artifact."""

    model_config = ConfigDict(extra="forbid")

    subset: Subset
    seed: int
    window_size: int = Field(ge=1)
    features_per_sensor: int = Field(ge=1)
    failure_horizon_cycles: int = Field(ge=1)
    rul_clip: int = Field(ge=1)
    forecast_horizon_cycles: int = Field(ge=1)


class DataProvenance(BaseModel):
    """Where the training data came from (FR-16)."""

    model_config = ConfigDict(extra="forbid")

    subset: Subset
    source: str
    downloaded_at: datetime | None = None
    sha256: str | None = None
    row_count: int | None = Field(default=None, ge=0)


class ArtifactManifest(BaseModel):
    """Manifest stored alongside every serialized model artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    artifact_id: str = Field(min_length=1)
    family: ModelFamily
    version: int = Field(ge=1)
    stage: ModelStage = "staging"
    created_at: datetime
    run_id: str = Field(min_length=1)
    subset: Subset
    metrics: dict[str, float] = Field(default_factory=dict)
    features: list[str] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)
    training_config: TrainingConfig
    data_provenance: DataProvenance
    git_sha: str | None = None
    artifact_path: str = Field(min_length=1)
    files: list[str] = Field(default_factory=list)
