"""Versioned model registry protocol (frozen contract).

Backend implementations must satisfy this interface exactly. Stage transitions
follow the documented lifecycle: staging -> production -> archived. FR-10.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from pipeline.config import ModelFamily, ModelStage
from pipeline.schemas.artifact import ArtifactManifest


class RegistryError(RuntimeError):
    """Raised for illegal stage transitions or unknown artifacts."""


class StageTransition(BaseModel):
    """A single auditable stage change for one artifact."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    from_stage: ModelStage | None
    to_stage: ModelStage
    reason: str
    occurred_at: datetime


class RegistryEntry(BaseModel):
    """Current state of one artifact plus its full transition history."""

    model_config = ConfigDict(extra="forbid")

    manifest: ArtifactManifest
    registered_at: datetime
    history: list[StageTransition] = Field(default_factory=list)


class ModelRegistry(Protocol):
    """Interface every registry backend must implement."""

    def register(self, manifest: ArtifactManifest) -> RegistryEntry: ...

    def promote(self, artifact_id: str, *, reason: str = "shadow gate passed") -> RegistryEntry: ...

    def archive(self, artifact_id: str, *, reason: str) -> RegistryEntry: ...

    def list(
        self,
        family: ModelFamily | None = None,
        stage: ModelStage | None = None,
    ) -> list[RegistryEntry]: ...

    def get(self, artifact_id: str) -> RegistryEntry: ...

    def get_production(self, family: ModelFamily) -> RegistryEntry | None: ...

    def history(self, artifact_id: str) -> list[StageTransition]: ...
