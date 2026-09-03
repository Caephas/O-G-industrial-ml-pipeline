"""Lifecycle event log contract (frozen contract). FR-10/11/12.

Lifecycle events are written as JSONL so monitoring and dashboards can replay
the full drift -> retrain -> shadow -> promotion story.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline.config import ModelStage


def utc_now() -> datetime:
    return datetime.now(UTC)


class BaseEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    occurred_at: datetime = Field(default_factory=utc_now)
    run_id: str = ""


class StageTransitionEvent(BaseEvent):
    event_type: Literal["stage_transition"] = "stage_transition"
    artifact_id: str
    from_stage: ModelStage | None
    to_stage: ModelStage
    reason: str = ""


class DriftEvent(BaseEvent):
    event_type: Literal["drift"] = "drift"
    feature: str
    psi: float = Field(ge=0.0)
    threshold: float = Field(gt=0.0)
    triggered: bool


class ShadowEvalEvent(BaseEvent):
    event_type: Literal["shadow_eval"] = "shadow_eval"
    champion_artifact_id: str
    challenger_artifact_id: str
    predictions_evaluated: int = Field(ge=1)
    quality_pass: bool
    latency_pass: bool
    promoted: bool


class RetrainEvent(BaseEvent):
    event_type: Literal["retrain"] = "retrain"
    trigger: Literal["drift", "manual"]
    artifact_id: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)


LifecycleEvent = Annotated[
    StageTransitionEvent | DriftEvent | ShadowEvalEvent | RetrainEvent,
    Field(discriminator="event_type"),
]
