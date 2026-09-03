"""Frozen inter-module contracts.

These schemas are the single source of truth for data crossing module
boundaries. Changes require re-planning per the development plan.
"""

from pipeline.schemas.api import (
    AnomalyResult,
    CycleReading,
    FailureResult,
    ForecastPoint,
    ModelVersionInfo,
    PredictRequest,
    PredictResponse,
    RulResult,
)
from pipeline.schemas.artifact import (
    ArtifactManifest,
    DataProvenance,
    TrainingConfig,
)
from pipeline.schemas.events import (
    DriftEvent,
    LifecycleEvent,
    RetrainEvent,
    ShadowEvalEvent,
    StageTransitionEvent,
)
from pipeline.schemas.metrics import MetricNames, MetricRecord
from pipeline.schemas.registry import (
    ModelRegistry,
    RegistryEntry,
    RegistryError,
    StageTransition,
)

__all__ = [
    "AnomalyResult",
    "ArtifactManifest",
    "CycleReading",
    "DataProvenance",
    "DriftEvent",
    "FailureResult",
    "ForecastPoint",
    "LifecycleEvent",
    "MetricNames",
    "MetricRecord",
    "ModelRegistry",
    "ModelVersionInfo",
    "PredictRequest",
    "PredictResponse",
    "RegistryEntry",
    "RegistryError",
    "RetrainEvent",
    "RulResult",
    "ShadowEvalEvent",
    "StageTransition",
    "StageTransitionEvent",
    "TrainingConfig",
]
