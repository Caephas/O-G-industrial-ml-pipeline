"""Model lifecycle: registry, drift, shadow mode, auto-retraining."""

from pipeline.lifecycle.drift import DriftDetector, DriftReport
from pipeline.lifecycle.events import EventLog
from pipeline.lifecycle.registry import SqliteRegistry
from pipeline.lifecycle.shadow import ShadowEvaluator
from pipeline.lifecycle.trainer import RetrainWorker

__all__ = [
    "DriftDetector",
    "DriftReport",
    "EventLog",
    "RetrainWorker",
    "ShadowEvaluator",
    "SqliteRegistry",
]
