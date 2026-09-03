"""Canonical evaluation metrics contract (frozen contract). FR-04."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline.config import ModelFamily
from pipeline.schemas.events import utc_now

Split = Literal["train", "validation", "test", "shadow"]


class MetricNames:
    """Canonical metric keys shared by evaluators and registries."""

    AUC = "auc"
    AVERAGE_PRECISION = "average_precision"
    BRIER = "brier"
    LOG_LOSS = "log_loss"
    RMSE = "rmse"
    NASA_SCORE = "nasa_score"
    INTERVAL_COVERAGE = "interval_coverage"
    INTERVAL_WIDTH = "interval_width"
    MAPE = "mape"
    PSI = "psi"
    P50_LATENCY_MS = "p50_latency_ms"
    P95_LATENCY_MS = "p95_latency_ms"


class MetricRecord(BaseModel):
    """One evaluation result for one model family on one split."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    family: ModelFamily
    split: Split
    metrics: dict[str, float] = Field(default_factory=dict)
    passed: bool | None = None
    recorded_at: datetime = Field(default_factory=utc_now)
