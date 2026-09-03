"""REST API request/response contracts (frozen contract). FR-13."""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pipeline.config import ModelFamily, Subset

SENSOR_COUNT = 21
SETTINGS_COUNT = 3


class CycleReading(BaseModel):
    """One cycle of operating settings and sensor readings for an engine."""

    model_config = ConfigDict(extra="forbid")

    cycle: int = Field(ge=1)
    operating_settings: tuple[float, float, float]
    sensors: tuple[float, ...]

    @field_validator("operating_settings")
    @classmethod
    def _settings_count(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) != SETTINGS_COUNT:
            raise ValueError(f"expected {SETTINGS_COUNT} operating settings, got {len(value)}")
        return value

    @field_validator("sensors")
    @classmethod
    def _sensor_count(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) != SENSOR_COUNT:
            raise ValueError(f"expected {SENSOR_COUNT} sensor readings, got {len(value)}")
        return value


class PredictRequest(BaseModel):
    """Inference request: a chronological window for one engine."""

    model_config = ConfigDict(extra="forbid")

    subset: Subset
    engine_id: str | int
    readings: list[CycleReading] = Field(min_length=1)
    forecast_horizon_cycles: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _readings_chronological(self) -> PredictRequest:
        for previous, current in pairwise(self.readings):
            if current.cycle <= previous.cycle:
                raise ValueError("readings must be in strictly increasing cycle order")
        return self


class AnomalyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0)
    flag: bool


class FailureResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probability: float = Field(ge=0.0, le=1.0)
    horizon_cycles: int = Field(ge=1)
    calibrated: bool = True


class RulResult(BaseModel):
    """Remaining-useful-life point estimate with a quantile interval."""

    model_config = ConfigDict(extra="forbid")

    point: float = Field(ge=0.0)
    lower: float = Field(ge=0.0)
    upper: float = Field(ge=0.0)
    interval_level: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _interval_ordering(self) -> RulResult:
        if not self.lower <= self.point <= self.upper:
            raise ValueError("interval must satisfy lower <= point <= upper")
        return self


class ForecastPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sensor_index: int = Field(ge=1, le=SENSOR_COUNT)
    cycle_offset: int = Field(ge=1)
    value: float


class ModelVersionInfo(BaseModel):
    """Family + artifact version that produced a response."""

    model_config = ConfigDict(extra="forbid")

    family: ModelFamily
    artifact_id: str
    version: int = Field(ge=1)
    stage: Literal["production"]


class PredictResponse(BaseModel):
    """Schema-validated inference response (FR-13)."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    subset: Subset
    engine_id: str | int
    final_cycle: int = Field(ge=1)
    anomaly: AnomalyResult
    failure: FailureResult
    rul: RulResult
    forecasts: list[ForecastPoint] = Field(default_factory=list)
    model_versions: list[ModelVersionInfo] = Field(min_length=1)
    latency_ms: float = Field(ge=0.0)
    generated_at: datetime
