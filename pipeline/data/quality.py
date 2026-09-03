"""Data quality gates run before training (FR-02).

Severity semantics:
- Blocking (``fail``): malformed data that would poison training, such as
  missing values or a pathological class imbalance.
- Advisory (``warn``): conditions the feature pipeline handles downstream,
  such as constant sensors and per-regime zero variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from pipeline.config import Subset, get_settings
from pipeline.data.loader import COLUMNS, SENSOR_COLUMNS, SETTINGS_COLUMNS

Status = Literal["pass", "warn", "fail"]

VARIANCE_EPSILON = 1e-8
DEFAULT_IMBALANCE_WARN_RATIO = 0.05
DEFAULT_IMBALANCE_FAIL_RATIO = 0.005


@dataclass(frozen=True)
class QualityResult:
    """Outcome of one quality gate."""

    name: str
    status: Status
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class QualityReport:
    """Aggregate outcome of all quality gates for one subset."""

    subset: Subset
    passed: bool
    results: tuple[QualityResult, ...]
    generated_at: datetime = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subset": self.subset,
            "passed": self.passed,
            "generated_at": self.generated_at.isoformat(),
            "results": [result.to_dict() for result in self.results],
        }


def check_missing_values(frame: pd.DataFrame) -> QualityResult:
    counts = frame.isna().sum()
    missing = {column: int(count) for column, count in counts.items() if count > 0}
    if missing:
        return QualityResult(
            name="nan",
            status="fail",
            detail={"message": "missing values found", "columns": missing},
        )
    return QualityResult(name="nan", status="pass", detail={"message": "no missing values"})


def check_constant_sensors(frame: pd.DataFrame) -> QualityResult:
    constant = [column for column in SENSOR_COLUMNS if frame[column].nunique() == 1]
    if constant:
        return QualityResult(
            name="constant_sensors",
            status="warn",
            detail={
                "message": "constant sensors detected; they carry no signal and are "
                "dropped during feature engineering",
                "sensors": constant,
            },
        )
    return QualityResult(name="constant_sensors", status="pass", detail={"message": "none"})


def _regime_groups(frame: pd.DataFrame) -> pd.Series:
    """Coarse operating-regime key from rounded settings columns."""
    rounded = np.round(frame[SETTINGS_COLUMNS].to_numpy(), decimals=1)
    return pd.Series(list(map(tuple, rounded)), index=frame.index, name="_regime")


def check_zero_variance_per_regime(frame: pd.DataFrame) -> QualityResult:
    groups = frame.groupby(_regime_groups(frame), sort=False)
    per_regime: dict[str, list[Any]] = {}
    for regime, group in groups:
        zero_variance = [
            column for column in SENSOR_COLUMNS if group[column].var() <= VARIANCE_EPSILON
        ]
        if zero_variance:
            per_regime[str(regime)] = zero_variance
    if per_regime:
        return QualityResult(
            name="zero_variance_per_regime",
            status="warn",
            detail={
                "message": "sensors with no variance within an operating regime; "
                "they carry no signal there",
                "regimes": per_regime,
            },
        )
    return QualityResult(name="zero_variance_per_regime", status="pass", detail={"message": "none"})


def compute_failure_fraction(
    frame: pd.DataFrame,
    horizon_cycles: int,
) -> dict[str, float]:
    """Fraction of cycles within ``horizon_cycles`` of failure (label imbalance)."""
    max_cycle = frame.groupby("unit")["cycle"].transform("max")
    rul = max_cycle - frame["cycle"]
    failure_cycles = int((rul <= horizon_cycles).sum())
    healthy_cycles = int((rul > horizon_cycles).sum())
    total = failure_cycles + healthy_cycles
    return {
        "failure_fraction": failure_cycles / total if total else 0.0,
        "failure_cycles": failure_cycles,
        "healthy_cycles": healthy_cycles,
    }


def check_class_imbalance(
    frame: pd.DataFrame,
    horizon_cycles: int,
    *,
    warn_ratio: float = DEFAULT_IMBALANCE_WARN_RATIO,
    fail_ratio: float = DEFAULT_IMBALANCE_FAIL_RATIO,
) -> QualityResult:
    stats = compute_failure_fraction(frame, horizon_cycles)
    fraction = stats["failure_fraction"]
    if fraction < fail_ratio:
        status: Status = "fail"
        message = "failure class is pathologically rare; training would be unreliable"
    elif fraction < warn_ratio:
        status = "warn"
        message = "failure class is rare; class weights or sampling are advised"
    else:
        status = "pass"
        message = "class balance acceptable"
    return QualityResult(
        name="class_imbalance",
        status=status,
        detail={"message": message, **stats},
    )


def run_quality_gates(
    train_frame: pd.DataFrame,
    subset: Subset = "FD002",
    *,
    horizon_cycles: int | None = None,
    warn_ratio: float = DEFAULT_IMBALANCE_WARN_RATIO,
    fail_ratio: float = DEFAULT_IMBALANCE_FAIL_RATIO,
) -> QualityReport:
    """Run every quality gate on the training split of one subset."""
    settings = get_settings()
    horizon = horizon_cycles or settings.failure_horizon_cycles

    missing_columns = [column for column in COLUMNS if column not in train_frame.columns]
    if missing_columns:
        results = (
            QualityResult(
                name="schema",
                status="fail",
                detail={"message": "missing columns", "columns": missing_columns},
            ),
        )
        return QualityReport(subset=subset, passed=False, results=results)

    results = (
        check_missing_values(train_frame),
        check_constant_sensors(train_frame),
        check_zero_variance_per_regime(train_frame),
        check_class_imbalance(
            train_frame,
            horizon_cycles=horizon,
            warn_ratio=warn_ratio,
            fail_ratio=fail_ratio,
        ),
    )
    passed = all(result.status != "fail" for result in results)
    return QualityReport(subset=subset, passed=passed, results=results)
