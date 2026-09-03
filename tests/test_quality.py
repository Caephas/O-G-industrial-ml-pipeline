"""Quality gate tests: clean data passes, injected defects fail/warn."""

from __future__ import annotations

import pandas as pd

from pipeline.data.loader import SENSOR_COLUMNS
from pipeline.data.quality import (
    QualityReport,
    check_class_imbalance,
    check_constant_sensors,
    check_zero_variance_per_regime,
    compute_failure_fraction,
    run_quality_gates,
)
from tests.fixtures.cmapss_synthetic import SyntheticCmapss


def _result_by_name(report: QualityReport, name: str):
    for result in report.results:
        if result.name == name:
            return result
    raise AssertionError(f"missing quality result: {name}")


def test_clean_data_passes(synthetic_cmapss: SyntheticCmapss) -> None:
    report = run_quality_gates(synthetic_cmapss.train, "FD002")
    assert report.passed is True
    assert _result_by_name(report, "nan").status == "pass"


def test_missing_values_are_blocking(synthetic_cmapss: SyntheticCmapss) -> None:
    corrupted = synthetic_cmapss.train.copy()
    corrupted.loc[corrupted.index[0], "s2"] = None
    report = run_quality_gates(corrupted, "FD002")
    assert report.passed is False
    nan_result = _result_by_name(report, "nan")
    assert nan_result.status == "fail"
    assert "s2" in nan_result.detail["columns"]


def test_constant_sensor_detection(synthetic_cmapss: SyntheticCmapss) -> None:
    altered = synthetic_cmapss.train.copy()
    altered["s2"] = 123.0
    result = check_constant_sensors(altered)
    assert result.status == "warn"
    assert "s2" in result.detail["sensors"]


def test_zero_variance_per_regime_detection(synthetic_cmapss: SyntheticCmapss) -> None:
    altered = synthetic_cmapss.train.copy()
    settings = altered[["op1", "op2", "op3"]]
    regime_key = settings.round(1).astype(str).agg("|".join, axis=1)
    for regime, indices in altered.groupby(regime_key).groups.items():
        sensor = SENSOR_COLUMNS[6]
        altered.loc[indices, sensor] = float(regime.split("|")[0])  # constant per regime
    result = check_zero_variance_per_regime(altered)
    assert result.status == "warn"
    assert any(
        SENSOR_COLUMNS[6] in sensors for sensors in result.detail["regimes"].values()
    )


def test_class_imbalance_ratio_computed(synthetic_cmapss: SyntheticCmapss) -> None:
    stats = compute_failure_fraction(synthetic_cmapss.train, horizon_cycles=30)
    assert 0.0 <= stats["failure_fraction"] <= 1.0
    assert stats["failure_cycles"] + stats["healthy_cycles"] == len(synthetic_cmapss.train)


def test_pathological_imbalance_is_blocking() -> None:
    healthy = pd.DataFrame(
        {
            "unit": [1, 2, 3] * 200,
            "cycle": list(range(1, 201)) * 3,
            **{f"s{i}": 1.0 for i in range(1, 22)},
            "op1": 0.0,
            "op2": 0.0,
            "op3": 100.0,
        }
    )
    healthy = healthy.sort_values(["unit", "cycle"]).reset_index(drop=True)
    result = check_class_imbalance(healthy, horizon_cycles=1, fail_ratio=0.02)
    assert result.status == "fail"
    assert 0.0 < result.detail["failure_fraction"] <= 0.01
