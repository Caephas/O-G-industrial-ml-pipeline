"""Synthetic fixture generator tests."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.fixtures.cmapss_synthetic import (
    COLUMNS,
    SyntheticCmapss,
    generate_synthetic_cmapss,
)


def test_layout_matches_cmapss(synthetic_cmapss: SyntheticCmapss) -> None:
    assert list(synthetic_cmapss.train.columns) == COLUMNS
    assert list(synthetic_cmapss.test.columns) == COLUMNS
    assert synthetic_cmapss.train["unit"].nunique() == 6
    assert synthetic_cmapss.test["unit"].nunique() == 3


def test_rul_alignment(small_synthetic_cmapss: SyntheticCmapss) -> None:
    data = small_synthetic_cmapss
    expected_rul = data.test.groupby("unit")["cycle"].max()
    for unit, max_cycle in expected_rul.items():
        lifecycle = data.lifecycle_by_unit[unit]
        assert max_cycle < lifecycle  # test trajectories never reach failure
        recorded = data.rul_test.set_index("unit").loc[unit, "rul"]
        assert recorded == lifecycle - max_cycle
    for unit, lifecycle in data.lifecycle_by_unit.items():
        if unit <= data.train["unit"].max():
            assert data.train.query("unit == @unit")["cycle"].max() == lifecycle


def test_constant_sensors_are_constant(synthetic_cmapss: SyntheticCmapss) -> None:
    for sensor in synthetic_cmapss.constant_sensors:
        column = f"s{sensor}"
        assert synthetic_cmapss.train[column].nunique() == 1
        assert synthetic_cmapss.test[column].nunique() == 1


def test_regime_diversity(synthetic_cmapss: SyntheticCmapss) -> None:
    settings = synthetic_cmapss.train[["op1", "op2", "op3"]].round(1)
    assert settings.drop_duplicates().shape[0] >= 2


def test_deterministic_generation() -> None:
    first = generate_synthetic_cmapss(seed=11)
    second = generate_synthetic_cmapss(seed=11)
    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.test, second.test)
    pd.testing.assert_frame_equal(first.rul_test, second.rul_test)


def test_different_seeds_differ() -> None:
    first = generate_synthetic_cmapss(seed=11)
    second = generate_synthetic_cmapss(seed=12)
    assert not first.train.equals(second.train)


def test_generator_rejects_unknown_subset() -> None:
    with pytest.raises(NotImplementedError):
        generate_synthetic_cmapss(subset="FD001")
