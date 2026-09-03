"""Feature engineering tests: layout, rolling correctness, leakage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.data.features import (
    RegimeEncoder,
    _series_features,
    _window_stats,
    build_feature_frame,
    feature_columns,
    prepare_features,
    save_features,
)
from pipeline.data.loader import SENSOR_COLUMNS
from tests.fixtures.cmapss_synthetic import SyntheticCmapss


def test_feature_column_layout(small_synthetic_cmapss: SyntheticCmapss) -> None:
    train, test, _ = prepare_features(small_synthetic_cmapss.train, small_synthetic_cmapss.test)
    expected = ["unit", "cycle", "regime", *feature_columns()]
    assert list(train.columns) == expected
    assert len(feature_columns()) == len(SENSOR_COLUMNS) * 6
    assert len(test) == len(small_synthetic_cmapss.test)
    assert train.isna().sum().sum() == 0
    assert test.isna().sum().sum() == 0


def test_regime_normalization_removes_regime_offset(
    small_synthetic_cmapss: SyntheticCmapss,
) -> None:
    train, _, encoder = prepare_features(small_synthetic_cmapss.train)
    assert set(train["regime"].unique()).issubset(set(range(6)))
    assert encoder.means.shape == (6, len(SENSOR_COLUMNS))
    assert encoder.stds.shape == (6, len(SENSOR_COLUMNS))
    assert np.all(encoder.stds > 0)


def test_rolling_statistics_are_exact() -> None:
    series = np.arange(1, 11, dtype=float)
    features = _series_features(series, window_size=3)
    np.testing.assert_allclose(features[0], [1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        features[9],
        [9.0, 1.0, 8.0, 10.0, 2.0, 1.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(_window_stats(np.array([5.0, 7.0, 9.0]))[-1], 2.0)


def test_no_future_leakage(small_synthetic_cmapss: SyntheticCmapss) -> None:
    train, _, encoder = prepare_features(small_synthetic_cmapss.train)
    unit = int(train["unit"].iloc[0])
    cutoff = 10
    assert train.query("unit == @unit")["cycle"].max() > cutoff + 5

    perturbed = small_synthetic_cmapss.train.copy()
    future_rows = (perturbed["unit"] == unit) & (perturbed["cycle"] > cutoff)
    perturbed.loc[future_rows, SENSOR_COLUMNS] += 1000.0

    recomputed = build_feature_frame(perturbed, encoder)
    original_rows = train.query("unit == @unit and cycle <= @cutoff")
    recomputed_rows = recomputed.query("unit == @unit and cycle <= @cutoff")
    pd.testing.assert_frame_equal(
        original_rows.reset_index(drop=True),
        recomputed_rows.reset_index(drop=True),
    )
    changed = recomputed.query("unit == @unit and cycle == @cutoff + 1")
    assert len(changed) == 1


def test_save_features_writes_manifest(
    tmp_path: Path,
    small_synthetic_cmapss: SyntheticCmapss,
) -> None:
    train, test, encoder = prepare_features(
        small_synthetic_cmapss.train,
        small_synthetic_cmapss.test,
    )
    processed_dir = tmp_path / "processed"
    paths = save_features(
        train,
        encoder,
        test_features=test,
        processed_dir=processed_dir,
        window_size=5,
    )
    assert paths["train"].exists()
    assert paths["test"].exists()
    assert paths["encoder"].exists()
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["window_size"] == 5
    assert manifest["features_per_sensor"] == 6
    assert len(manifest["columns"]) == 3 + len(SENSOR_COLUMNS) * 6
    assert manifest["normalization"] == "per-regime z-score (train-fit)"

    restored: RegimeEncoder = RegimeEncoder.load(processed_dir)
    original_labels = encoder.transform(small_synthetic_cmapss.train)["regime"].to_numpy()
    restored_labels = restored.transform(small_synthetic_cmapss.train)["regime"].to_numpy()
    np.testing.assert_array_equal(original_labels, restored_labels)
