"""Loader tests: raw text parsing, RUL alignment, Parquet round-trips."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.data.loader import (
    COLUMNS,
    DataValidationError,
    load_dataset,
    load_processed,
    save_processed,
)
from pipeline.schemas.artifact import DataProvenance
from tests.fixtures.cmapss_synthetic import SyntheticCmapss


def _write_raw_files(tmp_path: Path, data: SyntheticCmapss) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    data.train.to_csv(raw_dir / "train_FD002.txt", sep=" ", header=False, index=False)
    data.test.to_csv(raw_dir / "test_FD002.txt", sep=" ", header=False, index=False)
    data.rul_test.to_csv(raw_dir / "RUL_FD002.txt", sep=" ", header=False, index=False)
    return raw_dir


def test_load_dataset_parses_layout(
    tmp_path: Path,
    small_synthetic_cmapss: SyntheticCmapss,
) -> None:
    raw_dir = _write_raw_files(tmp_path, small_synthetic_cmapss)
    loaded = load_dataset("FD002", raw_dir=raw_dir)

    assert list(loaded.train.columns) == COLUMNS
    assert list(loaded.test.columns) == COLUMNS
    assert loaded.rul_test.columns.tolist() == ["unit", "rul"]
    pd.testing.assert_frame_equal(loaded.train, small_synthetic_cmapss.train, check_exact=False)
    pd.testing.assert_frame_equal(loaded.test, small_synthetic_cmapss.test, check_exact=False)
    pd.testing.assert_frame_equal(
        loaded.rul_test,
        small_synthetic_cmapss.rul_test,
        check_dtype=False,
    )


def test_load_dataset_rejects_rul_count_mismatch(
    tmp_path: Path,
    small_synthetic_cmapss: SyntheticCmapss,
) -> None:
    raw_dir = _write_raw_files(tmp_path, small_synthetic_cmapss)
    (raw_dir / "RUL_FD002.txt").write_text("10\n20\n30\n", encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_dataset("FD002", raw_dir=raw_dir)


def test_processed_roundtrip(
    tmp_path: Path,
    small_synthetic_cmapss: SyntheticCmapss,
) -> None:
    raw_dir = _write_raw_files(tmp_path, small_synthetic_cmapss)
    data = load_dataset("FD002", raw_dir=raw_dir)
    processed_dir = tmp_path / "processed"

    provenance = DataProvenance(
        subset="FD002",
        source="synthetic",
        row_count=len(data.train),
    )
    save_processed(data, processed_dir=processed_dir, provenance=provenance)

    restored = load_processed("FD002", processed_dir=processed_dir)
    pd.testing.assert_frame_equal(restored.train, data.train, check_exact=False)
    pd.testing.assert_frame_equal(restored.test, data.test, check_exact=False)

    manifest_path = processed_dir / "FD002_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["engine_counts"] == {"train": 4, "test": 2}
    assert manifest["provenance"]["source"] == "synthetic"


def test_load_dataset_requires_raw_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset("FD002", raw_dir=tmp_path)
