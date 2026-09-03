"""C-MAPSS raw loader and processed Parquet writer (FR-01).

Layout: each trajectory file is whitespace-separated with no header and 26
columns per cycle: engine id, cycle number, 3 operating settings, and 21 sensor
readings. Training trajectories run to failure; test trajectories are
truncated, with remaining useful life (RUL) provided in a separate
single-column file per subset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from pipeline.config import Subset, get_settings
from pipeline.schemas.artifact import DataProvenance

SETTINGS_COLUMNS = ["op1", "op2", "op3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS_COLUMNS, *SENSOR_COLUMNS]

RawKind = Literal["train", "test", "rul"]
_RAW_FILENAMES: dict[RawKind, str] = {
    "train": "train_{subset}.txt",
    "test": "test_{subset}.txt",
    "rul": "RUL_{subset}.txt",
}


class DataValidationError(ValueError):
    """Raised when raw files do not match the documented C-MAPSS layout."""


@dataclass(frozen=True)
class CmapssData:
    """Loaded dataset for one subset, mirroring the raw file layout."""

    subset: Subset
    train: pd.DataFrame
    test: pd.DataFrame
    rul_test: pd.DataFrame
    source_paths: dict[RawKind, Path]


def _find_raw_file(raw_dir: Path, subset: Subset, kind: RawKind) -> Path:
    filename = _RAW_FILENAMES[kind].format(subset=subset)
    matches = list(raw_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"{filename} not found under {raw_dir}. Run `make fetch` first."
        )
    if len(matches) > 1:
        raise DataValidationError(f"ambiguous raw file location: {matches}")
    return matches[0]


def _read_trajectories(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    if frame.shape[1] != len(COLUMNS):
        raise DataValidationError(f"{path} has {frame.shape[1]} columns, expected {len(COLUMNS)}")
    return frame.astype({"unit": "int64", "cycle": "int64"})


def _read_rul(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=["rul"])
    return frame.astype({"rul": "int64"})


def load_dataset(subset: Subset = "FD002", raw_dir: Path | None = None) -> CmapssData:
    """Parse the raw text files for one subset.

    Layout facts from the NASA C-MAPSS documentation are enforced here so
    malformed downloads fail loudly instead of poisoning training later.
    """
    root = raw_dir or get_settings().raw_dir
    train_path = _find_raw_file(root, subset, "train")
    test_path = _find_raw_file(root, subset, "test")
    rul_path = _find_raw_file(root, subset, "rul")

    train = _read_trajectories(train_path)
    test = _read_trajectories(test_path)
    rul_raw = _read_rul(rul_path)

    # The RUL file has one row per test engine, ordered by ascending engine id.
    test_units = sorted(test["unit"].unique())
    if len(rul_raw) != len(test_units):
        raise DataValidationError(
            f"RUL file has {len(rul_raw)} rows but the test set has "
            f"{len(test_units)} engines"
        )
    if rul_raw["rul"].min() < 0:
        raise DataValidationError("RUL values must be non-negative")
    rul_test = pd.DataFrame({"unit": test_units, "rul": rul_raw["rul"].to_numpy()})

    return CmapssData(
        subset=subset,
        train=train,
        test=test,
        rul_test=rul_test,
        source_paths={"train": train_path, "test": test_path, "rul": rul_path},
    )


def save_processed(
    data: CmapssData,
    processed_dir: Path | None = None,
    provenance: DataProvenance | None = None,
) -> dict[str, Path]:
    """Write tidy Parquet files plus a dataset manifest for one subset."""
    root = processed_dir or get_settings().processed_dir
    root.mkdir(parents=True, exist_ok=True)
    subset = data.subset

    paths = {
        "train": root / f"{subset}_train.parquet",
        "test": root / f"{subset}_test.parquet",
        "rul_test": root / f"{subset}_rul_test.parquet",
    }
    data.train.to_parquet(paths["train"], index=False)
    data.test.to_parquet(paths["test"], index=False)
    data.rul_test.to_parquet(paths["rul_test"], index=False)

    manifest = {
        "subset": subset,
        "columns": COLUMNS,
        "row_counts": {
            "train": int(len(data.train)),
            "test": int(len(data.test)),
            "rul_test": int(len(data.rul_test)),
        },
        "engine_counts": {
            "train": int(data.train["unit"].nunique()),
            "test": int(data.test["unit"].nunique()),
        },
        "files": {kind: str(path) for kind, path in paths.items()},
        "provenance": provenance.model_dump(mode="json") if provenance else None,
        "saved_at": datetime.now(UTC).isoformat(),
    }
    manifest_path = root / f"{subset}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


def load_processed(subset: Subset = "FD002", processed_dir: Path | None = None) -> CmapssData:
    """Reload a subset written by :func:`save_processed`."""
    root = processed_dir or get_settings().processed_dir
    paths = {
        "train": root / f"{subset}_train.parquet",
        "test": root / f"{subset}_test.parquet",
        "rul_test": root / f"{subset}_rul_test.parquet",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"missing processed file: {path}")
    return CmapssData(
        subset=subset,
        train=pd.read_parquet(paths["train"]),
        test=pd.read_parquet(paths["test"]),
        rul_test=pd.read_parquet(paths["rul_test"]),
        source_paths={},
    )
