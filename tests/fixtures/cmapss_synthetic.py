"""Deterministic FD002-style synthetic C-MAPSS generator.

Tests must never depend on the ~12 MB NASA download, so this module generates
small run-to-failure trajectories with the same layout (unit, cycle, 3
operating settings, 21 sensors), a multi-regime operating profile, constant
sensors, and per-engine RUL ground truth. FR-17.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

REGIME_CENTROIDS = np.array(
    [
        [0.0, 0.0, 100.0],
        [10.0, 0.25, 120.0],
        [20.0, 0.5, 140.0],
        [30.0, 0.75, 160.0],
        [42.0, 0.6, 180.0],
        [50.0, 0.85, 200.0],
    ],
    dtype=float,
)

SETTINGS_COLUMNS = ["op1", "op2", "op3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS_COLUMNS, *SENSOR_COLUMNS]
N_REGIMES = REGIME_CENTROIDS.shape[0]


@dataclass(frozen=True)
class SyntheticCmapss:
    """Generated dataset mirroring the C-MAPSS file layout."""

    subset: str
    train: pd.DataFrame
    test: pd.DataFrame
    rul_test: pd.DataFrame
    regime_centroids: np.ndarray
    constant_sensors: tuple[int, ...]
    degradation_sensors: tuple[int, ...]
    lifecycle_by_unit: dict[int, int]


def _as_one_based_indices(sensors: Sequence[int], label: str) -> tuple[int, ...]:
    indices: list[int] = []
    for sensor in sensors:
        if not 1 <= int(sensor) <= 21:
            raise ValueError(f"{label} sensor index out of range 1..21: {sensor}")
        indices.append(int(sensor) - 1)
    return tuple(sorted(set(indices)))


def generate_synthetic_cmapss(
    *,
    subset: str = "FD002",
    n_train_engines: int = 4,
    n_test_engines: int = 2,
    seed: int = 0,
    min_lifecycle: int = 25,
    max_lifecycle: int = 80,
    degradation_sensors: Sequence[int] = (7, 8, 11, 12),
    constant_sensors: Sequence[int] = (1, 5, 10, 16, 18, 19),
) -> SyntheticCmapss:
    """Generate a deterministic FD002-style dataset.

    Train engines run to failure; test engines are truncated and accompanied by
    a per-engine RUL table, matching the real C-MAPSS layout.
    """
    if subset != "FD002":
        raise NotImplementedError("generator only models FD002-style trajectories")
    if n_train_engines < 1 or n_test_engines < 1:
        raise ValueError("need at least one train and one test engine")

    degradation = _as_one_based_indices(degradation_sensors, "degradation")
    constant = _as_one_based_indices(constant_sensors, "constant")
    overlap = set(degradation).intersection(constant)
    if overlap:
        raise ValueError(f"sensor indices cannot be both degradation and constant: {overlap}")

    rng = np.random.default_rng(seed)
    # Constant-sensor values are fixed per subset (not per engine), like the
    # real dataset's non-informative channels.
    constant_values = rng.normal(loc=100.0, scale=5.0, size=21)
    degradation_slope = np.zeros(21)
    degradation_slope[list(degradation)] = rng.uniform(low=15.0, high=45.0, size=len(degradation))

    lifecycle_by_unit: dict[int, int] = {}

    def make_engine(unit_id: int, lifecycle: int, last_cycle: int) -> pd.DataFrame:
        engine_base = rng.normal(loc=100.0, scale=10.0, size=21)
        regime_effect = rng.normal(loc=0.0, scale=1.0, size=(N_REGIMES, 21))
        regime = int(rng.integers(0, N_REGIMES))
        rows: list[list[float | int]] = []
        for cycle in range(1, last_cycle + 1):
            if rng.random() < 0.03:
                regime = int(rng.integers(0, N_REGIMES))
            progress = (cycle - 1) / max(lifecycle - 1, 1)
            settings = REGIME_CENTROIDS[regime] + rng.normal(loc=0.0, scale=0.05, size=3)
            sensor = (
                engine_base
                + degradation_slope * progress**1.6
                + regime_effect[regime] * 0.3
                + rng.normal(loc=0.0, scale=0.5, size=21)
            )
            sensor[list(constant)] = constant_values[list(constant)]
            rows.append([unit_id, cycle, *settings.tolist(), *sensor.tolist()])
        return pd.DataFrame(rows, columns=COLUMNS)

    train_frames: list[pd.DataFrame] = []
    for unit_id in range(1, n_train_engines + 1):
        lifecycle = int(rng.integers(min_lifecycle, max_lifecycle + 1))
        lifecycle_by_unit[unit_id] = lifecycle
        train_frames.append(make_engine(unit_id, lifecycle, lifecycle))

    test_frames: list[pd.DataFrame] = []
    rul_rows: list[list[int]] = []
    for unit_id in range(n_train_engines + 1, n_train_engines + n_test_engines + 1):
        lifecycle = int(rng.integers(min_lifecycle, max_lifecycle + 1))
        lifecycle_by_unit[unit_id] = lifecycle
        truncation = rng.uniform(low=0.35, high=0.8)
        last_cycle = max(1, min(lifecycle - 1, int(lifecycle * truncation)))
        test_frames.append(make_engine(unit_id, lifecycle, last_cycle))
        rul_rows.append([unit_id, lifecycle - last_cycle])

    train = pd.concat(train_frames, ignore_index=True)
    test = pd.concat(test_frames, ignore_index=True)
    rul_test = pd.DataFrame(rul_rows, columns=["unit", "rul"])
    for frame in (train, test):
        frame["unit"] = frame["unit"].astype(np.int64)
        frame["cycle"] = frame["cycle"].astype(np.int64)

    return SyntheticCmapss(
        subset=subset,
        train=train,
        test=test,
        rul_test=rul_test,
        regime_centroids=REGIME_CENTROIDS.copy(),
        constant_sensors=tuple(sorted(int(s) + 1 for s in constant)),
        degradation_sensors=tuple(sorted(int(s) + 1 for s in degradation)),
        lifecycle_by_unit=lifecycle_by_unit,
    )
