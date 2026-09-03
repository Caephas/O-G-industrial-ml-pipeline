"""Regime-aware rolling-window feature engineering (FR-03).

The six C-MAPSS operating regimes are identified with seeded KMeans over the
three operating settings. Sensor channels are z-scored per regime so regime
effects are removed before rolling statistics are computed. Every rolling
window is engine-scoped and strictly chronological: a row at cycle ``c`` only
uses cycles ``max(1, c - window_size + 1) .. c``. The regime encoder is fit on
the training split only (no test information leaks into features).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans

from pipeline.config import Subset, get_settings
from pipeline.data.loader import SENSOR_COLUMNS, SETTINGS_COLUMNS

FEATURE_STATS = ["mean", "std", "min", "max", "range", "slope"]
EPSILON = 1e-6


def feature_columns() -> list[str]:
    return [f"{sensor}_{stat}" for sensor in SENSOR_COLUMNS for stat in FEATURE_STATS]


@dataclass
class RegimeEncoder:
    """KMeans regime model plus per-regime, per-sensor normalization stats.

    Fit on training data only; persisted alongside the feature manifest so
    inference transforms new windows with identical statistics.
    """

    kmeans: KMeans
    means: np.ndarray
    stds: np.ndarray

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        n_regimes: int = 6,
        seed: int = 42,
    ) -> RegimeEncoder:
        kmeans = KMeans(n_clusters=n_regimes, random_state=seed, n_init=10)
        kmeans.fit(frame[SETTINGS_COLUMNS].to_numpy())
        labels = kmeans.labels_
        means = np.stack(
            [
                frame.loc[labels == regime, SENSOR_COLUMNS].mean().to_numpy()
                for regime in range(n_regimes)
            ]
        )
        stds = np.stack(
            [
                frame.loc[labels == regime, SENSOR_COLUMNS].std(ddof=1).to_numpy()
                for regime in range(n_regimes)
            ]
        )
        stds = np.where(stds < EPSILON, 1.0, stds)
        return cls(kmeans=kmeans, means=means, stds=stds)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with regime ids and per-regime normalized sensors."""
        labels = self.kmeans.predict(frame[SETTINGS_COLUMNS].to_numpy())
        normalized = pd.DataFrame(index=frame.index, columns=SENSOR_COLUMNS, dtype=float)
        values = frame[SENSOR_COLUMNS].to_numpy()
        normalized_values = (values - self.means[labels]) / self.stds[labels]
        normalized[SENSOR_COLUMNS] = normalized_values
        normalized["regime"] = labels
        result = frame.drop(columns=SENSOR_COLUMNS).copy()
        result = pd.concat([result, normalized], axis=1)
        return result

    def save(self, directory: Path, name: str = "regime_encoder") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.pkl"
        with path.open("wb") as handle:
            pickle.dump(
                {"kmeans": self.kmeans, "means": self.means, "stds": self.stds},
                handle,
            )
        return path

    @classmethod
    def load(cls, directory: Path, name: str = "regime_encoder") -> RegimeEncoder:
        with (directory / f"{name}.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        return cls(kmeans=payload["kmeans"], means=payload["means"], stds=payload["stds"])


def _window_stats(values: np.ndarray) -> np.ndarray:
    """[mean, std(ddof=1), min, max, range, slope] for a 1-D window."""
    count = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if count > 1 else 0.0
    if np.isnan(std):
        std = 0.0
    minimum = float(values.min())
    maximum = float(values.max())
    if count > 1:
        xs = np.arange(count, dtype=float)
        centered = xs - xs.mean()
        denominator = float(np.dot(centered, centered))
        slope = float(np.dot(centered, values) / denominator)
    else:
        slope = 0.0
    return np.array([mean, std, minimum, maximum, maximum - minimum, slope])


def _series_features(values: np.ndarray, window_size: int) -> np.ndarray:
    """Per-row feature matrix (n, 6) over chronological windows."""
    count = len(values)
    output = np.full((count, len(FEATURE_STATS)), np.nan)
    warmup = min(window_size - 1, count)
    for index in range(warmup):
        output[index] = _window_stats(values[: index + 1])
    if count >= window_size:
        windows = sliding_window_view(values, window_size)
        positions = np.arange(window_size, dtype=float)
        centered = positions - positions.mean()
        denominator = float(np.dot(centered, centered))
        means = windows.mean(axis=1)
        stds = windows.std(axis=1, ddof=1)
        minimums = windows.min(axis=1)
        maximums = windows.max(axis=1)
        slopes = windows @ centered / denominator
        output[window_size - 1 :] = np.column_stack(
            [means, stds, minimums, maximums, maximums - minimums, slopes]
        )
    output[:, 1] = np.nan_to_num(output[:, 1], nan=0.0)
    return output


def build_feature_frame(
    frame: pd.DataFrame,
    encoder: RegimeEncoder,
    *,
    window_size: int = 5,
) -> pd.DataFrame:
    """Rolling features for one subset, engine-scoped and chronological."""
    normalized = encoder.transform(frame)
    columns = ["unit", "cycle", "regime", *feature_columns()]
    blocks: list[pd.DataFrame] = []
    for unit, engine in normalized.groupby("unit", sort=True):
        engine = engine.sort_values("cycle")
        cycles = engine["cycle"].to_numpy()
        regimes = engine["regime"].to_numpy()
        per_sensor = [
            _series_features(engine[sensor].to_numpy(dtype=float), window_size)
            for sensor in SENSOR_COLUMNS
        ]
        values = np.hstack(per_sensor)
        block = pd.DataFrame(values, columns=feature_columns())
        block.insert(0, "regime", regimes)
        block.insert(0, "cycle", cycles)
        block.insert(0, "unit", int(unit))
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)[columns]


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    *,
    n_regimes: int = 6,
    window_size: int | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, RegimeEncoder]:
    """Fit the regime encoder on train, then transform train and test."""
    settings = get_settings()
    encoder = RegimeEncoder.fit(
        train,
        n_regimes=n_regimes,
        seed=seed if seed is not None else settings.seed,
    )
    window = window_size or settings.window_size
    train_features = build_feature_frame(train, encoder, window_size=window)
    test_features = (
        build_feature_frame(test, encoder, window_size=window) if test is not None else None
    )
    return train_features, test_features, encoder


def save_features(
    train_features: pd.DataFrame,
    encoder: RegimeEncoder,
    *,
    subset: Subset = "FD002",
    test_features: pd.DataFrame | None = None,
    processed_dir: Path | None = None,
    window_size: int | None = None,
) -> dict[str, Path]:
    """Persist feature Parquet files, the encoder, and the feature manifest."""
    root = processed_dir or get_settings().processed_dir
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / f"{subset}_features_train.parquet"
    train_features.to_parquet(train_path, index=False)
    encoder_path = encoder.save(root)
    paths: dict[str, Path] = {
        "train": train_path,
        "encoder": encoder_path,
    }
    if test_features is not None:
        test_path = root / f"{subset}_features_test.parquet"
        test_features.to_parquet(test_path, index=False)
        paths["test"] = test_path

    manifest = {
        "subset": subset,
        "window_size": window_size or get_settings().window_size,
        "features_per_sensor": len(FEATURE_STATS),
        "feature_stats": FEATURE_STATS,
        "regime_method": "kmeans-6-seeded",
        "normalization": "per-regime z-score (train-fit)",
        "leakage_policy": "engine-scoped chronological windows; encoder fit on train only",
        "columns": train_features.columns.tolist(),
        "row_counts": {
            "train": int(len(train_features)),
            "test": int(len(test_features)) if test_features is not None else None,
        },
        "files": {kind: str(path) for kind, path in paths.items()},
        "generated_at": datetime.now(UTC).isoformat(),
    }
    manifest_path = root / f"{subset}_feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["manifest"] = manifest_path
    return paths
