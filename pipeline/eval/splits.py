"""Engine-disjoint, deterministic data splits (FR-04)."""

from __future__ import annotations

import numpy as np
import pandas as pd


class SplitError(ValueError):
    """Raised when split invariants are violated."""


def split_engine_ids(
    engine_ids: set[int] | list[int],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[set[int], set[int]]:
    """Split engine ids into disjoint train/validation sets."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    ids = np.array(sorted(engine_ids), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    cutoff = max(1, int(len(ids) * (1.0 - val_fraction)))
    train = set(int(value) for value in ids[:cutoff])
    validation = set(int(value) for value in ids[cutoff:])
    assert_disjoint_sets(train, validation)
    return train, validation


def assert_disjoint_sets(train: set[int], validation: set[int]) -> None:
    overlap = train.intersection(validation)
    if overlap:
        raise SplitError(f"engines appear in both splits: {sorted(overlap)[:10]}")


def split_features(
    features: pd.DataFrame,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    engine_column: str = "unit",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an engine-level feature frame without sharing engines."""
    train_ids, validation_ids = split_engine_ids(
        set(features[engine_column].unique()),
        val_fraction=val_fraction,
        seed=seed,
    )
    train = features[features[engine_column].isin(train_ids)].copy()
    validation = features[features[engine_column].isin(validation_ids)].copy()
    if len(train) == 0 or len(validation) == 0:
        raise SplitError("split produced an empty partition")
    return train, validation
