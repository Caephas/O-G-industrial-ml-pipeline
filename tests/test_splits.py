"""Split tests: engine disjointness and determinism."""

from __future__ import annotations

from pipeline.eval.splits import split_engine_ids, split_features


def test_split_engine_ids_are_disjoint_and_complete() -> None:
    engine_ids = set(range(1, 41))
    train, validation = split_engine_ids(engine_ids, val_fraction=0.2, seed=42)
    assert train.isdisjoint(validation)
    assert train.union(validation) == engine_ids
    assert 0 < len(validation) < len(engine_ids)


def test_split_is_deterministic() -> None:
    engine_ids = set(range(1, 41))
    first = split_engine_ids(engine_ids, seed=42)
    second = split_engine_ids(engine_ids, seed=42)
    assert first == second


def test_split_features_shares_no_engines(synthetic_cmapss) -> None:
    frame = synthetic_cmapss.train
    train, validation = split_features(frame, val_fraction=0.2, seed=42)
    train_units = set(train["unit"].unique())
    validation_units = set(validation["unit"].unique())
    assert train_units.isdisjoint(validation_units)
    assert train_units.union(validation_units) == set(frame["unit"].unique())
