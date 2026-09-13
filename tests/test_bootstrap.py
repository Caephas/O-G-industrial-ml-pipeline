"""Bootstrap marker logic tests (container one-command run)."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.lifecycle.registry import SqliteRegistry
from scripts.bootstrap import needs_features, needs_fetch, needs_sim, needs_train
from tests.test_registry import _manifest


def test_fetch_marker(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    assert needs_fetch(raw) is True
    raw.mkdir()
    (raw / "CMAPSSData.zip").touch()
    assert needs_fetch(raw) is False


def test_feature_marker(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    assert needs_features(processed) is True
    processed.mkdir()
    (processed / "FD002_feature_manifest.json").touch()
    assert needs_features(processed) is False


def test_train_marker(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    assert needs_train(runs) is True
    run_dir = runs / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"all_passed": False}),
        encoding="utf-8",
    )
    assert needs_train(runs) is True
    (run_dir / "run.json").write_text(
        json.dumps({"all_passed": True}),
        encoding="utf-8",
    )
    assert needs_train(runs) is False


def test_sim_marker_requires_four_production_families(tmp_path: Path) -> None:
    db = tmp_path / "registry.db"
    assert needs_sim(db) is True
    registry = SqliteRegistry(db)
    for family in ("anomaly", "failure", "performance", "forecasting"):
        entry = registry.register(_manifest(f"run-{family}", family=family))
        registry.promote(entry.manifest.artifact_id, reason="test")
    assert needs_sim(db) is False
