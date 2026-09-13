"""One-command container bootstrap: data -> features -> train -> serve.

Each stage is skipped when its durable marker already exists in the shared
volumes, so the first run performs the full pipeline and later starts are
fast. Serves the REST API and dashboard once the pipeline is ready.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def needs_fetch(raw_dir: Path) -> bool:
    return not (raw_dir / "CMAPSSData.zip").exists()


def needs_features(processed_dir: Path) -> bool:
    return not (processed_dir / "FD002_feature_manifest.json").exists()


def needs_train(runs_dir: Path) -> bool:
    for run_dir in sorted(runs_dir.glob("run-*"), reverse=True):
        run_json = run_dir / "run.json"
        if run_json.exists():
            summary = json.loads(run_json.read_text(encoding="utf-8"))
            if summary.get("all_passed") is True:
                return False
    return True


def needs_sim(registry_db: Path) -> bool:
    if not registry_db.exists():
        return True
    from pipeline.lifecycle.registry import SqliteRegistry

    try:
        registry = SqliteRegistry(registry_db)
        production_families = {entry.manifest.family for entry in registry.list(stage="production")}
        return production_families != {"anomaly", "failure", "performance", "forecasting"}
    except Exception:
        return True


def _run_module(module: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    from pipeline.config import get_settings

    settings = get_settings()
    if needs_fetch(settings.raw_dir):
        _run_module("scripts.fetch_cmapss")
    if needs_features(settings.processed_dir):
        _run_module("scripts.run_baselines")
    if needs_train(settings.runs_dir):
        _run_module("pipeline.cli", "train")
    if needs_sim(settings.registry_dir / "registry.db"):
        _run_module("scripts.lifecycle_sim")

    print("pipeline ready; starting API", flush=True)
    import uvicorn

    uvicorn.run(
        "pipeline.api.server:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    main()
