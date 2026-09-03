"""Seeded drift -> retrain -> shadow -> promote simulation (FR-10/11/12).

Replays the FD002 test split as a streaming corpus. The champion serves
windows while the PSI detector watches for drift; a trigger starts an isolated
retrain worker, the candidate enters the registry as staging, shadows the
champion over the final labeled windows, and is promoted or archived by the
dual quality/latency gates. All events are written to
``artifacts/lifecycle/sim_events.jsonl``.

Offline replay semantics: the test split ships with RUL ground truth, so
labels are available at evaluation time; a live deployment would score windows
as they arrive and apply the same gates once outcomes are known (ADR 0010).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pipeline.artifacts.manager import load_artifact  # noqa: E402
from pipeline.config import get_settings  # noqa: E402
from pipeline.eval.labels import add_targets  # noqa: E402
from pipeline.lifecycle.drift import DriftDetector  # noqa: E402
from pipeline.lifecycle.events import EventLog  # noqa: E402
from pipeline.lifecycle.registry import SqliteRegistry  # noqa: E402
from pipeline.lifecycle.shadow import ShadowEvaluator  # noqa: E402
from pipeline.lifecycle.trainer import RetrainWorker  # noqa: E402
from pipeline.models.orchestrator import FEATURE_COLUMNS  # noqa: E402
from pipeline.schemas.events import (  # noqa: E402
    DriftEvent,
    RetrainEvent,
    ShadowEvalEvent,
    StageTransitionEvent,
)


def _latest_passing_run() -> Path:
    runs_dir = get_settings().runs_dir
    candidates = sorted(runs_dir.glob("run-*"), reverse=True)
    for run_dir in candidates:
        run_json = run_dir / "run.json"
        if not run_json.exists():
            continue
        summary = json.loads(run_json.read_text(encoding="utf-8"))
        if summary.get("all_passed") is True:
            return run_dir
    raise RuntimeError("no passing training run found; run `make train` first")


def _per_row_latency_ms(model: object, row: np.ndarray) -> float:
    from pipeline.models.performance import RulModel

    start = time.perf_counter()
    if isinstance(model, RulModel):
        model.predict(row.reshape(1, -1))
    else:
        model.predict_proba(row.reshape(1, -1))
    return (time.perf_counter() - start) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="lifecycle simulation")
    parser.add_argument("--champion-run-id", default=None)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    settings = get_settings()
    horizon = settings.failure_horizon_cycles
    champion_run = (
        settings.runs_dir / args.champion_run_id if args.champion_run_id else _latest_passing_run()
    )
    models = {
        family: load_artifact(champion_run, family)[0]
        for family in ("failure", "performance")
    }

    test_features = pd.read_parquet(settings.processed_dir / "FD002_features_test.parquet")
    rul_map = pd.read_parquet(settings.processed_dir / "FD002_rul_test.parquet").set_index("unit")
    max_cycle = test_features.groupby("unit")["cycle"].max()
    failure_cycle = {
        int(unit): int(max_cycle.loc[unit]) + int(rul_map.loc[unit, "rul"])
        for unit in max_cycle.index
    }
    stream = test_features.sort_values(["cycle", "unit"]).reset_index(drop=True)
    X = stream[FEATURE_COLUMNS].to_numpy(dtype=float)
    cycles = stream["cycle"].to_numpy()
    units = stream["unit"].to_numpy()
    actual_rul = np.minimum(
        np.array(
            [
                failure_cycle[int(unit)] - int(cycle)
                for unit, cycle in zip(units, cycles, strict=True)
            ]
        ),
        settings.rul_clip,
    ).astype(float)
    failure_labels = actual_rul <= horizon

    champion_proba = models["failure"].predict_proba(X)
    champion_rul = models["performance"].predict(X)["point"]

    train_reference = add_targets(
        pd.read_parquet(settings.processed_dir / "FD002_features_train.parquet")
    )
    healthy_reference = train_reference[train_reference["rul"] >= 90]
    reference_std = healthy_reference[FEATURE_COLUMNS].std()
    active_columns = [column for column in FEATURE_COLUMNS if reference_std[column] > 1e-3]
    detector = DriftDetector(active_columns)
    detector.fit_reference(healthy_reference)

    # Fresh, deterministic registry and event log for the demo run.
    registry_path = settings.registry_dir / "registry.db"
    registry_path.unlink(missing_ok=True)
    registry = SqliteRegistry()
    events_path = settings.lifecycle_dir / "sim_events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("", encoding="utf-8")
    event_log = EventLog(events_path)

    # Register and promote the incumbent champion so the eventual promotion
    # archives a real prior production model.
    for family in ("anomaly", "failure", "performance", "forecasting"):
        _, manifest = load_artifact(champion_run, family)
        registry.register(manifest)
        registry.promote(manifest.artifact_id, reason="incumbent champion")
        event_log.append(
            StageTransitionEvent(
                artifact_id=manifest.artifact_id,
                from_stage=None,
                to_stage="production",
                reason="incumbent champion",
            )
        )

    challenger_models: dict[str, object] | None = None
    challenger_run_dir: Path | None = None
    worker: RetrainWorker | None = None
    drift_count = 0
    chunk_size = 50

    for start in range(0, len(stream), chunk_size):
        chunk = stream.iloc[start : start + chunk_size]
        report = detector.update(chunk)
        if report.triggered and worker is None:
            drift_count += 1
            top_feature = report.top_features[0]
            event_log.append(
                DriftEvent(
                    feature=top_feature,
                    psi=float(report.feature_psi[top_feature]),
                    threshold=detector.threshold,
                    triggered=True,
                )
            )
            event_log.append(RetrainEvent(trigger="drift"))
            worker = RetrainWorker().start(subset="FD002", seed=args.seed + drift_count)

    if worker is not None:
        new_run_id = worker.wait()
        challenger_run_dir = settings.runs_dir / new_run_id
        challenger_models = {
            family: load_artifact(challenger_run_dir, family)[0]
            for family in ("failure", "performance")
        }
        for family in ("anomaly", "failure", "performance", "forecasting"):
            _, manifest = load_artifact(challenger_run_dir, family)
            registry.register(manifest)
            event_log.append(
                StageTransitionEvent(
                    artifact_id=manifest.artifact_id,
                    from_stage=None,
                    to_stage="staging",
                    reason="retrained candidate",
                )
            )
        event_log.append(
            RetrainEvent(
                trigger="drift",
                artifact_id=f"failure-{new_run_id}",
                duration_seconds=0.0,
            )
        )

    if challenger_models is None:
        raise RuntimeError(
            "drift did not trigger a retrain during this replay; "
            "inspect sim_events and PSI thresholds"
        )

    # Shadow the challenger over the final labeled windows.
    tail = stream.iloc[-50:]
    tail_indices = tail.index.to_numpy()
    X_tail = X[tail_indices]
    challenger_proba = challenger_models["failure"].predict_proba(X_tail)
    challenger_rul = challenger_models["performance"].predict(X_tail)["point"]
    champion_tail_proba = champion_proba[tail_indices]
    champion_tail_rul = champion_rul[tail_indices]

    evaluator = ShadowEvaluator.from_settings()
    latencies_challenger = [
        _per_row_latency_ms(challenger_models["failure"], X_tail[i])
        + _per_row_latency_ms(challenger_models["performance"], X_tail[i])
        for i in range(min(50, len(X_tail)))
    ]
    for index in range(len(X_tail)):
        evaluator.add(
            champion_proba=float(champion_tail_proba[index]),
            champion_rul=float(champion_tail_rul[index]),
            challenger_proba=float(challenger_proba[index]),
            challenger_rul=float(challenger_rul[index]),
            label_available=True,
            failure_label=float(failure_labels[tail_indices[index]]),
            actual_rul=float(actual_rul[tail_indices[index]]),
            latency_ms=float(np.median(latencies_challenger)),
        )
    gate = evaluator.evaluate()
    event_log.append(
        ShadowEvalEvent(
            champion_artifact_id="champion",
            challenger_artifact_id=f"failure-{challenger_run_dir.name}",
            predictions_evaluated=evaluator.scored_count(),
            quality_pass=bool(gate["quality_pass"]),
            latency_pass=bool(gate["latency_pass"]),
            promoted=bool(gate["promote"]),
        )
    )

    promoted_families: list[str] = []
    if gate["promote"]:
        for family in ("anomaly", "failure", "performance", "forecasting"):
            _, manifest = load_artifact(challenger_run_dir, family)
            registry.promote(manifest.artifact_id, reason="shadow dual gates passed")
            promoted_families.append(family)
            event_log.append(
                StageTransitionEvent(
                    artifact_id=manifest.artifact_id,
                    from_stage="staging",
                    to_stage="production",
                    reason="shadow dual gates passed",
                )
            )
    else:
        for family in ("anomaly", "failure", "performance", "forecasting"):
            _, manifest = load_artifact(challenger_run_dir, family)
            registry.archive(manifest.artifact_id, reason="shadow gates failed")

    summary = {
        "champion_run": champion_run.name,
        "challenger_run": challenger_run_dir.name,
        "drift_events": drift_count,
        "gate": {
            key: value
            for key, value in gate.items()
            if not isinstance(value, (dict, list))
        },
        "promoted_families": promoted_families,
        "production": {
            entry.manifest.family: entry.manifest.artifact_id
            for entry in registry.list(stage="production")
        },
        "events_file": str(events_path),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
