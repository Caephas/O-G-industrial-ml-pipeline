"""Compute and record FD002 baselines (FR-04).

Produces the feature Parquet files used by later milestones, then evaluates
the majority-class, constant-RUL, and persistence baselines on an
engine-disjoint validation split. Outputs a JSON report under
``artifacts/baselines/`` that acceptance thresholds are calibrated against.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from pipeline.config import get_settings  # noqa: E402
from pipeline.data.features import feature_columns, prepare_features, save_features  # noqa: E402
from pipeline.data.loader import load_dataset, load_processed, save_processed  # noqa: E402
from pipeline.eval.baselines import (  # noqa: E402
    evaluate_failure_baseline,
    evaluate_rul_baseline,
    fit_failure_baseline,
    fit_rul_baseline,
    persistence_mape,
)
from pipeline.eval.labels import add_targets  # noqa: E402
from pipeline.eval.splits import split_features  # noqa: E402
from pipeline.schemas.artifact import DataProvenance  # noqa: E402


def _top_degradation_sensors(train_features, count: int) -> list[str]:
    slope_columns = [column for column in feature_columns() if column.endswith("_slope")]
    magnitudes = train_features[slope_columns].abs().mean().sort_values(ascending=False)
    return [column.replace("_slope", "") for column in magnitudes.head(count).index]


def main() -> None:
    settings = get_settings()
    if not (settings.processed_dir / "FD002_train.parquet").exists():
        # Self-contained processing: raw archive -> validated Parquet + manifest.
        raw_data = load_dataset("FD002")
        provenance = DataProvenance(
            subset="FD002",
            source="CMAPSSData.zip",
            row_count=len(raw_data.train),
        )
        save_processed(raw_data, provenance=provenance)
    data = load_processed("FD002")
    train_features, test_features, encoder = prepare_features(
        data.train,
        data.test,
        window_size=settings.window_size,
    )
    save_features(
        train_features,
        encoder,
        subset="FD002",
        test_features=test_features,
        window_size=settings.window_size,
    )

    labeled = add_targets(train_features, horizon_cycles=settings.failure_horizon_cycles)
    train_split, validation_split = split_features(
        labeled,
        val_fraction=0.2,
        seed=settings.seed,
    )

    failure_prior = fit_failure_baseline(train_split["failure"].to_numpy())
    failure = evaluate_failure_baseline(
        failure_prior,
        validation_split["failure"].to_numpy(),
    )

    rul_constant = fit_rul_baseline(train_split["rul"].to_numpy())
    rul = evaluate_rul_baseline(rul_constant, validation_split["rul"].to_numpy())

    sensors = _top_degradation_sensors(train_features, settings.forecast_sensor_count)
    horizon = settings.forecast_horizon_cycles
    sensor_mapes: dict[str, float] = {}
    for sensor in sensors:
        series = [
            data.train.query("unit == @unit")[sensor].to_numpy(dtype=float)
            for unit in sorted(data.train["unit"].unique())
        ]
        values = [persistence_mape(item, horizon) for item in series]
        sensor_mapes[sensor] = float(np.mean(values))

    report = {
        "subset": "FD002",
        "config": {
            "window_size": settings.window_size,
            "failure_horizon_cycles": settings.failure_horizon_cycles,
            "forecast_horizon_cycles": horizon,
            "val_fraction": 0.2,
            "seed": settings.seed,
        },
        "splits": {
            "train_engines": int(train_split["unit"].nunique()),
            "validation_engines": int(validation_split["unit"].nunique()),
            "train_rows": int(len(train_split)),
            "validation_rows": int(len(validation_split)),
        },
        "failure_baseline": {"prior": failure_prior, **failure},
        "rul_baseline": {"constant": rul_constant, **rul},
        "forecast_baseline": {
            "sensors": sensors,
            "persistence_mape": {sensor: round(value, 6) for sensor, value in sensor_mapes.items()},
            "mean_persistence_mape": float(np.mean(list(sensor_mapes.values()))),
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }
    output_dir = settings.artifacts_dir / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "FD002_baselines.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"train engines: {report['splits']['train_engines']}, "
          f"validation engines: {report['splits']['validation_engines']}")
    print(f"failure baseline (majority prior={failure_prior:.4f}): "
          f"auc={failure['auc']:.4f} brier={failure['brier']:.4f}")
    print(f"rul baseline (constant={rul_constant:.1f}): rmse={rul['rmse']:.2f} "
          f"nasa={rul['nasa_score']:.1f}")
    print(f"forecast persistence mape ({sensors}): "
          f"{report['forecast_baseline']['mean_persistence_mape']:.4f}")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
