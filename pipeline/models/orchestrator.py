"""4-model DAG training, evaluation, and artifact packaging (FR-09)."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.artifacts.manager import save_artifact, write_run_json
from pipeline.config import PROJECT_ROOT, Subset, get_settings
from pipeline.data.features import feature_columns
from pipeline.data.loader import load_processed
from pipeline.eval.baselines import (
    evaluate_rul_baseline,
    fit_rul_baseline,
)
from pipeline.eval.labels import add_targets
from pipeline.eval.metrics import evaluate_failure, evaluate_rul, load_thresholds
from pipeline.eval.splits import split_engine_ids, split_features
from pipeline.models.anomaly import AnomalyModel
from pipeline.models.failure import FailureModel
from pipeline.models.forecasting import ForecastingModel
from pipeline.models.performance import RulModel
from pipeline.schemas.artifact import DataProvenance, TrainingConfig

FEATURE_COLUMNS = feature_columns()


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _as_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[FEATURE_COLUMNS].to_numpy(dtype=float)


def _training_config(subset: Subset, seed: int) -> TrainingConfig:
    settings = get_settings()
    return TrainingConfig(
        subset=subset,
        seed=seed,
        window_size=settings.window_size,
        features_per_sensor=settings.features_per_sensor,
        failure_horizon_cycles=settings.failure_horizon_cycles,
        rul_clip=settings.rul_clip,
        forecast_horizon_cycles=settings.forecast_horizon_cycles,
    )


def run_dag(models: dict[str, Any], feature_row: np.ndarray) -> dict[str, Any]:
    """Execute the inference DAG on one feature row.

    Order is fixed: anomaly -> failure -> RUL; forecasting is not part of the
    per-window DAG (it runs independently per engine history).
    """
    x = np.asarray(feature_row, dtype=float).reshape(1, -1)
    anomaly_scores, anomaly_flags = models["anomaly"].predict(x)
    failure_proba = models["failure"].predict_proba(x)
    rul = models["rul"].predict(x)
    return {
        "dag_order": ["anomaly", "failure", "rul"],
        "anomaly": {
            "score": float(anomaly_scores[0]),
            "flag": bool(anomaly_flags[0]),
        },
        "failure": {"probability": float(failure_proba[0])},
        "rul": {
            "point": float(rul["point"][0]),
            "lower": float(rul["lower"][0]),
            "upper": float(rul["upper"][0]),
            "interval_level": float(rul["interval_level"][0]),
        },
    }


def train_all(
    *,
    subset: Subset = "FD002",
    seed: int | None = None,
    run_id: str | None = None,
    run_dir: Path | None = None,
    train_raw: pd.DataFrame | None = None,
    train_features: pd.DataFrame | None = None,
    forecast_engine_sample: int = 30,
) -> dict[str, Any]:
    """Train and evaluate all four families, then package artifacts."""
    settings = get_settings()
    seed = seed or settings.seed
    horizon = settings.failure_horizon_cycles
    if train_raw is None:
        train_raw = load_processed(subset).train
    if train_features is None:
        train_features = pd.read_parquet(
            settings.processed_dir / f"{subset}_features_train.parquet"
        )

    labeled = add_targets(train_features, horizon_cycles=horizon)
    train_split, validation_split = split_features(
        labeled,
        val_fraction=0.2,
        seed=seed,
    )
    X_train, y_rul_train = (
        _as_matrix(train_split),
        train_split["rul"].to_numpy(dtype=float),
    )
    X_validation, y_rul_validation, y_failure_validation = (
        _as_matrix(validation_split),
        validation_split["rul"].to_numpy(dtype=float),
        validation_split["failure"].to_numpy(),
    )

    # Anomaly detection (unsupervised; contamination sized from degradation).
    anomaly = AnomalyModel(seed=seed).fit(X_train, y_rul_train, horizon)
    anomaly_scores = anomaly.scores(X_validation)
    degraded = y_rul_validation <= 3 * horizon
    anomaly_proxy_auc = None
    if len(np.unique(degraded)) > 1:
        anomaly_proxy_auc = float(roc_auc_score(degraded, anomaly_scores))
    anomaly_metrics = {"proxy_auc": anomaly_proxy_auc}

    # Failure prediction with engine-disjoint calibration split.
    fit_engines, calibration_engines = split_engine_ids(
        set(train_split["unit"].unique()),
        val_fraction=0.2,
        seed=seed + 1,
    )
    fit_rows = train_split[train_split["unit"].isin(fit_engines)]
    calibration_rows = train_split[train_split["unit"].isin(calibration_engines)]
    failure = FailureModel(seed=seed).fit(
        _as_matrix(fit_rows),
        fit_rows["failure"].to_numpy(),
        _as_matrix(calibration_rows),
        calibration_rows["failure"].to_numpy(),
    )
    failure_metrics = evaluate_failure(y_failure_validation, failure.predict_proba(X_validation))

    # RUL point + gradient-boosted quantile interval.
    rul_model = RulModel(seed=seed).fit(
        _as_matrix(fit_rows),
        fit_rows["rul"].to_numpy(dtype=float),
        _as_matrix(calibration_rows),
        calibration_rows["rul"].to_numpy(dtype=float),
    )
    rul_predictions = rul_model.predict(X_validation)
    rul_metrics = evaluate_rul(y_rul_validation, rul_predictions["point"])
    rul_metrics["interval_coverage"] = float(
        np.mean(
            (rul_predictions["lower"] <= y_rul_validation)
            & (y_rul_validation <= rul_predictions["upper"])
        )
    )
    rul_metrics["interval_width"] = float(
        np.mean(rul_predictions["upper"] - rul_predictions["lower"])
    )

    # Independent forecasting leg on sampled train engines.
    sensors = ForecastingModel.select_sensors(train_features, settings.forecast_sensor_count)
    forecast_model = ForecastingModel(
        sensors=sensors,
        horizon=settings.forecast_horizon_cycles,
        seed=seed,
    )
    forecast_eval = forecast_model.evaluate(
        train_raw,
        train_features,
        engine_sample=forecast_engine_sample,
    )
    forecast_metrics = {
        "mape": float(forecast_eval["mape"]),
        "mape_ratio_vs_persistence": float(forecast_eval["mape_ratio_vs_persistence"]),
    }

    # Threshold gates (FR-04): every family must beat its committed baseline.
    thresholds = load_thresholds()
    baseline_rul_rmse = evaluate_rul_baseline(
        fit_rul_baseline(y_rul_train),
        y_rul_validation,
    )["rmse"]
    checks = {
        "failure": (
            failure_metrics["auc"] >= thresholds["failure"]["min_auc"]
            and failure_metrics["average_precision"]
            >= thresholds["failure"]["min_average_precision"]
            and failure_metrics["brier"] <= thresholds["failure"]["max_brier"]
            and failure_metrics["log_loss"] <= thresholds["failure"]["max_log_loss"]
        ),
        "rul": (
            rul_metrics["rmse"]
            <= baseline_rul_rmse
            * (1.0 - thresholds["rul"]["relative_rmse_gain_vs_baseline"])
            and rul_metrics["rmse"] <= thresholds["rul"]["max_rmse"]
            and thresholds["rul"]["interval_coverage_min"]
            <= rul_metrics["interval_coverage"]
            <= thresholds["rul"]["interval_coverage_max"]
        ),
        "forecast": (
            forecast_metrics["mape_ratio_vs_persistence"]
            <= thresholds["forecast"]["mape_ratio_vs_persistence_max"]
        ),
        "anomaly": (
            anomaly_proxy_auc is not None
            and anomaly_proxy_auc >= thresholds["anomaly"]["proxy_auc_min"]
        ),
    }

    run_id = run_id or (
        f"run-{datetime.now(UTC):%Y%m%d-%H%M%S}-{seed}"
    )
    run_dir = run_dir or settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    training_config = _training_config(subset, seed)
    provenance = DataProvenance(
        subset=subset,
        source="processed-parquet",
        row_count=int(len(train_features)),
    )
    git_sha = _git_sha()
    manifests = {
        "anomaly": save_artifact(
            family="anomaly",
            run_id=run_id,
            subset=subset,
            model=anomaly,
            metrics={key: value for key, value in anomaly_metrics.items() if value is not None},
            thresholds=dict(thresholds["anomaly"]),
            features=FEATURE_COLUMNS,
            training_config=training_config,
            data_provenance=provenance,
            git_sha=git_sha,
            run_dir=run_dir,
        ),
        "failure": save_artifact(
            family="failure",
            run_id=run_id,
            subset=subset,
            model=failure,
            metrics=failure_metrics,
            thresholds=dict(thresholds["failure"]),
            features=FEATURE_COLUMNS,
            training_config=training_config,
            data_provenance=provenance,
            git_sha=git_sha,
            run_dir=run_dir,
        ),
        "performance": save_artifact(
            family="performance",
            run_id=run_id,
            subset=subset,
            model=rul_model,
            metrics=rul_metrics,
            thresholds=dict(thresholds["rul"]),
            features=FEATURE_COLUMNS,
            training_config=training_config,
            data_provenance=provenance,
            git_sha=git_sha,
            run_dir=run_dir,
        ),
        "forecasting": save_artifact(
            family="forecasting",
            run_id=run_id,
            subset=subset,
            model=forecast_model,
            metrics=forecast_metrics,
            thresholds=dict(thresholds["forecast"]),
            features=sensors,
            training_config=training_config,
            data_provenance=provenance,
            git_sha=git_sha,
            run_dir=run_dir,
        ),
    }

    summary: dict[str, Any] = {
        "run_id": run_id,
        "subset": subset,
        "seed": seed,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "splits": {
            "train_engines": int(train_split["unit"].nunique()),
            "validation_engines": int(validation_split["unit"].nunique()),
        },
        "model_configs": {
            "anomaly": anomaly.params,
            "failure": failure.params,
            "performance": rul_model.params,
            "forecasting": forecast_model.params,
        },
        "metrics": {
            "anomaly": anomaly_metrics,
            "failure": failure_metrics,
            "performance": rul_metrics,
            "forecasting": forecast_metrics,
        },
        "forecast_detail": forecast_eval,
        "checks": checks,
        "all_passed": all(checks.values()),
        "artifact_paths": {
            family: str(manifest.artifact_path) for family, manifest in manifests.items()
        },
    }
    write_run_json(run_dir, summary)
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"run: {summary['run_id']} | subset: {summary['subset']} | seed: {summary['seed']}")
    print(f"all thresholds passed: {summary['all_passed']}")
    for family, metrics in summary["metrics"].items():
        readable = {key: round(value, 4) for key, value in metrics.items() if value is not None}
        print(f"  {family}: {readable}")
