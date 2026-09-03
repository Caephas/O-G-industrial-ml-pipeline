"""Command-line interface: train, evaluate, and predict (FR-09, FR-13)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.artifacts.manager import load_artifact
from pipeline.config import get_settings
from pipeline.data.features import (
    RegimeEncoder,
    build_feature_frame,
    feature_columns,
)
from pipeline.data.loader import COLUMNS
from pipeline.models.orchestrator import print_summary, run_dag, train_all


def _load_run_models(run_dir: Path) -> dict:
    return {
        family: load_artifact(run_dir, family)[0]
        for family in ("anomaly", "failure", "performance", "forecasting")
    }


def _cmd_train(args: argparse.Namespace) -> None:
    summary = train_all(
        subset=args.subset,
        seed=args.seed,
        run_id=args.run_id,
        forecast_engine_sample=args.forecast_engines,
    )
    print_summary(summary)
    print(f"artifacts: {get_settings().runs_dir / summary['run_id']}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    run_dir = get_settings().runs_dir / args.run_id
    summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    print_summary(summary)


def _cmd_predict(args: argparse.Namespace) -> None:
    settings = get_settings()
    run_dir = settings.runs_dir / args.run_id
    raw = pd.read_csv(args.window, sep=r"\s+", header=None, names=COLUMNS)
    raw = raw.astype({"unit": "int64", "cycle": "int64"})
    if raw["unit"].nunique() != 1:
        raise ValueError("window file must contain a single engine")
    encoder = RegimeEncoder.load(settings.processed_dir)
    features = build_feature_frame(raw, encoder, window_size=settings.window_size)
    models = _load_run_models(run_dir)
    last_row = features.iloc[[-1]]
    result = run_dag(models, last_row[feature_columns()].to_numpy(dtype=float).ravel())
    result["final_cycle"] = int(last_row["cycle"].iloc[0])
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="industrial-ml-pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train the 4-model DAG")
    train.add_argument("--subset", default="FD002")
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--run-id", default=None)
    train.add_argument("--forecast-engines", type=int, default=30)
    train.set_defaults(func=_cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="show stored run metrics")
    evaluate.add_argument("--run-id", required=True)
    evaluate.set_defaults(func=_cmd_evaluate)

    predict = subparsers.add_parser("predict", help="predict on a raw window CSV")
    predict.add_argument("--run-id", required=True)
    predict.add_argument("--window", type=Path, required=True)
    predict.set_defaults(func=_cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
