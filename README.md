# industrial-ml-pipeline

End-to-end predictive maintenance on the NASA C-MAPSS turbofan dataset (FD002,
six operating regimes): data quality gates → regime-aware features → a 4-model
inference DAG (anomaly detection, failure prediction, RUL estimation with
quantile intervals, and independent time-series forecasting) → versioned model
registry → shadow-mode rollouts → PSI drift detection with auto-retraining →
REST inference API + live dashboard → one-command Docker run.

## Status

Foundation complete: repository layout, pinned dependencies, frozen interface
contracts, deterministic test fixtures, and CI are in place.

System design and architecture diagrams: [docs/architecture.md](docs/architecture.md).

## Quickstart

```bash
make setup
make check
```

Fetch the NASA C-MAPSS dataset (12 MB download; verified by checksum, with an
automatic mirror fallback):

```bash
make fetch
```

Build regime-aware features and compute the evaluation baselines (writes
feature Parquet files under `data/processed/` and a baseline report under
`artifacts/baselines/`):

```bash
make baselines
```

Train and evaluate the four model families (anomaly, failure, RUL, forecasting)
against the committed thresholds. Artifacts and run metadata are written under
`artifacts/runs/<run-id>/`:

```bash
make train
```

Run the lifecycle simulation: the champion serves a replayed stream, PSI
drift triggers an isolated retrain worker, and the candidate shadows the
champion until the dual quality/latency gates decide promotion:

```bash
make demo
```

Events land in `artifacts/lifecycle/sim_events.jsonl` and registry state is
queryable from `artifacts/registry/registry.db`. The API service is in active
development.
