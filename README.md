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

## Quickstart

```bash
make setup
make check
```

Dataset download, model training, lifecycle simulation, and the API service are
in active development.
