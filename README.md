# industrial-ml-pipeline

End-to-end predictive maintenance on the NASA C-MAPSS turbofan dataset (FD002,
six operating regimes): data quality gates → regime-aware features → a 4-model
inference DAG (anomaly detection, failure prediction, RUL estimation with
quantile intervals, and independent time-series forecasting) → versioned model
registry → shadow-mode rollouts → PSI drift detection with auto-retraining →
REST inference API + live dashboard → one-command Docker run.

## What it demonstrates

- Isolation Forest anomaly detection with adaptive contamination
- Random Forest failure prediction with Platt (sigmoid) calibration
- RUL estimation: Random Forest point estimate + gradient-boosted quantile
  intervals, scale-calibrated on an engine-disjoint split
- Independent per-sensor ARIMA/SARIMA forecasting
- Shadow-mode rollouts with dual quality/latency gates over 50 predictions
- PSI drift detection (>0.20 per feature, aggregate trigger) with isolated
  auto-retraining and cooldown
- Versioned SQLite-backed model registry (staging → production → archived)
- REST inference API + htmx operations dashboard
- One-command Docker Compose run

## Status

The full pipeline is implemented end to end: data acquisition, quality gates,
regime-aware features, the four model families, registry-backed lifecycle
automation, and the serving layer. Docker Compose is the deployment path.

System design and architecture diagrams: [docs/architecture.md](docs/architecture.md).

Model cards per family: [anomaly](docs/model-cards/anomaly.md),
[failure](docs/model-cards/failure.md),
[performance / RUL](docs/model-cards/performance.md),
[forecasting](docs/model-cards/forecasting.md).

Evidence: [verification and acceptance matrix](docs/verification.md) ·
[end-to-end demo walkthrough](docs/demo.md).

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
queryable from `artifacts/registry/registry.db`.

Serve the REST inference API and the live htmx dashboard (loads the
production artifacts from the registry):

```bash
make serve
```

Then:

- `POST http://127.0.0.1:8000/v1/predict` — sensor window → anomaly, failure
  probability, RUL interval, optional forecasts
- `GET http://127.0.0.1:8000/health` — serving health and model versions
- `GET http://127.0.0.1:8000/dashboard` — registry, drift, lifecycle events,
  and recent predictions (auto-refreshing via htmx)

Interactive OpenAPI docs are available at `/docs`. htmx 2.0.10 is vendored
under `pipeline/monitoring/static/` (BSD-2-Clause).

## Docker (recommended)

```bash
docker compose up --build
```

The first start runs the full pipeline (checksummed dataset download, feature
build, model training, lifecycle simulation) into persistent named volumes,
then serves the API at `http://localhost:8000`. Later starts skip completed
stages. On-demand lifecycle tooling runs through the `worker` profile:

```bash
docker compose run --rm worker
```

## Data and attribution

The NASA C-MAPSS turbofan degradation dataset is public NASA data; the PCoE
Zenodo mirror is CC-BY-4.0. The downloader verifies a checksum and records
provenance in `data/provenance.json`.

## Repository layout

```text
pipeline/
  data/       loader, quality gates, regime-aware features
  models/     anomaly, failure, RUL, forecasting + DAG orchestrator
  lifecycle/  registry, drift, shadow mode, retrain worker
  api/        FastAPI inference service
  monitoring/ health tracker, htmx dashboard
scripts/      fetch, baselines, lifecycle simulation, container bootstrap
tests/        unit + integration suites (synthetic fixtures)
docs/         architecture, ADRs, model cards, runbooks
```
