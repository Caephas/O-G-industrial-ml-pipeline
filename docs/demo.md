# End-to-End Demo Walkthrough

Everything below was executed on macOS (arm64), Python 3.11.3, Docker 29.7.2.
Run ids are timestamped, so yours will differ; metrics should match.

## Path A — local

```bash
make setup          # venv + pinned dependencies
make fetch          # checksummed C-MAPSS download
```

Expected output (abridged):

```text
verified archive: .../data/raw/CMAPSSData.zip
```

```bash
make baselines      # regime features + baseline metrics
```

```text
train engines: 208, validation engines: 52
failure baseline (majority prior=0.1489): auc=0.5000 brier=0.1304
rul baseline (constant=104.0): rmse=45.53 nasa=17967116.7
forecast persistence mape (['s6', 's20', 's21']): 0.6148
```

```bash
make train          # 4-model DAG, threshold gates, artifacts
```

```text
run: run-20260903-095637-42 | subset: FD002 | seed: 42
all thresholds passed: True
  anomaly: {'proxy_auc': 0.7143}
  failure: {'auc': 0.9847, 'average_precision': 0.9331, 'brier': 0.0359, 'log_loss': 0.1192}
  performance: {'rmse': 20.8008, 'nasa_score': 145496.508, 'interval_coverage': 0.8771, 'interval_width': 53.445}
  forecasting: {'mape': 0.0638, 'mape_ratio_vs_persistence': 0.7414}
```

```bash
make demo           # drift -> retrain -> shadow -> promote -> archive
```

```json
{
  "challenger_run": "run-20260903-101506-44",
  "drift_events": 1,
  "gate": {
    "quality_pass": true,
    "latency_pass": true,
    "promote": true,
    "champion_brier": 0.239,
    "challenger_brier": 0.137,
    "champion_rmse": 10.64,
    "challenger_rmse": 9.91,
    "p50_ms": 47.4
  },
  "promoted_families": ["anomaly", "failure", "performance", "forecasting"]
}
```

```bash
make serve          # REST API + dashboard
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok","model_versions":{"failure":{"artifact_id":"failure-run-20260903-101506-44","version":2}}}
```

A prediction request returns the schema-valid response (abridged):

```json
{
  "failure": {"probability": 0.0008, "horizon_cycles": 30, "calibrated": true},
  "rul": {"point": 115.3, "lower": 73.4, "upper": 125.5, "interval_level": 0.9},
  "model_versions": [{"family": "failure", "version": 2, "stage": "production"}]
}
```

The dashboard is at `http://127.0.0.1:8000/dashboard`; interactive OpenAPI
docs at `/docs`.

## Path B — Docker (one command)

```bash
docker compose up --build
```

On a machine with an empty volume set the container bootstraps the entire
pipeline and then serves:

```text
verified archive: /app/data/raw/CMAPSSData.zip
train engines: 208, validation engines: 52
all thresholds passed: True
pipeline ready; starting API
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Observed: image build ≈ 2 minutes; full bootstrap to a healthy container
≈ 9–10 minutes on CPU; `/health`, `/v1/predict`, and `/dashboard` all return
200 from the container. Subsequent starts skip completed stages and come up
in seconds. Lifecycle tooling on demand: `docker compose run --rm worker`.

## Automated verification

```bash
make check
```

```text
79 passed in ~37s
```

Full requirement-by-requirement evidence lives in
[verification.md](verification.md).
