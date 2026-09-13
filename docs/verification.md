# Verification and Acceptance Evidence

This page maps every requirement in the machine contract to its implementation
and to observable evidence. Requirement IDs (FR-01…FR-18) are defined in the
project's development plan; the columns below are the public audit trail.

## Verification summary

| Check | Result |
|-------|--------|
| Automated test suite | 79 passed (`make check`, ~37 s) |
| Lint (ruff) | Clean |
| All model thresholds | Passed on FD002, seed 42 (`all_passed: true`) |
| One-command container run | Clean volumes → bootstrap → API healthy, verified over HTTP |
| Public vocabulary review | No proprietary or non-REST protocol references |

## Acceptance matrix

| ID | Requirement | Implementation | Evidence | Status |
|----|-------------|----------------|----------|--------|
| FR-01 | Checksummed multi-source C-MAPSS ingestion | `scripts/fetch_cmapss.py`, `pipeline/data/loader.py` | `make fetch` verified md5 `79a22f36…`; loader tests; FD002 260 train / 259 test engines | PASS |
| FR-02 | Data quality gates | `pipeline/data/quality.py` | `tests/test_quality.py`; FD002 report (`data/quality/FD002_report.json`) | PASS |
| FR-03 | Regime-aware rolling features | `pipeline/data/features.py` | `tests/test_features.py` (leakage, layout); feature manifest: 126 features | PASS |
| FR-04 | Leakage-safe evaluation harness + committed thresholds | `pipeline/eval/`, `metrics/thresholds.yaml` | `tests/test_splits.py`, `test_metrics.py`, `test_baselines.py`; baseline report | PASS |
| FR-05 | Isolation Forest anomaly detection | `pipeline/models/anomaly.py` | `tests/test_models.py`; validation proxy AUC 0.714 (≥ 0.70) | PASS |
| FR-06 | RF failure prediction + Platt calibration | `pipeline/models/failure.py` | AUC 0.985, AP 0.933, Brier 0.036, log loss 0.119 | PASS |
| FR-07 | RUL with gradient-boosted quantile intervals | `pipeline/models/performance.py` | RMSE 20.8 vs 45.5 baseline; coverage 0.877 (band 0.80–0.98) | PASS |
| FR-08 | Independent ARIMA/SARIMA forecasting | `pipeline/models/forecasting.py` | MAPE 0.064; ratio 0.741 vs persistence (≤ 0.97) | PASS |
| FR-09 | 4-model DAG orchestration | `pipeline/models/orchestrator.py`, `pipeline/cli.py` | `tests/test_orchestrator.py` (order + determinism); `run.json` metadata | PASS |
| FR-10 | Versioned registry (staging/production/archived) | `pipeline/lifecycle/registry.py` | `tests/test_registry.py`; production v2 ×4, archived v1 ×4 | PASS |
| FR-11 | Shadow-mode dual quality/latency gates | `pipeline/lifecycle/shadow.py` | `tests/test_shadow.py`; `shadow_eval` event: quality + latency passed over 50 predictions | PASS |
| FR-12 | PSI drift detection + isolated auto-retraining | `pipeline/lifecycle/drift.py`, `trainer.py`, `scripts/lifecycle_sim.py` | `tests/test_drift.py`, `test_trainer.py`; 1 drift, 2 retrain, 12 stage-transition events | PASS |
| FR-13 | REST inference API | `pipeline/api/server.py`, `pipeline/schemas/api.py` | `tests/test_api.py`; live `POST /v1/predict` 200 with versions and RUL interval | PASS |
| FR-14 | Health tracking + dashboard | `pipeline/monitoring/` | `tests/test_api.py`; `/health` p50/p95; `/dashboard` + partials 200 | PASS |
| FR-15 | One-command Docker run | `Dockerfile`, `docker-compose.yml`, `scripts/bootstrap.py` | Clean-volume run: bootstrap → healthy container → predict/dashboard over HTTP; `tests/test_bootstrap.py` | PASS |
| FR-16 | Reproducibility (pins, seeds, provenance) | `pyproject.toml`, `pipeline/config.py`, artifact manifests | Exact pins; seed 42/44 deterministic runs; checksummed dataset provenance; git SHA in manifests | PASS |
| FR-17 | Automated tests for core modules | `tests/` | 79 passing across data, features, eval, models, lifecycle, API, bootstrap | PASS |
| FR-18 | Documentation | `README.md`, `docs/architecture.md`, `docs/model-cards/`, `docs/runbooks/`, ADRs | Reviewed; model cards carry real metrics; architecture matches implementation | PASS |

## Reproduce

```bash
make setup
make fetch
make baselines
make train
make demo
make check
```

Or the containerized path: `docker compose up --build`.

## Notes and limitations

- Lifecycle evaluation uses offline replay with the test split's RUL ground
  truth; live deployments apply the same gates once outcomes are observed
  (ADR 0010).
- The demo is local-only: no authentication, rate limiting, or cloud
  infrastructure (ADR 0011, ADR 0012).
- Run ids embed timestamps, so metrics are reproducible but ids differ per
  execution.
