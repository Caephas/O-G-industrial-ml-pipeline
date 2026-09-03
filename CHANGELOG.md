# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added

- Repository scaffolding: exact dependency pins, Makefile targets, CI workflow
  (FR-16, FR-17)
- Frozen inter-module contracts: settings, artifact manifest, registry
  protocol, REST schemas, metric records, lifecycle events (FR-16)
- Deterministic synthetic C-MAPSS fixture generator (FR-17)
- Checksum-verified multi-source C-MAPSS downloader with provenance tracking
  (FR-01, FR-16)
- C-MAPSS loader with raw-file validation and processed Parquet output
  (FR-01)
- Data quality gates with blocking semantics: missing values, constant
  sensors, per-regime zero variance, class imbalance (FR-02)
- Loader, quality gate, and fetch script tests (FR-17)
- System architecture documentation with C4, data-flow, model-DAG, lifecycle,
  and deployment diagrams (FR-18)
- Regime-aware feature pipeline: seeded 6-regime encoder, per-regime
  normalization, chronological 6-stat rolling windows (FR-03, FR-16)
- Evaluation harness: labels, engine-disjoint splits, metric implementations,
  baseline models, committed thresholds (FR-04)
- Feature engineering and evaluation harness tests (FR-17)
- Model families: Isolation Forest anomaly detection with adaptive
  contamination, Random Forest + Platt-calibrated failure prediction, RUL
  point estimates with gradient-boosted quantile intervals (scale-calibrated
  on an engine-disjoint split), and per-sensor ARIMA/SARIMA forecasting
  (FR-05, FR-06, FR-07, FR-08)
- 4-model DAG orchestration with deterministic runs, threshold gates, and
  self-describing joblib artifacts (FR-09, FR-10, FR-16)
- CLI (`train` / `evaluate` / `predict`) and model tests (FR-17)
- Training runbook (FR-18)
- ADRs for typed contracts, dependency pinning, and repository licensing
  (FR-16)
