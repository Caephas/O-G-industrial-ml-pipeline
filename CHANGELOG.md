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
- ADRs for typed contracts, dependency pinning, and repository licensing
  (FR-16)
