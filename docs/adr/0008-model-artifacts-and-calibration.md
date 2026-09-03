# ADR 0008: Leakage-Safe Calibration and Self-Describing Artifacts

- Status: Accepted
- Date: 2026-09-03

## Context

Failure probabilities and RUL intervals must be calibrated without leaking
engine-level information across folds (FR-04, FR-06, FR-07). Model runs must
also be reproducible and portable: an artifact directory alone should describe
what was trained, on which data, with which metrics, and under which
thresholds.

## Decision

- Failure calibration uses an engine-disjoint calibration split: the Random
  Forest is fit on one set of engines, then a Platt (sigmoid) calibrator is
  fit on predictions for a disjoint set of engines. No engine appears in both.
- RUL intervals are produced by gradient-boosted quantile-loss regressors
  (5th/95th) and then scale-calibrated on the same engine-disjoint split so
  empirical coverage reaches the nominal level.
- Every run writes one directory per family containing `model.joblib` plus a
  `manifest.json` matching the frozen `ArtifactManifest` schema (family,
  version, stage, metrics, thresholds, features, training config, provenance,
  git SHA). `run.json` at the run root records split sizes, model configs, and
  the full metrics table.

## Consequences

- Probabilities and intervals stay honest under engine-disjoint evaluation.
- The registry (and any reviewer) can reconstruct a run from its directory
  without external state.
- Determinism is testable: two runs with the same seed produce identical
  metrics, verified in the test suite.
