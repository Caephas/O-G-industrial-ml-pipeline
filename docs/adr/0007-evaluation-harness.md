# ADR 0007: Evaluation Harness and Metrics-First Thresholds

- Status: Accepted
- Date: 2026-09-03

## Context

Evaluation must be defined before models are trained (FR-04): splits must not
share engines, metrics must be canonical, and "good enough" must be measurable
against baselines rather than asserted after the fact.

## Decision

- Engine-disjoint, deterministic train/validation splits (seeded shuffle of
  engine ids; no engine appears in both partitions).
- Canonical metrics per model family: AUC/AP/Brier/log loss for failure,
  RMSE/NASA score for RUL, MAPE/RMSE for forecasts, with keys defined in the
  frozen `MetricNames` contract.
- Baselines (majority class, constant RUL, persistence forecast) are measured
  on the validation split and recorded under `artifacts/baselines/`.
- Acceptance thresholds live in the committed `metrics/thresholds.yaml`,
  calibrated against those baseline numbers.

## Consequences

- Every model family has a number to beat before it can ship.
- Threshold changes are explicit, reviewed edits to a tracked file.
- The same metric implementations are reused by evaluation, shadow-mode
  gates, and registry manifests, so numbers stay comparable.
