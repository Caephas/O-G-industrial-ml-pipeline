# ADR 0010: Lifecycle Simulation and Drift Trigger Semantics

- Status: Accepted
- Date: 2026-09-03

## Context

The lifecycle demo replays the FD002 test split to exercise drift detection,
isolated retraining, shadow evaluation, and promotion (FR-11, FR-12). Two
behaviors needed explicit definitions: what "drift" means for this static
corpus, and how labels enter shadow evaluation.

## Decision

- The PSI reference is a healthy-deployment snapshot: training rows with
  RUL ≥ 90 cycles, matching the practice of taking the reference when a model
  deploys. Reference-constant features are excluded (variance below 1e-3),
  mirroring the quality-gate treatment of non-informative channels.
- A drift event fires when at least 15% of monitored features exceed
  PSI > 0.20 over a rolling 250-window buffer. A single noisy feature cannot
  trigger retraining; broad, late-life degradation does. Cooldown prevents
  repeated triggers.
- Shadow evaluation is an offline replay: test trajectories ship with RUL
  ground truth, so labels are available when the gates run. A live deployment
  would apply the identical dual gates once outcomes are observed, with the
  same 50-prediction window and p50/p95 latency budgets.
- The quality gate accepts a challenger that is no worse than the champion
  within the configured tolerance on failure Brier and RUL RMSE; promotion
  requires the latency gate to pass as well.

## Consequences

- The demo is deterministic (seeded replay and retraining) and shows a real
  drift -> retrain -> shadow -> promote -> archive cycle.
- The drift rule is deliberately documented so the threshold and trigger
  fraction are reviewable, not hidden tuning.
- Offline replay is clearly labeled as such in the simulation and runbook.
