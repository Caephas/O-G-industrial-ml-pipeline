# Model Card — Anomaly Detection

## Intended use

Early-warning signal per feature window: "is this engine behaving like the
healthy fleet?" Outputs are an anomaly score and a binary flag consumed by the
inference DAG before failure prediction.

## Training

- Algorithm: Isolation Forest, contamination estimated adaptively as the share
  of training windows in the final `3 × horizon` cycles of engine life
  (unsupervised at serve time).
- Inputs: 126 regime-normalized rolling features (6 statistics × 21 sensors),
  FD002 train split (260 engines, engine-disjoint validation).
- Seed 42; hyperparameters logged in run metadata.

## Evaluation (validation split)

| Metric | Value | Threshold |
--------|-------|-----------|
| Proxy AUC (late-life windows, RUL ≤ 90) | 0.714 | ≥ 0.70 |

## Limitations

- Ground truth for anomalies is a late-life proxy; real deployments need
  labeled event histories for proper calibration.
- Scores reflect deviation from the training reference; a new operating
  regime would legitimately score as anomalous until the reference is updated.

## Run provenance

Reference run: `run-20260903-095637-42` (FD002, seed 42). Production artifact
at the lifecycle milestone: `run-20260903-101506-44`.
