# Model Card — Failure Prediction

## Intended use

Calibrated probability that an engine will fail within the 30-cycle horizon,
given a regime-normalized feature window. Intended for maintenance triage and
as the quality signal for shadow-mode promotion.

## Training

- Base estimator: Random Forest (200 trees, balanced class weights).
- Calibration: Platt (sigmoid) fit on an engine-disjoint calibration split so
  no engine appears in both base training and calibration.
- Inputs: 126 regime-normalized rolling features; FD002 train split.
- Label: failure within 30 cycles (engine-wise, leakage-safe).

## Evaluation (validation split, 52 engines)

| Metric | Value | Baseline (majority) | Threshold |
|--------|-------|--------------------|-----------|
| AUC | 0.985 | 0.500 | ≥ 0.85 |
| Average precision | 0.933 | — | ≥ 0.35 |
| Brier | 0.036 | 0.130 | ≤ 0.124 |
| Log loss | 0.119 | ~0.421 | ≤ 0.40 |

## Limitations

- The 30-cycle horizon is fixed per deployment configuration; recalibrate if
  the horizon changes.
- Data is simulated NASA telemetry; absolute probabilities may not transfer to
  other equipment without retraining on local distributions.

## Run provenance

Reference run: `run-20260903-095637-42`. Production artifact:
`run-20260903-101506-44` (shadow evaluation improved Brier 0.137 vs 0.239 on
the replay segment).
