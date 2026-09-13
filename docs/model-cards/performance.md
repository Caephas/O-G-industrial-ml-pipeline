# Model Card — Performance / Remaining Useful Life

## Intended use

Point estimate and calibrated 90% interval for remaining useful life in
cycles, used to size maintenance windows and as the second quality signal in
shadow-mode promotion.

## Training

- Point estimator: Random Forest regressor on RUL clipped at 125 cycles.
- Interval: gradient-boosted quantile-loss regressors (5th/95th percentile),
  scale-calibrated to nominal coverage on an engine-disjoint calibration
  split (ADR 0008).
- Inputs: 126 regime-normalized rolling features; FD002 train split.

## Evaluation (validation split)

| Metric | Value | Baseline (constant 104) | Threshold |
|--------|-------|------------------------|-----------|
| RMSE | 20.8 | 45.5 | ≥10% better + ≤ 50 |
| NASA PHM08 score | 145,497 | 17,967,117 | — |
| Interval coverage (90%) | 0.877 | — | 0.80–0.98 |
| Mean interval width | 53.4 cycles | — | — |

## Limitations

- RUL is clipped at 125 cycles per the C-MAPSS convention; very healthy
  engines are indistinguishable by design.
- Coverage is calibrated on FD002 validation; recalibrate on local data for
  other equipment.

## Run provenance

Reference run: `run-20260903-095637-42`. Production artifact:
`run-20260903-101506-44`.
