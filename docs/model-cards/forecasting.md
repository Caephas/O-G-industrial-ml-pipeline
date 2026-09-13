# Model Card — Sensor Forecasting

## Intended use

Independent per-sensor forecasts over a 10-cycle horizon for the three most
consistently trending sensors, supporting trend analysis alongside the main
DAG (which it never blocks).

## Training / inference

- Sensor selection: highest mean absolute window slope per engine over the
  training split (s11, s4, s15 for the reference run).
- Method: pmdarima AutoARIMA with explicit linear trend, differencing
  disabled, capped order (deterministic; ADR 0010 companion tuning notes in
  run metadata).
- Models are fit per engine/sensor at evaluation or request time.

## Evaluation (sampled train engines)

| Metric | Value | Baseline (persistence) | Threshold |
|--------|-------|------------------------|-----------|
| MAPE | 0.064 | 0.086 (same sample) | ratio ≤ 0.97 |

## Limitations

- Engine cycles are not calendar time; seasonality detection is intentionally
  disabled for this domain.
- Forecasts on the noisiest channels can be worse than persistence; sensor
  selection therefore favors consistent trends.

## Run provenance

Reference run: `run-20260903-095637-42`. Production artifact:
`run-20260903-101506-44`.
