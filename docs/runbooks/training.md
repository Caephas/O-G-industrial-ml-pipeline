# Runbook: Training the 4-Model DAG

## Prerequisites

- `make setup` (virtualenv with pinned dependencies)
- `make fetch` (checksummed C-MAPSS archive)
- `make baselines` (regime features + baseline report; also writes the
  feature Parquet files training reads)

## Train

```bash
make train
```

Equivalent to:

```bash
.venv/bin/python -m pipeline.cli train --subset FD002 --seed 42
```

The run splits train engines into model-fit/calibration/validation partitions
(all engine-disjoint), trains anomaly, failure, RUL, and forecasting families,
and evaluates every family against `metrics/thresholds.yaml`.

## Verify a run

```bash
.venv/bin/python -m pipeline.cli evaluate --run-id <run-id>
```

## Predict on a raw window

Provide a single-engine window in C-MAPSS raw layout (26 whitespace-separated
columns per cycle, no header):

```bash
.venv/bin/python -m pipeline.cli predict \
  --run-id <run-id> \
  --window path/to/window.csv
```

## Manual intervention

- If `make train` reports `all thresholds passed: False`, inspect
  `artifacts/runs/<run-id>/run.json` for the failing family; a model that
  fails gates must not be promoted to production (shadow-mode evaluation in
  the lifecycle milestone enforces this automatically).
- Training is CPU-only; allow several minutes for a full FD002 run.
