# Runbook: Lifecycle Simulation (`make demo`)

## Prerequisites

- `make setup`, `make fetch`, `make baselines`
- A passing training run: `make train` (the demo auto-detects the latest run
  whose thresholds all passed, or use `--champion-run-id`)

## Run

```bash
make demo
```

Or with an explicit champion:

```bash
.venv/bin/python scripts/lifecycle_sim.py --champion-run-id <run-id>
```

## What happens

1. The champion's failure and RUL models serve the FD002 test split as a
   replayed stream.
2. The PSI detector compares recent windows against the healthy-deployment
   reference (training rows with RUL ≥ 90). When ≥15% of monitored features
   exceed PSI > 0.20, a drift event fires.
3. An isolated retrain worker trains a new candidate in a separate process
   (inference is unaffected while it runs).
4. The candidate registers as staging, shadows the champion over the final 50
   labeled windows, and is promoted if Brier/RMSE parity (within tolerance)
   and p50/p95 latency budgets pass.
5. Promotion archives the prior production model per family.

## Verify

- `artifacts/lifecycle/sim_events.jsonl` contains `drift`, `retrain`,
  `shadow_eval`, and `stage_transition` events.
- `artifacts/registry/registry.db` lists version 2 artifacts as production and
  version 1 artifacts as archived.

## Manual intervention

- The simulation resets `registry.db` and `sim_events.jsonl` at the start of
  every run (both are regenerable demo state).
- If the run reports that drift did not trigger, the reference or trigger
  rule needs review — see ADR 0010 before tuning thresholds.
