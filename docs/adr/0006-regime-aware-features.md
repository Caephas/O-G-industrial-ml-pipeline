# ADR 0006: Regime-Aware Feature Engineering

- Status: Accepted
- Date: 2026-09-03

## Context

FD002 engines operate across six operating regimes. Sensor readings mix
regime effects with degradation, so features computed on raw values would
confound the two. Rolling statistics also create a temporal-leakage risk if
windows are not strictly chronological and engine-scoped (FR-03, FR-04).

## Decision

- Identify regimes with seeded KMeans (6 clusters) over the three operating
  settings, fit on the training split only.
- Z-score each sensor per regime using training statistics; persist the
  encoder alongside the feature manifest so inference transforms identically.
- Compute six rolling statistics per sensor (mean, std, min, max, range,
  slope) over windows of at most `window_size` cycles, engine-scoped and
  chronological. Rows before the window is full use expanding windows rather
  than padding, so no row invents past data.

## Consequences

- Regime effects are removed before model training; the DAG sees normalized
  sensor dynamics.
- The encoder is part of every model artifact's lineage, keeping train and
  inference transforms identical.
- The leakage test (perturbing future rows must not change past features) is
  part of the test suite.
