# ADR 0005: Quality Gate Severity Semantics

- Status: Accepted
- Date: 2026-09-03

## Context

Quality gates must decide whether a dataset is trainable (FR-02). Treating
every anomaly as fatal would block on conditions the feature pipeline handles
by design; treating everything as advisory would let corrupt data through.

## Decision

Each gate returns `pass`, `warn`, or `fail`:

- Blocking (`fail`): missing values, a missing schema column, or a
  pathological class imbalance.
- Advisory (`warn`): constant sensors and per-regime zero variance, because
  feature engineering drops non-informative channels.

Training proceeds only when no gate reports `fail`.

## Consequences

- The boundary between "handle downstream" and "stop the pipeline" is
  explicit and testable.
- Gate severity can be tuned per deployment without changing gate logic.
