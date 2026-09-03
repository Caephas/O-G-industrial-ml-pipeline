# ADR 0009: SQLite-Backed Model Registry

- Status: Accepted
- Date: 2026-09-03

## Context

The registry decision gate (open since the contracts were frozen) had two
candidates: a filesystem JSON index or a SQLite index over the self-describing
artifact files. The registry protocol is identical either way.

## Decision

The default backend is a SQLite index (`artifacts/registry/registry.db`)
over the self-describing joblib + manifest artifact directories:

- `artifacts` stores the model and its JSON manifest;
- `artifacts` transitions are recorded as rows and applied transactionally;
- promoting a staging model automatically archives the current production
  model of the same family, keeping exactly one production artifact;
- the registry assigns sequential per-family versions.

## Consequences

- Stage changes are atomic and auditable via the `transitions` table.
- The filesystem-JSON alternative remains implementable behind the same
  frozen protocol without touching any caller.
- Registry state is disposable: it can be rebuilt from artifact manifests.
