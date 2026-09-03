# ADR 0001: Frozen Typed Contracts

- Status: Accepted
- Date: 2026-09-03

## Context

The foundation of this project must freeze the interfaces between modules
before independent features are built on them. Without explicit contracts,
different parts of the codebase can drift apart (e.g., the registry backend and
the model artifacts disagreeing about what a manifest contains).

## Decision

All cross-module data is typed with pydantic models (`extra="forbid"`) defined
in `pipeline/schemas/`, and the model registry is defined as a `Protocol` in
`pipeline/schemas/registry.py`. Settings live in `pipeline/config.py` backed by
pydantic-settings.

## Consequences

- Validation happens at module boundaries, so failures surface early.
- Any interface change requires re-planning (documented in the development
  plan), which keeps independent features safe to build in parallel.
- FR-16 and FR-17 are satisfied by construction: contracts are inspectable and
  testable.
