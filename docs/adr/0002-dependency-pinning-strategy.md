# ADR 0002: Dependency Pinning Strategy

- Status: Accepted
- Date: 2026-09-03

## Context

FR-16 requires reproducible runs. Unpinned dependencies make the same code
behave differently across environments and over time.

## Decision

Dependencies are declared in `pyproject.toml` with exact `==` pins. New
dependencies are added when a feature first needs them, after their official
docs are verified for the pinned version. Python 3.11+ is the project floor.

## Consequences

- Upgrades are deliberate, reviewed changes rather than incidental ones.
- The direct-dependency set stays small and readable; transitive resolution is
  handled by pip at install time.
- A lockfile tool (e.g., uv) can be adopted later without changing this
  decision's structure.
