# ADR 0011: REST Serving and Registry-Authoritative Versions

- Status: Accepted
- Date: 2026-09-03

## Context

The serving layer must expose the frozen predict contract over REST (FR-13),
report health (FR-14), and load exactly the artifacts the lifecycle promoted
to production.

## Decision

- FastAPI serves `POST /v1/predict` (schema-first from the frozen pydantic
  contract), `GET /v1/models`, and `GET /health`. URI versioning: breaking
  changes move to `/v2/`.
- Errors follow RFC 7807 Problem Details (`type`, `title`, `status`,
  `detail`, `instance`); validation errors are JSON-encoded before serialization.
- At startup the app reads production entries from the SQLite registry. The
  registry entry is authoritative for version/stage metadata; the artifact
  path provides the serialized model. This keeps the served version echo
  consistent with the lifecycle that promoted the model.
- The dashboard is server-rendered with vendored htmx 2.0.10 (BSD-2-Clause,
  pinned and stored under `pipeline/monitoring/static/`), polling registry,
  drift, gates, and predictions every 10 seconds.
- Authentication and rate limiting remain out of scope for the local demo
  (documented in the architecture reference); the API binds to localhost by
  default.

## Consequences

- Response versions always match the registry, avoiding file/registry drift.
- Invalid input returns a consistent problem-details envelope (tested).
- The dashboard needs no JavaScript build step and works offline because
  htmx is vendored.
