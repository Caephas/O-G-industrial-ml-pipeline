# ADR 0012: Containerization and Idempotent Bootstrap

- Status: Accepted
- Date: 2026-09-03

## Context

The portfolio must run with one command (FR-15). The full pipeline (download,
features, training, lifecycle simulation, serving) is CPU-heavy on first run,
and data/artifacts must persist between starts without committing generated
files to the repository.

## Decision

- One Python 3.11-slim image serves every stage. A non-root user owns the
  application tree.
- `scripts/bootstrap.py` is the container entrypoint: it checks durable
  markers (raw archive, feature manifest, passing training run, four
  production registry families) and runs only the stages that are missing,
  then serves the API via uvicorn.
- Named volumes (`data`, `artifacts`) persist across restarts and initialize
  with image-directory ownership so the non-root user can write.
- The `worker` service runs the lifecycle simulation on demand through a
  compose profile (`docker compose run --rm worker`) rather than as a second
  always-on daemon; the API container itself performs the one-time bootstrap.
- Compose healthcheck polls `/health` with a long start period to cover the
  first-run pipeline.

## Consequences

- `docker compose up --build` works from a clean clone; subsequent starts are
  fast because completed stages are skipped.
- No generated data or model weights are committed; `docker compose down -v`
  resets the demo.
- No secrets or host-specific paths are required.
