# Runbook: REST API and Dashboard (`make serve`)

## Prerequisites

- `make setup`, `make fetch`, `make baselines`, `make train`
- Production artifacts in the registry: `make demo` (promotes the retrained
  candidate to production)

## Start

```bash
make serve
```

The service binds to `127.0.0.1:8000` via the uvicorn factory:
`uvicorn pipeline.api.server:create_app --factory`.

## Smoke test

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"subset":"FD002","engine_id":1,"readings":[...]}'
```

`/v1/predict` accepts a chronological window of raw C-MAPSS cycles
(3 operating settings, 21 sensors). See `/docs` for the interactive schema.

## Dashboard

Open `http://127.0.0.1:8000/dashboard`. Panels refresh every 10 seconds and
show the registry (family/version/stage/metrics), drift events, lifecycle
gates, and recent predictions.

## Manual intervention

- If startup fails with "no production ... artifact in the registry", the
  registry is empty — run `make demo` (or `make train` first).
- If served versions look wrong, the registry entry is authoritative; verify
  with `artifacts/registry/registry.db` (see ADR 0011).
- Bind is intentionally localhost; change `--host` only for a controlled
  network deployment (auth is out of scope for the demo).
