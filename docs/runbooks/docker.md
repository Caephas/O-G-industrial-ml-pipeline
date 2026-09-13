# Runbook: Docker Compose (`docker compose up`)

## First run

```bash
docker compose up --build
```

The image builds (pinned Python dependencies), then `scripts/bootstrap.py`
executes any missing stages into the named volumes:

1. dataset download + checksum verification,
2. regime features + baselines,
3. 4-model training,
4. lifecycle simulation (drift → retrain → shadow → promote).

The first run is CPU-bound and can take 10–20 minutes; the API healthcheck has
a long start period for this reason. The service is then available at
`http://localhost:8000` (`/docs`, `/dashboard`, `/v1/predict`).

## Later starts

Completed stages are skipped via durable markers, so the API starts in
seconds. The lifecycle simulation can be re-run on demand:

```bash
docker compose run --rm worker
```

## Reset

```bash
docker compose down
docker volume rm industrial-ml-pipeline_data industrial-ml-pipeline_artifacts
```

Then start again for a fully fresh pipeline.

## Troubleshooting

- Port 8000 in use: change the published port (`8001:8000`).
- Code changed but image stale: rebuild with `docker compose up --build`.
- Healthcheck keeps failing after first boot: inspect logs with
  `docker compose logs api`; the bootstrap prints progress and finally
  "pipeline ready; starting API".
