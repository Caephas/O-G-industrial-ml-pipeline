FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY pyproject.toml README.md ./
COPY pipeline ./pipeline
COPY scripts ./scripts
COPY metrics ./metrics
COPY tests ./tests

RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/raw /app/data/processed /app/data/quality \
    /app/artifacts/runs /app/artifacts/registry /app/artifacts/lifecycle \
    /app/artifacts/baselines \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["python", "-m", "scripts.bootstrap"]
