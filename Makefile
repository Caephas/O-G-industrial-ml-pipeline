PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help setup install test lint check fetch baselines train demo serve clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

setup: ## Create the virtualenv and install package + dev dependencies
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

install: ## Install package + dev dependencies into the existing virtualenv
	$(PIP) install -e ".[dev]"

test: ## Run the test suite
	$(PY) -m pytest

lint: ## Lint the codebase
	$(PY) -m ruff check .

check: lint test ## Lint + test

fetch: ## Download and verify the NASA C-MAPSS dataset
	$(PY) scripts/fetch_cmapss.py

baselines: ## Build features and compute FD002 baselines
	$(PY) scripts/run_baselines.py

train: ## Train and evaluate the 4-model DAG
	$(PY) -m pipeline.cli train

demo: ## Run the drift -> retrain -> shadow -> promote lifecycle simulation
	$(PY) scripts/lifecycle_sim.py

serve: ## Serve the REST API and htmx dashboard (uvicorn factory)
	$(VENV)/bin/uvicorn pipeline.api.server:create_app --factory --host 127.0.0.1 --port 8000

clean: ## Remove local caches and build outputs
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
