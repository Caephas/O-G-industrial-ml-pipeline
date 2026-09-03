PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# fetch/train/serve/demo targets are added by their feature milestones.
.PHONY: help setup install test lint check clean

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

clean: ## Remove local caches and build outputs
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
