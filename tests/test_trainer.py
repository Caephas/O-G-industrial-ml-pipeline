"""Retrain worker isolation tests (FR-12)."""

from __future__ import annotations

import sys
import time

from pipeline.lifecycle.trainer import RetrainWorker


def test_worker_runs_in_separate_process_and_reports_run_id() -> None:
    code = (
        "import time; time.sleep(1.2); "
        "print('run: run-abc123-42'); print('artifacts: /tmp/x')"
    )
    worker = RetrainWorker().start_command([sys.executable, "-c", code])
    assert worker.is_running() is True
    started = time.monotonic()
    run_id = worker.wait(timeout=30)
    assert run_id == "run-abc123-42"
    assert time.monotonic() - started >= 1.0
    assert worker.is_running() is False


def test_worker_failure_raises() -> None:
    worker = RetrainWorker().start_command([sys.executable, "-c", "raise SystemExit(1)"])
    try:
        worker.wait(timeout=30)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError for failing worker")
