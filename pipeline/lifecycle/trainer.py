"""Isolated retraining worker (FR-12)."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class RetrainWorker:
    """Runs training in a separate process so inference is never starved."""

    process: subprocess.Popen[str] | None = None

    def start(self, *, subset: str = "FD002", seed: int) -> RetrainWorker:
        command = [
            sys.executable,
            "-m",
            "pipeline.cli",
            "train",
            "--subset",
            subset,
            "--seed",
            str(seed),
        ]
        return self.start_command(command)

    def start_command(self, command: list[str]) -> RetrainWorker:
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return self

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def wait(self, timeout: float | None = None) -> str:
        """Wait for completion and return the produced run id."""
        if self.process is None:
            raise RuntimeError("no worker started")
        output, _ = self.process.communicate(timeout=timeout)
        if self.process.returncode != 0:
            raise RuntimeError(f"retrain worker failed:\n{output[-2000:]}")
        match = re.search(r"run: (run-[A-Za-z0-9-]+)", output)
        if match is None:
            raise RuntimeError(f"could not parse run id from worker output:\n{output[-2000:]}")
        return match.group(1)
