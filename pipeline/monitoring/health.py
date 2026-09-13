"""In-process serving health metrics (FR-14)."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from threading import Lock

import numpy as np


class HealthTracker:
    """Rolling request counters, latency percentiles, and recent predictions."""

    def __init__(self, max_recent: int = 100) -> None:
        self.started_at = datetime.now(UTC)
        self.total_predictions = 0
        self.errors = 0
        self.latencies_ms: deque[float] = deque(maxlen=500)
        self.recent_predictions: deque[dict[str, object]] = deque(maxlen=max_recent)
        self._lock = Lock()

    def record_prediction(self, prediction: dict[str, object], latency_ms: float) -> None:
        with self._lock:
            self.total_predictions += 1
            self.latencies_ms.append(float(latency_ms))
            self.recent_predictions.appendleft(prediction)

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def summary(self) -> dict[str, object]:
        with self._lock:
            latencies = np.asarray(list(self.latencies_ms), dtype=float)
            uptime_seconds = (datetime.now(UTC) - self.started_at).total_seconds()
            return {
                "uptime_seconds": round(uptime_seconds, 1),
                "total_predictions": self.total_predictions,
                "errors": self.errors,
                "p50_latency_ms": float(np.percentile(latencies, 50)) if len(latencies) else None,
                "p95_latency_ms": float(np.percentile(latencies, 95)) if len(latencies) else None,
                "recent_predictions": list(self.recent_predictions)[:10],
            }
