"""Append-only lifecycle event log (frozen event contract)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from pipeline.schemas.events import LifecycleEvent

_adapter = TypeAdapter(LifecycleEvent)


class EventLog:
    """Writes lifecycle events as JSONL for replay and dashboards."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: LifecycleEvent) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.model_dump(mode="json"), separators=(",", ":")) + "\n")

    def read(self) -> list[LifecycleEvent]:
        if not self.path.exists():
            return []
        events: list[LifecycleEvent] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    events.append(_adapter.validate_python(json.loads(line)))
        return events
