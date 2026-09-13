"""htmx dashboard over registry, drift, gates, and serving health (FR-14)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from pipeline.lifecycle.events import EventLog
from pipeline.lifecycle.registry import SqliteRegistry
from pipeline.monitoring.health import HealthTracker

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def build_dashboard_router(
    *,
    registry: SqliteRegistry,
    tracker: HealthTracker,
    event_log: EventLog,
) -> APIRouter:
    router = APIRouter(prefix="/dashboard", tags=["dashboard"])

    def context() -> dict[str, object]:
        events = event_log.read()
        return {
            "registry_entries": registry.list(),
            "events": events[-25:],
            "drift_events": [event for event in events if event.event_type == "drift"],
            "health": tracker.summary(),
        }

    @router.get("", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            context(),
        )

    @router.get("/partials/registry", response_class=HTMLResponse)
    def registry_partial(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "partials/registry.html",
            context(),
        )

    @router.get("/partials/drift", response_class=HTMLResponse)
    def drift_partial(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "partials/drift.html",
            context(),
        )

    @router.get("/partials/gates", response_class=HTMLResponse)
    def gates_partial(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "partials/gates.html",
            context(),
        )

    @router.get("/partials/predictions", response_class=HTMLResponse)
    def predictions_partial(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "partials/predictions.html",
            context(),
        )

    return router
