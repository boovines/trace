"""Runner HTTP routes: /run/* and /runs/* (X-019).

Mounted by the gateway. Endpoints:

* ``POST /run/start`` — enqueue a new run; returns ``{run_id}``.
* ``GET /run/{run_id}`` — current ``run_metadata.json`` contents.
* ``WS /run/{run_id}/stream`` — status/event/confirmation stream.
* ``POST /run/{run_id}/confirm`` — deliver a UI confirmation decision.
* ``POST /run/{run_id}/abort`` — trigger the kill switch.
* ``GET /runs`` — list of run summaries (newest first).
* ``GET /run/{run_id}/events`` — parsed ``events.jsonl`` array.
* ``GET /run/{run_id}/screenshots/{filename}`` — serve a screenshot PNG.

The router reads a :class:`RunManager` from the FastAPI app state; the
gateway's ``lifespan`` is expected to populate it. The ``_run_manager_for``
dependency falls back to a lazily-initialized default so the router is also
mountable standalone for tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from runner.confirmation import ConfirmationDecision
from runner.kill_switch import KillSwitch, get_global_kill_switch
from runner.run_manager import (
    DailyCapExceeded,
    InvalidRunState,
    LiveModeNotEnabled,
    RunManager,
    RunNotFound,
)

logger = logging.getLogger(__name__)

WEBSOCKET_KEEPALIVE_SECONDS: float = 15.0

router = APIRouter()


class StartRunRequest(BaseModel):
    """Body of ``POST /run/start``."""

    model_config = ConfigDict(extra="forbid")

    skill_slug: str = Field(min_length=1)
    parameters: dict[str, str] = Field(default_factory=dict)
    mode: Literal["execute", "dry_run"]


class StartRunResponse(BaseModel):
    run_id: str


class ConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["confirm", "abort"]
    reason: str | None = None


def _run_manager(request: Request) -> RunManager:
    manager = getattr(request.app.state, "run_manager", None)
    if manager is None:
        manager = RunManager()
        request.app.state.run_manager = manager
    assert isinstance(manager, RunManager)
    return manager


def _run_manager_ws(websocket: WebSocket) -> RunManager:
    manager = getattr(websocket.app.state, "run_manager", None)
    if manager is None:
        manager = RunManager()
        websocket.app.state.run_manager = manager
    assert isinstance(manager, RunManager)
    return manager


@router.get("/run/status")
def status() -> dict[str, str]:
    """Liveness probe used by the gateway's scaffold."""
    return {"module": "runner", "status": "ok"}


@router.post("/run/start", response_model=StartRunResponse)
async def start_run(
    body: StartRunRequest,
    request: Request,
) -> StartRunResponse:
    manager = _run_manager(request)
    try:
        run_id = await manager.start_run(
            skill_slug=body.skill_slug,
            parameters=body.parameters,
            mode=body.mode,
        )
    except LiveModeNotEnabled as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DailyCapExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except RunNotFound as exc:
        raise HTTPException(
            status_code=404, detail=f"skill not found: {body.skill_slug}"
        ) from exc
    return StartRunResponse(run_id=run_id)


@router.get("/run/{run_id}")
def get_run(run_id: str, request: Request) -> dict[str, Any]:
    manager = _run_manager(request)
    try:
        return manager.get_metadata(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}") from exc


@router.post("/run/{run_id}/confirm")
def confirm_run(
    run_id: str,
    body: ConfirmRequest,
    request: Request,
) -> dict[str, bool]:
    manager = _run_manager(request)
    decision = ConfirmationDecision(action=body.decision, reason=body.reason)
    try:
        manager.submit_decision(run_id, decision)
    except RunNotFound as exc:
        raise HTTPException(
            status_code=404, detail=f"run not found: {run_id}"
        ) from exc
    except InvalidRunState as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"accepted": True}


@router.post("/run/{run_id}/abort")
def abort_run(run_id: str, request: Request = None) -> dict[str, object]:  # type: ignore[assignment]
    """Trigger the kill switch for ``run_id``. Idempotent.

    Always 200s: the caller (Tauri hotkey handler) has no way to distinguish
    "already finished" from "never started" and either way there's nothing
    further to do.
    """
    # ``request`` is optional so callers that only have the global kill switch
    # (e.g. a synchronous hotkey handler shimmed through HTTP) still succeed.
    manager: RunManager | None = None
    if request is not None:
        manager = getattr(request.app.state, "run_manager", None)
    if manager is not None:
        killed = manager.abort(run_id)
    else:
        switch: KillSwitch = get_global_kill_switch()
        killed = switch.kill(run_id, reason="user_abort")
    return {"run_id": run_id, "killed": killed, "aborted": True}


@router.get("/runs")
def list_runs(
    request: Request,
    skill_slug: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    manager = _run_manager(request)
    return manager.list_runs(
        skill_slug=skill_slug, limit=limit, offset=offset
    )


@router.get("/run/{run_id}/events")
def get_events(run_id: str, request: Request) -> list[dict[str, Any]]:
    manager = _run_manager(request)
    try:
        return manager.get_events(run_id)
    except RunNotFound as exc:
        raise HTTPException(
            status_code=404, detail=f"run not found: {run_id}"
        ) from exc


@router.get("/run/{run_id}/screenshots/{filename}")
def get_screenshot(
    run_id: str, filename: str, request: Request
) -> FileResponse:
    manager = _run_manager(request)
    try:
        path: Path = manager.screenshot_path(run_id, filename)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="image/png")


@router.websocket("/run/{run_id}/stream")
async def stream_run(websocket: WebSocket, run_id: str) -> None:
    """WebSocket feed of a run's status changes, events, and confirmations.

    Accepts unconditionally (subscribing before the run exists is legal — the
    UI may open the socket and then POST ``/run/start``). Sends a keepalive
    ping every :data:`WEBSOCKET_KEEPALIVE_SECONDS` when no event arrives.
    Closes cleanly after the broadcaster's ``done`` sentinel.
    """
    await websocket.accept()
    manager = _run_manager_ws(websocket)
    subscription = manager.broadcaster.subscribe(run_id)
    try:
        while True:
            try:
                event = await subscription.get(
                    timeout=WEBSOCKET_KEEPALIVE_SECONDS
                )
            except TimeoutError:
                await websocket.send_json({"type": "keepalive", "run_id": run_id})
                continue
            if event is None:
                break
            await websocket.send_json(event)
    except WebSocketDisconnect:
        logger.debug("ws disconnected for run %s", run_id)
    finally:
        subscription.close()
        with contextlib.suppress(RuntimeError):
            await websocket.close()


def build_app() -> FastAPI:
    """Return a standalone FastAPI app serving just the runner routes.

    Used by tests and by ``python -m runner`` dev-mode launches. The gateway
    uses the :data:`router` directly rather than this app.
    """
    app = FastAPI(title="Trace runner", version="0.1.0")
    manager = RunManager()
    try:
        manager.reconcile_index()
    except Exception:  # pragma: no cover - startup resilience
        logger.exception("reconcile_index at startup failed")
    app.state.run_manager = manager
    app.include_router(router)
    return app


app = build_app()


async def _aclose_tasks(manager: RunManager) -> None:
    """Best-effort cancel all outstanding run tasks on app shutdown."""
    for handle in list(manager._runs.values()):
        if not handle.task.done():
            handle.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await handle.task
    manager.shutdown()


__all__ = [
    "WEBSOCKET_KEEPALIVE_SECONDS",
    "StartRunRequest",
    "StartRunResponse",
    "app",
    "build_app",
    "router",
]
