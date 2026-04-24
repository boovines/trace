"""HTTP endpoints for the Recorder service.

Two routers are exported:

* :data:`router` — mounted by the gateway under the ``/recorder`` prefix.
  Owns ``POST /recorder/start``, ``POST /recorder/stop`` and
  ``GET /recorder/status``.
* :data:`trajectories_router` — mounted by the gateway at the root.  Owns
  ``GET /trajectories``, ``GET /trajectories/{id}``,
  ``DELETE /trajectories/{id}`` and the on-demand
  ``GET /trajectories/{id}/screenshots/{filename}`` static-style asset
  endpoint.

The per-process recording state and the on-disk root directory live on a
:class:`RecorderState` singleton resolved via the
:func:`get_recorder_state` FastAPI dependency.  Tests override the
dependency to inject a state with fakes for ``RecordingSession`` so the API
surface can be exercised without macOS permissions or real PyObjC.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from recorder.index_db import IndexDB
from recorder.session import (
    PermissionsMissingError,
    RecordingSession,
    SessionAlreadyActiveError,
    SessionNotActiveError,
    SessionSummary,
)
from recorder.storage import (
    default_index_db_path,
    default_trajectories_root,
    ensure_trajectories_root,
    remove_trajectory,
    trajectory_dir,
)

__all__ = [
    "RecorderState",
    "configure_state",
    "get_recorder_state",
    "router",
    "trajectories_router",
]

logger = logging.getLogger(__name__)

SessionFactory = Callable[[Path, "IndexDB | None"], RecordingSession]


class StartRequest(BaseModel):
    """Body for ``POST /recorder/start``.  ``label`` is optional."""

    label: str | None = Field(default=None, max_length=200)


class StartResponse(BaseModel):
    trajectory_id: str


class StopResponse(BaseModel):
    trajectory_id: str
    event_count: int
    duration_ms: int


class StatusResponse(BaseModel):
    recording: bool
    current_trajectory_id: str | None
    started_at: str | None


class TrajectorySummary(BaseModel):
    id: str
    label: str | None
    started_at: str | None
    stopped_at: str | None
    duration_ms: int | None
    event_count: int


class TrajectoryDetail(BaseModel):
    id: str
    metadata: dict[str, Any]
    events: list[dict[str, Any]]
    screenshots: list[str]


class _DeleteResponse(BaseModel):
    deleted: str


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _default_session_factory(
    root: Path, index_db: IndexDB | None
) -> RecordingSession:
    return RecordingSession(root, index_db=index_db)


class RecorderState:
    """Process-wide state for the Recorder HTTP layer.

    Owns the root directory, the at-most-one in-flight
    :class:`RecordingSession`, and a lock that serialises lifecycle changes
    so two simultaneous ``POST /recorder/start`` calls cannot create two
    sessions.
    """

    def __init__(
        self,
        root: Path,
        *,
        session_factory: SessionFactory | None = None,
        index_db: IndexDB | None = None,
        index_db_path: Path | None = None,
        reconcile_on_init: bool = True,
    ) -> None:
        self.root: Path = Path(root)
        self._session_factory: SessionFactory = (
            session_factory if session_factory is not None else _default_session_factory
        )
        self._lock = threading.Lock()
        self._session: RecordingSession | None = None
        self._started_at: str | None = None
        ensure_trajectories_root(self.root)

        if index_db is not None:
            self.index_db: IndexDB = index_db
        else:
            db_path = index_db_path if index_db_path is not None else self.root.parent / "index.db"
            self.index_db = IndexDB(db_path)

        if reconcile_on_init:
            try:
                self.index_db.reconcile(self.root)
            except Exception:
                logger.exception("IndexDB.reconcile on startup failed")

    # ------------------------------------------------------------- lifecycle

    def start(self, label: str | None) -> str:
        """Start a recording.  Returns the new trajectory id."""
        with self._lock:
            if self._session is not None and self._session.is_active():
                raise SessionAlreadyActiveError(
                    "A recording session is already in progress."
                )
            session = self._session_factory(self.root, self.index_db)
            tid = session.start(label or "")
            self._session = session
            self._started_at = _utc_now_iso()
            return tid

    def stop(self) -> SessionSummary:
        with self._lock:
            session = self._session
            if session is None or not session.is_active():
                raise SessionNotActiveError("No recording session is in progress.")
            summary = session.stop()
            self._session = None
            self._started_at = None
            return summary

    def status(self) -> StatusResponse:
        with self._lock:
            session = self._session
            if session is None or not session.is_active():
                return StatusResponse(
                    recording=False,
                    current_trajectory_id=None,
                    started_at=None,
                )
            return StatusResponse(
                recording=True,
                current_trajectory_id=session.trajectory_id,
                started_at=self._started_at,
            )

    def active_trajectory_id(self) -> str | None:
        """Trajectory id currently being recorded, or ``None``."""
        with self._lock:
            session = self._session
            if session is None or not session.is_active():
                return None
            return session.trajectory_id


# --------------------------------------------------------- state plumbing

# Module-level holder for the configured state.  In production the gateway
# (or `recorder.__main__`) calls `configure_state(...)` once at startup;
# tests build their own state and inject it via FastAPI dependency overrides.
_STATE: RecorderState | None = None


def configure_state(state: RecorderState) -> None:
    """Install the process-wide :class:`RecorderState` singleton."""
    global _STATE
    _STATE = state


def get_recorder_state() -> RecorderState:
    """FastAPI dependency that returns the configured :class:`RecorderState`.

    Initialises a default state on first call if none was configured — this
    keeps ``uv run uvicorn gateway.main:app`` working out-of-the-box.
    """
    global _STATE
    if _STATE is None:
        _STATE = RecorderState(
            default_trajectories_root(),
            index_db_path=default_index_db_path(),
        )
    return _STATE


# `Depends(...)` evaluated once at import time and reused as the default
# argument value for every endpoint — keeps `B008` happy without losing
# FastAPI's dependency-injection semantics (FastAPI inspects the default for
# the `Depends` instance, not its return value).
StateDep = Depends(get_recorder_state)


# ----------------------------------------------------------------- helpers


def _read_metadata(traj_dir: Path) -> dict[str, Any] | None:
    meta_path = traj_dir / "metadata.json"
    if not meta_path.is_file():
        return None
    try:
        with meta_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read metadata for %s", traj_dir.name, exc_info=True)
        return None
    if not isinstance(data, dict):
        return None
    return data


def _read_events(traj_dir: Path) -> list[dict[str, Any]]:
    events_path = traj_dir / "events.jsonl"
    if not events_path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with events_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed event line in %s", traj_dir.name)
                continue
            if isinstance(event, dict):
                out.append(event)
    return out


def _list_screenshots(traj_dir: Path) -> list[str]:
    screenshots_dir = traj_dir / "screenshots"
    if not screenshots_dir.is_dir():
        return []
    names = sorted(p.name for p in screenshots_dir.iterdir() if p.is_file())
    return [f"screenshots/{name}" for name in names]


def _safe_screenshot_path(root: Path, trajectory_id: str, filename: str) -> Path | None:
    """Return the resolved screenshot path iff it lives inside the trajectory dir.

    Guards against ``..`` traversal and absolute paths in ``filename``.
    """
    if "/" in filename or "\\" in filename or filename in ("", ".", ".."):
        return None
    if not filename.endswith(".png"):
        return None
    candidate = (trajectory_dir(root, trajectory_id) / "screenshots" / filename).resolve()
    expected_parent = (trajectory_dir(root, trajectory_id) / "screenshots").resolve()
    try:
        candidate.relative_to(expected_parent)
    except ValueError:
        return None
    return candidate


# ------------------------------------------------------------------- routers


router = APIRouter()
trajectories_router = APIRouter()


@router.post(
    "/start",
    response_model=StartResponse,
    responses={
        403: {"description": "missing macOS permissions"},
        409: {"description": "a recording session is already in progress"},
    },
)
def post_start(
    body: StartRequest,
    state: RecorderState = StateDep,
) -> StartResponse:
    try:
        tid = state.start(body.label)
    except PermissionsMissingError as exc:
        # exc.error is the structured PermissionsError dict the UI expects.
        return JSONResponse(  # type: ignore[return-value]
            status_code=403, content=dict(exc.error)
        )
    except SessionAlreadyActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return StartResponse(trajectory_id=tid)


@router.post(
    "/stop",
    response_model=StopResponse,
    responses={409: {"description": "no recording session is in progress"}},
)
def post_stop(
    state: RecorderState = StateDep,
) -> StopResponse:
    try:
        summary = state.stop()
    except SessionNotActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return StopResponse(**summary)


@router.get("/status", response_model=StatusResponse)
def get_status(
    state: RecorderState = StateDep,
) -> StatusResponse:
    return state.status()


@trajectories_router.get(
    "/trajectories",
    response_model=list[TrajectorySummary],
)
def list_trajectories(
    state: RecorderState = StateDep,
) -> list[TrajectorySummary]:
    rows = state.index_db.list_all()
    return [
        TrajectorySummary(
            id=row["id"],
            label=row["label"],
            started_at=row["started_at"],
            stopped_at=row["stopped_at"],
            duration_ms=row["duration_ms"],
            event_count=row["event_count"],
        )
        for row in rows
    ]


@trajectories_router.get(
    "/trajectories/{trajectory_id}",
    response_model=TrajectoryDetail,
    responses={404: {"description": "trajectory not found"}},
)
def get_trajectory(
    trajectory_id: str,
    request: Request,
    state: RecorderState = StateDep,
) -> TrajectoryDetail:
    traj_dir = trajectory_dir(state.root, trajectory_id)
    if not traj_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"trajectory {trajectory_id} not found")
    metadata = _read_metadata(traj_dir)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"trajectory {trajectory_id} not found")
    events = _read_events(traj_dir)
    screenshot_paths = _list_screenshots(traj_dir)
    base = str(request.base_url).rstrip("/")
    screenshots = [f"{base}/trajectories/{trajectory_id}/{path}" for path in screenshot_paths]
    return TrajectoryDetail(
        id=trajectory_id,
        metadata=metadata,
        events=events,
        screenshots=screenshots,
    )


@trajectories_router.delete(
    "/trajectories/{trajectory_id}",
    response_model=_DeleteResponse,
    responses={
        404: {"description": "trajectory not found"},
        409: {"description": "trajectory is the active recording"},
    },
)
def delete_trajectory(
    trajectory_id: str,
    state: RecorderState = StateDep,
) -> _DeleteResponse:
    if state.active_trajectory_id() == trajectory_id:
        raise HTTPException(
            status_code=409,
            detail="trajectory is the active recording; stop it first",
        )
    removed_dir = remove_trajectory(state.root, trajectory_id)
    removed_row = state.index_db.delete(trajectory_id)
    if not removed_dir and not removed_row:
        raise HTTPException(status_code=404, detail=f"trajectory {trajectory_id} not found")
    return _DeleteResponse(deleted=trajectory_id)


@trajectories_router.get(
    "/trajectories/{trajectory_id}/screenshots/{filename}",
    responses={404: {"description": "screenshot not found"}},
)
def get_screenshot(
    trajectory_id: str,
    filename: str,
    state: RecorderState = StateDep,
) -> FileResponse:
    path = _safe_screenshot_path(state.root, trajectory_id, filename)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="screenshot not found")
    return FileResponse(path, media_type="image/png")
