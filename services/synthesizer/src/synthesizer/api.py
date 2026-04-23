"""Synthesizer HTTP routes.

FastAPI router mounted by the gateway at ``/``; paths themselves carry their
``/synthesize/...`` and ``/skills/...`` prefixes. The Tauri app talks to this
on ``127.0.0.1:8765`` — binding to loopback is the gateway's concern, not the
router's (see ``gateway/main.py`` + CLAUDE.md safety invariants).

Two concern areas:

* **Synthesis sessions** — ``POST /synthesize/start`` creates a
  :class:`~synthesizer.session.SynthesisSession` keyed by a new session id,
  kicks off phase-1 draft generation in a background task, and returns the
  id. The UI then subscribes to the SSE stream at
  ``GET /synthesize/{session_id}/stream`` for state-change / draft_ready /
  question_ready / revision_ready / saved / error events. Answers flow through
  ``POST /synthesize/{session_id}/answer`` (revision runs async — the UI
  reconnects to the stream for the result), approval through
  ``POST /synthesize/{session_id}/approve`` (synchronous; returns the written
  skill slug), cancel through ``DELETE /synthesize/{session_id}``.
* **Skill library** — ``GET /skills``, ``GET /skills/{slug}``,
  ``GET /skills/{slug}/preview/{filename}``, ``DELETE /skills/{slug}``. Backed
  by the shared SQLite ``index.db`` (see :mod:`synthesizer.writer`) plus the
  skill directory on disk.

Test hooks: module-level ``_DRAFT_FN`` / ``_REVISE_FN`` / ``_STORE`` /
``_SKILLS_ROOT`` / ``_TRAJECTORIES_ROOT`` are the injection points. Tests
monkeypatch these to swap in deterministic stubs for the LLM-backed draft and
revision callables without touching the real Anthropic SDK.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sqlite3
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from synthesizer.draft import generate_draft
from synthesizer.llm_client import LLMClient, trace_data_dir
from synthesizer.revise import generate_revision
from synthesizer.session import (
    DraftFn,
    IllegalStateTransition,
    ReviseFn,
    SessionState,
    SessionStore,
    SynthesisSession,
)
from synthesizer.slug import SlugError
from synthesizer.trajectory_reader import TrajectoryReader, TrajectoryReadError
from synthesizer.writer import WrittenSkill, index_db_path

LOGGER = logging.getLogger(__name__)

__all__ = [
    "SSE_PING_INTERVAL_SECONDS",
    "SSE_POLL_INTERVAL_SECONDS",
    "router",
]


SSE_POLL_INTERVAL_SECONDS: float = 0.05
"""How often the SSE generator re-checks session snapshot for changes."""

SSE_PING_INTERVAL_SECONDS: int = 15
"""Keepalive comment cadence (sse-starlette's ``ping`` kwarg). 15s matches the
PRD AC and stays under typical proxy idle timeouts."""


# --- Module-level state (overridable for tests) -----------------------------

_STORE: SessionStore = SessionStore()
_DRAFT_FN: DraftFn = generate_draft
_REVISE_FN: ReviseFn = generate_revision
_WRITTEN_RESULTS: dict[str, WrittenSkill] = {}
# Strong refs to background tasks so they aren't GC'd mid-flight. asyncio.create_task
# returns a weak reference inside the loop's task set; see Python 3.11 docs note.
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()


def _schedule_background(coro: Any) -> None:
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


def _skills_root() -> Path:
    return trace_data_dir() / "skills"


def _trajectories_root() -> Path:
    return trace_data_dir() / "trajectories"


def _make_client() -> LLMClient:
    return LLMClient()


# --- Request / response schemas --------------------------------------------


class StartRequest(BaseModel):
    trajectory_id: str = Field(..., min_length=1)


class AnswerRequest(BaseModel):
    question_id: str = Field(..., min_length=1)
    answer: str


class ApproveRequest(BaseModel):
    slug: str | None = None


# --- Router -----------------------------------------------------------------

router = APIRouter()


@router.get("/synthesize/status")
def status() -> dict[str, str]:
    """Liveness probe. Declared before ``/synthesize/{session_id}`` so the
    path-parameter route doesn't swallow the literal ``/status``."""
    return {"module": "synthesizer", "status": "ok"}


@router.post("/synthesize/start")
async def start_synthesis(req: StartRequest) -> dict[str, str]:
    """Create a new synthesis session and kick off phase-1 draft generation."""
    trajectory_dir = _trajectories_root() / req.trajectory_id
    if not trajectory_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"trajectory not found: {req.trajectory_id}",
        )
    try:
        reader = TrajectoryReader(trajectory_dir)
    except (FileNotFoundError, TrajectoryReadError) as err:
        raise HTTPException(status_code=404, detail=str(err)) from err

    client = _make_client()
    session = SynthesisSession(
        trajectory_id=req.trajectory_id,
        reader=reader,
        client=client,
        skills_root=_skills_root(),
        draft_fn=_DRAFT_FN,
        revise_fn=_REVISE_FN,
    )
    _STORE.add(session)
    # Run the blocking draft call on a worker thread so the HTTP response
    # returns immediately; the SSE stream will surface state transitions.
    _schedule_background(_run_in_thread(session.start_draft))
    return {"session_id": session.session_id}


@router.get("/synthesize/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    session = _require_session(session_id)
    return session.snapshot()


@router.get("/synthesize/{session_id}/stream")
async def stream_session(session_id: str, request: Request) -> EventSourceResponse:
    session = _require_session(session_id)
    generator = _sse_event_generator(session, request)
    return EventSourceResponse(generator, ping=SSE_PING_INTERVAL_SECONDS)


@router.post("/synthesize/{session_id}/answer")
async def post_answer(session_id: str, req: AnswerRequest) -> dict[str, Any]:
    session = _require_session(session_id)
    if not req.answer.strip():
        raise HTTPException(status_code=400, detail="answer must be non-empty")
    if session.state != SessionState.AWAITING_ANSWER:
        raise HTTPException(
            status_code=409,
            detail=(
                f"session is in state {session.state.value}; "
                f"answer requires state {SessionState.AWAITING_ANSWER.value}"
            ),
        )
    # Revision runs async on a worker thread. The caller reconnects to /stream
    # to pick up the revision_ready event.
    _schedule_background(
        _run_in_thread(session.answer_question, req.question_id, req.answer)
    )
    return {"accepted": True}


@router.post("/synthesize/{session_id}/approve")
async def post_approve(session_id: str, req: ApproveRequest) -> dict[str, Any]:
    session = _require_session(session_id)
    if session.state != SessionState.AWAITING_APPROVAL:
        raise HTTPException(
            status_code=409,
            detail=(
                f"session is in state {session.state.value}; "
                f"approve requires state {SessionState.AWAITING_APPROVAL.value}"
            ),
        )
    try:
        written = await asyncio.to_thread(session.approve, slug=req.slug)
    except SlugError as err:
        raise HTTPException(status_code=409, detail=str(err)) from err
    except IllegalStateTransition as err:
        raise HTTPException(status_code=409, detail=str(err)) from err

    _WRITTEN_RESULTS[session.session_id] = written
    return {"skill_slug": written.slug, "path": str(written.path)}


@router.delete("/synthesize/{session_id}")
async def cancel_session(session_id: str) -> dict[str, bool]:
    session = _require_session(session_id)
    try:
        session.cancel()
    except IllegalStateTransition as err:
        raise HTTPException(status_code=409, detail=str(err)) from err
    return {"cancelled": True}


@router.get("/skills")
async def list_skills() -> list[dict[str, Any]]:
    db_path = index_db_path()
    if not db_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as conn:
        try:
            cur = conn.execute(
                "SELECT slug, name, trajectory_id, created_at, step_count, "
                "destructive_step_count FROM skills ORDER BY created_at DESC"
            )
        except sqlite3.OperationalError:
            # Table not created yet — no skills written.
            return []
        for slug, name, trajectory_id, created_at, step_count, destructive in cur:
            rows.append(
                {
                    "slug": slug,
                    "name": name,
                    "trajectory_id": trajectory_id,
                    "created_at": created_at,
                    "step_count": step_count,
                    "destructive_step_count": destructive,
                }
            )
    return rows


@router.get("/skills/{slug}")
async def get_skill(slug: str) -> dict[str, Any]:
    skill_dir = _resolve_skill_dir(slug)
    markdown = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    meta = json.loads((skill_dir / "skill.meta.json").read_text(encoding="utf-8"))
    preview_dir = skill_dir / "preview"
    preview_urls: list[str] = []
    if preview_dir.is_dir():
        for png in sorted(preview_dir.glob("*.png")):
            preview_urls.append(f"/skills/{slug}/preview/{png.name}")
    return {"markdown": markdown, "meta": meta, "preview_urls": preview_urls}


@router.get("/skills/{slug}/preview/{filename}")
async def get_skill_preview(slug: str, filename: str) -> FileResponse:
    skill_dir = _resolve_skill_dir(slug)
    # Prevent path traversal: filename must be a plain name, not a path.
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="invalid preview filename")
    target = skill_dir / "preview" / filename
    if not target.is_file():
        raise HTTPException(status_code=404, detail="preview not found")
    return FileResponse(target, media_type="image/png")


@router.delete("/skills/{slug}")
async def delete_skill(slug: str) -> dict[str, bool]:
    skill_dir = _skills_root() / slug
    db_path = index_db_path()
    deleted_row = False
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            try:
                cur = conn.execute("DELETE FROM skills WHERE slug = ?", (slug,))
            except sqlite3.OperationalError:
                cur = None
            if cur is not None:
                deleted_row = cur.rowcount > 0
                conn.commit()
    dir_existed = skill_dir.is_dir()
    if dir_existed:
        shutil.rmtree(skill_dir, ignore_errors=True)
    if not deleted_row and not dir_existed:
        raise HTTPException(status_code=404, detail=f"skill not found: {slug}")
    return {"deleted": True}


# --- Helpers ---------------------------------------------------------------


def _require_session(session_id: str) -> SynthesisSession:
    session = _STORE.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"session not found: {session_id}"
        )
    return session


def _resolve_skill_dir(slug: str) -> Path:
    if "/" in slug or "\\" in slug or slug.startswith("."):
        raise HTTPException(status_code=400, detail="invalid slug")
    skill_dir = _skills_root() / slug
    if not skill_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"skill not found: {slug}")
    return skill_dir


async def _run_in_thread(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Run ``fn(*args, **kwargs)`` in a worker thread. Exceptions are swallowed
    after logging — the session state machine has already transitioned to
    ERRORED by the time any exception propagates out of start_draft /
    answer_question, so the caller observes the error via the snapshot."""
    try:
        await asyncio.to_thread(fn, *args, **kwargs)
    except Exception:  # already recorded on the session; don't crash the loop
        LOGGER.exception("background synthesis task failed")


async def _sse_event_generator(
    session: SynthesisSession, request: Request
) -> AsyncIterator[dict[str, Any]]:
    """Yield SSE events tracking a single synthesis session to completion."""
    last_state: str | None = None
    last_markdown: str | None = None
    emitted_draft_ready = False

    while True:
        if await request.is_disconnected():
            return

        snap = session.snapshot()
        state = snap["state"]
        draft = snap["draft"]
        draft_markdown = draft["markdown"] if isinstance(draft, dict) else None

        if state != last_state:
            yield {"event": "state_change", "data": json.dumps({"state": state})}
            last_state = state

        if draft is not None and not emitted_draft_ready:
            yield {"event": "draft_ready", "data": json.dumps({"draft": draft})}
            emitted_draft_ready = True
            last_markdown = draft_markdown
            if state == SessionState.AWAITING_ANSWER.value and snap["next_question"]:
                yield {
                    "event": "question_ready",
                    "data": json.dumps({"question": snap["next_question"]}),
                }
        elif (
            draft is not None
            and draft_markdown is not None
            and draft_markdown != last_markdown
        ):
            yield {"event": "revision_ready", "data": json.dumps({"draft": draft})}
            last_markdown = draft_markdown
            if state == SessionState.AWAITING_ANSWER.value and snap["next_question"]:
                yield {
                    "event": "question_ready",
                    "data": json.dumps({"question": snap["next_question"]}),
                }

        if state == SessionState.COMPLETED.value:
            written = _WRITTEN_RESULTS.get(session.session_id)
            payload: dict[str, Any] = {}
            if written is not None:
                payload = {"slug": written.slug, "path": str(written.path)}
            yield {"event": "saved", "data": json.dumps(payload)}
            return
        if state == SessionState.ERRORED.value:
            yield {
                "event": "error",
                "data": json.dumps({"error": snap["error"]}),
            }
            return
        if state == SessionState.CANCELLED.value:
            return

        await asyncio.sleep(SSE_POLL_INTERVAL_SECONDS)


# --- Test / lifecycle helpers ----------------------------------------------


def reset_for_tests() -> None:
    """Clear in-memory session + written-result registries.

    Useful when tests monkeypatch ``TRACE_DATA_DIR`` and want a clean
    :class:`~synthesizer.session.SessionStore`. Not part of the public API —
    importing this in production code is a bug.
    """
    global _STORE
    _STORE = SessionStore()
    _WRITTEN_RESULTS.clear()


def _use_for_tests(
    *,
    draft_fn: DraftFn | None = None,
    revise_fn: ReviseFn | None = None,
) -> None:
    """Override default draft/revise callables. Test-only escape hatch."""
    global _DRAFT_FN, _REVISE_FN
    if draft_fn is not None:
        _DRAFT_FN = draft_fn
    if revise_fn is not None:
        _REVISE_FN = revise_fn


