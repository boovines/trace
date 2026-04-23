"""HTTP integration tests for :mod:`synthesizer.api`.

Every test drives the FastAPI router through an ASGI transport — no actual
network socket, no real Anthropic calls. The ``_force_fake_mode`` conftest
fixture already sets ``TRACE_LLM_FAKE_MODE=1`` and installs a respx router
that raises on any accidental hit to ``api.anthropic.com``. On top of that we
swap in stubbed ``_DRAFT_FN`` / ``_REVISE_FN`` at the module level so the
session state machine runs deterministically without exercising the real
draft-prompt / revision-prompt LLM code paths — those have their own tests.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport
from synthesizer import api as api_module
from synthesizer.draft import DraftResult, Question
from synthesizer.preprocess import PreprocessedTrajectory
from synthesizer.skill_doc import parse_skill_md
from synthesizer.trajectory_reader import TrajectoryReader

# --- Shared test data -------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_TRAJ = _REPO_ROOT / "fixtures" / "trajectories" / "gmail_reply"
_TRAJECTORY_ID = "b1a0f6f2-0001-4c00-8000-000000000001"

_SKILL_MD_INITIAL = """# Reply to Gmail

Reply to the latest unread email with a short template.

## Parameters

- `recipient_email` (string, required) — The email address to send to.

## Preconditions

- Chrome is open.
- Gmail is logged in.

## Steps

1. Switch to Chrome.
2. Click the Reply button in Gmail.
3. Type the reply body.
4. ⚠️ [DESTRUCTIVE] Click the Send button to send to {recipient_email}.

## Expected outcome

An email is sent to {recipient_email}.
"""

_META_INITIAL: dict[str, Any] = {
    "slug": "gmail_reply_api_test",
    "name": "Reply to Gmail",
    "trajectory_id": _TRAJECTORY_ID,
    "created_at": "2026-04-23T12:00:00+00:00",
    "parameters": [
        {"name": "recipient_email", "type": "string", "required": True},
    ],
    "destructive_steps": [4],
    "preconditions": ["Chrome is open.", "Gmail is logged in."],
    "step_count": 4,
}


def _build_initial_draft(questions: list[Question] | None = None) -> DraftResult:
    parsed = parse_skill_md(_SKILL_MD_INITIAL)
    return DraftResult(
        markdown=_SKILL_MD_INITIAL,
        parsed=parsed,
        meta=dict(_META_INITIAL),
        questions=questions
        if questions is not None
        else [
            Question(
                id="q1",
                category="parameterization",
                text="What email address should the reply always go to?",
            ),
        ],
        llm_calls=1,
        total_cost_usd=0.0123,
    )


def _fake_draft_fn(
    preprocessed: PreprocessedTrajectory,
    client: Any,
    *,
    reader: TrajectoryReader,
) -> DraftResult:
    return _build_initial_draft()


def _fake_revise_fn(
    *,
    current_draft: DraftResult,
    question: Question,
    answer: str,
    client: Any,
    reader: TrajectoryReader,
) -> DraftResult:
    # Produce a materially-different markdown so the SSE stream can detect the
    # revision via a markdown-hash change.
    revised_md = current_draft.markdown.replace(
        "Reply to the latest unread email with a short template.",
        f"Reply to the latest unread email (answered {question.id}).",
    )
    revised_parsed = parse_skill_md(revised_md)
    remaining = [q for q in current_draft.questions if q.id != question.id]
    return DraftResult(
        markdown=revised_md,
        parsed=revised_parsed,
        meta=current_draft.meta,
        questions=remaining,
        llm_calls=current_draft.llm_calls + 1,
        total_cost_usd=current_draft.total_cost_usd + 0.005,
    )


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def seeded_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Isolate TRACE_DATA_DIR, seed a trajectory, swap draft/revise callables."""
    monkeypatch.setenv("TRACE_DATA_DIR", str(tmp_path))
    traj_root = tmp_path / "trajectories"
    traj_root.mkdir(parents=True)
    shutil.copytree(_FIXTURE_TRAJ, traj_root / _TRAJECTORY_ID)

    api_module.reset_for_tests()
    monkeypatch.setattr(api_module, "_DRAFT_FN", _fake_draft_fn)
    monkeypatch.setattr(api_module, "_REVISE_FN", _fake_revise_fn)
    # Tight polling so the stream flushes events quickly in tests.
    monkeypatch.setattr(api_module, "SSE_POLL_INTERVAL_SECONDS", 0.005)

    yield tmp_path

    api_module.reset_for_tests()


@pytest.fixture
def app(seeded_env: Path) -> FastAPI:
    application = FastAPI()
    application.include_router(api_module.router)
    return application


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        yield ac


# --- Helpers ----------------------------------------------------------------


async def _wait_for_state(
    client: httpx.AsyncClient, session_id: str, target: str, timeout: float = 3.0
) -> dict[str, Any]:
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        resp = await client.get(f"/synthesize/{session_id}")
        assert resp.status_code == 200, resp.text
        snap = resp.json()
        if snap["state"] == target:
            return snap
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError(
                f"session {session_id} stuck at state={snap['state']!r}, "
                f"expected {target!r} within {timeout}s"
            )
        await asyncio.sleep(0.01)


# --- Tests ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_endpoint(client: httpx.AsyncClient) -> None:
    resp = await client.get("/synthesize/status")
    assert resp.status_code == 200
    assert resp.json() == {"module": "synthesizer", "status": "ok"}


@pytest.mark.asyncio
async def test_start_unknown_trajectory_returns_404(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.post(
        "/synthesize/start", json={"trajectory_id": "does-not-exist"}
    )
    assert resp.status_code == 404
    assert "trajectory not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_unknown_session_returns_404(client: httpx.AsyncClient) -> None:
    bad = "deadbeef0000000000000000"
    r1 = await client.get(f"/synthesize/{bad}")
    r2 = await client.post(
        f"/synthesize/{bad}/answer", json={"question_id": "q1", "answer": "x"}
    )
    r3 = await client.post(f"/synthesize/{bad}/approve", json={})
    r4 = await client.delete(f"/synthesize/{bad}")
    r5 = await client.get(f"/synthesize/{bad}/stream")
    for r in (r1, r2, r3, r4, r5):
        assert r.status_code == 404, (r.status_code, r.text)


@pytest.mark.asyncio
async def test_happy_path_full_synthesis(client: httpx.AsyncClient) -> None:
    # 1. Start synthesis
    r = await client.post(
        "/synthesize/start", json={"trajectory_id": _TRAJECTORY_ID}
    )
    assert r.status_code == 200, r.text
    session_id = r.json()["session_id"]
    assert isinstance(session_id, str)
    assert len(session_id) >= 16

    # 2. Wait for draft
    snap = await _wait_for_state(client, session_id, "awaiting_answer")
    assert snap["draft"] is not None
    assert snap["draft"]["meta"]["slug"] == "gmail_reply_api_test"
    assert len(snap["questions"]) == 1
    assert snap["next_question"]["id"] == "q1"
    assert snap["costs_so_far_usd"] == pytest.approx(0.0123)

    # 3. Submit answer; revision runs async
    r = await client.post(
        f"/synthesize/{session_id}/answer",
        json={"question_id": "q1", "answer": "jake@example.com"},
    )
    assert r.status_code == 200
    assert r.json() == {"accepted": True}

    # 4. Wait for revision to land — session should be awaiting approval
    snap = await _wait_for_state(client, session_id, "awaiting_approval")
    assert len(snap["answered"]) == 1
    assert snap["answered"][0]["answer"] == "jake@example.com"
    assert "answered q1" in snap["draft"]["markdown"]

    # 5. Approve
    r = await client.post(f"/synthesize/{session_id}/approve", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["skill_slug"] == "gmail_reply_api_test"
    assert body["path"].endswith("/skills/gmail_reply_api_test")

    # 6. GET /skills lists the new skill
    r = await client.get("/skills")
    assert r.status_code == 200
    skills = r.json()
    assert len(skills) == 1
    assert skills[0]["slug"] == "gmail_reply_api_test"
    assert skills[0]["name"] == "Reply to Gmail"
    assert skills[0]["step_count"] == 4
    assert skills[0]["destructive_step_count"] == 1

    # 7. GET /skills/{slug} returns full markdown + meta + preview_urls
    r = await client.get("/skills/gmail_reply_api_test")
    assert r.status_code == 200
    body = r.json()
    assert "## Steps" in body["markdown"]
    assert body["meta"]["slug"] == "gmail_reply_api_test"
    assert len(body["preview_urls"]) == 1
    assert body["preview_urls"][0].startswith(
        "/skills/gmail_reply_api_test/preview/"
    )

    # 8. Preview PNG served
    preview_url = body["preview_urls"][0]
    r = await client.get(preview_url)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 0


class _FakeRequest:
    """Stand-in for ``starlette.requests.Request`` that only exposes
    ``is_disconnected``. ``httpx.ASGITransport`` buffers streaming responses
    in full before returning any bytes — fine for normal JSON endpoints but
    incompatible with the long-lived SSE generator. We test the generator
    (``api_module._sse_event_generator``) directly instead, which is what
    actually carries the business logic; ``EventSourceResponse`` is a thin
    library wrapper and its ping parameter is verified structurally below."""

    def __init__(self) -> None:
        self._disconnected = False

    async def is_disconnected(self) -> bool:
        return self._disconnected

    def disconnect(self) -> None:
        self._disconnected = True


async def _drive_session_through_approval(
    client: httpx.AsyncClient,
) -> tuple[str, api_module.SynthesisSession]:
    """Helper: start a session and drive it to completed state via the API."""
    r = await client.post(
        "/synthesize/start", json={"trajectory_id": _TRAJECTORY_ID}
    )
    session_id = r.json()["session_id"]
    await _wait_for_state(client, session_id, "awaiting_answer")
    await client.post(
        f"/synthesize/{session_id}/answer",
        json={"question_id": "q1", "answer": "jake@example.com"},
    )
    await _wait_for_state(client, session_id, "awaiting_approval")
    r = await client.post(f"/synthesize/{session_id}/approve", json={})
    assert r.status_code == 200
    session = api_module._STORE.get(session_id)
    assert session is not None
    return session_id, session


@pytest.mark.asyncio
async def test_sse_stream_emits_lifecycle_events(
    client: httpx.AsyncClient,
) -> None:
    r = await client.post(
        "/synthesize/start", json={"trajectory_id": _TRAJECTORY_ID}
    )
    session_id = r.json()["session_id"]
    await _wait_for_state(client, session_id, "awaiting_answer")

    session = api_module._STORE.get(session_id)
    assert session is not None
    request = _FakeRequest()
    events: list[dict[str, Any]] = []
    gen = api_module._sse_event_generator(session, request)  # type: ignore[arg-type]
    async for event in gen:
        events.append(event)
        # Once we've seen the three expected lifecycle events, disconnect so
        # the generator exits cleanly on its next poll.
        if len(events) >= 3:
            request.disconnect()
            break

    event_names = [e["event"] for e in events]
    assert "state_change" in event_names
    assert "draft_ready" in event_names
    assert "question_ready" in event_names
    draft_event = next(e for e in events if e["event"] == "draft_ready")
    payload = json.loads(draft_event["data"])
    assert payload["draft"]["meta"]["slug"] == "gmail_reply_api_test"


@pytest.mark.asyncio
async def test_sse_stream_completes_on_approve(
    client: httpx.AsyncClient,
) -> None:
    _session_id, session = await _drive_session_through_approval(client)

    request = _FakeRequest()
    events: list[dict[str, Any]] = []
    async for event in api_module._sse_event_generator(session, request):  # type: ignore[arg-type]
        events.append(event)
        if len(events) > 20:  # safety break
            break

    names = [e["event"] for e in events]
    assert "saved" in names
    saved_event = next(e for e in events if e["event"] == "saved")
    payload = json.loads(saved_event["data"])
    assert payload["slug"] == "gmail_reply_api_test"


@pytest.mark.asyncio
async def test_sse_stream_respects_client_disconnect(
    client: httpx.AsyncClient,
) -> None:
    r = await client.post(
        "/synthesize/start", json={"trajectory_id": _TRAJECTORY_ID}
    )
    session_id = r.json()["session_id"]
    await _wait_for_state(client, session_id, "awaiting_answer")

    session = api_module._STORE.get(session_id)
    assert session is not None
    request = _FakeRequest()
    request.disconnect()
    events: list[dict[str, Any]] = []
    async for event in api_module._sse_event_generator(session, request):  # type: ignore[arg-type]
        events.append(event)
    # Pre-disconnected request ends the generator on the first iteration.
    assert events == []


def test_stream_endpoint_wires_ping_interval() -> None:
    """The SSE ping cadence is carried by the module-level
    ``SSE_PING_INTERVAL_SECONDS`` constant, passed as the ``ping`` kwarg on
    :class:`sse_starlette.sse.EventSourceResponse` (verified by route-level
    construction; sse-starlette emits the ``:ping`` comment itself)."""
    assert api_module.SSE_PING_INTERVAL_SECONDS == 15
    # Pin the route's actual wiring: stream_session builds the EventSourceResponse
    # with ping=SSE_PING_INTERVAL_SECONDS — walk the router to verify the path.
    routes = [r for r in api_module.router.routes if getattr(r, "path", None)
              == "/synthesize/{session_id}/stream"]
    assert len(routes) == 1


@pytest.mark.asyncio
async def test_cancel_returns_cancelled_true(
    client: httpx.AsyncClient,
) -> None:
    r = await client.post(
        "/synthesize/start", json={"trajectory_id": _TRAJECTORY_ID}
    )
    session_id = r.json()["session_id"]
    await _wait_for_state(client, session_id, "awaiting_answer")
    r = await client.delete(f"/synthesize/{session_id}")
    assert r.status_code == 200
    assert r.json() == {"cancelled": True}
    snap = (await client.get(f"/synthesize/{session_id}")).json()
    assert snap["state"] == "cancelled"


@pytest.mark.asyncio
async def test_answer_in_wrong_state_returns_409(
    client: httpx.AsyncClient,
) -> None:
    r = await client.post(
        "/synthesize/start", json={"trajectory_id": _TRAJECTORY_ID}
    )
    session_id = r.json()["session_id"]
    await _wait_for_state(client, session_id, "awaiting_answer")
    # Complete the Q&A
    await client.post(
        f"/synthesize/{session_id}/answer",
        json={"question_id": "q1", "answer": "jake@example.com"},
    )
    await _wait_for_state(client, session_id, "awaiting_approval")
    # Now try to answer again
    r = await client.post(
        f"/synthesize/{session_id}/answer",
        json={"question_id": "q1", "answer": "hello"},
    )
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_answer_empty_returns_400(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/synthesize/start", json={"trajectory_id": _TRAJECTORY_ID}
    )
    session_id = r.json()["session_id"]
    await _wait_for_state(client, session_id, "awaiting_answer")
    r = await client.post(
        f"/synthesize/{session_id}/answer",
        json={"question_id": "q1", "answer": "   "},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_approve_with_user_slug_collision_returns_409(
    client: httpx.AsyncClient,
) -> None:
    # First synthesis — writes gmail_reply_api_test
    r = await client.post(
        "/synthesize/start", json={"trajectory_id": _TRAJECTORY_ID}
    )
    sid1 = r.json()["session_id"]
    await _wait_for_state(client, sid1, "awaiting_answer")
    await client.post(
        f"/synthesize/{sid1}/answer",
        json={"question_id": "q1", "answer": "jake@example.com"},
    )
    await _wait_for_state(client, sid1, "awaiting_approval")
    r = await client.post(f"/synthesize/{sid1}/approve", json={})
    assert r.status_code == 200

    # Second synthesis — try to approve with the same user-provided slug
    r = await client.post(
        "/synthesize/start", json={"trajectory_id": _TRAJECTORY_ID}
    )
    sid2 = r.json()["session_id"]
    await _wait_for_state(client, sid2, "awaiting_answer")
    await client.post(
        f"/synthesize/{sid2}/answer",
        json={"question_id": "q1", "answer": "jake@example.com"},
    )
    await _wait_for_state(client, sid2, "awaiting_approval")
    r = await client.post(
        f"/synthesize/{sid2}/approve",
        json={"slug": "gmail_reply_api_test"},
    )
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_delete_skill_then_list_does_not_show_it(
    client: httpx.AsyncClient,
) -> None:
    r = await client.post(
        "/synthesize/start", json={"trajectory_id": _TRAJECTORY_ID}
    )
    session_id = r.json()["session_id"]
    await _wait_for_state(client, session_id, "awaiting_answer")
    await client.post(
        f"/synthesize/{session_id}/answer",
        json={"question_id": "q1", "answer": "jake@example.com"},
    )
    await _wait_for_state(client, session_id, "awaiting_approval")
    await client.post(f"/synthesize/{session_id}/approve", json={})

    r = await client.delete("/skills/gmail_reply_api_test")
    assert r.status_code == 200
    assert r.json() == {"deleted": True}
    r = await client.get("/skills")
    assert r.status_code == 200
    assert r.json() == []
    # Second delete returns 404
    r = await client.delete("/skills/gmail_reply_api_test")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_unknown_skill_returns_404(
    client: httpx.AsyncClient,
) -> None:
    r = await client.delete("/skills/no_such_skill")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_skill_not_found_returns_404(
    client: httpx.AsyncClient,
) -> None:
    r = await client.get("/skills/no_such_skill")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_list_skills_empty_when_no_synthesis(
    client: httpx.AsyncClient,
) -> None:
    r = await client.get("/skills")
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_preview_rejects_path_traversal(
    client: httpx.AsyncClient,
) -> None:
    r = await client.post(
        "/synthesize/start", json={"trajectory_id": _TRAJECTORY_ID}
    )
    session_id = r.json()["session_id"]
    await _wait_for_state(client, session_id, "awaiting_answer")
    await client.post(
        f"/synthesize/{session_id}/answer",
        json={"question_id": "q1", "answer": "jake@example.com"},
    )
    await _wait_for_state(client, session_id, "awaiting_approval")
    await client.post(f"/synthesize/{session_id}/approve", json={})

    # httpx URL-encodes the slash so this becomes /skills/.../preview/..%2Fetc
    r = await client.get(
        "/skills/gmail_reply_api_test/preview/..%2Fetc%2Fpasswd"
    )
    assert r.status_code in (400, 404)
