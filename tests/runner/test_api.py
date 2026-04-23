"""Tests for the runner HTTP API (X-019).

Covers:

* POST /run/start happy path (dry_run, returns run_id)
* POST /run/start with mode=execute and TRACE_ALLOW_LIVE unset → 400
* POST /run/start with unknown skill_slug → 404
* GET /run/{run_id} returns run metadata
* POST /run/{run_id}/confirm on an awaiting run → 200
* POST /run/{run_id}/confirm on a non-awaiting run → 409
* POST /run/{run_id}/confirm for an unknown run → 404
* POST /run/{run_id}/abort on an active run → 200, transitions to aborted
* POST /run/{run_id}/abort on an already-finished run → 200 (idempotent)
* GET /runs returns newest first, supports skill_slug filter
* GET /run/{run_id}/events returns events.jsonl as a JSON array
* GET /run/{run_id}/screenshots/{filename} serves the PNG
* WebSocket: start → stream → confirm → done happy path
* WebSocket sends keepalive ping when idle past the threshold
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from runner import api as api_module
from runner.agent_runtime import AgentResponse
from runner.api import router
from runner.coords import ImageMapping
from runner.input_adapter import DryRunInputAdapter
from runner.kill_switch import KillSwitch
from runner.pre_action_gate import AXTarget
from runner.run_manager import AdapterBundle, RunManager
from runner.screen_source import TrajectoryScreenSource, blank_canvas_png

_FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "skills"


# ---------- fakes ----------


class _FakeAgentRuntime:
    """Serve canned ``AgentResponse`` turns."""

    def __init__(self, responses: list[AgentResponse]) -> None:
        self._responses = list(responses)
        self._turn = 0

    async def run_turn(
        self,
        system_prompt: str,
        messages: list[Any],
        max_tokens: int = 4096,
    ) -> AgentResponse:
        self._turn += 1
        if self._turn > len(self._responses):
            # Keep the run alive waiting for external interruption (abort).
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")
        base = self._responses[self._turn - 1]
        return AgentResponse(
            content_blocks=base.content_blocks,
            stop_reason=base.stop_reason,
            input_tokens=base.input_tokens,
            output_tokens=base.output_tokens,
            turn_number=self._turn,
        )

    def set_image_mapping(self, mapping: ImageMapping | None) -> None:
        return None


class _FakeAXResolver:
    def resolve_at(self, x_pt: float, y_pt: float) -> AXTarget | None:
        return None


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _screenshot_block(block_id: str) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": block_id,
        "name": "computer",
        "input": {"action": "screenshot"},
    }


def _click_block(block_id: str, x: int = 100, y: int = 100) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": block_id,
        "name": "computer",
        "input": {"action": "left_click", "coordinate": [x, y]},
    }


def _response(
    blocks: list[dict[str, Any]],
    stop_reason: str = "tool_use",
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> AgentResponse:
    return AgentResponse(
        content_blocks=blocks,
        stop_reason=stop_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        turn_number=0,
    )


def _script_non_destructive() -> list[AgentResponse]:
    return [
        _response([_screenshot_block("t1")], stop_reason="tool_use"),
        _response([_click_block("t2")], stop_reason="tool_use"),
        _response(
            [_text_block("done <workflow_complete/>")], stop_reason="end_turn"
        ),
    ]


def _script_destructive(step: int) -> list[AgentResponse]:
    return [
        _response([_screenshot_block("t1")], stop_reason="tool_use"),
        _response([_click_block("t2")], stop_reason="tool_use"),
        _response(
            [_text_block(f'ready <needs_confirmation step="{step}"/>')],
            stop_reason="end_turn",
        ),
        _response(
            [_text_block("done <workflow_complete/>")], stop_reason="end_turn"
        ),
    ]


def _script_hang() -> list[AgentResponse]:
    """One screenshot turn, then the runtime hangs so abort can fire."""
    return [_response([_screenshot_block("t1")], stop_reason="tool_use")]


# ---------- fixtures ----------


@pytest.fixture
def skills_root(tmp_path: Path) -> Path:
    dest = tmp_path / "skills"
    dest.mkdir()
    for slug in ("gmail_reply", "notes_daily"):
        shutil.copytree(_FIXTURES_ROOT / slug, dest / slug)
    return dest


@pytest.fixture
def trajectories_root(tmp_path: Path) -> Path:
    root = tmp_path / "trajectories"
    png = blank_canvas_png()
    for trajectory_id in (
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000005",
    ):
        screenshots = root / trajectory_id / "screenshots"
        screenshots.mkdir(parents=True)
        for i in range(1, 21):
            (screenshots / f"{i:04d}.png").write_bytes(png)
    return root


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    return root


@pytest.fixture
def kill_switch() -> KillSwitch:
    return KillSwitch()


def _adapter_factory(trajectories_root: Path) -> Any:
    def factory(skill: Any, mode: str) -> AdapterBundle:
        trajectory_id = str(skill.meta.get("trajectory_ref", ""))
        return AdapterBundle(
            input_adapter=DryRunInputAdapter(),
            screen_source=TrajectoryScreenSource(
                trajectory_id, trajectories_root=trajectories_root
            ),
            ax_resolver=_FakeAXResolver(),
        )

    return factory


def _build_app(
    *,
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    kill_switch: KillSwitch,
    runtime_factory: Any,
    costs_path: Path | None = None,
) -> FastAPI:
    app = FastAPI()
    manager = RunManager(
        runs_root=runs_root,
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        costs_path=costs_path or (runs_root.parent / "costs.jsonl"),
        kill_switch=kill_switch,
        adapter_factory=_adapter_factory(trajectories_root),
        runtime_factory=runtime_factory,
    )
    app.state.run_manager = manager
    app.include_router(router)
    return app


@pytest.fixture
async def client(
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    kill_switch: KillSwitch,
) -> AsyncIterator[AsyncClient]:
    def runtime_factory(run_id: str) -> _FakeAgentRuntime:
        return _FakeAgentRuntime(_script_non_destructive())

    app = _build_app(
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        runs_root=runs_root,
        kill_switch=kill_switch,
        runtime_factory=runtime_factory,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        c.app = app  # type: ignore[attr-defined]
        yield c


# ---------- start ----------


async def test_start_run_dry_run_returns_run_id(
    client: AsyncClient,
) -> None:
    resp = await client.post(
        "/run/start",
        json={"skill_slug": "notes_daily", "parameters": {}, "mode": "dry_run"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "run_id" in data
    uuid.UUID(data["run_id"])
    # Wait for the run to complete.
    manager: RunManager = client.app.state.run_manager  # type: ignore[attr-defined]
    handle = manager.get(data["run_id"])
    assert handle is not None
    await handle.task


async def test_start_run_execute_without_trace_allow_live_returns_400(
    client: AsyncClient,
) -> None:
    resp = await client.post(
        "/run/start",
        json={"skill_slug": "notes_daily", "parameters": {}, "mode": "execute"},
    )
    assert resp.status_code == 400
    assert "TRACE_ALLOW_LIVE" in resp.json()["detail"]


async def test_start_run_unknown_skill_returns_404(
    client: AsyncClient,
) -> None:
    resp = await client.post(
        "/run/start",
        json={"skill_slug": "does_not_exist", "parameters": {}, "mode": "dry_run"},
    )
    assert resp.status_code == 404


# ---------- get run ----------


async def test_get_run_returns_metadata(client: AsyncClient) -> None:
    resp = await client.post(
        "/run/start",
        json={"skill_slug": "notes_daily", "parameters": {}, "mode": "dry_run"},
    )
    run_id = resp.json()["run_id"]
    manager: RunManager = client.app.state.run_manager  # type: ignore[attr-defined]
    await manager.get(run_id).task  # type: ignore[union-attr]

    resp = await client.get(f"/run/{run_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == run_id
    assert data["skill_slug"] == "notes_daily"
    assert data["status"] in {"succeeded", "failed", "aborted"}


async def test_get_run_unknown_returns_404(client: AsyncClient) -> None:
    resp = await client.get(f"/run/{uuid.uuid4()}")
    assert resp.status_code == 404


# ---------- confirm ----------


async def test_confirm_on_non_awaiting_run_returns_409(
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    kill_switch: KillSwitch,
) -> None:
    """Confirm fires 409 when the run isn't awaiting_confirmation."""

    def runtime_factory(run_id: str) -> _FakeAgentRuntime:
        return _FakeAgentRuntime(_script_hang())

    app = _build_app(
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        runs_root=runs_root,
        kill_switch=kill_switch,
        runtime_factory=runtime_factory,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/run/start",
            json={
                "skill_slug": "notes_daily",
                "parameters": {},
                "mode": "dry_run",
            },
        )
        run_id = resp.json()["run_id"]
        await asyncio.sleep(0.05)  # let the run start

        resp = await client.post(
            f"/run/{run_id}/confirm", json={"decision": "confirm"}
        )
        assert resp.status_code == 409

        # Clean up: abort the hung run.
        await client.post(f"/run/{run_id}/abort")
        manager: RunManager = app.state.run_manager
        handle = manager.get(run_id)
        assert handle is not None
        await handle.task


async def test_confirm_unknown_run_returns_404(client: AsyncClient) -> None:
    resp = await client.post(
        f"/run/{uuid.uuid4()}/confirm", json={"decision": "confirm"}
    )
    assert resp.status_code == 404


# ---------- abort ----------


async def test_abort_active_run_transitions_to_aborted(
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    kill_switch: KillSwitch,
) -> None:
    def runtime_factory(run_id: str) -> _FakeAgentRuntime:
        return _FakeAgentRuntime(_script_hang())

    app = _build_app(
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        runs_root=runs_root,
        kill_switch=kill_switch,
        runtime_factory=runtime_factory,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/run/start",
            json={
                "skill_slug": "notes_daily",
                "parameters": {},
                "mode": "dry_run",
            },
        )
        run_id = resp.json()["run_id"]
        # Wait for the run to be live.
        await asyncio.sleep(0.05)

        resp = await client.post(f"/run/{run_id}/abort")
        assert resp.status_code == 200

        manager: RunManager = app.state.run_manager
        handle = manager.get(run_id)
        assert handle is not None
        await asyncio.wait_for(handle.task, timeout=2.0)
        metadata_path = runs_root / run_id / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        assert metadata["status"] == "aborted"


async def test_abort_already_finished_run_is_idempotent(
    client: AsyncClient,
) -> None:
    resp = await client.post(
        "/run/start",
        json={"skill_slug": "notes_daily", "parameters": {}, "mode": "dry_run"},
    )
    run_id = resp.json()["run_id"]
    manager: RunManager = client.app.state.run_manager  # type: ignore[attr-defined]
    await manager.get(run_id).task  # type: ignore[union-attr]

    resp = await client.post(f"/run/{run_id}/abort")
    assert resp.status_code == 200
    # Second call is also fine.
    resp = await client.post(f"/run/{run_id}/abort")
    assert resp.status_code == 200


async def test_abort_unknown_run_is_idempotent_200(client: AsyncClient) -> None:
    resp = await client.post(f"/run/{uuid.uuid4()}/abort")
    assert resp.status_code == 200


# ---------- list runs ----------


async def test_list_runs_newest_first_and_filter(
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    kill_switch: KillSwitch,
) -> None:
    """Create three runs and assert /runs orders by started_at desc, filters by slug."""

    def runtime_factory(run_id: str) -> _FakeAgentRuntime:
        return _FakeAgentRuntime(_script_non_destructive())

    app = _build_app(
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        runs_root=runs_root,
        kill_switch=kill_switch,
        runtime_factory=runtime_factory,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        run_ids = []
        for slug in ("notes_daily", "gmail_reply", "notes_daily"):
            params = (
                {"sender": "a@b.co", "template": "hi"} if slug == "gmail_reply" else {}
            )
            # gmail_reply has a destructive step, so we'd need to confirm — use
            # notes_daily primarily for the listing test but include one gmail_reply
            # which the hang-script will block forever; we abort it instead.
            if slug == "gmail_reply":
                # Swap runtime to hang for this run only.
                manager: RunManager = app.state.run_manager
                manager._runtime_factory = lambda rid: _FakeAgentRuntime(
                    _script_hang()
                )
            else:
                manager = app.state.run_manager
                manager._runtime_factory = lambda rid: _FakeAgentRuntime(
                    _script_non_destructive()
                )
            resp = await client.post(
                "/run/start",
                json={"skill_slug": slug, "parameters": params, "mode": "dry_run"},
            )
            rid = resp.json()["run_id"]
            run_ids.append(rid)
            await asyncio.sleep(0.02)
            if slug == "gmail_reply":
                await client.post(f"/run/{rid}/abort")
                await asyncio.wait_for(
                    manager.get(rid).task, timeout=2.0  # type: ignore[union-attr]
                )
            else:
                await asyncio.wait_for(
                    manager.get(rid).task, timeout=5.0  # type: ignore[union-attr]
                )
            # Tiny sleep to ensure started_at timestamps are distinct.
            await asyncio.sleep(0.01)

        resp = await client.get("/runs")
        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) == 3
        assert [r["run_id"] for r in rows] == list(reversed(run_ids))

        resp = await client.get("/runs", params={"skill_slug": "gmail_reply"})
        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) == 1
        assert rows[0]["skill_slug"] == "gmail_reply"


# ---------- events ----------


async def test_get_events_returns_array(client: AsyncClient) -> None:
    resp = await client.post(
        "/run/start",
        json={"skill_slug": "notes_daily", "parameters": {}, "mode": "dry_run"},
    )
    run_id = resp.json()["run_id"]
    manager: RunManager = client.app.state.run_manager  # type: ignore[attr-defined]
    await manager.get(run_id).task  # type: ignore[union-attr]

    resp = await client.get(f"/run/{run_id}/events")
    assert resp.status_code == 200
    events = resp.json()
    assert isinstance(events, list)
    assert any(e.get("type") == "step_start" for e in events)


async def test_get_events_unknown_run_returns_404(client: AsyncClient) -> None:
    resp = await client.get(f"/run/{uuid.uuid4()}/events")
    assert resp.status_code == 404


# ---------- screenshots ----------


async def test_get_screenshot_serves_png(
    runs_root: Path, client: AsyncClient
) -> None:
    resp = await client.post(
        "/run/start",
        json={"skill_slug": "notes_daily", "parameters": {}, "mode": "dry_run"},
    )
    run_id = resp.json()["run_id"]
    manager: RunManager = client.app.state.run_manager  # type: ignore[attr-defined]
    await manager.get(run_id).task  # type: ignore[union-attr]

    screenshots_dir = runs_root / run_id / "screenshots"
    files = sorted(screenshots_dir.iterdir())
    assert files, "expected at least one screenshot"
    filename = files[0].name

    resp = await client.get(f"/run/{run_id}/screenshots/{filename}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(b"\x89PNG")


async def test_get_screenshot_missing_returns_404(
    client: AsyncClient,
) -> None:
    resp = await client.post(
        "/run/start",
        json={"skill_slug": "notes_daily", "parameters": {}, "mode": "dry_run"},
    )
    run_id = resp.json()["run_id"]
    manager: RunManager = client.app.state.run_manager  # type: ignore[attr-defined]
    await manager.get(run_id).task  # type: ignore[union-attr]

    resp = await client.get(f"/run/{run_id}/screenshots/9999.png")
    assert resp.status_code == 404


# ---------- WebSocket ----------


def _destructive_runtime_factory(step: int) -> Any:
    def factory(run_id: str) -> _FakeAgentRuntime:
        return _FakeAgentRuntime(_script_destructive(step))

    return factory


def test_websocket_happy_path_start_stream_confirm_done(
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    kill_switch: KillSwitch,
) -> None:
    """Full WebSocket happy path: start → stream events → confirm → done."""

    app = _build_app(
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        runs_root=runs_root,
        kill_switch=kill_switch,
        runtime_factory=_destructive_runtime_factory(7),
    )
    client = TestClient(app)
    start_resp = client.post(
        "/run/start",
        json={
            "skill_slug": "gmail_reply",
            "parameters": {"sender": "a@b.co", "template": "hi"},
            "mode": "dry_run",
        },
    )
    run_id = start_resp.json()["run_id"]

    saw_confirmation = False
    saw_done = False

    with client.websocket_connect(f"/run/{run_id}/stream") as ws:
        for _ in range(200):
            message = ws.receive_json()
            if message.get("type") == "confirmation_request":
                saw_confirmation = True
                # Submit the confirm decision via REST.
                resp = client.post(
                    f"/run/{run_id}/confirm", json={"decision": "confirm"}
                )
                assert resp.status_code == 200
            if message.get("type") == "done":
                saw_done = True
                break

    assert saw_confirmation
    assert saw_done


def test_websocket_keepalive_fires_when_idle(
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    kill_switch: KillSwitch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keepalive ping is sent when no events arrive within the threshold."""

    # Speed up the keepalive so the test stays fast.
    monkeypatch.setattr(api_module, "WEBSOCKET_KEEPALIVE_SECONDS", 0.05)

    app = _build_app(
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        runs_root=runs_root,
        kill_switch=kill_switch,
        runtime_factory=lambda rid: _FakeAgentRuntime([]),
    )
    client = TestClient(app)

    # Use a run_id that doesn't exist yet — subscription will receive no
    # events and must emit a keepalive.
    with client.websocket_connect(f"/run/{uuid.uuid4()}/stream") as ws:
        message = ws.receive_json()
        assert message["type"] == "keepalive"
