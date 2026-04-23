"""End-to-end dry-run pipeline smoke tests (X-021).

Drives the full runner stack — POST /run/start → WebSocket stream →
/run/{id}/confirm → final run_metadata.json — against fake-mode LLM fixtures
for each of the five reference skills, plus dedicated tests for the abort,
kill-switch, budget-breach, and pre-action-gate paths.

Every test uses the dry-run adapter pair (``DryRunInputAdapter`` +
``TrajectoryScreenSource``) so no real CGEventPost fires and no real screen
is captured. ``anthropic_mock`` is used to catch any accidental network
traffic to ``api.anthropic.com``.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from runner.agent_runtime import AgentResponse
from runner.api import router
from runner.claude_runtime import FAKE_MODE_ENV_VAR, ClaudeRuntime
from runner.coords import ImageMapping
from runner.input_adapter import DryRunInputAdapter
from runner.kill_switch import KillSwitch
from runner.pre_action_gate import AXTarget
from runner.run_manager import AdapterBundle, RunManager
from runner.safety import LIVE_MODE_ENV_VAR
from runner.screen_source import TrajectoryScreenSource, blank_canvas_png

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES_SKILLS = _REPO_ROOT / "fixtures" / "skills"

# (slug, params, destructive_step_or_None, expected_turn_count, trajectory_ref)
_SKILL_SCENARIOS = [
    (
        "gmail_reply",
        {"sender": "a@b.co", "template": "hi"},
        7,
        4,
        "00000000-0000-0000-0000-000000000001",
    ),
    (
        "calendar_block",
        {},
        7,
        4,
        "00000000-0000-0000-0000-000000000002",
    ),
    (
        "finder_organize",
        {},
        5,
        4,
        "00000000-0000-0000-0000-000000000003",
    ),
    (
        "slack_status",
        {},
        7,
        4,
        "00000000-0000-0000-0000-000000000004",
    ),
    (
        "notes_daily",
        {},
        None,
        4,
        "00000000-0000-0000-0000-000000000005",
    ),
]


# ---------- fakes ----------


class _FakeAgentRuntime:
    """Serve canned ``AgentResponse`` turns without hitting the API."""

    def __init__(self, responses: list[AgentResponse], *, hang_after: bool = False) -> None:
        self._responses = list(responses)
        self._hang_after = hang_after
        self._turn = 0

    async def run_turn(
        self,
        system_prompt: str,
        messages: list[Any],
        max_tokens: int = 4096,
    ) -> AgentResponse:
        self._turn += 1
        if self._turn > len(self._responses):
            if self._hang_after:
                import asyncio as _asyncio

                await _asyncio.sleep(3600)
            raise AssertionError(
                f"Fake runtime exhausted: asked for turn {self._turn} "
                f"but only have {len(self._responses)} canned"
            )
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


class _NullAXResolver:
    def resolve_at(self, x_pt: float, y_pt: float) -> AXTarget | None:
        return None


class _SendButtonAXResolver:
    """Always claim the element under the cursor is an AXButton labeled 'Send'."""

    def resolve_at(self, x_pt: float, y_pt: float) -> AXTarget | None:
        return AXTarget(role="AXButton", label="Send")


# ---------- canned-turn helpers ----------


def _screenshot(block_id: str) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": block_id,
        "name": "computer",
        "input": {"action": "screenshot"},
    }


def _click(block_id: str, x: int = 100, y: int = 100) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": block_id,
        "name": "computer",
        "input": {"action": "left_click", "coordinate": [x, y]},
    }


def _response(blocks: list[dict[str, Any]], stop: str = "tool_use") -> AgentResponse:
    return AgentResponse(
        content_blocks=blocks,
        stop_reason=stop,
        input_tokens=100,
        output_tokens=20,
        turn_number=0,
    )


# ---------- shared fixtures ----------


@pytest.fixture
def skills_root(tmp_path: Path) -> Path:
    dest = tmp_path / "skills"
    dest.mkdir()
    for slug_dir in _FIXTURES_SKILLS.iterdir():
        if slug_dir.is_dir():
            shutil.copytree(slug_dir, dest / slug_dir.name)
    return dest


@pytest.fixture
def trajectories_root(tmp_path: Path) -> Path:
    root = tmp_path / "trajectories"
    png = blank_canvas_png()
    for (_slug, _params, _step, _turns, traj_id) in _SKILL_SCENARIOS:
        screenshots = root / traj_id / "screenshots"
        screenshots.mkdir(parents=True)
        for i in range(1, 11):
            (screenshots / f"{i:04d}.png").write_bytes(png)
    return root


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    return root


@pytest.fixture
def costs_path(tmp_path: Path) -> Path:
    return tmp_path / "costs.jsonl"


@pytest.fixture
def kill_switch() -> KillSwitch:
    return KillSwitch()


@pytest.fixture
def anthropic_mock_strict() -> Iterator[respx.MockRouter]:
    """respx mock that fails loudly if any network call is attempted."""
    with respx.mock(
        base_url="https://api.anthropic.com", assert_all_called=False
    ) as router_:
        yield router_


def _adapter_factory(
    trajectories_root: Path, *, ax_resolver: Any = None
) -> Any:
    resolver = ax_resolver or _NullAXResolver()

    def factory(skill: Any, mode: str) -> AdapterBundle:
        trajectory_id = str(skill.meta.get("trajectory_ref", ""))
        return AdapterBundle(
            input_adapter=DryRunInputAdapter(),
            screen_source=TrajectoryScreenSource(
                trajectory_id, trajectories_root=trajectories_root
            ),
            ax_resolver=resolver,
        )

    return factory


def _build_app(
    *,
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    costs_path: Path,
    kill_switch: KillSwitch,
    runtime_factory: Any,
    ax_resolver: Any = None,
) -> FastAPI:
    app = FastAPI()
    manager = RunManager(
        runs_root=runs_root,
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        costs_path=costs_path,
        kill_switch=kill_switch,
        adapter_factory=_adapter_factory(
            trajectories_root, ax_resolver=ax_resolver
        ),
        runtime_factory=runtime_factory,
    )
    app.state.run_manager = manager
    app.include_router(router)
    return app


_TERMINAL = {"succeeded", "failed", "aborted", "budget_exceeded", "rate_limited"}


def _wait_for_terminal(runs_root: Path, run_id: str, timeout: float = 15.0) -> dict[str, Any]:
    """Poll the on-disk run_metadata.json until status is terminal."""
    metadata_path = runs_root / run_id / "run_metadata.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if metadata_path.is_file():
            try:
                data = json.loads(metadata_path.read_text())
            except json.JSONDecodeError:
                time.sleep(0.02)
                continue
            if data.get("status") in _TERMINAL:
                return dict(data)
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not reach a terminal status within {timeout}s")


def _drive_run_to_completion(
    client: TestClient, run_id: str, *, max_messages: int = 400
) -> tuple[bool, bool]:
    """Stream WebSocket events; confirm on any destructive pause; wait for done."""
    saw_confirmation = False
    saw_done = False
    with client.websocket_connect(f"/run/{run_id}/stream") as ws:
        for _ in range(max_messages):
            message = ws.receive_json()
            if message.get("type") == "confirmation_request":
                saw_confirmation = True
                resp = client.post(
                    f"/run/{run_id}/confirm", json={"decision": "confirm"}
                )
                assert resp.status_code == 200
            if message.get("type") == "done":
                saw_done = True
                break
    return saw_confirmation, saw_done


# ---------- per-skill fake-mode e2e ----------


@pytest.mark.parametrize(
    ("slug", "params", "destructive_step", "expected_turns", "trajectory_id"),
    _SKILL_SCENARIOS,
    ids=[s[0] for s in _SKILL_SCENARIOS],
)
def test_e2e_fake_mode_runs_each_skill_to_success(
    slug: str,
    params: dict[str, str],
    destructive_step: int | None,
    expected_turns: int,
    trajectory_id: str,
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    costs_path: Path,
    kill_switch: KillSwitch,
    anthropic_mock_strict: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each reference skill runs end-to-end in fake-mode to status=succeeded."""

    monkeypatch.setenv(FAKE_MODE_ENV_VAR, "1")

    def runtime_factory(run_id: str) -> ClaudeRuntime:
        return ClaudeRuntime(run_id=run_id, costs_path=costs_path)

    started = time.monotonic()
    app = _build_app(
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        runs_root=runs_root,
        costs_path=costs_path,
        kill_switch=kill_switch,
        runtime_factory=runtime_factory,
    )
    client = TestClient(app)

    resp = client.post(
        "/run/start",
        json={"skill_slug": slug, "parameters": params, "mode": "dry_run"},
    )
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    saw_confirmation, saw_done = _drive_run_to_completion(client, run_id)
    _wait_for_terminal(runs_root, run_id)

    if destructive_step is not None:
        assert saw_confirmation, f"{slug}: expected a confirmation_request"
    assert saw_done, f"{slug}: expected a done event"

    # Per-run e2e time budget.
    elapsed = time.monotonic() - started
    assert elapsed < 15.0, f"{slug} e2e took {elapsed:.2f}s (>15s)"

    # run_metadata.json: schema-valid, status=succeeded, ended_at set.
    run_dir = runs_root / run_id
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "succeeded"
    assert metadata["skill_slug"] == slug
    assert metadata["ended_at"] is not None
    assert metadata["mode"] == "dry_run"

    # transcript.jsonl has at least as many turns as the canned script.
    transcript_lines = [
        json.loads(line)
        for line in (run_dir / "transcript.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(transcript_lines) >= expected_turns

    # events.jsonl contains step_start + (for destructive skills) the
    # confirmation_requested / confirmed pair.
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    event_types = [e["type"] for e in events]
    assert "step_start" in event_types
    if destructive_step is not None:
        assert any(
            e["type"] == "confirmation_requested"
            and e.get("step_number") == destructive_step
            for e in events
        )
        assert any(
            e["type"] == "confirmed" and e.get("step_number") == destructive_step
            for e in events
        )

    # screenshots/ has at least one PNG for each screenshot tool_call.
    screenshots = sorted((run_dir / "screenshots").iterdir())
    assert screenshots, "expected at least one screenshot"
    for p in screenshots:
        assert p.read_bytes().startswith(b"\x89PNG")

    # costs.jsonl totals match metadata.total_cost_usd within $0.001.
    cost_lines = [
        json.loads(line)
        for line in costs_path.read_text().splitlines()
        if line.strip()
    ]
    run_costs = [
        record for record in cost_lines if record.get("run_id") == run_id
    ]
    assert run_costs, "fake-mode runtime should log at least one cost line"
    total = sum(float(r["cost_estimate_usd"]) for r in run_costs)
    assert abs(total - float(metadata["total_cost_usd"])) < 0.001

    # Zero real API calls went to api.anthropic.com.
    assert len(anthropic_mock_strict.calls) == 0


# ---------- abort mid-run ----------


def test_e2e_abort_midrun_transitions_to_aborted(
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    costs_path: Path,
    kill_switch: KillSwitch,
    anthropic_mock_strict: respx.MockRouter,
) -> None:
    """POST /run/{id}/abort interrupts a live run; no further tool calls fire."""

    recorded_adapters: list[DryRunInputAdapter] = []

    def recording_adapter_factory(skill: Any, mode: str) -> AdapterBundle:
        adapter = DryRunInputAdapter()
        recorded_adapters.append(adapter)
        trajectory_id = str(skill.meta.get("trajectory_ref", ""))
        return AdapterBundle(
            input_adapter=adapter,
            screen_source=TrajectoryScreenSource(
                trajectory_id, trajectories_root=trajectories_root
            ),
            ax_resolver=_NullAXResolver(),
        )

    # Canned: screenshot, then hang forever — the abort must cut in.
    def runtime_factory(run_id: str) -> _FakeAgentRuntime:
        return _FakeAgentRuntime(
            [_response([_screenshot("t1")])], hang_after=True
        )

    app = FastAPI()
    manager = RunManager(
        runs_root=runs_root,
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        costs_path=costs_path,
        kill_switch=kill_switch,
        adapter_factory=recording_adapter_factory,
        runtime_factory=runtime_factory,
    )
    app.state.run_manager = manager
    app.include_router(router)
    client = TestClient(app)

    resp = client.post(
        "/run/start",
        json={"skill_slug": "notes_daily", "parameters": {}, "mode": "dry_run"},
    )
    run_id = resp.json()["run_id"]

    # Give the run a moment to start its first turn.
    time.sleep(0.1)

    resp = client.post(f"/run/{run_id}/abort")
    assert resp.status_code == 200

    _wait_for_terminal(runs_root, run_id)

    metadata = json.loads(
        (runs_root / run_id / "run_metadata.json").read_text()
    )
    assert metadata["status"] == "aborted"

    # No second click / type action ran — the only side effect was the
    # single screenshot in turn 1 (which doesn't hit the input adapter).
    assert recorded_adapters
    assert recorded_adapters[0].get_recorded_calls() == []


# ---------- kill-switch (direct) ----------


def test_e2e_kill_switch_direct_abort(
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    costs_path: Path,
    anthropic_mock_strict: respx.MockRouter,
) -> None:
    """Calling KillSwitch.kill directly (bypassing the API) aborts the run."""

    kill_switch = KillSwitch()

    def runtime_factory(run_id: str) -> _FakeAgentRuntime:
        return _FakeAgentRuntime(
            [_response([_screenshot("t1")])], hang_after=True
        )

    app = _build_app(
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        runs_root=runs_root,
        costs_path=costs_path,
        kill_switch=kill_switch,
        runtime_factory=runtime_factory,
    )
    client = TestClient(app)

    resp = client.post(
        "/run/start",
        json={"skill_slug": "notes_daily", "parameters": {}, "mode": "dry_run"},
    )
    run_id = resp.json()["run_id"]
    time.sleep(0.1)

    assert kill_switch.kill(run_id, reason="kill_switch_hotkey") is True

    _wait_for_terminal(runs_root, run_id)

    metadata = json.loads(
        (runs_root / run_id / "run_metadata.json").read_text()
    )
    assert metadata["status"] == "aborted"
    assert metadata.get("abort_reason") == "kill_switch_hotkey"


# ---------- budget breach ----------


def test_e2e_budget_breach_transitions_to_budget_exceeded(
    tmp_path: Path,
    trajectories_root: Path,
    runs_root: Path,
    costs_path: Path,
    kill_switch: KillSwitch,
    anthropic_mock_strict: respx.MockRouter,
) -> None:
    """A tight max_total_actions cap aborts with status=budget_exceeded."""

    # Clone notes_daily into a test-owned skills root and override
    # runtime_limits so total_actions trips on the third tool call.
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    shutil.copytree(_FIXTURES_SKILLS / "notes_daily", skills_root / "notes_daily")
    meta_path = skills_root / "notes_daily" / "skill.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["runtime_limits"] = {"max_total_actions": 3}
    meta_path.write_text(json.dumps(meta, indent=2))

    # Script: 5 screenshot turns. On action #3 the budget trips.
    def runtime_factory(run_id: str) -> _FakeAgentRuntime:
        return _FakeAgentRuntime(
            [_response([_screenshot(f"t{i}")]) for i in range(1, 6)]
        )

    app = _build_app(
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        runs_root=runs_root,
        costs_path=costs_path,
        kill_switch=kill_switch,
        runtime_factory=runtime_factory,
    )
    client = TestClient(app)

    resp = client.post(
        "/run/start",
        json={"skill_slug": "notes_daily", "parameters": {}, "mode": "dry_run"},
    )
    run_id = resp.json()["run_id"]

    _wait_for_terminal(runs_root, run_id)

    metadata = json.loads(
        (runs_root / run_id / "run_metadata.json").read_text()
    )
    assert metadata["status"] == "budget_exceeded"
    assert metadata.get("abort_reason", "").startswith("budget_exceeded")


# ---------- pre-action gate false-negative ----------


def test_e2e_pre_action_gate_false_negative_forces_abort(
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    costs_path: Path,
    kill_switch: KillSwitch,
    anthropic_mock_strict: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate flags an unflagged click on a 'Send' AXButton → user aborts."""

    monkeypatch.setenv(LIVE_MODE_ENV_VAR, "1")

    # Script: one screenshot, then a click that the skill didn't flag.
    def runtime_factory(run_id: str) -> _FakeAgentRuntime:
        return _FakeAgentRuntime(
            [
                _response([_screenshot("t1")]),
                _response([_click("t2", x=120, y=240)]),
            ],
            hang_after=True,
        )

    app = _build_app(
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        runs_root=runs_root,
        costs_path=costs_path,
        kill_switch=kill_switch,
        runtime_factory=runtime_factory,
        ax_resolver=_SendButtonAXResolver(),
    )
    client = TestClient(app)

    # notes_daily has no destructive steps — the gate is the only reason a
    # confirmation can appear.
    resp = client.post(
        "/run/start",
        json={"skill_slug": "notes_daily", "parameters": {}, "mode": "execute"},
    )
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    saw_gate_request = False
    with client.websocket_connect(f"/run/{run_id}/stream") as ws:
        for _ in range(200):
            message = ws.receive_json()
            if message.get("type") == "confirmation_request":
                saw_gate_request = True
                resp = client.post(
                    f"/run/{run_id}/confirm", json={"decision": "abort"}
                )
                assert resp.status_code == 200
            if message.get("type") == "done":
                break

    assert saw_gate_request

    _wait_for_terminal(runs_root, run_id)
    metadata = json.loads(
        (runs_root / run_id / "run_metadata.json").read_text()
    )
    assert metadata["status"] == "aborted"
