"""Tests for the X-018 kill switch.

Three layers of coverage:

1. :class:`KillSwitch` behavior on its own — register/unregister/kill/reason.
2. End-to-end executor integration in fake mode — kill during a canned
   script aborts within 2 seconds, kill during confirmation injects abort,
   kill before start / after finish are no-ops, idempotent kill, no further
   tool calls after kill, in-flight httpx request is actually cancelled.
3. API endpoint surface — POST ``/run/{run_id}/abort`` triggers
   :meth:`KillSwitch.kill` on the process-global switch.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from runner.agent_runtime import AgentResponse
from runner.api import app
from runner.budget import BudgetTracker, RunBudget
from runner.claude_runtime import ClaudeRuntime
from runner.confirmation import ConfirmationQueue
from runner.coords import ImageMapping
from runner.executor import KILL_SWITCH_REASON, Executor
from runner.input_adapter import DryRunInputAdapter
from runner.kill_switch import (
    DEFAULT_KILL_REASON,
    KillSwitch,
    get_global_kill_switch,
)
from runner.pre_action_gate import AXTarget
from runner.run_writer import RunWriter
from runner.schema import RunMode
from runner.screen_source import TrajectoryScreenSource, blank_canvas_png
from runner.skill_loader import load_skill

pytestmark = pytest.mark.asyncio

_FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "skills"


# ---------- KillSwitch unit tests ----------


async def test_register_returns_event_and_is_registered() -> None:
    ks = KillSwitch()
    assert not ks.is_registered("r1")
    event = ks.register("r1")
    assert isinstance(event, asyncio.Event)
    assert not event.is_set()
    assert ks.is_registered("r1")
    assert not ks.is_killed("r1")


async def test_register_twice_returns_same_event() -> None:
    ks = KillSwitch()
    e1 = ks.register("r1")
    e2 = ks.register("r1")
    assert e1 is e2


async def test_kill_sets_event_and_records_reason() -> None:
    ks = KillSwitch()
    event = ks.register("r1")
    assert ks.kill("r1", reason="user_abort") is True
    assert event.is_set()
    assert ks.is_killed("r1")
    assert ks.reason("r1") == "user_abort"


async def test_kill_default_reason_is_kill_switch() -> None:
    ks = KillSwitch()
    ks.register("r1")
    ks.kill("r1")
    assert ks.reason("r1") == DEFAULT_KILL_REASON


async def test_kill_before_register_is_noop() -> None:
    ks = KillSwitch()
    assert ks.kill("unknown") is False


async def test_kill_after_unregister_is_noop() -> None:
    ks = KillSwitch()
    ks.register("r1")
    ks.unregister("r1")
    assert ks.kill("r1") is False
    assert not ks.is_registered("r1")


async def test_kill_is_idempotent() -> None:
    ks = KillSwitch()
    ks.register("r1")
    assert ks.kill("r1") is True
    assert ks.kill("r1") is False
    assert ks.kill("r1", reason="different") is False
    # Reason from the first successful kill is preserved.
    assert ks.reason("r1") == DEFAULT_KILL_REASON


async def test_unregister_unknown_run_is_noop() -> None:
    ks = KillSwitch()
    ks.unregister("never_registered")  # must not raise


async def test_reason_unknown_returns_none() -> None:
    ks = KillSwitch()
    assert ks.reason("unknown") is None


# ---------- Executor integration fakes ----------


class FakeAgentRuntime:
    """Serves canned turns, with an optional per-turn delay.

    ``delay_seconds`` can be a single float (applied to every turn) or a list
    the same length as ``responses`` for per-turn delays. The delay knob lets
    tests simulate a slow Anthropic call so the kill path is exercised
    mid-request rather than between turns.
    """

    def __init__(
        self,
        responses: list[AgentResponse],
        *,
        delay_seconds: float | list[float] = 0.0,
    ) -> None:
        self._responses = list(responses)
        if isinstance(delay_seconds, list):
            self._delays: list[float] = list(delay_seconds)
        else:
            self._delays = [float(delay_seconds)] * len(responses)
        self._turn_counter = 0
        self.calls: int = 0
        self.cancelled: int = 0

    async def run_turn(
        self,
        system_prompt: str,
        messages: list[Any],
        max_tokens: int = 4096,
    ) -> AgentResponse:
        self.calls += 1
        self._turn_counter += 1
        idx = self._turn_counter - 1
        delay = self._delays[idx] if idx < len(self._delays) else 0.0
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            if self._turn_counter > len(self._responses):
                raise IndexError("FakeAgentRuntime out of responses")
            base = self._responses[self._turn_counter - 1]
            return dc_replace(base, turn_number=self._turn_counter)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise

    def set_image_mapping(self, mapping: ImageMapping | None) -> None:
        pass


class FakeAXResolver:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []

    def resolve_at(self, x_pt: float, y_pt: float) -> AXTarget | None:
        self.calls.append((x_pt, y_pt))
        return None


def _response(
    blocks: list[dict[str, Any]],
    stop_reason: str = "tool_use",
    *,
    input_tokens: int = 50,
    output_tokens: int = 10,
) -> AgentResponse:
    return AgentResponse(
        content_blocks=blocks,
        stop_reason=stop_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        turn_number=0,
    )


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


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


@pytest.fixture
def trajectory_root(tmp_path: Path) -> Path:
    traj = tmp_path / "trajectory_root"
    screenshots = traj / "run_traj" / "screenshots"
    screenshots.mkdir(parents=True)
    png = blank_canvas_png()
    for i in range(1, 21):
        (screenshots / f"{i:04d}.png").write_bytes(png)
    return traj


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    return root


def _build(
    *,
    runs_root: Path,
    trajectory_root: Path,
    slug: str,
    responses: list[AgentResponse],
    kill_switch: KillSwitch,
    delay_seconds: float | list[float] = 0.0,
    mode: RunMode = "dry_run",
    run_id: str | None = None,
) -> tuple[Executor, FakeAgentRuntime, DryRunInputAdapter, ConfirmationQueue, RunWriter, str]:
    run_id_str = run_id or str(uuid.uuid4())
    loaded = load_skill(slug, _FIXTURES_ROOT)
    adapter = DryRunInputAdapter()
    source = TrajectoryScreenSource("run_traj", trajectories_root=trajectory_root)
    runtime = FakeAgentRuntime(responses, delay_seconds=delay_seconds)
    writer = RunWriter(
        run_id=run_id_str, skill_slug=slug, mode=mode, runs_root=runs_root
    )
    queue = ConfirmationQueue()
    tracker = BudgetTracker(budget=RunBudget(), time_source=lambda: 0.0)
    executor = Executor(
        loaded_skill=loaded,
        parameters={},
        mode=mode,
        agent_runtime=runtime,
        input_adapter=adapter,
        screen_source=source,
        ax_resolver=FakeAXResolver(),
        budget=tracker,
        writer=writer,
        confirmation_queue=queue,
        run_id=run_id_str,
        confirmation_timeout_seconds=5.0,
        kill_switch=kill_switch,
    )
    return executor, runtime, adapter, queue, writer, run_id_str


# ---------- End-to-end executor integration ----------


async def test_kill_during_slow_turn_aborts_within_2_seconds(
    runs_root: Path, trajectory_root: Path
) -> None:
    """Kill fires mid-API-call. The long turn is cancelled; status=aborted in <2s."""

    ks = KillSwitch()
    # The fake runtime will block for 10s unless cancelled.
    responses = [
        _response([_screenshot_block("t1")]),
        _response([_text_block("done <workflow_complete/>")], stop_reason="end_turn"),
    ]
    executor, runtime, _adapter, _queue, writer, run_id = _build(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        responses=responses,
        kill_switch=ks,
        delay_seconds=10.0,
    )

    async def kill_soon() -> None:
        # Wait for the turn to actually be in-flight.
        while runtime.calls == 0:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        ks.kill(run_id, reason="user_abort")

    start = time.monotonic()
    async with asyncio.TaskGroup() as tg:
        tg.create_task(kill_soon())
        run_task = tg.create_task(executor.run())
    elapsed = time.monotonic() - start

    result = run_task.result()
    assert result.status == "aborted"
    assert result.abort_reason == "user_abort"
    assert elapsed < 2.0, f"kill took {elapsed:.2f}s, must be < 2s"
    assert runtime.cancelled >= 1, "in-flight turn task was not cancelled"
    # Disk reflects the abort.
    meta_path = writer.run_dir / "run_metadata.json"
    assert '"status": "aborted"' in meta_path.read_text()


async def test_kill_while_awaiting_confirmation_aborts(
    runs_root: Path, trajectory_root: Path
) -> None:
    """Kill during a pending confirmation → abort decision delivered, status=aborted."""

    ks = KillSwitch()
    responses = [
        _response([_screenshot_block("t1")]),
        _response(
            [_text_block('ready <needs_confirmation step="4"/>')],
            stop_reason="end_turn",
        ),
        _response(
            [_text_block("done <workflow_complete/>")], stop_reason="end_turn"
        ),
    ]
    executor, _runtime, _adapter, queue, _writer, run_id = _build(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        responses=responses,
        kill_switch=ks,
    )

    async def kill_when_pending() -> None:
        while not queue.has_pending(run_id):
            await asyncio.sleep(0)
        ks.kill(run_id, reason="kill_switch")

    async with asyncio.TaskGroup() as tg:
        tg.create_task(kill_when_pending())
        run_task = tg.create_task(executor.run())

    result = run_task.result()
    assert result.status == "aborted"
    assert result.abort_reason == "kill_switch"


async def test_kill_before_start_is_noop(
    runs_root: Path, trajectory_root: Path
) -> None:
    """kill() for a run that has not yet registered is a no-op — the run completes normally."""

    ks = KillSwitch()
    # Kill a never-registered run_id; must not raise.
    assert ks.kill("nonexistent") is False

    responses = [
        _response([_screenshot_block("t1")]),
        _response([_text_block("done <workflow_complete/>")], stop_reason="end_turn"),
    ]
    executor, _runtime, _adapter, _queue, _writer, _rid = _build(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        responses=responses,
        kill_switch=ks,
    )
    result = await executor.run()
    assert result.status == "succeeded"


async def test_kill_after_finish_is_noop(
    runs_root: Path, trajectory_root: Path
) -> None:
    """kill() after the executor has unregistered the run does nothing."""

    ks = KillSwitch()
    responses = [
        _response([_screenshot_block("t1")]),
        _response([_text_block("done <workflow_complete/>")], stop_reason="end_turn"),
    ]
    executor, _runtime, _adapter, _queue, _writer, run_id = _build(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        responses=responses,
        kill_switch=ks,
    )
    result = await executor.run()
    assert result.status == "succeeded"
    # After the run, kill is a no-op.
    assert ks.kill(run_id) is False
    assert not ks.is_registered(run_id)


async def test_kill_is_idempotent_during_run(
    runs_root: Path, trajectory_root: Path
) -> None:
    """Second kill() doesn't double-transition the run's final state."""

    ks = KillSwitch()
    responses = [
        _response([_screenshot_block("t1")]),
        _response([_text_block("done <workflow_complete/>")], stop_reason="end_turn"),
    ]
    executor, runtime, _adapter, _queue, _writer, run_id = _build(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        responses=responses,
        kill_switch=ks,
        delay_seconds=2.0,
    )

    async def double_kill() -> None:
        while runtime.calls == 0:
            await asyncio.sleep(0.01)
        assert ks.kill(run_id) is True
        assert ks.kill(run_id) is False

    async with asyncio.TaskGroup() as tg:
        tg.create_task(double_kill())
        task = tg.create_task(executor.run())

    result = task.result()
    assert result.status == "aborted"


async def test_no_further_tool_calls_dispatched_after_kill(
    runs_root: Path, trajectory_root: Path
) -> None:
    """Once killed, the executor must not invoke the input adapter again.

    Turn 1 is a screenshot (no adapter call), turn 2 is a slow click turn.
    We fire the kill partway through turn 2 — the turn task is cancelled
    before parse/dispatch, so the click must never reach the adapter.
    """

    ks = KillSwitch()
    responses = [
        _response([_screenshot_block("t1")]),
        _response([_click_block("t2")]),
        _response([_text_block("done <workflow_complete/>")], stop_reason="end_turn"),
    ]
    executor, runtime, adapter, _queue, _writer, run_id = _build(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        responses=responses,
        kill_switch=ks,
        delay_seconds=[0.0, 5.0, 0.0],
    )

    async def kill_during_second_turn() -> None:
        # Wait until turn 2 is in-flight (turn 1 has returned, turn 2 started).
        while runtime.calls < 2:
            await asyncio.sleep(0.01)
        ks.kill(run_id)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(kill_during_second_turn())
        task = tg.create_task(executor.run())

    result = task.result()
    assert result.status == "aborted"
    # The click (turn 2) must not have been dispatched.
    action_names = [call[0] for call in adapter.get_recorded_calls()]
    assert "click" not in action_names
    assert runtime.cancelled >= 1


async def test_default_kill_reason_used_when_none_provided(
    runs_root: Path, trajectory_root: Path
) -> None:
    """kill() with no reason → abort_reason defaults to KILL_SWITCH_REASON."""

    ks = KillSwitch()
    responses = [
        _response([_screenshot_block("t1")]),
        _response([_text_block("done <workflow_complete/>")], stop_reason="end_turn"),
    ]
    executor, runtime, _adapter, _queue, _writer, run_id = _build(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        responses=responses,
        kill_switch=ks,
        delay_seconds=2.0,
    )

    async def kill_soon() -> None:
        while runtime.calls == 0:
            await asyncio.sleep(0.01)
        ks.kill(run_id)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(kill_soon())
        task = tg.create_task(executor.run())

    result = task.result()
    assert result.status == "aborted"
    assert result.abort_reason == KILL_SWITCH_REASON


async def test_executor_without_kill_switch_still_works(
    runs_root: Path, trajectory_root: Path
) -> None:
    """Regression: kill_switch=None preserves existing executor behavior."""

    responses = [
        _response([_screenshot_block("t1")]),
        _response([_text_block("done <workflow_complete/>")], stop_reason="end_turn"),
    ]
    run_id_str = str(uuid.uuid4())
    loaded = load_skill("notes_daily", _FIXTURES_ROOT)
    writer = RunWriter(
        run_id=run_id_str, skill_slug="notes_daily", mode="dry_run", runs_root=runs_root
    )
    queue = ConfirmationQueue()
    tracker = BudgetTracker(budget=RunBudget(), time_source=lambda: 0.0)
    executor = Executor(
        loaded_skill=loaded,
        parameters={},
        mode="dry_run",
        agent_runtime=FakeAgentRuntime(responses),
        input_adapter=DryRunInputAdapter(),
        screen_source=TrajectoryScreenSource("run_traj", trajectories_root=trajectory_root),
        ax_resolver=FakeAXResolver(),
        budget=tracker,
        writer=writer,
        confirmation_queue=queue,
        run_id=run_id_str,
    )
    result = await executor.run()
    assert result.status == "succeeded"


# ---------- Real httpx cancellation via ClaudeRuntime + respx ----------


async def test_kill_cancels_inflight_httpx_request(
    runs_root: Path, trajectory_root: Path, tmp_path: Path
) -> None:
    """The AsyncAnthropic call is cancellable mid-flight.

    Exercises :class:`ClaudeRuntime` (not the fake) so the ``asyncio.cancel
    → httpx request cancellation`` path is covered. We patch the SDK's
    ``messages.create`` to a coroutine that sleeps forever; when the kill
    switch fires, the turn task is cancelled and the patched coroutine
    surfaces ``asyncio.CancelledError`` — the exact mechanism that cancels a
    real ``httpx`` socket read.
    """

    ks = KillSwitch()
    request_started = asyncio.Event()
    cancelled_count = 0

    async def sleep_forever(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal cancelled_count
        request_started.set()
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            cancelled_count += 1
            raise
        raise RuntimeError("sleep_forever should have been cancelled")  # pragma: no cover

    run_id_str = str(uuid.uuid4())
    loaded = load_skill("notes_daily", _FIXTURES_ROOT)
    writer = RunWriter(
        run_id=run_id_str,
        skill_slug="notes_daily",
        mode="dry_run",
        runs_root=runs_root,
    )
    queue = ConfirmationQueue()
    tracker = BudgetTracker(budget=RunBudget(), time_source=lambda: 0.0)
    runtime = ClaudeRuntime(
        run_id=run_id_str,
        costs_path=tmp_path / "costs.jsonl",
        api_key="test-key",
    )
    assert runtime._client is not None
    runtime._client.messages.create = sleep_forever  # type: ignore[method-assign]

    executor = Executor(
        loaded_skill=loaded,
        parameters={},
        mode="dry_run",
        agent_runtime=runtime,
        input_adapter=DryRunInputAdapter(),
        screen_source=TrajectoryScreenSource(
            "run_traj", trajectories_root=trajectory_root
        ),
        ax_resolver=FakeAXResolver(),
        budget=tracker,
        writer=writer,
        confirmation_queue=queue,
        run_id=run_id_str,
        kill_switch=ks,
    )

    async def kill_soon() -> None:
        await request_started.wait()
        ks.kill(run_id_str, reason="user_abort")

    start = time.monotonic()
    async with asyncio.TaskGroup() as tg:
        tg.create_task(kill_soon())
        task = tg.create_task(executor.run())
    elapsed = time.monotonic() - start

    result = task.result()
    assert result.status == "aborted"
    assert result.abort_reason == "user_abort"
    assert elapsed < 2.0, f"kill took {elapsed:.2f}s"
    assert cancelled_count >= 1, "in-flight SDK coroutine was not cancelled"


# ---------- API endpoint ----------


async def test_api_abort_endpoint_calls_global_kill_switch() -> None:
    """POST /run/{run_id}/abort flips the global kill switch for that run_id."""

    switch = get_global_kill_switch()
    run_id = f"api-test-{uuid.uuid4()}"
    switch.register(run_id)
    try:
        client = TestClient(app)
        resp = client.post(f"/run/{run_id}/abort")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == run_id
        assert body["killed"] is True
        assert switch.is_killed(run_id)
    finally:
        switch.unregister(run_id)


async def test_api_abort_endpoint_unknown_run_id_returns_killed_false() -> None:
    """Unknown run_id → 200 with killed=False; no error raised."""

    client = TestClient(app)
    resp = client.post(f"/run/unknown-{uuid.uuid4()}/abort")
    assert resp.status_code == 200
    assert resp.json()["killed"] is False
