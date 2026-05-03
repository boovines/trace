"""Tests for the main execution loop (X-017).

Covers the happy path for each reference skill, confirmation/abort handling
(both parser-level and pre-action-gate-level), budget breach, agent_stuck
recovery, and the dry-run/execute mode split. Every test uses fake
collaborators — no real LLM, no real input, no real screen. The dispatcher
itself is real so the full tool-call → adapter → screen-source path is
exercised end-to-end.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import pytest

from runner.agent_runtime import AgentResponse, AgentRuntime
from runner.budget import BudgetTracker, RunBudget
from runner.confirmation import ConfirmationDecision, ConfirmationQueue
from runner.coords import ImageMapping
from runner.executor import (
    AGENT_STUCK_REASON,
    Executor,
)
from runner.input_adapter import DryRunInputAdapter
from runner.pre_action_gate import AXTarget
from runner.run_writer import RunWriter
from runner.schema import RunMode
from runner.screen_source import TrajectoryScreenSource, blank_canvas_png
from runner.skill_loader import load_skill

pytestmark = pytest.mark.asyncio

_FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "skills"


# ---------- fakes ----------


class FakeAgentRuntime:
    """Minimal ``AgentRuntime`` that serves canned turns from a list.

    Tracks ``set_image_mapping`` calls so assertions can confirm the
    executor is propagating the mapping after each screenshot. Turn counter
    resets on construction so each test starts fresh.
    """

    def __init__(self, responses: list[AgentResponse]) -> None:
        self._responses = list(responses)
        self._turn_counter = 0
        self.image_mappings_set: list[ImageMapping | None] = []
        self.system_prompts: list[str] = []

    async def run_turn(
        self,
        system_prompt: str,
        messages: list[Any],
        max_tokens: int = 4096,
    ) -> AgentResponse:
        self._turn_counter += 1
        self.system_prompts.append(system_prompt)
        if self._turn_counter > len(self._responses):
            raise IndexError(
                f"FakeAgentRuntime ran out of responses at turn "
                f"{self._turn_counter}"
            )
        base = self._responses[self._turn_counter - 1]
        return dc_replace(base, turn_number=self._turn_counter)

    def set_image_mapping(self, mapping: ImageMapping | None) -> None:
        self.image_mappings_set.append(mapping)


class FakeAXResolver:
    """Maps display-point coordinates to AX targets via a list of rules."""

    def __init__(self, rules: list[tuple[tuple[float, float], AXTarget]] | None = None) -> None:
        self.rules = rules or []
        self.calls: list[tuple[float, float]] = []

    def resolve_at(self, x_pt: float, y_pt: float) -> AXTarget | None:
        self.calls.append((x_pt, y_pt))
        for ((rx, ry), target) in self.rules:
            if abs(rx - x_pt) < 5.0 and abs(ry - y_pt) < 5.0:
                return target
        return None


# ---------- helpers ----------


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


def _response(
    content_blocks: list[dict[str, Any]],
    stop_reason: str = "tool_use",
    *,
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> AgentResponse:
    return AgentResponse(
        content_blocks=content_blocks,
        stop_reason=stop_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        turn_number=0,
    )


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


def _build_executor(
    *,
    runs_root: Path,
    trajectory_root: Path,
    slug: str,
    parameters: dict[str, str],
    responses: list[AgentResponse],
    ax_rules: list[tuple[tuple[float, float], AXTarget]] | None = None,
    budget: RunBudget | None = None,
    mode: RunMode = "dry_run",
    time_source: Any = None,
    run_id: str | None = None,
) -> tuple[
    Executor,
    FakeAgentRuntime,
    DryRunInputAdapter,
    ConfirmationQueue,
    FakeAXResolver,
    RunWriter,
    str,
]:
    run_id_str = run_id or str(uuid.uuid4())
    loaded = load_skill(slug, _FIXTURES_ROOT)
    adapter = DryRunInputAdapter()
    source = TrajectoryScreenSource("run_traj", trajectories_root=trajectory_root)
    resolver = FakeAXResolver(ax_rules)
    runtime = FakeAgentRuntime(responses)
    writer = RunWriter(
        run_id=run_id_str, skill_slug=slug, mode=mode, runs_root=runs_root
    )
    queue = ConfirmationQueue()
    tracker = BudgetTracker(
        budget=budget or RunBudget(),
        time_source=time_source or (lambda: 0.0),
    )
    executor = Executor(
        loaded_skill=loaded,
        parameters=parameters,
        mode=mode,
        agent_runtime=runtime,
        input_adapter=adapter,
        screen_source=source,
        ax_resolver=resolver,
        budget=tracker,
        writer=writer,
        confirmation_queue=queue,
        run_id=run_id_str,
        confirmation_timeout_seconds=5.0,
    )
    return executor, runtime, adapter, queue, resolver, writer, run_id_str


def _skill_params(slug: str) -> dict[str, str]:
    return {
        "gmail_reply": {"recipient_name": "Alice", "reply_body": "Thanks!"},
        "calendar_block": {},
        "finder_organize": {},
        "slack_status": {},
        # Canonical notes_daily declares ``note_template`` as required.
        "notes_daily": {"note_template": "- [ ] focus block\n"},
    }[slug]


def _read_metadata(writer: RunWriter) -> dict[str, Any]:
    with (writer.run_dir / "run_metadata.json").open() as f:
        return json.loads(f.read())


def _read_events(writer: RunWriter) -> list[dict[str, Any]]:
    path = writer.run_dir / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# ---------- happy-path fixtures-driven coverage ----------


def _script_for_non_destructive_skill() -> list[AgentResponse]:
    """Three turns: screenshot → click → workflow_complete."""
    return [
        _response([_screenshot_block("t1")], stop_reason="tool_use"),
        _response([_click_block("t2")], stop_reason="tool_use"),
        _response([_text_block("done <workflow_complete/>")], stop_reason="end_turn"),
    ]


def _script_for_destructive_skill(step: int) -> list[AgentResponse]:
    """Four turns: screenshot → click → needs_confirmation → workflow_complete."""
    return [
        _response([_screenshot_block("t1")], stop_reason="tool_use"),
        _response([_click_block("t2")], stop_reason="tool_use"),
        _response(
            [_text_block(f"ready <needs_confirmation step=\"{step}\"/>")],
            stop_reason="end_turn",
        ),
        _response(
            [_text_block("done <workflow_complete/>")], stop_reason="end_turn"
        ),
    ]


async def test_happy_path_notes_daily_succeeds(
    runs_root: Path, trajectory_root: Path
) -> None:
    """notes_daily has no destructive step; runs to completion without pauses."""
    executor, _runtime, adapter, _queue, _ax, writer, _rid = _build_executor(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        parameters=_skill_params("notes_daily"),
        responses=_script_for_non_destructive_skill(),
    )
    result = await executor.run()
    assert result.status == "succeeded"
    assert result.final_step_reached == 4  # notes_daily has 4 steps
    meta = _read_metadata(writer)
    assert meta["status"] == "succeeded"
    # Screenshot + click were both dispatched.
    actions = [call[0] for call in adapter.get_recorded_calls()]
    assert actions == ["click"]  # screenshot doesn't go through adapter


@pytest.mark.parametrize(
    ("slug", "destructive_step"),
    [
        ("gmail_reply", 7),
        ("calendar_block", 7),
        ("finder_organize", 5),
        ("slack_status", 7),
    ],
)
async def test_happy_path_destructive_skills_confirmed(
    slug: str,
    destructive_step: int,
    runs_root: Path,
    trajectory_root: Path,
) -> None:
    """Each destructive reference skill pauses, confirms, completes."""
    executor, _runtime, _adapter, queue, _ax, writer, run_id = _build_executor(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug=slug,
        parameters=_skill_params(slug),
        responses=_script_for_destructive_skill(destructive_step),
    )

    async def confirm_when_pending() -> None:
        while not queue.has_pending(run_id):
            await asyncio.sleep(0)
        assert queue.submit_decision(
            run_id, ConfirmationDecision(action="confirm")
        )

    async with asyncio.TaskGroup() as tg:
        tg.create_task(confirm_when_pending())
        run_task = tg.create_task(executor.run())

    result = run_task.result()
    assert result.status == "succeeded"
    assert result.confirmation_count == 1
    assert result.destructive_actions_executed == [destructive_step]
    assert result.final_step_reached is not None
    meta = _read_metadata(writer)
    assert meta["status"] == "succeeded"


# ---------- confirmation flow ----------


async def test_confirmation_flow_confirm_proceeds(
    runs_root: Path, trajectory_root: Path
) -> None:
    executor, _runtime, _adapter, queue, _ax, writer, run_id = _build_executor(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="gmail_reply",
        parameters=_skill_params("gmail_reply"),
        responses=_script_for_destructive_skill(7),
    )

    async def confirm() -> None:
        while not queue.has_pending(run_id):
            await asyncio.sleep(0)
        queue.submit_decision(run_id, ConfirmationDecision(action="confirm"))

    async with asyncio.TaskGroup() as tg:
        tg.create_task(confirm())
        task = tg.create_task(executor.run())

    result = task.result()
    assert result.status == "succeeded"
    events = _read_events(writer)
    event_types = [e["type"] for e in events]
    assert "confirmation_requested" in event_types
    assert "confirmed" in event_types


async def test_abort_flow_ends_with_status_aborted(
    runs_root: Path, trajectory_root: Path
) -> None:
    executor, _runtime, _adapter, queue, _ax, writer, run_id = _build_executor(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="gmail_reply",
        parameters=_skill_params("gmail_reply"),
        responses=_script_for_destructive_skill(7),
    )

    async def abort() -> None:
        while not queue.has_pending(run_id):
            await asyncio.sleep(0)
        queue.submit_decision(
            run_id, ConfirmationDecision(action="abort", reason="user_abort")
        )

    async with asyncio.TaskGroup() as tg:
        tg.create_task(abort())
        task = tg.create_task(executor.run())

    result = task.result()
    assert result.status == "aborted"
    assert result.abort_reason == "user_abort"
    # The destructive step was NOT executed; the list is empty.
    assert result.destructive_actions_executed == []
    events = _read_events(writer)
    assert any(e["type"] == "aborted" for e in events)


# ---------- budget breach ----------


async def test_budget_breach_by_excessive_total_actions(
    runs_root: Path, trajectory_root: Path
) -> None:
    """A skill-level cap of max_total_actions=2 trips on the third dispatch."""
    budget = RunBudget(max_total_actions=2)
    responses = [
        _response([_screenshot_block(f"t{i}")], stop_reason="tool_use")
        for i in range(1, 10)
    ]
    executor, _runtime, _adapter, _queue, _ax, _writer, _rid = _build_executor(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        parameters=_skill_params("notes_daily"),
        responses=responses,
        budget=budget,
    )
    result = await executor.run()
    assert result.status == "budget_exceeded"
    assert result.abort_reason is not None
    assert "actions" in result.abort_reason


async def test_budget_breach_by_token_cap(
    runs_root: Path, trajectory_root: Path
) -> None:
    """A token cap of 50 trips on the very first turn (100 input tokens)."""
    budget = RunBudget(max_input_tokens=50)
    responses = [_response([_screenshot_block("t1")], stop_reason="tool_use")]
    executor, _runtime, _adapter, _queue, _ax, _writer, _rid = _build_executor(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        parameters=_skill_params("notes_daily"),
        responses=responses,
        budget=budget,
    )
    result = await executor.run()
    assert result.status == "budget_exceeded"
    assert "input_tokens" in (result.abort_reason or "")


# ---------- agent_stuck ----------


async def test_agent_stuck_after_four_unknown_turns(
    runs_root: Path, trajectory_root: Path
) -> None:
    """Four plain-text turns in a row → status=failed reason='agent_stuck'."""
    responses = [
        _response([_text_block(f"just narrating turn {i}")], stop_reason="end_turn")
        for i in range(1, 6)
    ]
    executor, _runtime, _adapter, _queue, _ax, _writer, _rid = _build_executor(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        parameters=_skill_params("notes_daily"),
        responses=responses,
    )
    result = await executor.run()
    assert result.status == "failed"
    assert result.error_message == AGENT_STUCK_REASON
    assert result.abort_reason == AGENT_STUCK_REASON


# ---------- pre-action gate ----------


async def test_pre_action_gate_fires_on_unflagged_destructive_click(
    runs_root: Path, trajectory_root: Path
) -> None:
    """The skill didn't mark the step destructive, but the gate catches a 'Send' button."""
    # After turn 1's screenshot, the trajectory source updates the mapping to
    # the 1440x900 blank canvas (which falls under the 1568 resize cap, so
    # scale_from_resized_to_points = (1440/1440)/2.0 = 0.5). A resized click at
    # [100, 100] therefore maps to display points (50.0, 50.0) — the AX rule is
    # registered at that location so the gate fires on the 'Send' button.
    responses = [
        _response([_screenshot_block("t1")], stop_reason="tool_use"),
        _response([_click_block("t2", x=100, y=100)], stop_reason="tool_use"),
        _response([_text_block("unexpected <workflow_complete/>")], stop_reason="end_turn"),
    ]
    # notes_daily is non-destructive; we run in execute mode so the gate fires.
    send_target = AXTarget(role="AXButton", label="Send")
    executor, _runtime, adapter, queue, _ax, writer, run_id = _build_executor(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        parameters=_skill_params("notes_daily"),
        responses=responses,
        ax_rules=[((50.0, 50.0), send_target)],
        mode="execute",
    )

    async def abort_when_pending() -> None:
        while not queue.has_pending(run_id):
            await asyncio.sleep(0)
        queue.submit_decision(
            run_id, ConfirmationDecision(action="abort", reason="user_abort")
        )

    async with asyncio.TaskGroup() as tg:
        tg.create_task(abort_when_pending())
        task = tg.create_task(executor.run())

    result = task.result()
    assert result.status == "aborted"
    # Assert the click was NOT dispatched (the adapter should only have
    # recorded zero click calls — the confirmation halted it).
    calls = [c[0] for c in adapter.get_recorded_calls()]
    assert "click" not in calls
    events = _read_events(writer)
    assert any("pre_action_gate" in e.get("message", "") for e in events)


async def test_dry_run_mode_skips_pre_action_gate(
    runs_root: Path, trajectory_root: Path
) -> None:
    """Same scenario as above but mode=dry_run → gate short-circuits → click dispatched."""
    responses = _script_for_non_destructive_skill()
    send_target = AXTarget(role="AXButton", label="Send")
    executor, _runtime, adapter, _queue, resolver, _writer, _rid = _build_executor(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        parameters=_skill_params("notes_daily"),
        responses=responses,
        ax_rules=[((50.0, 50.0), send_target)],
        mode="dry_run",
    )
    result = await executor.run()
    assert result.status == "succeeded"
    # The gate was short-circuited in dry_run — resolver was never called.
    assert resolver.calls == []
    # And the click was dispatched normally.
    assert any(c[0] == "click" for c in adapter.get_recorded_calls())


# ---------- mode split: execute vs dry_run share the same executor ----------


async def test_execute_mode_writes_live_prompt_and_dry_run_writes_dry_run(
    runs_root: Path, trajectory_root: Path
) -> None:
    """The prompt differs by mode; otherwise the executor is identical."""
    responses = _script_for_non_destructive_skill()
    ex_dry, rt_dry, *_ = _build_executor(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        parameters=_skill_params("notes_daily"),
        responses=responses,
        mode="dry_run",
    )
    await ex_dry.run()
    assert any("DRY RUN" in p for p in rt_dry.system_prompts)

    responses2 = _script_for_non_destructive_skill()
    ex_live, rt_live, *_ = _build_executor(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        parameters=_skill_params("notes_daily"),
        responses=responses2,
        mode="execute",
        run_id=str(uuid.uuid4()),
    )
    await ex_live.run()
    assert all("DRY RUN" not in p for p in rt_live.system_prompts)


# ---------- image mapping propagation ----------


async def test_image_mapping_propagated_after_each_screenshot(
    runs_root: Path, trajectory_root: Path
) -> None:
    responses = [
        _response([_screenshot_block("t1")], stop_reason="tool_use"),
        _response([_text_block("done <workflow_complete/>")], stop_reason="end_turn"),
    ]
    executor, runtime, _adapter, _queue, _ax, _writer, _rid = _build_executor(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        parameters=_skill_params("notes_daily"),
        responses=responses,
    )
    await executor.run()
    # Once at start + once after the screenshot dispatch.
    assert len(runtime.image_mappings_set) >= 2
    # The post-screenshot mapping is not None.
    assert runtime.image_mappings_set[-1] is not None


# ---------- transcript + screenshots on disk ----------


async def test_transcript_and_screenshot_persisted(
    runs_root: Path, trajectory_root: Path
) -> None:
    responses = [
        _response([_screenshot_block("t1")], stop_reason="tool_use"),
        _response([_text_block("done <workflow_complete/>")], stop_reason="end_turn"),
    ]
    executor, _runtime, _adapter, _queue, _ax, writer, _rid = _build_executor(
        runs_root=runs_root,
        trajectory_root=trajectory_root,
        slug="notes_daily",
        parameters=_skill_params("notes_daily"),
        responses=responses,
    )
    await executor.run()
    transcript = (writer.run_dir / "transcript.jsonl").read_text().splitlines()
    assert len(transcript) == 2
    # Screenshot persisted after the first turn's dispatch.
    screenshots = sorted((writer.run_dir / "screenshots").glob("*.png"))
    assert len(screenshots) == 1
    assert screenshots[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


# ---------- AgentRuntime protocol still passes isinstance ----------


async def test_fake_runtime_satisfies_protocol() -> None:
    runtime: AgentRuntime = FakeAgentRuntime([])
    assert isinstance(runtime, AgentRuntime)
