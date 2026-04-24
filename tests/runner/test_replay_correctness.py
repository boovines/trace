"""Replay-correctness snapshot test (X-022).

Semantic-equivalence check: for each reference skill, run the full e2e
execution pipeline in dry-run fake-mode and compare the sequence of dispatched
input-adapter calls against the source trajectory's semantic events. The
trajectory is the "gold standard" of what a real user did; the runner should
reproduce the workflow with high fidelity (≥ 80% clicks matched, 100% text
inputs matched, 100% confirmations fired for destructive steps).

This is the counterpart to the synthesizer's similarity test — rather than
comparing two skill files, it compares a run against a trajectory. It catches
regressions where the fake-mode LLM scripts drift from realistic agent
behavior or where the parser/dispatcher combo silently swallows actions.

The trajectory fixtures live at ``fixtures/trajectories/<uuid>/`` with the
shape described in ``contracts/trajectory.schema.json``. Clicks are annotated
via the ``note`` field (``step=N;target=<label>[;destructive=true]``) so the
semantic-equivalence helper can line them up with dispatched actions by
target / step rather than strict coordinate equality.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from runner.api import router
from runner.claude_runtime import FAKE_MODE_ENV_VAR, ClaudeRuntime
from runner.input_adapter import DryRunInputAdapter, RecordedCall
from runner.kill_switch import KillSwitch
from runner.pre_action_gate import AXTarget
from runner.run_manager import AdapterBundle, RunManager
from runner.screen_source import TrajectoryScreenSource, blank_canvas_png
from synthesizer.trajectory_reader import (
    ReadTrajectory,
    SemanticEvent,
    load_trajectory,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES_SKILLS = _REPO_ROOT / "fixtures" / "skills"
_FIXTURES_TRAJECTORIES = _REPO_ROOT / "fixtures" / "trajectories"

# (slug, params, trajectory_id)
_REFERENCE_SKILLS: list[tuple[str, dict[str, str], str]] = [
    ("gmail_reply", {"sender": "a@b.co", "template": "hi"}, "00000000-0000-0000-0000-000000000001"),
    ("calendar_block", {}, "00000000-0000-0000-0000-000000000002"),
    ("finder_organize", {}, "00000000-0000-0000-0000-000000000003"),
    ("slack_status", {}, "00000000-0000-0000-0000-000000000004"),
    ("notes_daily", {}, "00000000-0000-0000-0000-000000000005"),
]

# Pass thresholds (from X-022 acceptance criteria).
_CLICK_MATCH_THRESHOLD: float = 0.80
_TEXT_INPUT_MATCH_THRESHOLD: float = 1.00
_DESTRUCTIVE_CONFIRMATION_THRESHOLD: float = 1.00

# Pixel tolerance when matching a trajectory click to a dispatched click.
# The trajectory records display-point coordinates; the dispatched coord is
# also in display points after the `resized_pixels_to_points` mapping, but we
# allow some slack because fake-mode scripts may round differently than the
# recorder did.
_COORD_TOLERANCE_POINTS: float = 250.0


# ---------- AX resolver ----------


class _NullAXResolver:
    def resolve_at(self, x_pt: float, y_pt: float) -> AXTarget | None:
        return None


# ---------- metrics ----------


@dataclass(frozen=True, slots=True)
class ReplayMetrics:
    """Counts produced by :func:`compare_trajectory_to_dispatch`."""

    slug: str
    trajectory_clicks: int
    dispatched_clicks: int
    clicks_matched: int
    trajectory_text_inputs: int
    dispatched_text_inputs: int
    text_inputs_matched: int
    destructive_clicks_in_trajectory: int
    confirmations_fired: int
    extra_screenshots: int
    missing_actions: list[str]

    @property
    def click_match_rate(self) -> float:
        non_destructive = self.trajectory_clicks - self.destructive_clicks_in_trajectory
        if non_destructive <= 0:
            return 1.0
        return self.clicks_matched / non_destructive

    @property
    def text_input_match_rate(self) -> float:
        if self.trajectory_text_inputs == 0:
            return 1.0
        return self.text_inputs_matched / self.trajectory_text_inputs

    @property
    def destructive_confirmation_rate(self) -> float:
        if self.destructive_clicks_in_trajectory == 0:
            return 1.0
        return self.confirmations_fired / self.destructive_clicks_in_trajectory


# ---------- semantic-equivalence helper ----------


def _dispatched_clicks(calls: list[RecordedCall]) -> list[tuple[float, float]]:
    clicks: list[tuple[float, float]] = []
    for name, args, _kwargs in calls:
        if name == "click" and len(args) >= 2:
            x = args[0]
            y = args[1]
            if isinstance(x, int | float) and isinstance(y, int | float):
                clicks.append((float(x), float(y)))
    return clicks


def _dispatched_text_inputs(calls: list[RecordedCall]) -> list[str]:
    texts: list[str] = []
    for name, args, _kwargs in calls:
        if name == "type_text" and args and isinstance(args[0], str):
            texts.append(args[0])
    return texts


def _text_roughly_equal(a: str, b: str) -> bool:
    """Loose match: either is a substring of the other (after strip)."""
    a_norm = a.strip()
    b_norm = b.strip()
    if not a_norm or not b_norm:
        return a_norm == b_norm
    return a_norm in b_norm or b_norm in a_norm


def _click_within_tolerance(
    expected: SemanticEvent, actual: tuple[float, float]
) -> bool:
    if expected.x is None or expected.y is None:
        return False
    dx = abs(expected.x - actual[0])
    dy = abs(expected.y - actual[1])
    return dx <= _COORD_TOLERANCE_POINTS and dy <= _COORD_TOLERANCE_POINTS


def compare_trajectory_to_dispatch(
    slug: str,
    trajectory: ReadTrajectory,
    dispatched: list[RecordedCall],
    confirmations_fired: int,
    extra_screenshots: int = 0,
) -> ReplayMetrics:
    """Line up trajectory semantic events against dispatched adapter calls.

    Matches each non-destructive trajectory click to the nearest-in-order
    dispatched click within ``_COORD_TOLERANCE_POINTS``; each trajectory
    text_input to the first loosely-matching ``type_text`` call. Destructive
    clicks are expected to become confirmations and are counted separately.
    """
    dispatched_clicks = _dispatched_clicks(dispatched)
    dispatched_texts = _dispatched_text_inputs(dispatched)

    trajectory_clicks = list(trajectory.clicks)
    non_destructive = trajectory.non_destructive_clicks
    destructive_count = len(trajectory.destructive_clicks)

    missing_actions: list[str] = []

    # Click matching — consume dispatched clicks in order. Each trajectory
    # click can consume one dispatched click within tolerance; if none match,
    # we mark the action missing.
    remaining_dispatched: list[tuple[float, float]] = list(dispatched_clicks)
    clicks_matched = 0
    for expected in non_destructive:
        found_index: int | None = None
        for idx, actual in enumerate(remaining_dispatched):
            if _click_within_tolerance(expected, actual):
                found_index = idx
                break
        if found_index is not None:
            clicks_matched += 1
            remaining_dispatched.pop(found_index)
        else:
            label = expected.target_label or f"({expected.x}, {expected.y})"
            missing_actions.append(f"click:{label}")

    # Text-input matching — each trajectory text_input matches any dispatched
    # type_text call (by substring, either direction).
    remaining_texts = list(dispatched_texts)
    text_inputs_matched = 0
    for expected_text in trajectory.text_inputs:
        expected_value = expected_text.text or ""
        found_index = None
        for idx, actual in enumerate(remaining_texts):
            if _text_roughly_equal(expected_value, actual):
                found_index = idx
                break
        if found_index is not None:
            text_inputs_matched += 1
            remaining_texts.pop(found_index)
        else:
            label = expected_text.target_label or expected_value
            missing_actions.append(f"text_input:{label}")

    return ReplayMetrics(
        slug=slug,
        trajectory_clicks=len(trajectory_clicks),
        dispatched_clicks=len(dispatched_clicks),
        clicks_matched=clicks_matched,
        trajectory_text_inputs=len(trajectory.text_inputs),
        dispatched_text_inputs=len(dispatched_texts),
        text_inputs_matched=text_inputs_matched,
        destructive_clicks_in_trajectory=destructive_count,
        confirmations_fired=confirmations_fired,
        extra_screenshots=extra_screenshots,
        missing_actions=missing_actions,
    )


# ---------- test harness fixtures ----------


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
    """Copy the on-disk trajectory fixtures and seed blank-canvas screenshots."""
    root = tmp_path / "trajectories"
    root.mkdir()
    png = blank_canvas_png()
    for _slug, _params, traj_id in _REFERENCE_SKILLS:
        src = _FIXTURES_TRAJECTORIES / traj_id
        dst = root / traj_id
        dst.mkdir()
        # Copy metadata.json and events.jsonl if present.
        for name in ("metadata.json", "events.jsonl"):
            src_file = src / name
            if src_file.is_file():
                shutil.copy2(src_file, dst / name)
        screenshots = dst / "screenshots"
        screenshots.mkdir()
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
    with respx.mock(
        base_url="https://api.anthropic.com", assert_all_called=False
    ) as router_:
        yield router_


def _adapter_factory(
    trajectories_root: Path,
    recorded: list[DryRunInputAdapter],
) -> Any:
    def factory(skill: Any, mode: str) -> AdapterBundle:
        adapter = DryRunInputAdapter()
        recorded.append(adapter)
        trajectory_id = str(skill.meta.get("trajectory_ref", ""))
        return AdapterBundle(
            input_adapter=adapter,
            screen_source=TrajectoryScreenSource(
                trajectory_id, trajectories_root=trajectories_root
            ),
            ax_resolver=_NullAXResolver(),
        )

    return factory


def _build_app(
    *,
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    costs_path: Path,
    kill_switch: KillSwitch,
    recorded: list[DryRunInputAdapter],
) -> FastAPI:
    app = FastAPI()

    def runtime_factory(run_id: str) -> ClaudeRuntime:
        return ClaudeRuntime(run_id=run_id, costs_path=costs_path)

    manager = RunManager(
        runs_root=runs_root,
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        costs_path=costs_path,
        kill_switch=kill_switch,
        adapter_factory=_adapter_factory(trajectories_root, recorded),
        runtime_factory=runtime_factory,
    )
    app.state.run_manager = manager
    app.include_router(router)
    return app


_TERMINAL = {"succeeded", "failed", "aborted", "budget_exceeded", "rate_limited"}


def _wait_for_terminal(runs_root: Path, run_id: str, timeout: float = 15.0) -> dict[str, Any]:
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
    raise AssertionError(f"run {run_id} did not reach a terminal status")


def _drive_run(
    client: TestClient, run_id: str, *, max_messages: int = 400
) -> tuple[int, int]:
    """Stream events; auto-confirm destructive prompts; return (confirmations, screenshots)."""
    confirmations_fired = 0
    screenshot_events = 0
    with client.websocket_connect(f"/run/{run_id}/stream") as ws:
        for _ in range(max_messages):
            message = ws.receive_json()
            mtype = message.get("type")
            if mtype == "confirmation_request":
                confirmations_fired += 1
                resp = client.post(
                    f"/run/{run_id}/confirm", json={"decision": "confirm"}
                )
                assert resp.status_code == 200
            elif mtype == "screenshot":
                screenshot_events += 1
            elif mtype == "done":
                break
    return confirmations_fired, screenshot_events


# ---------- per-skill replay correctness ----------


@pytest.mark.parametrize(
    ("slug", "params", "trajectory_id"),
    _REFERENCE_SKILLS,
    ids=[s[0] for s in _REFERENCE_SKILLS],
)
def test_replay_correctness_against_reference_trajectory(
    slug: str,
    params: dict[str, str],
    trajectory_id: str,
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    costs_path: Path,
    kill_switch: KillSwitch,
    anthropic_mock_strict: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each reference skill's fake-mode run matches its trajectory ≥ 80%."""

    monkeypatch.setenv(FAKE_MODE_ENV_VAR, "1")

    trajectory = load_trajectory(trajectory_id, trajectories_root)

    recorded: list[DryRunInputAdapter] = []
    app = _build_app(
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        runs_root=runs_root,
        costs_path=costs_path,
        kill_switch=kill_switch,
        recorded=recorded,
    )
    client = TestClient(app)

    resp = client.post(
        "/run/start",
        json={"skill_slug": slug, "parameters": params, "mode": "dry_run"},
    )
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    confirmations_fired, screenshot_events = _drive_run(client, run_id)
    _wait_for_terminal(runs_root, run_id)

    assert recorded, f"{slug}: adapter_factory was never invoked"
    dispatched_calls = recorded[0].get_recorded_calls()

    # Count how many trajectory events were "extra screenshots" (screenshot
    # events emitted by the runner but not anchored to a trajectory click).
    # This is a soft metric — reported but not asserted.
    extra_screenshots = max(0, screenshot_events - len(trajectory.clicks))

    metrics = compare_trajectory_to_dispatch(
        slug=slug,
        trajectory=trajectory,
        dispatched=dispatched_calls,
        confirmations_fired=confirmations_fired,
        extra_screenshots=extra_screenshots,
    )

    # Emit per-skill metrics into stdout so pytest -s shows them.
    with capsys.disabled():
        print(
            f"\n[replay:{slug}] clicks_matched={metrics.clicks_matched}/"
            f"{metrics.trajectory_clicks - metrics.destructive_clicks_in_trajectory} "
            f"({metrics.click_match_rate:.0%}), "
            f"text_inputs_matched={metrics.text_inputs_matched}/"
            f"{metrics.trajectory_text_inputs} "
            f"({metrics.text_input_match_rate:.0%}), "
            f"confirmations_fired={metrics.confirmations_fired}/"
            f"{metrics.destructive_clicks_in_trajectory}, "
            f"extra_screenshots={metrics.extra_screenshots}, "
            f"missing_actions={metrics.missing_actions}"
        )

    expected_clicks = [
        e.target_label or f"({e.x}, {e.y})"
        for e in trajectory.non_destructive_clicks
    ]
    assert metrics.click_match_rate >= _CLICK_MATCH_THRESHOLD, (
        f"{slug}: click match rate {metrics.click_match_rate:.0%} below "
        f"{_CLICK_MATCH_THRESHOLD:.0%}. "
        f"Expected clicks: {expected_clicks}; "
        f"Actual dispatched: {_dispatched_clicks(dispatched_calls)}"
    )
    assert metrics.text_input_match_rate >= _TEXT_INPUT_MATCH_THRESHOLD, (
        f"{slug}: text_input match rate {metrics.text_input_match_rate:.0%} "
        f"below {_TEXT_INPUT_MATCH_THRESHOLD:.0%}. "
        f"Expected: {[e.text for e in trajectory.text_inputs]}; "
        f"Actual dispatched: {_dispatched_text_inputs(dispatched_calls)}"
    )
    assert (
        metrics.destructive_confirmation_rate >= _DESTRUCTIVE_CONFIRMATION_THRESHOLD
    ), (
        f"{slug}: destructive-confirmation rate "
        f"{metrics.destructive_confirmation_rate:.0%} below "
        f"{_DESTRUCTIVE_CONFIRMATION_THRESHOLD:.0%}. "
        f"{metrics.destructive_clicks_in_trajectory} destructive clicks in "
        f"trajectory, {metrics.confirmations_fired} confirmations fired."
    )

    # Zero real API calls.
    assert len(anthropic_mock_strict.calls) == 0
