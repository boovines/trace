"""Tests for :mod:`synthesizer.budget` — S-018 cost budget guardrails.

Covers both the standalone :class:`BudgetMonitor` surface (config loading,
per-session + daily classification, stderr warnings) and integration into the
:class:`SynthesisSession` state machine and the ``POST /synthesize/start``
HTTP endpoint.
"""

from __future__ import annotations

import json
import struct
import uuid
import zlib
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from synthesizer import api as api_module
from synthesizer.budget import (
    DEFAULT_DAILY_CAP_USD,
    DEFAULT_SESSION_CAP_USD,
    BudgetConfig,
    BudgetMonitor,
    BudgetStatus,
    budget_config_path,
    load_budget_config,
)
from synthesizer.draft import DraftResult, Question
from synthesizer.llm_client import ConfigurationError, LLMClient
from synthesizer.session import SessionState, SynthesisSession
from synthesizer.skill_doc import Parameter, ParsedSkill, Step, render_skill_md
from synthesizer.trajectory_reader import TrajectoryReader

# --- shared helpers --------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data_dir = tmp_path / "trace_data"
    data_dir.mkdir()
    monkeypatch.setenv("TRACE_DATA_DIR", str(data_dir))
    return data_dir


def _iso(seconds_offset: float = 0.0) -> str:
    base = datetime(2026, 4, 22, 14, 0, 0, tzinfo=UTC)
    return (base + timedelta(seconds=seconds_offset)).isoformat().replace(
        "+00:00", "Z"
    )


def _one_pixel_png() -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"

    def _chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00", 9)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _make_trajectory(root: Path, trajectory_id: str | None = None) -> TrajectoryReader:
    tid = trajectory_id or str(uuid.uuid4())
    traj_dir = root / f"traj-{uuid.uuid4().hex[:8]}"
    traj_dir.mkdir(parents=True)
    ss_dir = traj_dir / "screenshots"
    ss_dir.mkdir()
    (ss_dir / "0001.png").write_bytes(_one_pixel_png())
    (ss_dir / "0002.png").write_bytes(_one_pixel_png())
    events: list[dict[str, Any]] = [
        {
            "seq": 1,
            "t": _iso(0.0),
            "kind": "app_switch",
            "bundle_id": "com.google.Chrome",
            "screenshot_ref": "screenshots/0001.png",
        },
        {
            "seq": 2,
            "t": _iso(1.0),
            "kind": "click",
            "x": 100,
            "y": 200,
            "button": "left",
            "bundle_id": "com.google.Chrome",
            "target": {
                "label": "Send",
                "role": "button",
                "bundle_id": "com.google.Chrome",
            },
            "screenshot_ref": "screenshots/0002.png",
        },
    ]
    metadata = {
        "id": tid,
        "started_at": _iso(0.0),
        "stopped_at": _iso(2.0),
        "label": "test",
        "display_info": {"width": 2560, "height": 1440, "scale": 2.0},
        "app_focus_history": [
            {"at": _iso(0.0), "bundle_id": "com.google.Chrome", "title": "Test"}
        ],
    }
    (traj_dir / "metadata.json").write_text(json.dumps(metadata))
    with (traj_dir / "events.jsonl").open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    return TrajectoryReader(traj_dir)


def _make_parsed_skill() -> ParsedSkill:
    return ParsedSkill(
        title="Reply to Gmail",
        description="Reply to the newest unread Gmail message.",
        parameters=[
            Parameter(
                name="message_body",
                type="string",
                required=True,
                default=None,
                description="Body text of the reply.",
            )
        ],
        preconditions=["Chrome is open"],
        steps=[
            Step(number=1, text="Type {message_body}.", destructive=False),
            Step(number=2, text='Click "Send" to deliver the reply.', destructive=True),
        ],
        expected_outcome="The reply is sent successfully.",
        notes=None,
    )


def _make_draft_result(
    trajectory_id: str,
    *,
    slug: str = "gmail_reply",
    questions: list[Question] | None = None,
    total_cost_usd: float = 0.0,
) -> DraftResult:
    parsed = _make_parsed_skill()
    meta: dict[str, Any] = {
        "slug": slug,
        "name": "Reply to Gmail",
        "trajectory_id": trajectory_id,
        "created_at": _iso(0.0).replace("Z", "+00:00"),
        "parameters": [
            {"name": "message_body", "type": "string", "required": True}
        ],
        "destructive_steps": [2],
        "preconditions": ["Chrome is open"],
        "step_count": 2,
    }
    return DraftResult(
        markdown=render_skill_md(parsed),
        parsed=parsed,
        meta=meta,
        questions=list(questions) if questions is not None else [],
        llm_calls=1,
        total_cost_usd=total_cost_usd,
    )


def _write_costs_line(
    path: Path,
    *,
    when: datetime,
    cost: float,
    module: str = "synthesizer",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp_iso": when.isoformat(),
        "module": module,
        "model": "claude-sonnet-4-5",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_estimate_usd": cost,
        "context_label": "test",
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# --- Config loading --------------------------------------------------------


def test_load_budget_config_missing_file_uses_defaults(_isolated_data_dir: Path) -> None:
    # No config.json on disk → defaults.
    config = load_budget_config()
    assert config.synthesis_per_session_usd_cap == DEFAULT_SESSION_CAP_USD
    assert config.synthesis_daily_usd_cap == DEFAULT_DAILY_CAP_USD


def test_load_budget_config_reads_disk(_isolated_data_dir: Path) -> None:
    budget_config_path().write_text(
        json.dumps(
            {
                "synthesis_per_session_usd_cap": 0.25,
                "synthesis_daily_usd_cap": 2.00,
            }
        )
    )
    config = load_budget_config()
    assert config.synthesis_per_session_usd_cap == 0.25
    assert config.synthesis_daily_usd_cap == 2.00


def test_load_budget_config_partial_overrides_merge_with_defaults(
    _isolated_data_dir: Path,
) -> None:
    budget_config_path().write_text(
        json.dumps({"synthesis_per_session_usd_cap": 0.50})
    )
    config = load_budget_config()
    assert config.synthesis_per_session_usd_cap == 0.50
    assert config.synthesis_daily_usd_cap == DEFAULT_DAILY_CAP_USD


@pytest.mark.parametrize(
    "bad_value",
    ["string", None, [1, 2], {"nested": 1}, True, 0, -1.5],
)
def test_load_budget_config_invalid_session_cap_raises(
    _isolated_data_dir: Path, bad_value: object
) -> None:
    budget_config_path().write_text(
        json.dumps({"synthesis_per_session_usd_cap": bad_value})
    )
    with pytest.raises(ConfigurationError):
        load_budget_config()


def test_load_budget_config_invalid_daily_cap_raises(_isolated_data_dir: Path) -> None:
    budget_config_path().write_text(
        json.dumps({"synthesis_daily_usd_cap": "expensive"})
    )
    with pytest.raises(ConfigurationError):
        load_budget_config()


def test_load_budget_config_non_object_raises(_isolated_data_dir: Path) -> None:
    budget_config_path().write_text("[1, 2, 3]")
    with pytest.raises(ConfigurationError):
        load_budget_config()


def test_load_budget_config_malformed_json_raises(_isolated_data_dir: Path) -> None:
    budget_config_path().write_text("{ not valid json")
    with pytest.raises(ConfigurationError):
        load_budget_config()


# --- classify / per-session -----------------------------------------------


def _monitor(
    *,
    session_cap: float = 1.00,
    daily_cap: float = 5.00,
    costs_path: Path | None = None,
    today: date | None = None,
) -> BudgetMonitor:
    fixed = datetime.combine(
        today or date(2026, 4, 22), datetime.min.time(), tzinfo=UTC
    )
    return BudgetMonitor(
        config=BudgetConfig(
            synthesis_per_session_usd_cap=session_cap,
            synthesis_daily_usd_cap=daily_cap,
        ),
        costs_path=costs_path,
        clock=lambda: fixed,
    )


def test_check_session_cost_ok_under_80_percent() -> None:
    monitor = _monitor(session_cap=1.00)
    result = monitor.check_session_cost("sid-1", 0.10)
    assert result.status == BudgetStatus.OK
    assert result.current_usd == 0.10
    assert result.cap_usd == 1.00
    assert result.percent_used == pytest.approx(0.10)


def test_check_session_cost_warning_at_80_percent(capsys: pytest.CaptureFixture[str]) -> None:
    monitor = _monitor(session_cap=1.00)
    result = monitor.check_session_cost("sid-1", 0.80)
    assert result.status == BudgetStatus.WARNING
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "sid-1" in captured.err


def test_check_session_cost_exceeded_at_100_percent() -> None:
    monitor = _monitor(session_cap=1.00)
    result = monitor.check_session_cost("sid-1", 1.00)
    assert result.status == BudgetStatus.EXCEEDED


def test_check_session_cost_exceeded_over_cap() -> None:
    monitor = _monitor(session_cap=0.20)
    result = monitor.check_session_cost("sid-1", 0.25)
    assert result.status == BudgetStatus.EXCEEDED


def test_check_session_cost_warning_is_deduplicated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    monitor = _monitor(session_cap=1.00)
    monitor.check_session_cost("sid-1", 0.80)
    capsys.readouterr()  # drain
    monitor.check_session_cost("sid-1", 0.85)
    captured = capsys.readouterr()
    assert "WARNING" not in captured.err  # not re-emitted for same session


def test_check_session_cost_warning_per_session(
    capsys: pytest.CaptureFixture[str],
) -> None:
    monitor = _monitor(session_cap=1.00)
    monitor.check_session_cost("sid-1", 0.80)
    monitor.check_session_cost("sid-2", 0.80)
    captured = capsys.readouterr()
    assert captured.err.count("WARNING") == 2


# --- daily --------------------------------------------------------------


def test_check_daily_cost_no_log_file_returns_zero(tmp_path: Path) -> None:
    monitor = _monitor(daily_cap=5.00, costs_path=tmp_path / "does-not-exist.jsonl")
    result = monitor.check_daily_cost()
    assert result.status == BudgetStatus.OK
    assert result.current_usd == 0.0


def test_check_daily_cost_sums_todays_synthesizer_entries(tmp_path: Path) -> None:
    log = tmp_path / "costs.jsonl"
    today = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    _write_costs_line(log, when=today, cost=0.10)
    _write_costs_line(log, when=today.replace(hour=13), cost=0.25)
    # Different day — not counted.
    _write_costs_line(
        log, when=today - timedelta(days=1), cost=10.0
    )
    # Different module — not counted.
    _write_costs_line(log, when=today, cost=1.0, module="recorder")

    monitor = _monitor(
        daily_cap=5.00, costs_path=log, today=today.date()
    )
    result = monitor.check_daily_cost()
    assert result.current_usd == pytest.approx(0.35)
    assert result.status == BudgetStatus.OK


def test_check_daily_cost_exceeded_returns_exceeded_status(tmp_path: Path) -> None:
    log = tmp_path / "costs.jsonl"
    today = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    _write_costs_line(log, when=today, cost=3.00)
    _write_costs_line(log, when=today, cost=2.50)

    monitor = _monitor(daily_cap=5.00, costs_path=log, today=today.date())
    result = monitor.check_daily_cost()
    assert result.status == BudgetStatus.EXCEEDED
    assert result.current_usd == pytest.approx(5.50)


def test_check_daily_cost_warning_at_80_percent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "costs.jsonl"
    today = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    _write_costs_line(log, when=today, cost=4.00)
    monitor = _monitor(daily_cap=5.00, costs_path=log, today=today.date())
    result = monitor.check_daily_cost()
    assert result.status == BudgetStatus.WARNING
    captured = capsys.readouterr()
    assert "daily cap" in captured.err


def test_check_daily_cost_warning_deduplicated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "costs.jsonl"
    today = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    _write_costs_line(log, when=today, cost=4.00)
    monitor = _monitor(daily_cap=5.00, costs_path=log, today=today.date())
    monitor.check_daily_cost()
    capsys.readouterr()
    monitor.check_daily_cost()
    captured = capsys.readouterr()
    assert "WARNING" not in captured.err


def test_check_daily_cost_accumulation_across_multiple_sessions(tmp_path: Path) -> None:
    log = tmp_path / "costs.jsonl"
    today = datetime(2026, 4, 22, 8, 0, 0, tzinfo=UTC)
    # Simulate three sessions over the course of today.
    for hour_offset, cost in [(0, 0.30), (2, 0.45), (6, 0.25)]:
        _write_costs_line(
            log, when=today + timedelta(hours=hour_offset), cost=cost
        )
    monitor = _monitor(daily_cap=5.00, costs_path=log, today=today.date())
    result = monitor.check_daily_cost()
    assert result.current_usd == pytest.approx(1.00)
    assert result.status == BudgetStatus.OK


def test_check_daily_cost_skips_malformed_lines(tmp_path: Path) -> None:
    log = tmp_path / "costs.jsonl"
    today = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        f.write("{ broken json\n")
        f.write("\n")  # empty line
    _write_costs_line(log, when=today, cost=0.50)
    monitor = _monitor(daily_cap=5.00, costs_path=log, today=today.date())
    result = monitor.check_daily_cost()
    assert result.current_usd == pytest.approx(0.50)


# --- SynthesisSession integration ------------------------------------------


def test_session_transitions_to_errored_when_draft_exceeds_cap(
    tmp_path: Path, _isolated_data_dir: Path
) -> None:
    trajectory = _make_trajectory(tmp_path)
    trajectory_id = trajectory.metadata["id"]
    # Draft cost exceeds the per-session cap.
    expensive_draft = _make_draft_result(trajectory_id, total_cost_usd=1.50)

    def _fake_draft(
        preprocessed: Any, client: LLMClient, *, reader: TrajectoryReader
    ) -> DraftResult:
        return expensive_draft

    monitor = BudgetMonitor(
        config=BudgetConfig(
            synthesis_per_session_usd_cap=1.00,
            synthesis_daily_usd_cap=50.00,
        ),
    )
    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=LLMClient(),
        skills_root=tmp_path / "skills",
        draft_fn=_fake_draft,
        budget_monitor=monitor,
    )
    session.start_draft()
    assert session.state == SessionState.ERRORED
    assert session.error == "per-session cost cap exceeded"


def test_session_under_cap_proceeds_normally(
    tmp_path: Path, _isolated_data_dir: Path
) -> None:
    trajectory = _make_trajectory(tmp_path)
    trajectory_id = trajectory.metadata["id"]
    cheap_draft = _make_draft_result(trajectory_id, total_cost_usd=0.10)

    def _fake_draft(
        preprocessed: Any, client: LLMClient, *, reader: TrajectoryReader
    ) -> DraftResult:
        return cheap_draft

    monitor = BudgetMonitor(
        config=BudgetConfig(
            synthesis_per_session_usd_cap=1.00,
            synthesis_daily_usd_cap=50.00,
        ),
    )
    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=LLMClient(),
        skills_root=tmp_path / "skills",
        draft_fn=_fake_draft,
        budget_monitor=monitor,
    )
    session.start_draft()
    assert session.state == SessionState.AWAITING_APPROVAL
    assert session.error is None


def test_session_revision_cost_trip_transitions_to_errored(
    tmp_path: Path, _isolated_data_dir: Path
) -> None:
    trajectory = _make_trajectory(tmp_path)
    trajectory_id = trajectory.metadata["id"]
    first_draft = _make_draft_result(
        trajectory_id,
        questions=[Question(id="q1", category="parameterization", text="?")],
        total_cost_usd=0.30,
    )
    expensive_revision = _make_draft_result(
        trajectory_id,
        questions=[],
        total_cost_usd=1.20,  # > 1.00 cap
    )

    def _fake_draft(
        preprocessed: Any, client: LLMClient, *, reader: TrajectoryReader
    ) -> DraftResult:
        return first_draft

    def _fake_revise(
        *,
        current_draft: DraftResult,
        question: Question,
        answer: str,
        client: LLMClient,
        reader: TrajectoryReader,
    ) -> DraftResult:
        return expensive_revision

    monitor = BudgetMonitor(
        config=BudgetConfig(
            synthesis_per_session_usd_cap=1.00,
            synthesis_daily_usd_cap=50.00,
        ),
    )
    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=LLMClient(),
        skills_root=tmp_path / "skills",
        draft_fn=_fake_draft,
        revise_fn=_fake_revise,
        budget_monitor=monitor,
    )
    session.start_draft()
    assert session.state == SessionState.AWAITING_ANSWER
    session.answer_question("q1", "some answer")
    assert session.state == SessionState.ERRORED
    assert session.error == "per-session cost cap exceeded"


def test_session_without_budget_monitor_is_unaffected(
    tmp_path: Path, _isolated_data_dir: Path
) -> None:
    trajectory = _make_trajectory(tmp_path)
    trajectory_id = trajectory.metadata["id"]
    expensive_draft = _make_draft_result(trajectory_id, total_cost_usd=99.99)

    def _fake_draft(
        preprocessed: Any, client: LLMClient, *, reader: TrajectoryReader
    ) -> DraftResult:
        return expensive_draft

    # No budget_monitor — session must not error.
    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=LLMClient(),
        skills_root=tmp_path / "skills",
        draft_fn=_fake_draft,
    )
    session.start_draft()
    assert session.state == SessionState.AWAITING_APPROVAL


# --- HTTP /synthesize/start daily cap ---------------------------------------


@pytest.fixture
def seeded_api_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _isolated_data_dir: Path
) -> Path:
    # Seed a trajectory under TRACE_DATA_DIR/trajectories/<id>/.
    trajectory_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    trajectories_root = _isolated_data_dir / "trajectories"
    trajectories_root.mkdir(parents=True, exist_ok=True)
    traj_dir = trajectories_root / trajectory_id
    traj_dir.mkdir()
    (traj_dir / "screenshots").mkdir()
    (traj_dir / "screenshots" / "0001.png").write_bytes(_one_pixel_png())
    events = [
        {
            "seq": 1,
            "t": _iso(0.0),
            "kind": "app_switch",
            "bundle_id": "com.google.Chrome",
            "screenshot_ref": "screenshots/0001.png",
        }
    ]
    metadata = {
        "id": trajectory_id,
        "started_at": _iso(0.0),
        "stopped_at": _iso(2.0),
        "label": "test",
        "display_info": {"width": 2560, "height": 1440, "scale": 2.0},
        "app_focus_history": [
            {"at": _iso(0.0), "bundle_id": "com.google.Chrome", "title": "Test"}
        ],
    }
    (traj_dir / "metadata.json").write_text(json.dumps(metadata))
    with (traj_dir / "events.jsonl").open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    api_module.reset_for_tests()
    yield _isolated_data_dir
    api_module.reset_for_tests()


async def _start_request(trajectory_id: str) -> httpx.Response:
    app = FastAPI()
    app.include_router(api_module.router)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        return await ac.post(
            "/synthesize/start", json={"trajectory_id": trajectory_id}
        )


@pytest.mark.asyncio
async def test_start_returns_429_when_daily_cap_exceeded(
    monkeypatch: pytest.MonkeyPatch, seeded_api_env: Path
) -> None:
    # Inject a monitor whose daily check always returns EXCEEDED.
    fixed_today = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
    log = seeded_api_env / "costs.jsonl"
    _write_costs_line(log, when=fixed_today, cost=10.00)

    def _factory() -> BudgetMonitor:
        return BudgetMonitor(
            config=BudgetConfig(
                synthesis_per_session_usd_cap=1.00,
                synthesis_daily_usd_cap=5.00,
            ),
            costs_path=log,
            clock=lambda: fixed_today,
        )

    monkeypatch.setattr(api_module, "_BUDGET_MONITOR_FACTORY", _factory)

    resp = await _start_request("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    assert resp.status_code == 429
    assert "daily" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_start_succeeds_when_daily_cap_not_exceeded(
    monkeypatch: pytest.MonkeyPatch, seeded_api_env: Path
) -> None:
    fixed_today = datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)

    def _factory() -> BudgetMonitor:
        return BudgetMonitor(
            config=BudgetConfig(
                synthesis_per_session_usd_cap=1.00,
                synthesis_daily_usd_cap=5.00,
            ),
            costs_path=seeded_api_env / "costs.jsonl",
            clock=lambda: fixed_today,
        )

    monkeypatch.setattr(api_module, "_BUDGET_MONITOR_FACTORY", _factory)

    # Inject a non-expensive fake draft so the background task doesn't crash.
    def _fake_draft(
        preprocessed: Any, client: LLMClient, *, reader: TrajectoryReader
    ) -> DraftResult:
        return _make_draft_result(reader.metadata["id"], total_cost_usd=0.01)

    monkeypatch.setattr(api_module, "_DRAFT_FN", _fake_draft)

    resp = await _start_request("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
