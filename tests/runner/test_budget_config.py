"""Tests for runner cost budget configuration (X-023).

Covers:

* ``load_runner_budget`` default paths (dev profile, prod profile, missing).
* Malformed ``config.json`` raises ``ConfigurationError``.
* ``sum_daily_runner_cost_usd`` filters by module+date and handles junk.
* Per-run cap (plumbed via ``RunBudget.max_cost_usd``) aborts with
  ``abort_reason='per_run_cost_cap'``.
* Daily cap blocks ``POST /run/start`` with HTTP 429.
* 80% warning fires on crossing, exactly once.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from runner.agent_runtime import AgentResponse
from runner.api import router
from runner.budget import (
    BudgetReason,
    BudgetStatusKind,
    BudgetTracker,
    RunBudget,
)
from runner.budget_config import (
    DEV_DAILY_USD_CAP,
    DEV_PER_EXECUTION_USD_CAP,
    PROD_DAILY_USD_CAP,
    PROD_PER_EXECUTION_USD_CAP,
    ConfigurationError,
    RunnerBudgetConfig,
    crossed_warning_threshold,
    default_config_for_profile,
    load_runner_budget,
    sum_daily_runner_cost_usd,
)
from runner.coords import ImageMapping
from runner.executor import PER_RUN_COST_CAP_REASON
from runner.input_adapter import DryRunInputAdapter
from runner.kill_switch import KillSwitch
from runner.paths import PROFILE_ENV_VAR
from runner.pre_action_gate import AXTarget
from runner.run_manager import AdapterBundle, RunManager
from runner.screen_source import TrajectoryScreenSource, blank_canvas_png

_SKILL_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "skills"


# ---------- load_runner_budget ----------


def test_default_config_dev_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROFILE_ENV_VAR, raising=False)
    cfg = default_config_for_profile()
    assert cfg.run_per_execution_usd_cap == DEV_PER_EXECUTION_USD_CAP == 0.50
    assert cfg.run_daily_usd_cap == DEV_DAILY_USD_CAP == 2.00


def test_default_config_prod_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROFILE_ENV_VAR, "prod")
    cfg = default_config_for_profile()
    assert cfg.run_per_execution_usd_cap == PROD_PER_EXECUTION_USD_CAP == 2.00
    assert cfg.run_daily_usd_cap == PROD_DAILY_USD_CAP == 20.00


def test_load_runner_budget_missing_file_uses_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PROFILE_ENV_VAR, raising=False)
    cfg = load_runner_budget(tmp_path / "nonexistent.json")
    assert cfg == RunnerBudgetConfig(
        run_per_execution_usd_cap=DEV_PER_EXECUTION_USD_CAP,
        run_daily_usd_cap=DEV_DAILY_USD_CAP,
    )


def test_load_runner_budget_reads_overrides(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {"run_per_execution_usd_cap": 1.25, "run_daily_usd_cap": 5.0}
        ),
        encoding="utf-8",
    )
    cfg = load_runner_budget(path)
    assert cfg.run_per_execution_usd_cap == 1.25
    assert cfg.run_daily_usd_cap == 5.0


def test_load_runner_budget_partial_override_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PROFILE_ENV_VAR, raising=False)
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"run_daily_usd_cap": 3.0}), encoding="utf-8"
    )
    cfg = load_runner_budget(path)
    assert cfg.run_per_execution_usd_cap == DEV_PER_EXECUTION_USD_CAP
    assert cfg.run_daily_usd_cap == 3.0


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        "[1, 2]",
        json.dumps({"run_per_execution_usd_cap": "cheap"}),
        json.dumps({"run_daily_usd_cap": 0}),
        json.dumps({"run_daily_usd_cap": -1.0}),
        json.dumps({"run_per_execution_usd_cap": True}),
        json.dumps({"unknown_key": 1.0}),
    ],
)
def test_load_runner_budget_invalid_raises(tmp_path: Path, body: str) -> None:
    path = tmp_path / "config.json"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_runner_budget(path)


# ---------- sum_daily_runner_cost_usd ----------


def test_sum_daily_cost_missing_file_returns_zero(tmp_path: Path) -> None:
    assert sum_daily_runner_cost_usd(tmp_path / "costs.jsonl") == 0.0


def test_sum_daily_cost_filters_module_and_date(tmp_path: Path) -> None:
    import datetime as _dt

    path = tmp_path / "costs.jsonl"
    now = _dt.datetime(2026, 4, 23, 12, 0, 0, tzinfo=_dt.UTC)
    today_ms = int(now.timestamp() * 1000)
    yesterday = now - _dt.timedelta(days=1)
    yesterday_ms = int(yesterday.timestamp() * 1000)
    lines = [
        # today, runner → counts
        {"timestamp_ms": today_ms, "module": "runner", "cost_estimate_usd": 0.25},
        {"timestamp_ms": today_ms, "module": "runner", "cost_estimate_usd": 0.10},
        # today, synthesizer → excluded
        {
            "timestamp_ms": today_ms,
            "module": "synthesizer",
            "cost_estimate_usd": 1.00,
        },
        # yesterday, runner → excluded
        {
            "timestamp_ms": yesterday_ms,
            "module": "runner",
            "cost_estimate_usd": 5.00,
        },
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    total = sum_daily_runner_cost_usd(path, now=now)
    assert total == pytest.approx(0.35)


def test_sum_daily_cost_ignores_malformed_lines(tmp_path: Path) -> None:
    import datetime as _dt

    path = tmp_path / "costs.jsonl"
    now = _dt.datetime(2026, 4, 23, 12, 0, 0, tzinfo=_dt.UTC)
    today_ms = int(now.timestamp() * 1000)
    path.write_text(
        "\n".join(
            [
                "not json",
                json.dumps(
                    {
                        "timestamp_ms": today_ms,
                        "module": "runner",
                        "cost_estimate_usd": 0.40,
                    }
                ),
                json.dumps({"timestamp_ms": "bogus", "module": "runner", "cost_estimate_usd": 1.0}),
                json.dumps([1, 2, 3]),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    total = sum_daily_runner_cost_usd(path, now=now)
    assert total == pytest.approx(0.40)


# ---------- crossed_warning_threshold ----------


def test_crossed_warning_threshold_exact_boundary() -> None:
    assert crossed_warning_threshold(0.79, 0.80, 1.00) is True
    assert crossed_warning_threshold(0.80, 0.90, 1.00) is False  # already past
    assert crossed_warning_threshold(0.0, 0.0, 1.0) is False
    assert crossed_warning_threshold(0.0, 1.0, 0.0) is False


# ---------- BudgetTracker: cost cap ----------


def test_cost_cap_trips_after_recorded_cost_exceeds() -> None:
    budget = RunBudget(max_cost_usd=1.00)
    tracker = BudgetTracker(budget=budget)
    tracker.record_cost(0.40)
    tracker.record_cost(0.50)
    assert tracker.check().kind is BudgetStatusKind.OK
    tracker.record_cost(0.11)
    status = tracker.check()
    assert status.kind is BudgetStatusKind.BUDGET_EXCEEDED
    assert status.reason is BudgetReason.COST


def test_cost_cap_none_never_trips() -> None:
    budget = RunBudget(max_cost_usd=None)
    tracker = BudgetTracker(budget=budget)
    tracker.record_cost(1_000_000.0)
    assert tracker.check().kind is BudgetStatusKind.OK


def test_run_budget_rejects_non_positive_cost() -> None:
    with pytest.raises(ValueError, match="max_cost_usd"):
        RunBudget(max_cost_usd=0.0)
    with pytest.raises(ValueError, match="max_cost_usd"):
        RunBudget(max_cost_usd=-0.5)


def test_record_cost_rejects_negative() -> None:
    tracker = BudgetTracker(budget=RunBudget())
    with pytest.raises(ValueError):
        tracker.record_cost(-0.1)


# ---------- Per-run cap integrated via Executor ----------


class _FakeAgentRuntime:
    def __init__(self, responses: list[AgentResponse]) -> None:
        self._responses = list(responses)
        self._turn = 0

    async def run_turn(
        self, system_prompt: str, messages: list[Any], max_tokens: int = 4096
    ) -> AgentResponse:
        self._turn += 1
        if self._turn > len(self._responses):
            import asyncio

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


def _resp(
    blocks: list[dict[str, Any]],
    *,
    input_tokens: int = 100,
    output_tokens: int = 20,
    stop_reason: str = "tool_use",
) -> AgentResponse:
    return AgentResponse(
        content_blocks=blocks,
        stop_reason=stop_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        turn_number=0,
    )


@pytest.fixture
def skills_root(tmp_path: Path) -> Path:
    dest = tmp_path / "skills"
    dest.mkdir()
    for slug in ("notes_daily",):
        shutil.copytree(_SKILL_FIXTURES / slug, dest / slug)
    return dest


@pytest.fixture
def trajectories_root(tmp_path: Path) -> Path:
    root = tmp_path / "trajectories"
    png = blank_canvas_png()
    trajectory_id = "00000000-0000-0000-0000-000000000005"
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
    config_path: Path,
    costs_path: Path,
) -> FastAPI:
    app = FastAPI()
    manager = RunManager(
        runs_root=runs_root,
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        costs_path=costs_path,
        kill_switch=kill_switch,
        adapter_factory=_adapter_factory(trajectories_root),
        runtime_factory=runtime_factory,
        config_path=config_path,
    )
    app.state.run_manager = manager
    app.include_router(router)
    return app


@pytest.fixture
async def client_with_low_caps(
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    tmp_path: Path,
) -> AsyncIterator[AsyncClient]:
    # Per-execution cap of $0.01 — one $0.015 turn trips it.
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"run_per_execution_usd_cap": 0.01, "run_daily_usd_cap": 5.0}
        ),
        encoding="utf-8",
    )
    costs = tmp_path / "costs.jsonl"

    def runtime_factory(run_id: str) -> _FakeAgentRuntime:
        # One turn at 1000 input / 500 output = $0.0105 — just over $0.01 cap.
        return _FakeAgentRuntime(
            [
                _resp(
                    [_screenshot_block("t1")],
                    input_tokens=1000,
                    output_tokens=500,
                ),
                _resp(
                    [_text_block("done <workflow_complete/>")],
                    stop_reason="end_turn",
                    input_tokens=10,
                    output_tokens=5,
                ),
            ]
        )

    app = _build_app(
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        runs_root=runs_root,
        kill_switch=KillSwitch(),
        runtime_factory=runtime_factory,
        config_path=config,
        costs_path=costs,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_per_run_cap_triggers_budget_exceeded(
    client_with_low_caps: AsyncClient,
    runs_root: Path,
) -> None:
    resp = await client_with_low_caps.post(
        "/run/start",
        json={"skill_slug": "notes_daily", "parameters": {}, "mode": "dry_run"},
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    for _ in range(50):
        import asyncio

        await asyncio.sleep(0.1)
        got = await client_with_low_caps.get(f"/run/{run_id}")
        if got.json().get("status") in {"budget_exceeded", "succeeded", "failed"}:
            break
    final = got.json()
    assert final["status"] == "budget_exceeded"
    assert final["abort_reason"] == PER_RUN_COST_CAP_REASON


# ---------- Daily cap → 429 ----------


@pytest.fixture
async def client_with_daily_cap_hit(
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    tmp_path: Path,
) -> AsyncIterator[AsyncClient]:
    import datetime as _dt

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"run_per_execution_usd_cap": 2.0, "run_daily_usd_cap": 1.0}
        ),
        encoding="utf-8",
    )
    costs = tmp_path / "costs.jsonl"
    today_ms = int(_dt.datetime.now(_dt.UTC).timestamp() * 1000)
    costs.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp_ms": today_ms,
                        "module": "runner",
                        "cost_estimate_usd": 1.10,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def runtime_factory(run_id: str) -> _FakeAgentRuntime:
        return _FakeAgentRuntime(
            [_resp([_text_block("done <workflow_complete/>")], stop_reason="end_turn")]
        )

    app = _build_app(
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        runs_root=runs_root,
        kill_switch=KillSwitch(),
        runtime_factory=runtime_factory,
        config_path=config,
        costs_path=costs,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_daily_cap_hit_returns_429_on_start(
    client_with_daily_cap_hit: AsyncClient,
) -> None:
    resp = await client_with_daily_cap_hit.post(
        "/run/start",
        json={"skill_slug": "notes_daily", "parameters": {}, "mode": "dry_run"},
    )
    assert resp.status_code == 429
    assert "daily runner cost cap" in resp.json()["detail"]


# ---------- 80% warning fires at the right threshold ----------


@pytest.mark.asyncio
async def test_per_run_80pct_warning_fires_once(
    skills_root: Path,
    trajectories_root: Path,
    runs_root: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Cap $1.00, 80% threshold = $0.80.
    # Turn 1: $0.60 (no warning). Turn 2: $0.45 → total $1.05 crosses 80% AND cap.
    # Use 2 separate turns with screenshot then workflow_complete to verify
    # the warning fires *before* terminal, and only once.
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"run_per_execution_usd_cap": 1.00, "run_daily_usd_cap": 10.0}
        ),
        encoding="utf-8",
    )
    costs = tmp_path / "costs.jsonl"

    # Tokens: 100_000 input + 20_000 output = $0.30 + $0.30 = $0.60 per turn.
    # After turn 1: $0.60 (below 80%). After turn 2 on the screenshot turn:
    # $1.20 (crosses 80% AND $1.00 cap).
    def runtime_factory(run_id: str) -> _FakeAgentRuntime:
        return _FakeAgentRuntime(
            [
                _resp(
                    [_screenshot_block("t1")],
                    input_tokens=100_000,
                    output_tokens=20_000,
                ),
                _resp(
                    [_screenshot_block("t2")],
                    input_tokens=100_000,
                    output_tokens=20_000,
                ),
                _resp(
                    [_text_block("done <workflow_complete/>")],
                    stop_reason="end_turn",
                    input_tokens=10,
                    output_tokens=5,
                ),
            ]
        )

    app = _build_app(
        skills_root=skills_root,
        trajectories_root=trajectories_root,
        runs_root=runs_root,
        kill_switch=KillSwitch(),
        runtime_factory=runtime_factory,
        config_path=config,
        costs_path=costs,
    )
    transport = ASGITransport(app=app)
    caplog.set_level(logging.WARNING, logger="runner.executor")
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            "/run/start",
            json={"skill_slug": "notes_daily", "parameters": {}, "mode": "dry_run"},
        )
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]
        import asyncio

        for _ in range(50):
            await asyncio.sleep(0.1)
            got = await c.get(f"/run/{run_id}")
            if got.json().get("status") in {
                "budget_exceeded",
                "succeeded",
                "failed",
            }:
                break

    warnings = [
        r for r in caplog.records if "exceeded 80%" in r.getMessage()
    ]
    assert len(warnings) == 1
