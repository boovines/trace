"""Tests for the AgentRuntime protocol + ClaudeRuntime implementation.

Groups:

* Fake mode — each of the 5 canned scripts walks to ``end_turn`` cleanly.
* Live mode — 429 retries, 401 fails fast, cost log written.
* OpenAIRuntime stub raises ``NotImplementedError`` on construction.

Real network is impossible: ``AsyncAnthropic`` goes through httpx and every
httpx call is intercepted by ``respx``. The respx mock is scoped via a
fixture so tests that hit neither path don't need to configure it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from anthropic import AsyncAnthropic
from runner.agent_runtime import AgentResponse, AgentRuntime
from runner.claude_runtime import (
    API_KEY_ENV_VAR,
    DEFAULT_MODEL,
    DEFAULT_TARGET_LONGEST_EDGE,
    FAKE_MODE_ENV_VAR,
    INPUT_COST_PER_MTOK_USD,
    MAX_RETRIES,
    OUTPUT_COST_PER_MTOK_USD,
    RETRY_DELAYS_S,
    ClaudeRuntime,
    OpenAIRuntime,
    estimate_cost_usd,
)
from runner.coords import DryRunDisplayInfo, ImageMapping

_FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "llm_responses"
_CANNED_SLUGS: tuple[str, ...] = (
    "gmail_reply",
    "calendar_block",
    "finder_organize",
    "slack_status",
    "notes_daily",
)


# ---------- fixtures ----------


@pytest.fixture
def fake_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FAKE_MODE_ENV_VAR, "1")


@pytest.fixture
def costs_path(tmp_path: Path) -> Path:
    return tmp_path / "costs.jsonl"


@pytest.fixture
def respx_mock() -> Iterator[respx.MockRouter]:
    with respx.mock(
        base_url="https://api.anthropic.com", assert_all_called=False
    ) as router:
        yield router


@pytest.fixture
def claude_client() -> AsyncAnthropic:
    return AsyncAnthropic(api_key="test-key", max_retries=0)


def _success_body(input_tokens: int = 10, output_tokens: int = 5) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": DEFAULT_MODEL,
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


# ---------- fake-mode canned scripts ----------


def _load_script(slug: str) -> dict[str, Any]:
    with (_FIXTURES_ROOT / f"runner_{slug}.json").open(encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def test_fixture_dir_has_all_five_scripts() -> None:
    names = {p.stem for p in _FIXTURES_ROOT.glob("runner_*.json")}
    assert names == {f"runner_{slug}" for slug in _CANNED_SLUGS}


@pytest.mark.parametrize("slug", _CANNED_SLUGS)
async def test_fake_mode_walks_canned_script_to_end_turn(
    slug: str, fake_mode: None, costs_path: Path
) -> None:
    _ = fake_mode
    script = _load_script(slug)
    runtime = ClaudeRuntime(run_id=f"run_{slug}", costs_path=costs_path)

    responses: list[AgentResponse] = []
    for _ in script["turns"]:
        responses.append(
            await runtime.run_turn(
                system_prompt=f"... {script['marker']} ...",
                messages=[{"role": "user", "content": "start"}],
            )
        )
    assert len(responses) == len(script["turns"])
    # At least one tool_use turn, final turn is end_turn.
    stop_reasons = [r.stop_reason for r in responses]
    assert "tool_use" in stop_reasons
    assert stop_reasons[-1] == "end_turn"
    # Turn numbers are 1-based and contiguous.
    assert [r.turn_number for r in responses] == list(
        range(1, len(script["turns"]) + 1)
    )
    # Content blocks are the raw dicts from the fixture (no mutation).
    assert responses[0].content_blocks == script["turns"][0]["content_blocks"]


async def test_fake_mode_missing_marker_raises_keyerror(
    fake_mode: None, costs_path: Path
) -> None:
    _ = fake_mode
    runtime = ClaudeRuntime(run_id="r", costs_path=costs_path)
    with pytest.raises(KeyError, match="No fake-mode script"):
        await runtime.run_turn(
            system_prompt="a prompt with no known marker",
            messages=[{"role": "user", "content": "hi"}],
        )


async def test_fake_mode_running_past_end_raises_indexerror(
    fake_mode: None, costs_path: Path
) -> None:
    _ = fake_mode
    runtime = ClaudeRuntime(run_id="r", costs_path=costs_path)
    script = _load_script("gmail_reply")
    for _ in script["turns"]:
        await runtime.run_turn(
            system_prompt="Gmail Reply",
            messages=[{"role": "user", "content": "x"}],
        )
    with pytest.raises(IndexError, match="no canned response"):
        await runtime.run_turn(
            system_prompt="Gmail Reply",
            messages=[{"role": "user", "content": "x"}],
        )


async def test_fake_mode_logs_cost_per_turn(
    fake_mode: None, costs_path: Path
) -> None:
    _ = fake_mode
    runtime = ClaudeRuntime(run_id="run_cost", costs_path=costs_path)
    script = _load_script("notes_daily")
    for _ in script["turns"]:
        await runtime.run_turn(
            system_prompt="Notes Daily",
            messages=[{"role": "user", "content": "x"}],
        )
    lines = costs_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(script["turns"])
    for i, line in enumerate(lines):
        record = json.loads(line)
        assert record["module"] == "runner"
        assert record["run_id"] == "run_cost"
        assert record["model"] == DEFAULT_MODEL
        assert record["input_tokens"] == script["turns"][i]["input_tokens"]
        assert record["output_tokens"] == script["turns"][i]["output_tokens"]
        assert record["cost_estimate_usd"] == pytest.approx(
            estimate_cost_usd(
                script["turns"][i]["input_tokens"],
                script["turns"][i]["output_tokens"],
            )
        )
        assert isinstance(record["timestamp_ms"], int)


# ---------- real-transport tests (respx) ----------


async def test_runtime_conforms_to_protocol(costs_path: Path) -> None:
    runtime = ClaudeRuntime(
        run_id="r", costs_path=costs_path, api_key="test", client=AsyncAnthropic(api_key="t")
    )
    assert isinstance(runtime, AgentRuntime)


async def test_live_mode_missing_api_key_raises(
    costs_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match=API_KEY_ENV_VAR):
        ClaudeRuntime(run_id="r", costs_path=costs_path)


async def test_live_mode_401_fails_fast_without_retry(
    costs_path: Path,
    respx_mock: respx.MockRouter,
    claude_client: AsyncAnthropic,
) -> None:
    route = respx_mock.post("/v1/messages").respond(
        401,
        json={
            "type": "error",
            "error": {"type": "authentication_error", "message": "bad key"},
        },
    )
    runtime = ClaudeRuntime(
        run_id="r", costs_path=costs_path, client=claude_client
    )
    with pytest.raises(RuntimeError, match=API_KEY_ENV_VAR):
        await runtime.run_turn(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
        )
    # Single call — no retry on 401.
    assert route.call_count == 1
    # Cost log stays empty: no successful turn to log.
    assert not costs_path.exists() or costs_path.read_text() == ""


async def test_live_mode_429_retries_then_succeeds(
    costs_path: Path,
    respx_mock: respx.MockRouter,
    claude_client: AsyncAnthropic,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Skip the real sleeps — tests must not block on retry delays.
    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("runner.claude_runtime.asyncio.sleep", fake_sleep)

    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(
                429,
                json={
                    "type": "error",
                    "error": {"type": "rate_limit_error", "message": "rl"},
                },
            )
        return httpx.Response(200, json=_success_body(input_tokens=100, output_tokens=50))

    respx_mock.post("/v1/messages").mock(side_effect=handler)
    runtime = ClaudeRuntime(
        run_id="r429", costs_path=costs_path, client=claude_client
    )
    response = await runtime.run_turn(
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response.stop_reason == "end_turn"
    assert response.input_tokens == 100
    assert response.output_tokens == 50
    assert len(calls) == 3
    # Two backoff sleeps: 1s then 2s.
    assert sleep_calls == [RETRY_DELAYS_S[0], RETRY_DELAYS_S[1]]


async def test_live_mode_retries_exhausted_raises(
    costs_path: Path,
    respx_mock: respx.MockRouter,
    claude_client: AsyncAnthropic,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(seconds: float) -> None:
        _ = seconds

    monkeypatch.setattr("runner.claude_runtime.asyncio.sleep", fake_sleep)

    route = respx_mock.post("/v1/messages").respond(
        503,
        json={
            "type": "error",
            "error": {"type": "overloaded_error", "message": "busy"},
        },
    )
    runtime = ClaudeRuntime(
        run_id="r503", costs_path=costs_path, client=claude_client
    )
    from anthropic import APIStatusError

    with pytest.raises(APIStatusError):
        await runtime.run_turn(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
        )
    # Initial attempt + MAX_RETRIES retries = MAX_RETRIES + 1 calls.
    assert route.call_count == MAX_RETRIES + 1


async def test_live_mode_cost_logging_on_success(
    costs_path: Path,
    respx_mock: respx.MockRouter,
    claude_client: AsyncAnthropic,
) -> None:
    respx_mock.post("/v1/messages").respond(
        200, json=_success_body(input_tokens=2000, output_tokens=400)
    )
    runtime = ClaudeRuntime(
        run_id="run_abc", costs_path=costs_path, client=claude_client
    )
    await runtime.run_turn(
        system_prompt="sys", messages=[{"role": "user", "content": "hi"}]
    )
    lines = costs_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record == {
        "timestamp_ms": record["timestamp_ms"],
        "module": "runner",
        "model": DEFAULT_MODEL,
        "input_tokens": 2000,
        "output_tokens": 400,
        "cost_estimate_usd": pytest.approx(
            (2000 / 1e6) * INPUT_COST_PER_MTOK_USD
            + (400 / 1e6) * OUTPUT_COST_PER_MTOK_USD
        ),
        "run_id": "run_abc",
    }


async def test_tools_use_default_display_when_no_mapping(
    costs_path: Path,
    respx_mock: respx.MockRouter,
    claude_client: AsyncAnthropic,
) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_success_body())

    respx_mock.post("/v1/messages").mock(side_effect=handler)
    runtime = ClaudeRuntime(
        run_id="r", costs_path=costs_path, client=claude_client
    )
    await runtime.run_turn(
        system_prompt="sys", messages=[{"role": "user", "content": "hi"}]
    )
    tools = captured["body"]["tools"]
    assert len(tools) == 1
    tool = tools[0]
    assert tool["type"] == "computer_20250124"
    assert tool["name"] == "computer"
    assert tool["display_number"] == 1
    # DryRunDisplayInfo (2880x1800) downscaled to longest-edge 1568 ⇒ 1568x980.
    assert tool["display_width_px"] == DEFAULT_TARGET_LONGEST_EDGE
    expected_h = round(
        DryRunDisplayInfo.height_pixels
        * DEFAULT_TARGET_LONGEST_EDGE
        / max(DryRunDisplayInfo.width_pixels, DryRunDisplayInfo.height_pixels)
    )
    assert tool["display_height_px"] == expected_h


async def test_tools_use_current_image_mapping_when_set(
    costs_path: Path,
    respx_mock: respx.MockRouter,
    claude_client: AsyncAnthropic,
) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_success_body())

    respx_mock.post("/v1/messages").mock(side_effect=handler)
    runtime = ClaudeRuntime(
        run_id="r", costs_path=costs_path, client=claude_client
    )
    runtime.set_image_mapping(
        ImageMapping(
            original_pixels=(3840, 2160),
            resized_pixels=(1568, 882),
            scale_from_resized_to_points=1.224,
        )
    )
    await runtime.run_turn(
        system_prompt="sys", messages=[{"role": "user", "content": "hi"}]
    )
    tool = captured["body"]["tools"][0]
    assert tool["display_width_px"] == 1568
    assert tool["display_height_px"] == 882


def test_estimate_cost_usd_at_known_prices() -> None:
    # 1M input + 1M output ⇒ $3 + $15 = $18.
    assert estimate_cost_usd(1_000_000, 1_000_000) == pytest.approx(18.0)
    assert estimate_cost_usd(0, 0) == 0.0


# ---------- OpenAIRuntime stub ----------


def test_openai_runtime_raises_on_construction() -> None:
    with pytest.raises(NotImplementedError, match="stub"):
        OpenAIRuntime()
