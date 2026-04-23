"""ClaudeRuntime: AsyncAnthropic-backed AgentRuntime with retries + cost log.

The runner's default agent runtime. Calls ``messages.create`` with the
``computer_20250124`` tool, retries 429/5xx with exponential backoff, fails
fast on 401, and appends one line to ``costs.jsonl`` for every successful
turn (real or fake).

Fake mode: set ``TRACE_LLM_FAKE_MODE=1`` and ClaudeRuntime will serve turns
from ``fixtures/llm_responses/runner_*.json`` instead of the live API. Each
fixture declares a ``marker`` substring that must appear in the
``system_prompt`` — whichever fixture matches drives the conversation. Turn
selection is by ``run_turn`` call count (1-based), NOT by message-list hash,
so callers always walk a canned script deterministically.

Cost estimate uses the published ``claude-sonnet-4-5`` prices ($3/M input,
$15/M output) and is purely informational — the log consumer can re-cost
later if prices change.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Final, cast

from anthropic import APIStatusError, AsyncAnthropic, AuthenticationError

from runner.agent_runtime import AgentResponse, Message
from runner.coords import DryRunDisplayInfo, ImageMapping

logger = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "claude-sonnet-4-5"
DEFAULT_TARGET_LONGEST_EDGE: Final[int] = 1568
INPUT_COST_PER_MTOK_USD: Final[float] = 3.0
OUTPUT_COST_PER_MTOK_USD: Final[float] = 15.0

MAX_RETRIES: Final[int] = 5
RETRY_DELAYS_S: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0, 16.0)
FAKE_MODE_ENV_VAR: Final[str] = "TRACE_LLM_FAKE_MODE"
API_KEY_ENV_VAR: Final[str] = "ANTHROPIC_API_KEY"

_FIXTURES_ROOT: Final[Path] = (
    Path(__file__).resolve().parents[4] / "fixtures" / "llm_responses"
)


def _default_display_dims() -> tuple[int, int]:
    """Resized dimensions for DryRunDisplayInfo at longest-edge 1568.

    Matches what ``coords.capture_and_normalize`` would produce for a
    ``2880x1800`` source image, i.e. ``1568x980``. Used when no screenshot
    has been taken yet and thus no ``ImageMapping`` is available.
    """

    orig_w = DryRunDisplayInfo.width_pixels
    orig_h = DryRunDisplayInfo.height_pixels
    longest = max(orig_w, orig_h)
    if longest <= DEFAULT_TARGET_LONGEST_EDGE:
        return (orig_w, orig_h)
    ratio = DEFAULT_TARGET_LONGEST_EDGE / longest
    return (round(orig_w * ratio), round(orig_h * ratio))


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Return a dollar estimate for the given usage at current sonnet prices."""

    return (input_tokens / 1_000_000.0) * INPUT_COST_PER_MTOK_USD + (
        output_tokens / 1_000_000.0
    ) * OUTPUT_COST_PER_MTOK_USD


class ClaudeRuntime:
    """AsyncAnthropic-backed ``AgentRuntime``.

    Parameters
    ----------
    run_id:
        Run identifier included in every costs.jsonl line.
    costs_path:
        Where to append cost records. Parent directory is created on demand.
    api_key:
        Optional override; falls back to ``ANTHROPIC_API_KEY`` env var.
        Ignored in fake mode.
    client:
        Inject a pre-built ``AsyncAnthropic`` (e.g. for tests that set up a
        custom transport). If omitted, a fresh client is constructed.
    model:
        Defaults to ``claude-sonnet-4-5``.
    fixtures_root:
        Override the fake-mode fixture directory (tests use ``tmp_path``).
    """

    def __init__(
        self,
        *,
        run_id: str,
        costs_path: Path,
        api_key: str | None = None,
        client: AsyncAnthropic | None = None,
        model: str = DEFAULT_MODEL,
        fixtures_root: Path | None = None,
    ) -> None:
        self._model = model
        self._run_id = run_id
        self._costs_path = costs_path
        self._fixtures_root = fixtures_root if fixtures_root is not None else _FIXTURES_ROOT
        self._fake_mode = os.environ.get(FAKE_MODE_ENV_VAR) == "1"
        self._turn_counter = 0
        self.current_image_mapping: ImageMapping | None = None

        if self._fake_mode:
            self._client: AsyncAnthropic | None = None
            return

        if client is not None:
            self._client = client
            return

        resolved_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV_VAR)
        if not resolved_key:
            raise RuntimeError(
                f"{API_KEY_ENV_VAR} environment variable is not set. "
                "ClaudeRuntime requires an API key when fake mode is off "
                f"(set {FAKE_MODE_ENV_VAR}=1 for fixture-driven tests)."
            )
        # ``max_retries=0`` disables the SDK's own retry policy — we apply our
        # own exponential backoff so retry timing is observable and testable.
        self._client = AsyncAnthropic(api_key=resolved_key, max_retries=0)

    def set_image_mapping(self, mapping: ImageMapping | None) -> None:
        """Update the mapping used to derive the computer tool's display dims."""

        self.current_image_mapping = mapping

    def _current_display_dims(self) -> tuple[int, int]:
        if self.current_image_mapping is not None:
            return self.current_image_mapping.resized_pixels
        return _default_display_dims()

    def _tools(self) -> list[dict[str, Any]]:
        width, height = self._current_display_dims()
        return [
            {
                "type": "computer_20250124",
                "name": "computer",
                "display_width_px": width,
                "display_height_px": height,
                "display_number": 1,
            }
        ]

    async def run_turn(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 4096,
    ) -> AgentResponse:
        """Send one turn and return an ``AgentResponse``.

        In fake mode, returns a canned response from the matching fixture.
        Every successful turn (fake or real) appends one line to
        ``costs.jsonl``.
        """

        self._turn_counter += 1
        if self._fake_mode:
            response = _fake_response(
                system_prompt=system_prompt,
                turn_number=self._turn_counter,
                fixtures_root=self._fixtures_root,
            )
        else:
            response = await self._call_with_retry(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=max_tokens,
            )
        self._log_cost(response)
        return response

    async def _call_with_retry(
        self,
        *,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int,
    ) -> AgentResponse:
        assert self._client is not None
        attempt = 0
        while True:
            try:
                msg = await self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=messages,
                    tools=cast(Any, self._tools()),
                )
            except AuthenticationError as exc:
                raise RuntimeError(
                    "Anthropic API rejected the request with 401 Unauthorized. "
                    f"Check that {API_KEY_ENV_VAR} is set to a valid key."
                ) from exc
            except APIStatusError as exc:
                status = exc.status_code
                if status == 429 or (500 <= status < 600):
                    if attempt >= MAX_RETRIES:
                        raise
                    delay = RETRY_DELAYS_S[min(attempt, len(RETRY_DELAYS_S) - 1)]
                    logger.warning(
                        "Anthropic returned %d; retrying %d/%d after %.1fs",
                        status,
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                raise
            return AgentResponse(
                content_blocks=[block.model_dump() for block in msg.content],
                stop_reason=msg.stop_reason,
                input_tokens=int(msg.usage.input_tokens),
                output_tokens=int(msg.usage.output_tokens),
                turn_number=self._turn_counter,
            )

    def _log_cost(self, response: AgentResponse) -> None:
        record = {
            "timestamp_ms": int(time.time() * 1000),
            "module": "runner",
            "model": self._model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_estimate_usd": estimate_cost_usd(
                response.input_tokens, response.output_tokens
            ),
            "run_id": self._run_id,
        }
        self._costs_path.parent.mkdir(parents=True, exist_ok=True)
        with self._costs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class OpenAIRuntime:
    """Stub demonstrating the ``AgentRuntime`` swap seam.

    Instantiating it raises ``NotImplementedError`` immediately. When we
    actually need OpenAI coverage, replace the ``__init__`` body with real
    SDK construction and flesh out ``run_turn``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "OpenAIRuntime is a stub. Implement OpenAI's chat completions "
            "call here (with the computer-use tool equivalent) when a "
            "second runtime is needed."
        )

    async def run_turn(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 4096,
    ) -> AgentResponse:
        raise NotImplementedError

    def set_image_mapping(self, mapping: ImageMapping | None) -> None:
        raise NotImplementedError


def _fake_response(
    *,
    system_prompt: str,
    turn_number: int,
    fixtures_root: Path,
) -> AgentResponse:
    script = _select_script(system_prompt, fixtures_root)
    turns = script["turns"]
    idx = turn_number - 1
    if idx >= len(turns):
        raise IndexError(
            f"Fake script {script.get('slug', '<unknown>')!r} has "
            f"{len(turns)} turns; run_turn call #{turn_number} has no canned "
            "response. Extend the fixture or stop the loop earlier."
        )
    turn = turns[idx]
    content_blocks = list(turn["content_blocks"])
    stop_reason = turn.get("stop_reason")
    return AgentResponse(
        content_blocks=content_blocks,
        stop_reason=stop_reason,
        input_tokens=int(turn.get("input_tokens", 0)),
        output_tokens=int(turn.get("output_tokens", 0)),
        turn_number=turn_number,
    )


def _select_script(system_prompt: str, fixtures_root: Path) -> dict[str, Any]:
    if not fixtures_root.is_dir():
        raise FileNotFoundError(
            f"Fake-mode fixtures directory missing: {fixtures_root}"
        )
    matches: list[dict[str, Any]] = []
    for path in sorted(fixtures_root.glob("runner_*.json")):
        with path.open(encoding="utf-8") as f:
            script = json.load(f)
        marker = script.get("marker")
        if isinstance(marker, str) and marker and marker in system_prompt:
            matches.append(script)
    if not matches:
        raise KeyError(
            f"No fake-mode script in {fixtures_root} had a marker matching "
            f"the provided system_prompt (first 120 chars: "
            f"{system_prompt[:120]!r})."
        )
    if len(matches) > 1:
        slugs = [m.get("slug", "<unknown>") for m in matches]
        raise KeyError(
            f"Multiple fake-mode scripts matched the prompt: {slugs}. "
            "Make markers more specific."
        )
    return matches[0]


__all__ = [
    "API_KEY_ENV_VAR",
    "DEFAULT_MODEL",
    "DEFAULT_TARGET_LONGEST_EDGE",
    "FAKE_MODE_ENV_VAR",
    "INPUT_COST_PER_MTOK_USD",
    "MAX_RETRIES",
    "OUTPUT_COST_PER_MTOK_USD",
    "RETRY_DELAYS_S",
    "ClaudeRuntime",
    "OpenAIRuntime",
    "estimate_cost_usd",
]
