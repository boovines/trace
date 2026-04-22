"""Thin wrapper around the Anthropic SDK with fake-mode, retries, and cost logging.

The wrapper exists so the rest of the synthesizer never imports the SDK directly:

* ``TRACE_LLM_FAKE_MODE=1`` swaps real API calls for canned JSON responses keyed by
  a deterministic hash of the request — required for hermetic Ralph iteration.
* Every call (real or fake) appends a line to ``costs.jsonl`` so cost is observable
  without per-call wiring at the call sites.
* Retries with exponential backoff on 429/5xx; fails fast on 401 with a
  ``ConfigurationError`` so a misconfigured API key surfaces immediately.

The SDK's own retry layer is disabled (``max_retries=0``) so this wrapper owns the
retry policy end-to-end.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
from anthropic import Anthropic

from synthesizer.schema import _find_repo_root

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "MAX_RETRIES",
    "PRICING_USD_PER_MTOK",
    "RETRYABLE_STATUS_CODES",
    "RETRY_BACKOFF_SECONDS",
    "ConfigurationError",
    "FakeResponseNotFound",
    "LLMClient",
    "LLMResponse",
    "compute_request_hash",
    "costs_log_path",
    "estimate_cost_usd",
    "save_fake_response",
    "trace_data_dir",
]

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 4096
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# USD per million tokens, matching Anthropic's published list price. Update this
# table whenever Anthropic publishes new prices — it is the single source of
# truth for cost estimates module-wide.
PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "fake": {"input": 0.0, "output": 0.0},
}


class ConfigurationError(RuntimeError):
    """Raised when the LLM client cannot run because of misconfiguration.

    Covers both startup (missing ``ANTHROPIC_API_KEY``) and runtime (HTTP 401
    from the API). Both cases share a fail-fast story: do not retry, surface
    the problem to the operator immediately.
    """


class FakeResponseNotFound(RuntimeError):
    """Raised in fake mode when no canned response file matches the request.

    The error message tells the caller exactly which hash to register. Tests
    typically use :func:`save_fake_response` to write the missing fixture.
    """


@dataclass(frozen=True)
class LLMResponse:
    """A normalized LLM call result decoupled from the SDK's mutable types.

    ``raw`` holds the original Anthropic response payload (or the canned dict in
    fake mode) for callers that need fields beyond the normalized surface.
    """

    text: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    raw: Mapping[str, Any]


# --- Path helpers -----------------------------------------------------------


def trace_data_dir() -> Path:
    """Resolve the per-profile data directory.

    Priority: ``TRACE_DATA_DIR`` env override (used by tests) > ``TRACE_PROFILE``
    selecting between ``Trace`` (prod) and ``Trace-dev`` (dev). Defaults to prod.
    """
    override = os.environ.get("TRACE_DATA_DIR")
    if override:
        return Path(override)
    profile = os.environ.get("TRACE_PROFILE", "prod")
    suffix = "-dev" if profile == "dev" else ""
    return Path.home() / "Library" / "Application Support" / f"Trace{suffix}"


def costs_log_path() -> Path:
    """Path to the append-only ``costs.jsonl`` for the active profile."""
    return trace_data_dir() / "costs.jsonl"


def _fake_responses_dir() -> Path:
    """Resolve the directory holding canned fake-mode responses.

    ``TRACE_FAKE_RESPONSES_DIR`` lets tests point at a temp directory without
    touching the committed fixtures.
    """
    override = os.environ.get("TRACE_FAKE_RESPONSES_DIR")
    if override:
        return Path(override)
    repo_root = _find_repo_root(Path(__file__).resolve())
    return repo_root / "fixtures" / "llm_responses"


def _is_fake_mode() -> bool:
    return os.environ.get("TRACE_LLM_FAKE_MODE") == "1"


# --- Hashing & cost estimation ---------------------------------------------


def compute_request_hash(
    *,
    system: str,
    messages: Sequence[Any],
    model: str,
    max_tokens: int,
    tools: Any = None,
) -> str:
    """SHA-256 of a canonical JSON encoding of the request inputs.

    Keys are sorted and ``default=str`` accepts dataclasses / Path / datetime
    fields callers might pass in. The hash is the file-system key for canned
    fake-mode responses.
    """
    payload: dict[str, Any] = {
        "system": system,
        "messages": list(messages),
        "model": model,
        "max_tokens": max_tokens,
        "tools": tools,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def estimate_cost_usd(*, model: str, input_tokens: int, output_tokens: int) -> float:
    """Look up per-MTok pricing for ``model`` and return USD cost.

    Unknown models return ``0.0`` with a warning rather than crashing — the cost
    log is for observability, not authorization, and we'd rather have an
    incomplete log entry than a dropped call.
    """
    pricing = PRICING_USD_PER_MTOK.get(model)
    if pricing is None:
        LOGGER.warning("No pricing for model %s; cost recorded as 0.0", model)
        return 0.0
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


# --- Cost logging ----------------------------------------------------------


def _append_cost_line(line: Mapping[str, Any]) -> None:
    path = costs_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(line)) + "\n")


# --- Fake-mode helpers -----------------------------------------------------


def save_fake_response(
    *,
    request_hash: str,
    text: str,
    stop_reason: str | None = "end_turn",
    input_tokens: int = 0,
    output_tokens: int = 0,
    extra: Mapping[str, Any] | None = None,
    directory: Path | None = None,
) -> Path:
    """Write a canned response to ``<dir>/<request_hash>.json`` and return the path.

    Tests use this to register a known-good response for a given request hash.
    """
    base = directory if directory is not None else _fake_responses_dir()
    base.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "text": text,
        "stop_reason": stop_reason,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
    }
    if extra:
        payload.update(dict(extra))
    path = base / f"{request_hash}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def _load_fake_response(request_hash: str) -> dict[str, Any]:
    path = _fake_responses_dir() / f"{request_hash}.json"
    if not path.is_file():
        raise FakeResponseNotFound(
            f"No canned fake-mode response at {path}. Register one with "
            "synthesizer.llm_client.save_fake_response("
            f"request_hash='{request_hash}', text=..., input_tokens=..., output_tokens=...)."
        )
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


# --- The client -------------------------------------------------------------


class LLMClient:
    """Synthesizer-side wrapper around :class:`anthropic.Anthropic`.

    Reads ``ANTHROPIC_API_KEY`` from the environment at construction time and
    raises :class:`ConfigurationError` if missing. Honors ``TRACE_LLM_FAKE_MODE``
    at *call* time so tests can flip it per-test via ``monkeypatch``.

    Pass a custom ``sleep`` callable to make retry tests instantaneous.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        sleep: Callable[[float], None] | None = None,
        client: Anthropic | None = None,
    ) -> None:
        env_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        if not env_key:
            raise ConfigurationError(
                "ANTHROPIC_API_KEY is not set in the environment. Export it "
                "before constructing LLMClient (the synthesizer never reads it "
                "from disk)."
            )
        self._api_key = env_key
        self._sleep: Callable[[float], None] = sleep or time.sleep
        # Disable the SDK's built-in retries; we own the retry policy so the
        # backoff schedule and cost log entries stay consistent with the PRD.
        self._client = client or Anthropic(api_key=env_key, max_retries=0)

    def complete(
        self,
        *,
        messages: Sequence[Any],
        system: str,
        tools: Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model: str = DEFAULT_MODEL,
        context_label: str = "",
    ) -> LLMResponse:
        """Run a single ``messages.create`` call and return a normalized response.

        ``context_label`` is recorded in ``costs.jsonl`` so cost lines can be
        attributed to e.g. ``"draft:gmail_reply"`` or ``"revision:q2"`` after the
        fact. Empty string is fine.
        """
        if _is_fake_mode():
            return self._fake_complete(
                messages=messages,
                system=system,
                tools=tools,
                max_tokens=max_tokens,
                model=model,
                context_label=context_label,
            )
        return self._real_complete(
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            model=model,
            context_label=context_label,
        )

    # -- fake mode ---------------------------------------------------------

    def _fake_complete(
        self,
        *,
        messages: Sequence[Any],
        system: str,
        tools: Any,
        max_tokens: int,
        model: str,
        context_label: str,
    ) -> LLMResponse:
        request_hash = compute_request_hash(
            system=system,
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            tools=tools,
        )
        data = _load_fake_response(request_hash)
        response = LLMResponse(
            text=str(data["text"]),
            stop_reason=data.get("stop_reason"),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            raw=data,
        )
        _append_cost_line(
            {
                "timestamp_iso": datetime.now(UTC).isoformat(),
                "module": "synthesizer",
                "model": "fake",
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_estimate_usd": 0.0,
                "context_label": context_label,
            }
        )
        return response

    # -- real mode ---------------------------------------------------------

    def _real_complete(
        self,
        *,
        messages: Sequence[Any],
        system: str,
        tools: Any,
        max_tokens: int,
        model: str,
        context_label: str,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "messages": list(messages),
            "max_tokens": max_tokens,
            "model": model,
        }
        if system:
            kwargs["system"] = system
        if tools is not None:
            kwargs["tools"] = tools

        message = self._call_with_retry(kwargs)

        text = _extract_text(message)
        usage = getattr(message, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        raw: Mapping[str, Any]
        if hasattr(message, "model_dump"):
            raw = message.model_dump()
        else:
            raw = {"id": getattr(message, "id", None)}

        response = LLMResponse(
            text=text,
            stop_reason=getattr(message, "stop_reason", None),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw=raw,
        )
        _append_cost_line(
            {
                "timestamp_iso": datetime.now(UTC).isoformat(),
                "module": "synthesizer",
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_estimate_usd": estimate_cost_usd(
                    model=model, input_tokens=input_tokens, output_tokens=output_tokens
                ),
                "context_label": context_label,
            }
        )
        return response

    def _call_with_retry(self, kwargs: dict[str, Any]) -> Any:
        attempt = 0
        while True:
            try:
                return self._client.messages.create(**kwargs)
            except anthropic.AuthenticationError as exc:
                # 401 is a config bug, not transient — fail loudly.
                raise ConfigurationError(
                    "Anthropic returned 401 Unauthorized. Check that "
                    "ANTHROPIC_API_KEY points to a valid key for the "
                    "synthesizer (the dev key during Ralph iterations)."
                ) from exc
            except anthropic.APIStatusError as exc:
                status = getattr(exc, "status_code", None)
                if status not in RETRYABLE_STATUS_CODES or attempt >= MAX_RETRIES:
                    raise
                wait = RETRY_BACKOFF_SECONDS[attempt]
                LOGGER.warning(
                    "Anthropic call failed status=%s; retry %d/%d after %.1fs",
                    status,
                    attempt + 1,
                    MAX_RETRIES,
                    wait,
                )
                self._sleep(wait)
                attempt += 1
            except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
                # Network blips: same backoff schedule.
                if attempt >= MAX_RETRIES:
                    raise
                wait = RETRY_BACKOFF_SECONDS[attempt]
                LOGGER.warning(
                    "Anthropic call failed (%s); retry %d/%d after %.1fs",
                    type(exc).__name__,
                    attempt + 1,
                    MAX_RETRIES,
                    wait,
                )
                self._sleep(wait)
                attempt += 1


def _extract_text(message: Any) -> str:
    """Concatenate text blocks from an Anthropic ``Message`` content list."""
    content = getattr(message, "content", None) or []
    parts: list[str] = []
    for block in content:
        text_value = getattr(block, "text", None)
        if isinstance(text_value, str):
            parts.append(text_value)
    return "".join(parts)
