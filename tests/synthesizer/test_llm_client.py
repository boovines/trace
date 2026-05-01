"""Tests for ``synthesizer.llm_client``.

Covers the five PRD-required scenarios plus a few defenses that fell out of
implementation: cost-line shape, model-pricing math, fake-mode hash stability,
and that the retry path stops at the configured ceiling.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anthropic
import httpx
import pytest
import respx

from synthesizer.llm_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    MAX_RETRIES,
    PRICING_USD_PER_MTOK,
    ConfigurationError,
    FakeResponseNotFound,
    LLMClient,
    compute_request_hash,
    costs_log_path,
    estimate_cost_usd,
    save_fake_response,
)

# --- shared fixtures --------------------------------------------------------


@pytest.fixture
def isolated_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point ``TRACE_DATA_DIR`` at ``tmp_path`` so cost logs land under tmp."""
    monkeypatch.setenv("TRACE_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def isolated_fakes_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Use a per-test fake-responses dir so registrations don't leak."""
    fakes = tmp_path / "fakes"
    fakes.mkdir()
    monkeypatch.setenv("TRACE_FAKE_RESPONSES_DIR", str(fakes))
    return fakes


@pytest.fixture
def real_mode(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Disable fake mode for tests that exercise the real HTTP path (mocked)."""
    monkeypatch.delenv("TRACE_LLM_FAKE_MODE", raising=False)
    yield


def _no_sleep(_seconds: float) -> None:
    return None


def _success_message_payload(
    *, input_tokens: int = 10, output_tokens: int = 5, text: str = "hi"
) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": DEFAULT_MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _error_payload(kind: str, message: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": message}}


# --- fake mode --------------------------------------------------------------


def test_fake_mode_returns_canned_response(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    messages = [{"role": "user", "content": "hello"}]
    request_hash = compute_request_hash(
        system="sys",
        messages=messages,
        model=DEFAULT_MODEL,
        max_tokens=DEFAULT_MAX_TOKENS,
        tools=None,
    )
    save_fake_response(
        request_hash=request_hash,
        text="canned hello",
        input_tokens=42,
        output_tokens=7,
        directory=isolated_fakes_dir,
    )

    client = LLMClient()
    response = client.complete(messages=messages, system="sys")

    assert response.text == "canned hello"
    assert response.input_tokens == 42
    assert response.output_tokens == 7
    assert response.stop_reason == "end_turn"
    assert response.raw["text"] == "canned hello"


def test_fake_mode_missing_canned_response_raises(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    client = LLMClient()
    with pytest.raises(FakeResponseNotFound) as exc:
        client.complete(messages=[{"role": "user", "content": "x"}], system="s")
    assert "save_fake_response" in str(exc.value)


def test_fake_mode_logs_zero_cost_with_model_fake(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    messages = [{"role": "user", "content": "hello"}]
    request_hash = compute_request_hash(
        system="sys",
        messages=messages,
        model=DEFAULT_MODEL,
        max_tokens=DEFAULT_MAX_TOKENS,
        tools=None,
    )
    save_fake_response(
        request_hash=request_hash,
        text="ok",
        input_tokens=100,
        output_tokens=20,
        directory=isolated_fakes_dir,
    )
    LLMClient().complete(messages=messages, system="sys", context_label="unit-test")

    line = _read_last_cost_line(isolated_data_dir)
    assert line["module"] == "synthesizer"
    assert line["model"] == "fake"
    assert line["cost_estimate_usd"] == 0.0
    assert line["input_tokens"] == 100
    assert line["output_tokens"] == 20
    assert line["context_label"] == "unit-test"
    assert "timestamp_iso" in line


# --- real mode (mocked HTTP) -----------------------------------------------


def test_retry_on_429_then_succeeds(
    real_mode: None,
    isolated_data_dir: Path,
    anthropic_mock: respx.MockRouter,
) -> None:
    route = anthropic_mock.post("/v1/messages")
    route.side_effect = [
        httpx.Response(429, json=_error_payload("rate_limit_error", "slow down")),
        httpx.Response(429, json=_error_payload("rate_limit_error", "slow down")),
        httpx.Response(200, json=_success_message_payload(text="ok")),
    ]

    client = LLMClient(sleep=_no_sleep)
    response = client.complete(
        messages=[{"role": "user", "content": "ping"}],
        system="s",
    )

    assert route.call_count == 3
    assert response.text == "ok"
    assert response.input_tokens == 10
    assert response.output_tokens == 5


def test_401_fails_fast_without_retry(
    real_mode: None,
    isolated_data_dir: Path,
    anthropic_mock: respx.MockRouter,
) -> None:
    route = anthropic_mock.post("/v1/messages").mock(
        return_value=httpx.Response(
            401, json=_error_payload("authentication_error", "Invalid API Key")
        )
    )

    client = LLMClient(sleep=_no_sleep)
    with pytest.raises(ConfigurationError) as exc:
        client.complete(messages=[{"role": "user", "content": "x"}], system="s")

    assert route.call_count == 1
    assert "ANTHROPIC_API_KEY" in str(exc.value)
    # No cost line should be written when the call fails.
    assert not costs_log_path().exists()


def test_real_mode_logs_cost_after_success(
    real_mode: None,
    isolated_data_dir: Path,
    anthropic_mock: respx.MockRouter,
) -> None:
    anthropic_mock.post("/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json=_success_message_payload(input_tokens=1_000_000, output_tokens=500_000, text="hi"),
        )
    )

    client = LLMClient(sleep=_no_sleep)
    client.complete(
        messages=[{"role": "user", "content": "hi"}],
        system="s",
        context_label="draft:gmail_reply",
    )

    line = _read_last_cost_line(isolated_data_dir)
    assert line["module"] == "synthesizer"
    assert line["model"] == DEFAULT_MODEL
    assert line["input_tokens"] == 1_000_000
    assert line["output_tokens"] == 500_000
    # 1M input @ $3 + 0.5M output @ $15 = $3 + $7.5 = $10.5
    assert line["cost_estimate_usd"] == pytest.approx(10.5)
    assert line["context_label"] == "draft:gmail_reply"


def test_retry_gives_up_after_max_retries(
    real_mode: None,
    isolated_data_dir: Path,
    anthropic_mock: respx.MockRouter,
) -> None:
    route = anthropic_mock.post("/v1/messages").mock(
        return_value=httpx.Response(503, json=_error_payload("overloaded", "busy"))
    )

    client = LLMClient(sleep=_no_sleep)
    with pytest.raises(anthropic.APIStatusError) as exc:
        client.complete(messages=[{"role": "user", "content": "x"}], system="s")

    # Initial attempt + MAX_RETRIES retries == MAX_RETRIES + 1 total calls.
    assert route.call_count == MAX_RETRIES + 1
    # The final exception is the SDK's APIStatusError, not ConfigurationError.
    assert exc.value.status_code == 503


def test_non_retryable_status_raises_immediately(
    real_mode: None,
    isolated_data_dir: Path,
    anthropic_mock: respx.MockRouter,
) -> None:
    route = anthropic_mock.post("/v1/messages").mock(
        return_value=httpx.Response(400, json=_error_payload("invalid_request_error", "bad"))
    )
    client = LLMClient(sleep=_no_sleep)
    with pytest.raises(anthropic.APIStatusError):
        client.complete(messages=[{"role": "user", "content": "x"}], system="s")
    assert route.call_count == 1


# --- construction guards ---------------------------------------------------


def test_missing_api_key_raises_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigurationError) as exc:
        LLMClient()
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_explicit_api_key_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Should not raise when api_key is passed explicitly.
    LLMClient(api_key="test-explicit-key")


# --- pricing math ----------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "input_tokens", "output_tokens", "expected"),
    [
        ("claude-sonnet-4-5", 0, 0, 0.0),
        ("claude-sonnet-4-5", 1_000_000, 0, 3.0),
        ("claude-sonnet-4-5", 0, 1_000_000, 15.0),
        ("claude-haiku-4-5", 1_000_000, 1_000_000, 6.0),
        ("fake", 999_999, 999_999, 0.0),
    ],
)
def test_estimate_cost_usd(
    model: str, input_tokens: int, output_tokens: int, expected: float
) -> None:
    actual = estimate_cost_usd(
        model=model, input_tokens=input_tokens, output_tokens=output_tokens
    )
    assert actual == pytest.approx(expected)


def test_estimate_cost_usd_unknown_model_returns_zero() -> None:
    assert estimate_cost_usd(model="claude-mystery-9", input_tokens=10, output_tokens=10) == 0.0


def test_pricing_table_has_required_models() -> None:
    # The synthesizer ships against Sonnet 4.5; the similarity scorer (S-014)
    # ships against Haiku 4.5. Both must always have pricing entries.
    assert "claude-sonnet-4-5" in PRICING_USD_PER_MTOK
    assert "claude-haiku-4-5" in PRICING_USD_PER_MTOK
    for model, table in PRICING_USD_PER_MTOK.items():
        assert "input" in table, f"missing input pricing for {model}"
        assert "output" in table, f"missing output pricing for {model}"


# --- request hashing -------------------------------------------------------


def test_compute_request_hash_is_deterministic() -> None:
    a = compute_request_hash(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        model=DEFAULT_MODEL,
        max_tokens=4096,
    )
    b = compute_request_hash(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        model=DEFAULT_MODEL,
        max_tokens=4096,
    )
    assert a == b
    assert len(a) == 64  # sha256 hex length


def test_compute_request_hash_changes_on_input_change() -> None:
    base = compute_request_hash(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        model=DEFAULT_MODEL,
        max_tokens=4096,
    )
    other = compute_request_hash(
        system="s",
        messages=[{"role": "user", "content": "ho"}],
        model=DEFAULT_MODEL,
        max_tokens=4096,
    )
    assert base != other


# --- helpers ---------------------------------------------------------------


def _read_last_cost_line(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "costs.jsonl"
    assert path.is_file(), f"costs.jsonl was not written at {path}"
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert lines, "costs.jsonl is empty"
    parsed: dict[str, Any] = json.loads(lines[-1])
    return parsed
