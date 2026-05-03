"""Tests for runner.safety — the TRACE_ALLOW_LIVE gate.

Every live-mode adapter depends on this gate. If these tests regress, every
downstream safety assumption breaks, so each behavior is pinned here.
"""

from __future__ import annotations

import pytest

from runner.safety import (
    LIVE_MODE_ENV_VAR,
    LiveModeNotAllowed,
    is_live_mode_allowed,
    require_live_mode,
)


def test_is_live_mode_allowed_default_is_false() -> None:
    assert is_live_mode_allowed() is False


def test_is_live_mode_allowed_true_only_for_exact_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(LIVE_MODE_ENV_VAR, "1")
    assert is_live_mode_allowed() is True


@pytest.mark.parametrize("value", ["0", "true", "True", "TRUE", "yes", "", "2", " 1", "1 "])
def test_is_live_mode_allowed_false_for_non_exact_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(LIVE_MODE_ENV_VAR, value)
    assert is_live_mode_allowed() is False


def test_require_live_mode_raises_when_unset() -> None:
    with pytest.raises(LiveModeNotAllowed) as excinfo:
        require_live_mode()
    assert LIVE_MODE_ENV_VAR in str(excinfo.value)


def test_require_live_mode_passes_when_set(live_mode_allowed: None) -> None:
    require_live_mode()


def test_require_live_mode_error_mentions_ralph() -> None:
    with pytest.raises(LiveModeNotAllowed) as excinfo:
        require_live_mode()
    msg = str(excinfo.value)
    assert LIVE_MODE_ENV_VAR in msg
    assert "Ralph" in msg or "ralph" in msg.lower()


def test_live_mode_not_allowed_is_runtime_error() -> None:
    assert issubclass(LiveModeNotAllowed, RuntimeError)
