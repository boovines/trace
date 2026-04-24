"""Tests for ``recorder.permissions``.

The real framework calls are never exercised here: every test patches either
the tight ``_check_*`` helpers (for composition coverage) or the PyObjC
symbols they import (for unit coverage of the helpers themselves). This keeps
the suite hermetic — it runs the same on a Ralph sandbox, CI, and a dev Mac.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

from recorder import permissions

# ---------------------------------------------------------------------------
# check_permissions() / get_missing_permissions_error() — combination tests
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_checks(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Factory that patches the three ``_check_*`` helpers to return given values."""

    def _apply(accessibility: bool, screen_recording: bool, input_monitoring: bool) -> None:
        monkeypatch.setattr(
            permissions, "_check_accessibility", lambda: accessibility
        )
        monkeypatch.setattr(
            permissions, "_check_screen_recording", lambda: screen_recording
        )
        monkeypatch.setattr(
            permissions, "_check_input_monitoring", lambda: input_monitoring
        )

    return _apply


@pytest.mark.parametrize(
    ("accessibility", "screen_recording", "input_monitoring"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_check_permissions_returns_all_three_booleans(
    patch_checks: Any,
    accessibility: bool,
    screen_recording: bool,
    input_monitoring: bool,
) -> None:
    patch_checks(accessibility, screen_recording, input_monitoring)

    status = permissions.check_permissions()

    assert status == {
        "accessibility": accessibility,
        "screen_recording": screen_recording,
        "input_monitoring": input_monitoring,
    }


def test_get_missing_permissions_error_returns_none_when_all_granted(
    patch_checks: Any,
) -> None:
    patch_checks(True, True, True)
    assert permissions.get_missing_permissions_error() is None


def test_get_missing_permissions_error_reports_single_missing(patch_checks: Any) -> None:
    patch_checks(True, False, True)

    err = permissions.get_missing_permissions_error()

    assert err is not None
    assert err["error"] == "missing_permission"
    assert err["permissions"] == ["screen_recording"]
    assert list(err["how_to_grant"].keys()) == ["screen_recording"]
    assert "Screen Recording" in err["how_to_grant"]["screen_recording"]


def test_get_missing_permissions_error_reports_all_missing_in_canonical_order(
    patch_checks: Any,
) -> None:
    patch_checks(False, False, False)

    err = permissions.get_missing_permissions_error()

    assert err is not None
    assert err["permissions"] == [
        "accessibility",
        "screen_recording",
        "input_monitoring",
    ]
    # how_to_grant must cover every missing permission and nothing else.
    assert set(err["how_to_grant"].keys()) == set(err["permissions"])
    for name in err["permissions"]:
        assert err["how_to_grant"][name], f"missing guidance for {name}"


def test_get_missing_permissions_error_preserves_order_when_some_missing(
    patch_checks: Any,
) -> None:
    patch_checks(False, True, False)

    err = permissions.get_missing_permissions_error()

    assert err is not None
    assert err["permissions"] == ["accessibility", "input_monitoring"]


# ---------------------------------------------------------------------------
# _check_accessibility — framework-level mocking
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_application_services(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Install a stand-in ``ApplicationServices`` module in ``sys.modules``."""
    module = types.ModuleType("ApplicationServices")
    module.kAXTrustedCheckOptionPrompt = "AXTrustedCheckOptionPrompt"  # type: ignore[attr-defined]
    module.AXIsProcessTrustedWithOptions = lambda _opts: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ApplicationServices", module)
    yield module


def test_check_accessibility_true_when_framework_returns_truthy(
    fake_application_services: types.ModuleType,
) -> None:
    captured: dict[str, Any] = {}

    def trusted(options: dict[Any, Any]) -> int:
        captured["options"] = options
        return 1

    fake_application_services.AXIsProcessTrustedWithOptions = trusted  # type: ignore[attr-defined]

    assert permissions._check_accessibility() is True
    # Must pass the prompt-option set to False — never prompt on inspection.
    assert captured["options"] == {
        fake_application_services.kAXTrustedCheckOptionPrompt: False  # type: ignore[attr-defined]
    }


def test_check_accessibility_false_when_framework_returns_falsey(
    fake_application_services: types.ModuleType,
) -> None:
    fake_application_services.AXIsProcessTrustedWithOptions = lambda _opts: 0  # type: ignore[attr-defined]
    assert permissions._check_accessibility() is False


def test_check_accessibility_false_when_framework_raises(
    fake_application_services: types.ModuleType,
) -> None:
    def boom(_opts: dict[Any, Any]) -> bool:
        raise RuntimeError("AX exploded")

    fake_application_services.AXIsProcessTrustedWithOptions = boom  # type: ignore[attr-defined]
    assert permissions._check_accessibility() is False


def test_check_accessibility_false_when_framework_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force an ImportError out of `from ApplicationServices import …`.
    broken = types.ModuleType("ApplicationServices")
    # Deliberately omit the two symbols we import.
    monkeypatch.setitem(sys.modules, "ApplicationServices", broken)

    assert permissions._check_accessibility() is False


# ---------------------------------------------------------------------------
# _check_screen_recording — framework-level mocking
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_quartz(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    module = types.ModuleType("Quartz")
    # Defaults — tests override per-case.
    module.CGPreflightScreenCaptureAccess = lambda: False  # type: ignore[attr-defined]
    module.CGMainDisplayID = lambda: 1  # type: ignore[attr-defined]
    module.CGDisplayCreateImageForRect = lambda _display, _rect: None  # type: ignore[attr-defined]
    module.CGRectMake = lambda x, y, w, h: (x, y, w, h)  # type: ignore[attr-defined]
    module.CGEventTapCreate = (  # type: ignore[attr-defined]
        lambda *args, **kwargs: None
    )
    module.CFMachPortInvalidate = lambda _tap: None  # type: ignore[attr-defined]
    module.kCGEventTapOptionListenOnly = 1  # type: ignore[attr-defined]
    module.kCGHeadInsertEventTap = 0  # type: ignore[attr-defined]
    module.kCGSessionEventTap = 1  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "Quartz", module)
    yield module


def test_check_screen_recording_uses_preflight_when_available(
    fake_quartz: types.ModuleType,
) -> None:
    fake_quartz.CGPreflightScreenCaptureAccess = lambda: True  # type: ignore[attr-defined]
    assert permissions._check_screen_recording() is True


def test_check_screen_recording_preflight_false(fake_quartz: types.ModuleType) -> None:
    fake_quartz.CGPreflightScreenCaptureAccess = lambda: False  # type: ignore[attr-defined]
    assert permissions._check_screen_recording() is False


def test_check_screen_recording_preflight_raises_falls_back_to_capture(
    fake_quartz: types.ModuleType,
) -> None:
    def boom() -> bool:
        raise RuntimeError("preflight broke")

    fake_quartz.CGPreflightScreenCaptureAccess = boom  # type: ignore[attr-defined]
    fake_quartz.CGDisplayCreateImageForRect = (  # type: ignore[attr-defined]
        lambda _display, _rect: object()
    )

    assert permissions._check_screen_recording() is True


def test_check_screen_recording_fallback_returns_false_when_capture_nil(
    fake_quartz: types.ModuleType,
) -> None:
    def boom() -> bool:
        raise RuntimeError("preflight broke")

    fake_quartz.CGPreflightScreenCaptureAccess = boom  # type: ignore[attr-defined]
    fake_quartz.CGDisplayCreateImageForRect = (  # type: ignore[attr-defined]
        lambda _display, _rect: None
    )

    assert permissions._check_screen_recording() is False


# ---------------------------------------------------------------------------
# _check_input_monitoring — framework-level mocking
# ---------------------------------------------------------------------------


def test_check_input_monitoring_true_when_tap_created(
    fake_quartz: types.ModuleType,
) -> None:
    calls: dict[str, Any] = {"invalidated": 0}

    sentinel_tap = object()
    fake_quartz.CGEventTapCreate = (  # type: ignore[attr-defined]
        lambda *args, **kwargs: sentinel_tap
    )

    def invalidate(tap: Any) -> None:
        calls["invalidated"] += 1
        assert tap is sentinel_tap

    fake_quartz.CFMachPortInvalidate = invalidate  # type: ignore[attr-defined]

    assert permissions._check_input_monitoring() is True
    assert calls["invalidated"] == 1


def test_check_input_monitoring_false_when_tap_is_none(
    fake_quartz: types.ModuleType,
) -> None:
    fake_quartz.CGEventTapCreate = (  # type: ignore[attr-defined]
        lambda *args, **kwargs: None
    )
    assert permissions._check_input_monitoring() is False


def test_check_input_monitoring_false_when_tap_create_raises(
    fake_quartz: types.ModuleType,
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("tap create exploded")

    fake_quartz.CGEventTapCreate = boom  # type: ignore[attr-defined]
    assert permissions._check_input_monitoring() is False


def test_check_input_monitoring_survives_invalidate_raising(
    fake_quartz: types.ModuleType,
) -> None:
    fake_quartz.CGEventTapCreate = (  # type: ignore[attr-defined]
        lambda *args, **kwargs: object()
    )

    def bad_invalidate(_tap: Any) -> None:
        raise RuntimeError("CFMachPortInvalidate is unhappy")

    fake_quartz.CFMachPortInvalidate = bad_invalidate  # type: ignore[attr-defined]

    # A clean-up failure does not taint the permission result — the tap did
    # get created, which is the signal we care about.
    assert permissions._check_input_monitoring() is True
