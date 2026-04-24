"""Tests for ``recorder.event_tap``.

The real ``CGEventTap`` is only ever exercised in tests marked
``@pytest.mark.macos`` — those require Accessibility + Input Monitoring
permission on a real Mac.  Everything else is hermetic: we call
:meth:`EventTap.handle_cg_event` directly with a stub ``CGEventRef`` and a
stub ``Quartz`` module injected via ``sys.modules``, so the full
normalisation + tap-reenable behaviour is covered on Ralph's sandbox.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any

import pytest

from recorder import event_tap
from recorder.event_tap import EventTap

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _CGPoint:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def _install_stub_quartz(
    monkeypatch: pytest.MonkeyPatch,
    *,
    location: tuple[float, float] = (100.0, 200.0),
    flags: int = 0,
    integer_fields: dict[int, int] | None = None,
) -> types.ModuleType:
    """Install a minimal ``Quartz`` stub into ``sys.modules``.

    Only the handful of symbols ``recorder.event_tap`` imports inside its
    functions are populated; anything else raises ``AttributeError`` so a
    new code path needing more API surface will fail loudly.
    """
    fields = integer_fields or {}
    stub = types.ModuleType("Quartz")

    def CGEventGetLocation(event: Any) -> _CGPoint:
        return _CGPoint(*location)

    def CGEventGetFlags(event: Any) -> int:
        return flags

    def CGEventGetIntegerValueField(event: Any, field: int) -> int:
        return fields.get(field, 0)

    stub.CGEventGetLocation = CGEventGetLocation  # type: ignore[attr-defined]
    stub.CGEventGetFlags = CGEventGetFlags  # type: ignore[attr-defined]
    stub.CGEventGetIntegerValueField = CGEventGetIntegerValueField  # type: ignore[attr-defined]
    # Field-id sentinels — only identity matters, so ints are fine.
    stub.kCGKeyboardEventKeycode = 9  # type: ignore[attr-defined]
    stub.kCGScrollWheelEventDeltaAxis1 = 11  # type: ignore[attr-defined]
    stub.kCGScrollWheelEventDeltaAxis2 = 12  # type: ignore[attr-defined]
    # Minimum run-loop constants so anything that does try to import them
    # from Quartz during teardown doesn't explode.
    stub.CGEventTapEnable = lambda tap, on: None  # type: ignore[attr-defined]
    stub.CFRunLoopStop = lambda loop: None  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "Quartz", stub)
    return stub


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_event_mask_covers_expected_types() -> None:
    mask = event_tap._event_mask_for_recorder()
    for t in (1, 2, 3, 4, 10, 11, 12, 22, 25, 26):
        assert mask & (1 << t), f"mask missing type {t}"
    # Mouse-moved (5) is deliberately NOT captured — too chatty to store.
    assert mask & (1 << 5) == 0
    # Key-up (11) IS captured so text aggregator can fingerprint release
    # timing; verify explicitly so future edits don't silently drop it.
    assert mask & (1 << 11)


def test_modifiers_from_flags_empty() -> None:
    assert event_tap._modifiers_from_flags(0) == []


def test_modifiers_from_flags_combined() -> None:
    # shift + command + fn
    flags = 0x00020000 | 0x00100000 | 0x00800000
    mods = event_tap._modifiers_from_flags(flags)
    assert set(mods) == {"shift", "command", "fn"}


def test_modifiers_from_flags_unknown_bits_ignored() -> None:
    # Random high bits shouldn't produce spurious modifier names.
    assert event_tap._modifiers_from_flags(0x40000000) == []


# ---------------------------------------------------------------------------
# _normalise_event
# ---------------------------------------------------------------------------


def test_normalise_click_event(monkeypatch: pytest.MonkeyPatch) -> None:
    quartz = _install_stub_quartz(
        monkeypatch, location=(300.5, 400.75), flags=0x00100000  # command
    )
    t0 = int(time.time() * 1000)
    payload = event_tap._normalise_event(1, object(), quartz)
    assert payload["cg_event_type"] == "left_mouse_down"
    assert payload["cg_event_type_code"] == 1
    assert payload["location_x"] == pytest.approx(300.5)
    assert payload["location_y"] == pytest.approx(400.75)
    assert payload["modifiers"] == ["command"]
    assert payload["timestamp_ms"] >= t0
    # No key_code / scroll fields on a click.
    assert "key_code" not in payload
    assert "scroll_delta_y" not in payload


def test_normalise_key_down_includes_key_code(monkeypatch: pytest.MonkeyPatch) -> None:
    quartz = _install_stub_quartz(monkeypatch, integer_fields={9: 42})
    payload = event_tap._normalise_event(10, object(), quartz)
    assert payload["cg_event_type"] == "key_down"
    assert payload["key_code"] == 42


def test_normalise_scroll_includes_both_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    quartz = _install_stub_quartz(monkeypatch, integer_fields={11: -3, 12: 7})
    payload = event_tap._normalise_event(22, object(), quartz)
    assert payload["cg_event_type"] == "scroll_wheel"
    assert payload["scroll_delta_y"] == -3
    assert payload["scroll_delta_x"] == 7


def test_normalise_unknown_event_type_preserves_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quartz = _install_stub_quartz(monkeypatch)
    payload = event_tap._normalise_event(99, object(), quartz)
    assert payload["cg_event_type"] == "unknown_99"
    assert payload["cg_event_type_code"] == 99


def test_normalise_handles_framework_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``CGEventGetLocation`` explodes we fall back to ``None`` rather than
    leaking an exception up to the run-loop thread."""
    quartz = _install_stub_quartz(monkeypatch)

    def boom(event: Any) -> None:
        raise RuntimeError("simulated PyObjC failure")

    quartz.CGEventGetLocation = boom  # type: ignore[attr-defined]
    payload = event_tap._normalise_event(1, object(), quartz)
    assert payload["location_x"] is None
    assert payload["location_y"] is None


# ---------------------------------------------------------------------------
# handle_cg_event — delivery + tap-reenable
# ---------------------------------------------------------------------------


def test_handle_cg_event_delivers_to_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stub_quartz(monkeypatch, integer_fields={9: 11})
    received: list[dict[str, Any]] = []
    tap = EventTap(callback=received.append)
    tap.handle_cg_event(10, object())
    assert len(received) == 1
    assert received[0]["cg_event_type"] == "key_down"
    assert received[0]["key_code"] == 11


def test_handle_cg_event_reenables_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the OS disables the tap under load we must re-enable it and
    surface a synthetic ``tap_reenabled`` event to the callback."""
    enable_calls: list[tuple[Any, bool]] = []

    def CGEventTapEnable(tap_arg: Any, on: bool) -> None:
        enable_calls.append((tap_arg, on))

    quartz = _install_stub_quartz(monkeypatch)
    quartz.CGEventTapEnable = CGEventTapEnable  # type: ignore[attr-defined]

    received: list[dict[str, Any]] = []
    tap = EventTap(callback=received.append)
    tap._tap = object()  # simulate "already started"

    tap.handle_cg_event(0xFFFFFFFE, object())  # kCGEventTapDisabledByTimeout

    assert len(received) == 1
    assert received[0]["cg_event_type"] == "tap_reenabled"
    assert received[0]["reason"] == "timeout"
    assert enable_calls and enable_calls[0][1] is True


def test_handle_cg_event_reenables_on_user_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quartz = _install_stub_quartz(monkeypatch)
    quartz.CGEventTapEnable = lambda tap_arg, on: None  # type: ignore[attr-defined]
    received: list[dict[str, Any]] = []
    tap = EventTap(callback=received.append)
    tap._tap = object()
    tap.handle_cg_event(-1, object())  # kCGEventTapDisabledByUserInput (signed form)
    assert received[0]["cg_event_type"] == "tap_reenabled"
    assert received[0]["reason"] == "user_input"


def test_handle_cg_event_suppressed_while_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once ``stop()`` has set the stopping flag, in-flight events are
    swallowed so the consumer never sees phantom events post-stop."""
    _install_stub_quartz(monkeypatch)
    received: list[dict[str, Any]] = []
    tap = EventTap(callback=received.append)
    tap._stopping = True
    tap.handle_cg_event(1, object())
    assert received == []


def test_tap_disabled_during_stop_is_silent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``CGEventTapEnable(False)`` triggers an echo ``kCGEventTapDisabled*``
    event back through the callback. While we're stopping, that's our own
    teardown speaking — no warning, no re-enable attempt, no synthetic
    ``tap_reenabled`` delivery (which would be dropped anyway)."""
    quartz = _install_stub_quartz(monkeypatch)
    enable_calls: list[tuple[Any, bool]] = []
    quartz.CGEventTapEnable = (  # type: ignore[attr-defined]
        lambda tap_arg, on: enable_calls.append((tap_arg, on))
    )
    received: list[dict[str, Any]] = []
    tap = EventTap(callback=received.append)
    tap._tap = object()
    tap._stopping = True

    import logging

    with caplog.at_level(logging.WARNING, logger="recorder.event_tap"):
        tap.handle_cg_event(-1, object())  # kCGEventTapDisabledByUserInput

    assert received == []
    assert enable_calls == []
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("CGEventTap disabled" in r.getMessage() for r in warnings)


def test_callback_exception_does_not_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misbehaving consumer callback must not tear down the tap."""
    _install_stub_quartz(monkeypatch)

    def bad(_: dict[str, Any]) -> None:
        raise ValueError("consumer bug")

    tap = EventTap(callback=bad)
    tap.handle_cg_event(1, object())  # should log + swallow


# ---------------------------------------------------------------------------
# start()/stop() using a stubbed Quartz that does not spin a real run-loop
# ---------------------------------------------------------------------------


def _install_runloop_stub_quartz(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A Quartz stub that emulates ``CFRunLoopRun`` with a ``threading.Event``.

    The tap thread blocks on ``run_event`` until ``CFRunLoopStop`` is called,
    which sets the event.  This lets us exercise the real start()/stop()
    orchestration without touching macOS frameworks.
    """
    stub = _install_stub_quartz(monkeypatch)

    state: dict[str, Any] = {
        "tap": object(),
        "source": object(),
        "run_loop": object(),
        "run_event": threading.Event(),
        "enable_calls": [],
        "stopped": False,
    }

    def CGEventTapCreate(*args: Any, **kwargs: Any) -> Any:
        return state["tap"]

    def CFMachPortCreateRunLoopSource(*args: Any) -> Any:
        return state["source"]

    def CFRunLoopGetCurrent() -> Any:
        return state["run_loop"]

    def CFRunLoopAddSource(*args: Any) -> None:
        return None

    def CFRunLoopRemoveSource(*args: Any) -> None:
        return None

    def CGEventTapEnable(tap_arg: Any, on: bool) -> None:
        state["enable_calls"].append((tap_arg, on))

    def CFRunLoopRun() -> None:
        state["run_event"].wait(timeout=5.0)

    def CFRunLoopStop(loop: Any) -> None:
        state["stopped"] = True
        state["run_event"].set()

    stub.CGEventTapCreate = CGEventTapCreate  # type: ignore[attr-defined]
    stub.CFMachPortCreateRunLoopSource = CFMachPortCreateRunLoopSource  # type: ignore[attr-defined]
    stub.CFRunLoopGetCurrent = CFRunLoopGetCurrent  # type: ignore[attr-defined]
    stub.CFRunLoopAddSource = CFRunLoopAddSource  # type: ignore[attr-defined]
    stub.CFRunLoopRemoveSource = CFRunLoopRemoveSource  # type: ignore[attr-defined]
    stub.CGEventTapEnable = CGEventTapEnable  # type: ignore[attr-defined]
    stub.CFRunLoopRun = CFRunLoopRun  # type: ignore[attr-defined]
    stub.CFRunLoopStop = CFRunLoopStop  # type: ignore[attr-defined]
    stub.kCGHIDEventTap = 0  # type: ignore[attr-defined]
    stub.kCGSessionEventTap = 1  # type: ignore[attr-defined]
    stub.kCGHeadInsertEventTap = 0  # type: ignore[attr-defined]
    stub.kCGEventTapOptionListenOnly = 1  # type: ignore[attr-defined]
    stub.kCFRunLoopCommonModes = object()  # type: ignore[attr-defined]

    stub._test_state = state  # type: ignore[attr-defined]
    return stub


def test_start_and_stop_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _install_runloop_stub_quartz(monkeypatch)
    received: list[dict[str, Any]] = []
    tap = EventTap(callback=received.append)
    tap.start()
    assert tap._tap is stub._test_state["tap"]  # type: ignore[attr-defined]
    # Tap should have been explicitly enabled after install.
    calls = stub._test_state["enable_calls"]  # type: ignore[attr-defined]
    assert any(on is True for _, on in calls)

    tap.stop()
    assert stub._test_state["stopped"] is True  # type: ignore[attr-defined]
    # A second stop() is a no-op.
    tap.stop()


def test_start_twice_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_runloop_stub_quartz(monkeypatch)
    tap = EventTap(callback=lambda _: None)
    tap.start()
    try:
        with pytest.raises(RuntimeError, match="more than once"):
            tap.start()
    finally:
        tap.stop()


def test_start_raises_when_tap_create_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_runloop_stub_quartz(monkeypatch)
    stub.CGEventTapCreate = lambda *a, **kw: None  # type: ignore[attr-defined]
    tap = EventTap(callback=lambda _: None)
    with pytest.raises(RuntimeError, match="permission"):
        tap.start()


# ---------------------------------------------------------------------------
# real-macOS integration smoke (skipped unless on darwin with permissions)
# ---------------------------------------------------------------------------


@pytest.mark.macos
def test_macos_real_tap_receives_synthesised_click() -> None:  # pragma: no cover
    """Create a real tap, post a synthetic click, verify it's received.

    Requires Accessibility + Input Monitoring permission for the terminal
    running pytest.  Skipped automatically off-darwin because the ``macos``
    marker filters it out of the default run.
    """
    Quartz = pytest.importorskip("Quartz")

    received: list[dict[str, Any]] = []
    tap = EventTap(callback=received.append)
    try:
        tap.start()
    except RuntimeError as exc:
        pytest.skip(f"real CGEventTap unavailable: {exc}")

    try:
        ev = Quartz.CGEventCreateMouseEvent(
            None, 1, (10.0, 10.0), 0  # kCGEventLeftMouseDown
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        # Give the tap thread a moment to process.
        deadline = time.time() + 2.0
        while time.time() < deadline and not received:
            time.sleep(0.05)
    finally:
        tap.stop()

    click_events = [e for e in received if e["cg_event_type"] == "left_mouse_down"]
    assert click_events, f"no click events received; got: {received}"
