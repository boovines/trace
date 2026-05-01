"""Tests for :mod:`runner.live_input` — the CGEventPost-backed live adapter.

Two test modes share the file:

* **Mock-based tests** (default). They patch the PyObjC seams at the module
  level (``runner.live_input.CGEventPost``, ``CGEventCreateMouseEvent``, etc.)
  and verify the call shapes without touching the real event system. These
  run on any platform as long as PyObjC resolves at import time.
* **Live tests** marked ``@pytest.mark.macos @pytest.mark.live_input``. They
  post real events through the HID tap and verify them via a secondary
  ``CGEventTap``. Skipped by default; enable with ``--run-live-input``.

The ``live_mode_allowed`` fixture is required for **every** test that
constructs a ``LiveInputAdapter``, because ``__init__`` refuses to run unless
``TRACE_ALLOW_LIVE=1`` is set.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from runner import live_input as live_input_module
from runner.coords import DisplayInfo
from runner.input_adapter import InputAdapter
from runner.live_input import (
    DEFAULT_EVENT_DELAY_SECONDS,
    LiveInputAdapter,
    LiveInputError,
)
from runner.safety import LiveModeNotAllowed

_FAKE_DISPLAY = DisplayInfo(
    width_points=1440.0,
    height_points=900.0,
    scale_factor=2.0,
    width_pixels=2880,
    height_pixels=1800,
)


@pytest.fixture
def mock_quartz(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Patch every PyObjC symbol used by ``LiveInputAdapter``.

    Returns a dict of mocks keyed by Quartz symbol name so tests can assert on
    call shapes. Each ``*Create*`` mock returns a unique sentinel object so
    tests can verify the exact CFObject passed downstream to
    ``CGEventPost`` / ``CGEventSetFlags``.
    """
    mocks: dict[str, MagicMock] = {}
    for name in (
        "CGEventCreateMouseEvent",
        "CGEventCreateKeyboardEvent",
        "CGEventCreateScrollWheelEvent",
        "CGEventKeyboardSetUnicodeString",
        "CGEventPost",
        "CGEventSetFlags",
    ):
        mock = MagicMock(name=name)
        if name.startswith("CGEventCreate"):
            mock.side_effect = lambda *_a, _name=name, **_kw: object()
        else:
            mock.return_value = None
        monkeypatch.setattr(live_input_module, name, mock)
        mocks[name] = mock

    # Skip the event delay for fast tests; still covered by a dedicated test.
    monkeypatch.setattr(live_input_module.time, "sleep", lambda _s: None)
    return mocks


# --------------------------------------------------------------------------- #
# Safety gate
# --------------------------------------------------------------------------- #


def test_init_raises_without_flag() -> None:
    with pytest.raises(LiveModeNotAllowed) as exc:
        LiveInputAdapter(display_info=_FAKE_DISPLAY)
    assert "TRACE_ALLOW_LIVE" in str(exc.value)


def test_init_succeeds_with_flag(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed, mock_quartz
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)
    assert adapter.display_info is _FAKE_DISPLAY


def test_adapter_satisfies_input_adapter_protocol(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed, mock_quartz
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)
    assert isinstance(adapter, InputAdapter)


# --------------------------------------------------------------------------- #
# Mock-based: click / modifiers / buttons
# --------------------------------------------------------------------------- #


def test_click_posts_mouse_down_then_up(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)

    adapter.click(100.0, 200.0)

    create = mock_quartz["CGEventCreateMouseEvent"]
    assert create.call_count == 2
    first_call_args = create.call_args_list[0].args
    assert first_call_args[0] is None  # event source
    assert first_call_args[1] == int(live_input_module.kCGEventLeftMouseDown)
    assert first_call_args[2] == (100.0, 200.0)
    assert first_call_args[3] == int(live_input_module.kCGMouseButtonLeft)

    second_call_args = create.call_args_list[1].args
    assert second_call_args[1] == int(live_input_module.kCGEventLeftMouseUp)
    assert second_call_args[2] == (100.0, 200.0)

    post = mock_quartz["CGEventPost"]
    assert post.call_count == 2
    for call in post.call_args_list:
        assert call.args[0] == live_input_module.kCGHIDEventTap

    assert mock_quartz["CGEventSetFlags"].call_count == 0  # no modifiers


def test_click_right_button_posts_right_mouse_events(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)

    adapter.click(50.0, 75.0, button="right")

    types = [c.args[1] for c in mock_quartz["CGEventCreateMouseEvent"].call_args_list]
    assert types == [
        int(live_input_module.kCGEventRightMouseDown),
        int(live_input_module.kCGEventRightMouseUp),
    ]
    buttons = [c.args[3] for c in mock_quartz["CGEventCreateMouseEvent"].call_args_list]
    assert all(b == int(live_input_module.kCGMouseButtonRight) for b in buttons)


def test_click_middle_button_posts_other_mouse_events(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)

    adapter.click(10.0, 20.0, button="middle")

    types = [c.args[1] for c in mock_quartz["CGEventCreateMouseEvent"].call_args_list]
    assert types == [
        int(live_input_module.kCGEventOtherMouseDown),
        int(live_input_module.kCGEventOtherMouseUp),
    ]


def test_click_with_modifiers_sets_flags(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)

    adapter.click(1.0, 2.0, modifiers=["cmd", "shift"])

    expected_mask = int(live_input_module.kCGEventFlagMaskCommand) | int(
        live_input_module.kCGEventFlagMaskShift
    )
    set_flags = mock_quartz["CGEventSetFlags"]
    assert set_flags.call_count == 2  # once per (down, up)
    for call in set_flags.call_args_list:
        assert call.args[1] == expected_mask


def test_click_with_unknown_modifier_raises(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed, mock_quartz
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)
    with pytest.raises(LiveInputError, match="unknown modifier"):
        adapter.click(0.0, 0.0, modifiers=["hyper"])


def test_click_with_unknown_button_raises(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed, mock_quartz
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)
    with pytest.raises(LiveInputError, match="unknown mouse button"):
        adapter.click(0.0, 0.0, button="fourth")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Mock-based: type_text (Unicode-aware)
# --------------------------------------------------------------------------- #


def test_type_text_uses_unicode_string(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)

    adapter.type_text("hello")

    create = mock_quartz["CGEventCreateKeyboardEvent"]
    assert create.call_count == 2  # down + up
    assert [c.args[2] for c in create.call_args_list] == [True, False]

    set_uni = mock_quartz["CGEventKeyboardSetUnicodeString"]
    assert set_uni.call_count == 2
    for call in set_uni.call_args_list:
        assert call.args[1] == 5
        assert call.args[2] == "hello"


def test_type_text_emoji_and_non_ascii(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)

    text = "héllo \U0001f44b"  # héllo 👋
    adapter.type_text(text)

    set_uni = mock_quartz["CGEventKeyboardSetUnicodeString"]
    for call in set_uni.call_args_list:
        assert call.args[2] == text
        assert call.args[1] == len(text)


def test_type_text_empty_is_noop(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)

    adapter.type_text("")

    assert mock_quartz["CGEventCreateKeyboardEvent"].call_count == 0
    assert mock_quartz["CGEventPost"].call_count == 0


# --------------------------------------------------------------------------- #
# Mock-based: key_press
# --------------------------------------------------------------------------- #


def test_key_press_posts_downs_then_ups_in_reverse(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)

    adapter.key_press(["cmd", "n"])

    create = mock_quartz["CGEventCreateKeyboardEvent"]
    assert create.call_count == 4
    # (keycode, key_down) sequence = [(cmd, True), (n, True), (n, False), (cmd, False)]
    observed = [(c.args[1], c.args[2]) for c in create.call_args_list]
    assert observed == [(0x37, True), (0x2D, True), (0x2D, False), (0x37, False)]

    assert mock_quartz["CGEventPost"].call_count == 4


def test_key_press_unknown_key_raises(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed, mock_quartz
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)
    with pytest.raises(LiveInputError, match="unknown key"):
        adapter.key_press(["banana"])


def test_key_press_empty_is_noop(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)
    adapter.key_press([])
    assert mock_quartz["CGEventCreateKeyboardEvent"].call_count == 0
    assert mock_quartz["CGEventPost"].call_count == 0


# --------------------------------------------------------------------------- #
# Mock-based: scroll and move_mouse
# --------------------------------------------------------------------------- #


def test_scroll_down_posts_move_then_scroll(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)

    adapter.scroll(640.0, 480.0, "down", 10)

    # First a move_mouse, then the scroll event.
    mouse_calls = mock_quartz["CGEventCreateMouseEvent"].call_args_list
    assert len(mouse_calls) == 1
    assert mouse_calls[0].args[1] == int(live_input_module.kCGEventMouseMoved)
    assert mouse_calls[0].args[2] == (640.0, 480.0)

    scroll_create = mock_quartz["CGEventCreateScrollWheelEvent"]
    assert scroll_create.call_count == 1
    args = scroll_create.call_args_list[0].args
    assert args[0] is None
    assert args[1] == int(live_input_module.kCGScrollEventUnitPixel)
    assert args[2] == 1  # wheelCount
    assert args[3] == -10


def test_scroll_up_positive_delta(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)

    adapter.scroll(0.0, 0.0, "up", 5)

    args = mock_quartz["CGEventCreateScrollWheelEvent"].call_args_list[0].args
    assert args[2] == 1
    assert args[3] == 5


def test_scroll_right_uses_two_wheel_axes(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)

    adapter.scroll(0.0, 0.0, "right", 7)

    args = mock_quartz["CGEventCreateScrollWheelEvent"].call_args_list[0].args
    assert args[2] == 2  # wheelCount
    assert args[3] == 0
    assert args[4] == 7


def test_scroll_negative_amount_raises(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed, mock_quartz
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)
    with pytest.raises(LiveInputError, match="non-negative"):
        adapter.scroll(0.0, 0.0, "down", -1)


def test_move_mouse_posts_mouse_moved(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)

    adapter.move_mouse(320.0, 240.0)

    mouse_calls = mock_quartz["CGEventCreateMouseEvent"].call_args_list
    assert len(mouse_calls) == 1
    assert mouse_calls[0].args[1] == int(live_input_module.kCGEventMouseMoved)
    assert mouse_calls[0].args[2] == (320.0, 240.0)
    assert mock_quartz["CGEventPost"].call_count == 1


# --------------------------------------------------------------------------- #
# Mock-based: error wrapping, rate limiting, tap-state check
# --------------------------------------------------------------------------- #


def test_cg_event_post_exception_becomes_live_input_error(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed
    mock_quartz["CGEventPost"].side_effect = RuntimeError("kaboom")
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)
    with pytest.raises(LiveInputError, match=r"click failed:.*kaboom"):
        adapter.click(0.0, 0.0)


def test_cg_event_create_returning_none_raises(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed
    mock_quartz["CGEventCreateMouseEvent"].side_effect = None
    mock_quartz["CGEventCreateMouseEvent"].return_value = None
    adapter = LiveInputAdapter(display_info=_FAKE_DISPLAY)
    with pytest.raises(LiveInputError, match="returned None"):
        adapter.click(0.0, 0.0)


def test_event_delay_is_honored(
    live_mode_allowed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = live_mode_allowed
    # Reset every PyObjC seam but keep a real-enough event delay.
    for name in (
        "CGEventCreateMouseEvent",
        "CGEventCreateKeyboardEvent",
        "CGEventCreateScrollWheelEvent",
        "CGEventKeyboardSetUnicodeString",
        "CGEventPost",
        "CGEventSetFlags",
    ):
        m = MagicMock(name=name)
        if name.startswith("CGEventCreate"):
            m.return_value = object()
        monkeypatch.setattr(live_input_module, name, m)
    sleeps: list[float] = []
    monkeypatch.setattr(live_input_module.time, "sleep", lambda s: sleeps.append(s))

    adapter = LiveInputAdapter(
        display_info=_FAKE_DISPLAY, event_delay_seconds=0.123
    )
    adapter.click(10.0, 20.0)

    assert sleeps == [0.123, 0.123]  # one per posted event


def test_default_event_delay_constant() -> None:
    assert DEFAULT_EVENT_DELAY_SECONDS == 0.020


def test_post_calls_tap_state_check(
    live_mode_allowed: None, mock_quartz: dict[str, MagicMock]
) -> None:
    _ = live_mode_allowed, mock_quartz
    calls: list[str] = []

    class RecordingAdapter(LiveInputAdapter):
        def _check_event_tap_state(self, operation: str) -> None:
            calls.append(operation)

    adapter = RecordingAdapter(display_info=_FAKE_DISPLAY)
    adapter.click(1.0, 2.0)

    assert calls == ["click", "click"]  # one per posted event


# --------------------------------------------------------------------------- #
# Live tests — real CGEventPost; skipped unless --run-live-input is passed.
# --------------------------------------------------------------------------- #


def _make_event_tap_listener() -> tuple[Any, list[tuple[int, tuple[float, float]]]]:
    """Create a passive listen-only event tap.

    Returns ``(run_source, events)`` where ``events`` is the list the
    callback appends to. The caller is responsible for adding the source to
    the current run loop and removing it after the test.
    """
    from Quartz import (
        CFMachPortCreateRunLoopSource,
        CGEventMaskBit,
        CGEventTapCreate,
        kCGEventLeftMouseDown,
        kCGEventLeftMouseUp,
        kCGEventTapOptionListenOnly,
        kCGHeadInsertEventTap,
        kCGSessionEventTap,
    )

    events: list[tuple[int, tuple[float, float]]] = []
    mask = CGEventMaskBit(kCGEventLeftMouseDown) | CGEventMaskBit(kCGEventLeftMouseUp)

    def callback(proxy: Any, event_type: int, event: Any, refcon: Any) -> Any:
        from Quartz import CGEventGetLocation

        loc = CGEventGetLocation(event)
        events.append((event_type, (float(loc.x), float(loc.y))))
        return event

    tap = CGEventTapCreate(
        kCGSessionEventTap,
        kCGHeadInsertEventTap,
        kCGEventTapOptionListenOnly,
        mask,
        callback,
        None,
    )
    if tap is None:
        pytest.skip("could not create event tap (Accessibility permission?)")
    source = CFMachPortCreateRunLoopSource(None, tap, 0)
    return source, events


@pytest.mark.macos
@pytest.mark.live_input
@pytest.mark.skipif(sys.platform != "darwin", reason="requires real macOS HID")
def test_live_click_observed_by_event_tap(
    live_mode_allowed: None,
) -> None:  # pragma: no cover - opt-in
    _ = live_mode_allowed
    from Quartz import (
        CFRunLoopAddSource,
        CFRunLoopGetCurrent,
        CFRunLoopRemoveSource,
        CFRunLoopRunInMode,
        kCFRunLoopDefaultMode,
    )

    source, events = _make_event_tap_listener()
    loop = CFRunLoopGetCurrent()
    CFRunLoopAddSource(loop, source, kCFRunLoopDefaultMode)
    try:
        adapter = LiveInputAdapter(event_delay_seconds=0.01)
        adapter.click(100.0, 100.0)
        # Pump the run loop briefly so the tap delivers callbacks.
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.2, False)
    finally:
        CFRunLoopRemoveSource(loop, source, kCFRunLoopDefaultMode)

    assert len(events) == 2  # down + up


@pytest.mark.macos
@pytest.mark.live_input
@pytest.mark.skipif(sys.platform != "darwin", reason="requires real macOS HID")
def test_live_modifiers_observable(live_mode_allowed: None) -> None:  # pragma: no cover - opt-in
    _ = live_mode_allowed
    from Quartz import (
        CFMachPortCreateRunLoopSource,
        CFRunLoopAddSource,
        CFRunLoopGetCurrent,
        CFRunLoopRemoveSource,
        CFRunLoopRunInMode,
        CGEventGetFlags,
        CGEventMaskBit,
        CGEventTapCreate,
        kCFRunLoopDefaultMode,
        kCGEventFlagMaskCommand,
        kCGEventLeftMouseDown,
        kCGEventTapOptionListenOnly,
        kCGHeadInsertEventTap,
        kCGSessionEventTap,
    )

    flags_seen: list[int] = []

    def callback(proxy: Any, event_type: int, event: Any, refcon: Any) -> Any:
        flags_seen.append(int(CGEventGetFlags(event)))
        return event

    mask = CGEventMaskBit(kCGEventLeftMouseDown)
    tap = CGEventTapCreate(
        kCGSessionEventTap,
        kCGHeadInsertEventTap,
        kCGEventTapOptionListenOnly,
        mask,
        callback,
        None,
    )
    if tap is None:
        pytest.skip("could not create event tap")
    source = CFMachPortCreateRunLoopSource(None, tap, 0)
    loop = CFRunLoopGetCurrent()
    CFRunLoopAddSource(loop, source, kCFRunLoopDefaultMode)
    try:
        adapter = LiveInputAdapter(event_delay_seconds=0.01)
        adapter.click(80.0, 80.0, modifiers=["cmd"])
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.2, False)
    finally:
        CFRunLoopRemoveSource(loop, source, kCFRunLoopDefaultMode)

    assert flags_seen
    assert flags_seen[0] & int(kCGEventFlagMaskCommand)


@pytest.mark.macos
@pytest.mark.live_input
@pytest.mark.skipif(sys.platform != "darwin", reason="requires real macOS HID")
def test_live_unicode_text_observable(live_mode_allowed: None) -> None:  # pragma: no cover - opt-in
    _ = live_mode_allowed
    from Quartz import (
        CFMachPortCreateRunLoopSource,
        CFRunLoopAddSource,
        CFRunLoopGetCurrent,
        CFRunLoopRemoveSource,
        CFRunLoopRunInMode,
        CGEventKeyboardGetUnicodeString,
        CGEventMaskBit,
        CGEventTapCreate,
        kCFRunLoopDefaultMode,
        kCGEventKeyDown,
        kCGEventTapOptionListenOnly,
        kCGHeadInsertEventTap,
        kCGSessionEventTap,
    )

    captured: list[str] = []

    def callback(proxy: Any, event_type: int, event: Any, refcon: Any) -> Any:
        length, _, buf = CGEventKeyboardGetUnicodeString(event, 64, None, None)
        if buf:
            captured.append("".join(buf)[:length])
        return event

    mask = CGEventMaskBit(kCGEventKeyDown)
    tap = CGEventTapCreate(
        kCGSessionEventTap,
        kCGHeadInsertEventTap,
        kCGEventTapOptionListenOnly,
        mask,
        callback,
        None,
    )
    if tap is None:
        pytest.skip("could not create event tap")
    source = CFMachPortCreateRunLoopSource(None, tap, 0)
    loop = CFRunLoopGetCurrent()
    CFRunLoopAddSource(loop, source, kCFRunLoopDefaultMode)
    try:
        adapter = LiveInputAdapter(event_delay_seconds=0.01)
        adapter.type_text("héllo \U0001f44b")
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.3, False)
    finally:
        CFRunLoopRemoveSource(loop, source, kCFRunLoopDefaultMode)

    assert any("héllo" in s for s in captured)
