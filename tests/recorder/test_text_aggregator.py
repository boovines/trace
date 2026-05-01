"""Tests for :mod:`recorder.text_aggregator`.

The aggregator is pure Python — no PyObjC in sight — so every test here
runs hermetically on the Ralph sandbox.  The one AX-backed helper
(``resolve_focused_field``) is exercised by stubbing
``ApplicationServices`` via ``sys.modules`` the same way the other
``recorder`` modules do it.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any

import pytest

from recorder.text_aggregator import (
    BACKSPACE_KEY_CODE,
    FORWARD_DELETE_KEY_CODE,
    TextAggregator,
    TextInputEvent,
    resolve_focused_field,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _key_event(
    chars: str | None = None,
    *,
    key_code: int | None = None,
    modifiers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "cg_event_type": "key_down",
        "chars": chars,
        "key_code": key_code,
        "modifiers": modifiers or [],
    }


def _emitter() -> tuple[list[TextInputEvent], Any]:
    received: list[TextInputEvent] = []

    def emit(payload: TextInputEvent) -> None:
        received.append(payload)

    return received, emit


def _type(agg: TextAggregator, text: str) -> None:
    for ch in text:
        agg.handle_key_event(_key_event(chars=ch))


# ---------------------------------------------------------------------------
# set_focus / flush behaviour
# ---------------------------------------------------------------------------


def test_flush_emits_current_buffer_and_clears_focus() -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit)

    agg.set_focus("com.google.Chrome", "search", "Search")
    _type(agg, "hello")
    agg.flush()

    assert received == [
        {"app_bundle_id": "com.google.Chrome", "text": "hello", "field_label": "Search"}
    ]
    # After flush there is nothing left — subsequent keystrokes have no
    # focused field to land in.
    _type(agg, "world")
    agg.flush()
    assert len(received) == 1


def test_flush_without_focus_is_noop() -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit)
    agg.flush()
    assert received == []


def test_empty_buffer_does_not_emit() -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit)
    agg.set_focus("com.apple.Notes", "body", "Body")
    _type(agg, "hi")
    agg.handle_key_event(_key_event(key_code=BACKSPACE_KEY_CODE))
    agg.handle_key_event(_key_event(key_code=BACKSPACE_KEY_CODE))
    agg.flush()
    assert received == []


def test_field_switch_flushes_previous_field_first() -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit)

    agg.set_focus("com.google.Chrome", "field_a", "A")
    _type(agg, "alpha")
    agg.set_focus("com.google.Chrome", "field_b", "B")
    _type(agg, "beta")
    agg.flush()

    assert received == [
        {"app_bundle_id": "com.google.Chrome", "text": "alpha", "field_label": "A"},
        {"app_bundle_id": "com.google.Chrome", "text": "beta", "field_label": "B"},
    ]


def test_app_switch_flushes_previous_field_first() -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit)

    agg.set_focus("com.google.Chrome", "addr", "Address bar")
    _type(agg, "trace.dev")
    agg.set_focus("com.apple.Notes", "body", "Body")
    _type(agg, "note")
    agg.flush()

    assert [e["app_bundle_id"] for e in received] == [
        "com.google.Chrome",
        "com.apple.Notes",
    ]
    assert received[0]["text"] == "trace.dev"
    assert received[1]["text"] == "note"


def test_set_focus_clears_when_bundle_id_none() -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit)
    agg.set_focus("com.apple.Notes", "body", None)
    _type(agg, "x")
    agg.set_focus(None, None, None)
    # The prior field flushed synchronously.
    assert received == [
        {"app_bundle_id": "com.apple.Notes", "text": "x", "field_label": None}
    ]
    _type(agg, "y")  # no current field — dropped
    agg.flush()
    assert len(received) == 1


def test_set_focus_refocuses_same_field_keeps_buffer_and_updates_label() -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit)
    agg.set_focus("com.apple.Notes", "body", None)
    _type(agg, "ab")
    # AX may resolve a richer label on the second pass — re-focusing the
    # same (app, key) must not flush the buffer, only refresh the label.
    agg.set_focus("com.apple.Notes", "body", "Body")
    _type(agg, "cd")
    agg.flush()
    assert received == [
        {"app_bundle_id": "com.apple.Notes", "text": "abcd", "field_label": "Body"}
    ]


def test_app_level_fallback_when_no_field_key() -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit)
    agg.set_focus("com.slack.Slack", None, None)
    _type(agg, "hey")
    # Re-focus the same app with still no field_key should be treated as
    # the same bucket (app-level) — no premature flush.
    agg.set_focus("com.slack.Slack", None, None)
    _type(agg, " team")
    agg.flush()
    assert received == [
        {
            "app_bundle_id": "com.slack.Slack",
            "text": "hey team",
            "field_label": None,
        }
    ]


# ---------------------------------------------------------------------------
# keystroke handling
# ---------------------------------------------------------------------------


def test_backspace_removes_last_character() -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit)
    agg.set_focus("com.apple.Notes", "body", "Body")
    _type(agg, "hello")
    agg.handle_key_event(_key_event(key_code=BACKSPACE_KEY_CODE))
    agg.handle_key_event(_key_event(key_code=BACKSPACE_KEY_CODE))
    agg.flush()
    assert received == [
        {"app_bundle_id": "com.apple.Notes", "text": "hel", "field_label": "Body"}
    ]


def test_forward_delete_also_removes_character() -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit)
    agg.set_focus("com.apple.Notes", "body", "Body")
    _type(agg, "ab")
    agg.handle_key_event(_key_event(key_code=FORWARD_DELETE_KEY_CODE))
    agg.flush()
    assert received == [
        {"app_bundle_id": "com.apple.Notes", "text": "a", "field_label": "Body"}
    ]


def test_backspace_on_empty_buffer_is_noop() -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit)
    agg.set_focus("com.apple.Notes", "body", "Body")
    agg.handle_key_event(_key_event(key_code=BACKSPACE_KEY_CODE))
    _type(agg, "x")
    agg.flush()
    assert received == [
        {"app_bundle_id": "com.apple.Notes", "text": "x", "field_label": "Body"}
    ]


@pytest.mark.parametrize("modifier", ["command", "control", "option"])
def test_shortcut_modifiers_skip_text_buffer(modifier: str) -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit)
    agg.set_focus("com.apple.Notes", "body", "Body")
    agg.handle_key_event(_key_event(chars="a", modifiers=[modifier]))
    _type(agg, "b")
    agg.flush()
    # Only the non-shortcut "b" landed in the buffer.
    assert received == [
        {"app_bundle_id": "com.apple.Notes", "text": "b", "field_label": "Body"}
    ]


def test_shift_and_fn_do_not_suppress_text() -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit)
    agg.set_focus("com.apple.Notes", "body", "Body")
    agg.handle_key_event(_key_event(chars="H", modifiers=["shift"]))
    agg.handle_key_event(_key_event(chars="i", modifiers=["fn"]))
    agg.flush()
    assert received == [
        {"app_bundle_id": "com.apple.Notes", "text": "Hi", "field_label": "Body"}
    ]


def test_unfocused_keystrokes_are_dropped() -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit)
    _type(agg, "nope")
    agg.flush()
    assert received == []


def test_non_string_chars_are_ignored() -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit)
    agg.set_focus("com.apple.Notes", "body", "Body")
    agg.handle_key_event(_key_event(chars=None))
    agg.handle_key_event(_key_event(chars=""))
    agg.handle_key_event({"key_code": 999})  # no chars, no special key_code
    _type(agg, "ok")
    agg.flush()
    assert received == [
        {"app_bundle_id": "com.apple.Notes", "text": "ok", "field_label": "Body"}
    ]


# ---------------------------------------------------------------------------
# idle timer
# ---------------------------------------------------------------------------


def test_idle_timeout_auto_flushes(monkeypatch: pytest.MonkeyPatch) -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit, idle_timeout=0.05)
    agg.set_focus("com.apple.Notes", "body", "Body")
    _type(agg, "idle")

    # Wait long enough for the Timer to fire.
    deadline = time.time() + 1.0
    while time.time() < deadline and not received:
        time.sleep(0.01)

    assert received == [
        {"app_bundle_id": "com.apple.Notes", "text": "idle", "field_label": "Body"}
    ]


def test_idle_timer_resets_on_new_keystroke() -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit, idle_timeout=0.2)
    agg.set_focus("com.apple.Notes", "body", "Body")
    _type(agg, "a")
    time.sleep(0.1)
    _type(agg, "b")
    time.sleep(0.1)
    # Only ~0.2s total elapsed since the *first* keystroke, but the second
    # keystroke reset the timer — so the buffer should still be live.
    assert agg.current_buffer() == "ab"
    agg.flush()
    assert received == [
        {"app_bundle_id": "com.apple.Notes", "text": "ab", "field_label": "Body"}
    ]


def test_stop_flushes_and_blocks_future_events() -> None:
    received, emit = _emitter()
    agg = TextAggregator(emit)
    agg.set_focus("com.apple.Notes", "body", "Body")
    _type(agg, "stopme")
    agg.stop()
    assert received == [
        {"app_bundle_id": "com.apple.Notes", "text": "stopme", "field_label": "Body"}
    ]
    # A post-stop focus/type round-trip is a silent no-op.
    agg.set_focus("com.apple.Notes", "body", "Body")
    _type(agg, "ignored")
    agg.flush()
    assert len(received) == 1
    # stop() a second time is idempotent.
    agg.stop()


def test_stop_without_focus_is_safe() -> None:
    _received, emit = _emitter()
    agg = TextAggregator(emit)
    agg.stop()  # no focus, no buffer — must not raise


def test_emit_exception_is_isolated() -> None:
    def boom(_payload: TextInputEvent) -> None:
        raise RuntimeError("consumer blew up")

    agg = TextAggregator(boom)
    agg.set_focus("com.apple.Notes", "body", "Body")
    _type(agg, "hello")
    # The raised error in the consumer must not tear down the aggregator.
    agg.flush()
    # Aggregator is still usable after a consumer failure.
    agg.set_focus("com.apple.Notes", "body", "Body")
    _type(agg, "again")
    agg.flush()


def test_invalid_idle_timeout_rejected() -> None:
    _received, emit = _emitter()
    with pytest.raises(ValueError):
        TextAggregator(emit, idle_timeout=0)
    with pytest.raises(ValueError):
        TextAggregator(emit, idle_timeout=-1.0)


# ---------------------------------------------------------------------------
# resolve_focused_field — AX helper
# ---------------------------------------------------------------------------


def _install_ax_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    create_app: Any,
    copy_attribute: Any,
) -> None:
    stub = types.ModuleType("ApplicationServices")
    stub.AXUIElementCreateApplication = create_app  # type: ignore[attr-defined]
    stub.AXUIElementCopyAttributeValue = copy_attribute  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ApplicationServices", stub)


def test_resolve_focused_field_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    app_elem = object()
    focused = object()

    def create_app(pid: int) -> object:
        assert pid == 1234
        return app_elem

    def copy_attribute(element: object, attribute: str, _none: Any) -> tuple[int, Any]:
        if element is app_elem and attribute == "AXFocusedUIElement":
            return (0, focused)
        if element is focused and attribute == "AXRole":
            return (0, "AXTextField")
        if element is focused and attribute == "AXIdentifier":
            return (0, "email_input")
        if element is focused and attribute == "AXTitle":
            return (0, "Email")
        return (0, None)

    _install_ax_stub(monkeypatch, create_app=create_app, copy_attribute=copy_attribute)

    key, label = resolve_focused_field(1234)
    assert key == "email_input"
    assert label == "Email"


def test_resolve_focused_field_non_text_role_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_elem = object()
    focused = object()

    def create_app(_pid: int) -> object:
        return app_elem

    def copy_attribute(element: object, attribute: str, _none: Any) -> tuple[int, Any]:
        if attribute == "AXFocusedUIElement":
            return (0, focused)
        if attribute == "AXRole":
            return (0, "AXButton")
        return (0, None)

    _install_ax_stub(monkeypatch, create_app=create_app, copy_attribute=copy_attribute)

    assert resolve_focused_field(42) == (None, None)


def test_resolve_focused_field_falls_back_to_description_then_role_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_elem = object()
    focused = object()

    def create_app(_pid: int) -> object:
        return app_elem

    def copy_attribute(element: object, attribute: str, _none: Any) -> tuple[int, Any]:
        if attribute == "AXFocusedUIElement":
            return (0, focused)
        if attribute == "AXRole":
            return (0, "AXTextArea")
        if attribute == "AXIdentifier":
            return (0, None)
        if attribute == "AXTitle":
            return (0, None)
        if attribute == "AXDescription":
            return (0, "Message body")
        return (0, None)

    _install_ax_stub(monkeypatch, create_app=create_app, copy_attribute=copy_attribute)

    key, label = resolve_focused_field(42)
    assert label == "Message body"
    # No identifier — key should fold the role + label together.
    assert key == "role:AXTextArea:label:Message body"


def test_resolve_focused_field_no_identifier_no_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_elem = object()
    focused = object()

    def create_app(_pid: int) -> object:
        return app_elem

    def copy_attribute(element: object, attribute: str, _none: Any) -> tuple[int, Any]:
        if attribute == "AXFocusedUIElement":
            return (0, focused)
        if attribute == "AXRole":
            return (0, "AXSearchField")
        return (0, None)

    _install_ax_stub(monkeypatch, create_app=create_app, copy_attribute=copy_attribute)

    key, label = resolve_focused_field(42)
    assert label is None
    assert key == "role:AXSearchField"


def test_resolve_focused_field_ax_error_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def create_app(_pid: int) -> object:
        return object()

    def copy_attribute(_element: object, _attribute: str, _none: Any) -> tuple[int, Any]:
        return (-25212, None)  # kAXErrorAPIDisabled

    _install_ax_stub(monkeypatch, create_app=create_app, copy_attribute=copy_attribute)

    assert resolve_focused_field(42) == (None, None)


def test_resolve_focused_field_handles_application_services_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = types.ModuleType("ApplicationServices")
    # No AX attributes → ``from ApplicationServices import ...`` raises.
    monkeypatch.setitem(sys.modules, "ApplicationServices", stub)

    assert resolve_focused_field(42) == (None, None)


def test_resolve_focused_field_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    slow_started = threading.Event()

    def create_app(_pid: int) -> object:
        slow_started.set()
        time.sleep(0.5)
        return object()

    def copy_attribute(_element: object, _attribute: str, _none: Any) -> tuple[int, Any]:
        return (0, None)

    _install_ax_stub(monkeypatch, create_app=create_app, copy_attribute=copy_attribute)

    start = time.monotonic()
    key, label = resolve_focused_field(42, timeout_seconds=0.05)
    elapsed = time.monotonic() - start
    assert (key, label) == (None, None)
    assert elapsed < 0.4  # well under the 0.5s sleep
    # The worker thread was allowed to start before we gave up — rules out
    # a "never dispatched" false positive.
    assert slow_started.is_set()


def test_resolve_focused_field_null_app_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def create_app(_pid: int) -> None:
        return None

    def copy_attribute(*_args: Any) -> tuple[int, Any]:
        raise AssertionError("should not be called when app element is None")

    _install_ax_stub(monkeypatch, create_app=create_app, copy_attribute=copy_attribute)

    assert resolve_focused_field(42) == (None, None)
