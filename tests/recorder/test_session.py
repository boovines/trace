"""Tests for :mod:`recorder.session` (R-010).

Hermetic coverage of the orchestrator:

* Every macOS-dependent dependency (``EventTap``, ``FocusTracker``,
  ``TextAggregator``, ``resolve_element_at``, ``capture_main_display``,
  ``get_main_display_info``, ``get_missing_permissions_error``) is
  injected via constructor hooks.  Tests never touch PyObjC.
* Synthetic ``EventTap``/``FocusTracker`` fakes drive the session via
  their registered callbacks, producing a real on-disk trajectory the
  test then re-reads.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from recorder.focus_tracker import (
    AppInfo,
    AppSwitchPayload,
    WindowFocusPayload,
)
from recorder.keyframe_policy import KeyframePolicy
from recorder.permissions import PermissionsError
from recorder.session import (
    PermissionsMissingError,
    RecordingSession,
    SessionAlreadyActiveError,
    SessionNotActiveError,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNG_BYTES = PNG_MAGIC + b"\x00" * 32  # only used as writer input; schema
#                                       never inspects contents.


# --------------------------------------------------------- fakes (no PyObjC)


class FakeEventTap:
    """Stand-in for :class:`recorder.event_tap.EventTap`.

    Tests dispatch synthetic event dicts via :meth:`fire`, which calls the
    callback the session registered.  ``start``/``stop`` just flip flags so
    assertions can verify lifecycle.
    """

    def __init__(self, callback: Any) -> None:
        self.callback = callback
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def fire(self, event: dict[str, Any]) -> None:
        self.callback(event)


class FakeFocusTracker:
    """In-memory ``FocusTracker`` that invokes callbacks synchronously."""

    def __init__(self) -> None:
        self.app_switch_cbs: list[Any] = []
        self.window_focus_cbs: list[Any] = []
        self.current_app: AppInfo | None = None
        self.history: list[dict[str, Any]] = []
        self.started = False
        self.stopped = False

    def on_app_switch(self, callback: Any) -> None:
        self.app_switch_cbs.append(callback)

    def on_window_focus_change(self, callback: Any) -> None:
        self.window_focus_cbs.append(callback)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        if self.history and self.history[-1].get("exited_at") is None:
            self.history[-1]["exited_at"] = "2026-04-22T10:10:00.000+00:00"

    def get_current_app(self) -> AppInfo | None:
        return self.current_app

    def get_app_focus_history(self) -> list[dict[str, Any]]:
        return list(self.history)

    # test helpers ---------------------------------------------------------

    def fire_app_switch(self, payload: AppSwitchPayload) -> None:
        new_app: AppInfo = {
            "bundle_id": payload["to_bundle_id"],
            "name": payload.get("to_name") or "",
            "pid": 1000 + len(self.history),
        }
        entered_at = f"2026-04-22T10:00:{len(self.history):02d}.000+00:00"
        if self.history and self.history[-1].get("exited_at") is None:
            self.history[-1]["exited_at"] = entered_at
        self.history.append(
            {
                "bundle_id": new_app["bundle_id"],
                "name": new_app["name"],
                "entered_at": entered_at,
                "exited_at": None,
            }
        )
        self.current_app = new_app
        for cb in self.app_switch_cbs:
            cb(payload)

    def fire_window_focus(self, payload: WindowFocusPayload) -> None:
        for cb in self.window_focus_cbs:
            cb(payload)


class FakeTextAggregator:
    """Minimal TextAggregator that records calls and supports manual emit."""

    def __init__(self, emit: Any) -> None:
        self.emit = emit
        self.set_focus_calls: list[tuple[Any, ...]] = []
        self.key_events: list[dict[str, Any]] = []
        self.stopped = False

    def set_focus(self, bundle_id: Any, key: Any, label: Any) -> None:
        self.set_focus_calls.append((bundle_id, key, label))

    def handle_key_event(self, event: dict[str, Any]) -> None:
        self.key_events.append(event)

    def stop(self) -> None:
        self.stopped = True

    def fire_text_input(self, text: str, field_label: str | None, bundle_id: str) -> None:
        self.emit(
            {
                "text": text,
                "field_label": field_label,
                "app_bundle_id": bundle_id,
            }
        )


# ------------------------------------------------------- common helpers


def _build_session(
    tmp_path: Path,
    *,
    permissions_missing: PermissionsError | None = None,
    display_info: dict[str, Any] | None = None,
    screenshot_bytes: bytes | None = PNG_BYTES,
    screenshot_raises: bool = False,
    ax_target: dict[str, Any] | None = None,
    ax_raises: bool = False,
    keyframe_policy: KeyframePolicy | None = None,
) -> tuple[RecordingSession, dict[str, Any]]:
    """Build a session with fakes and return `(session, bag)` for assertions."""
    bag: dict[str, Any] = {}

    def event_tap_factory(cb: Any) -> FakeEventTap:
        tap = FakeEventTap(cb)
        bag["event_tap"] = tap
        return tap

    def focus_tracker_factory() -> FakeFocusTracker:
        ft = FakeFocusTracker()
        bag["focus_tracker"] = ft
        return ft

    def text_aggregator_factory(emit: Any) -> FakeTextAggregator:
        ta = FakeTextAggregator(emit)
        bag["text_aggregator"] = ta
        return ta

    def permissions_check() -> PermissionsError | None:
        return permissions_missing

    def ax_resolver(x: float, y: float) -> dict[str, Any] | None:
        bag.setdefault("ax_calls", []).append((x, y))
        if ax_raises:
            raise RuntimeError("ax boom")
        return ax_target

    def screenshot_capturer() -> bytes | None:
        bag.setdefault("screenshot_calls", []).append(time.monotonic())
        if screenshot_raises:
            raise RuntimeError("screenshot boom")
        return screenshot_bytes

    def display_info_provider() -> dict[str, Any] | None:
        return display_info if display_info is not None else {
            "width": 1920,
            "height": 1080,
            "scale_factor": 2.0,
        }

    session = RecordingSession(
        tmp_path,
        event_tap_factory=event_tap_factory,
        focus_tracker_factory=focus_tracker_factory,
        text_aggregator_factory=text_aggregator_factory,
        permissions_check=permissions_check,
        ax_resolver=ax_resolver,
        screenshot_capturer=screenshot_capturer,
        display_info_provider=display_info_provider,
        keyframe_policy=keyframe_policy,
    )
    return session, bag


def _read_events(tmp_path: Path, trajectory_id: str) -> list[dict[str, Any]]:
    events_path = tmp_path / trajectory_id / "events.jsonl"
    with events_path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _read_metadata(tmp_path: Path, trajectory_id: str) -> dict[str, Any]:
    with (tmp_path / trajectory_id / "metadata.json").open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
        return data


# ------------------------------------------------------------- lifecycle tests


def test_start_creates_trajectory_and_writes_metadata(tmp_path: Path) -> None:
    session, bag = _build_session(tmp_path)
    trajectory_id = session.start("demo")

    try:
        assert session.is_active() is True
        assert session.trajectory_id == trajectory_id
        assert bag["event_tap"].started is True
        assert bag["focus_tracker"].started is True
        assert bag["focus_tracker"].app_switch_cbs, "session should subscribe to app_switch"
        assert bag["focus_tracker"].window_focus_cbs, "session should subscribe to window_focus"

        meta = _read_metadata(tmp_path, trajectory_id)
        assert meta["id"] == trajectory_id
        assert meta["label"] == "demo"
        assert meta["stopped_at"] is None
        assert meta["display_info"] == {"width": 1920, "height": 1080, "scale_factor": 2.0}
    finally:
        session.stop()


def test_start_while_active_raises(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path)
    session.start("demo")
    try:
        with pytest.raises(SessionAlreadyActiveError):
            session.start("demo2")
    finally:
        session.stop()


def test_session_is_single_shot(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path)
    session.start("demo")
    session.stop()
    with pytest.raises(SessionAlreadyActiveError):
        session.start("demo")


def test_stop_without_start_raises(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path)
    with pytest.raises(SessionNotActiveError):
        session.stop()


def test_stop_is_idempotent(tmp_path: Path) -> None:
    session, bag = _build_session(tmp_path)
    session.start("demo")
    first = session.stop()
    second = session.stop()
    assert first == second
    # Components should only be stopped once.
    assert bag["event_tap"].stopped is True
    assert bag["focus_tracker"].stopped is True
    assert bag["text_aggregator"].stopped is True


def test_permissions_missing_raises_structured_error(tmp_path: Path) -> None:
    missing: PermissionsError = {
        "error": "missing_permission",
        "permissions": ["accessibility"],
        "how_to_grant": {"accessibility": "go to system settings"},
    }
    session, _ = _build_session(tmp_path, permissions_missing=missing)
    with pytest.raises(PermissionsMissingError) as exc:
        session.start("demo")
    assert exc.value.error == missing
    assert session.is_active() is False
    # No trajectory directory should have been spawned.
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------- keyframes


def _no_periodic_policy() -> KeyframePolicy:
    """Policy that never fires a periodic keyframe in the test window."""
    return KeyframePolicy(periodic_interval_seconds=3600.0, post_click_delay_seconds=0.0)


def test_click_emits_pre_resolve_and_post_envelope(tmp_path: Path) -> None:
    target = {
        "role": "AXButton",
        "label": "Send",
        "description": None,
        "frame": {"x": 100.0, "y": 200.0, "w": 60.0, "h": 20.0},
        "ax_identifier": "send-btn",
    }
    session, bag = _build_session(
        tmp_path,
        ax_target=target,
        keyframe_policy=_no_periodic_policy(),
    )
    trajectory_id = session.start("demo")
    try:
        bag["event_tap"].fire(
            {
                "cg_event_type": "left_mouse_down",
                "timestamp_ms": 111,
                "location_x": 150.0,
                "location_y": 205.0,
                "modifiers": ["command"],
            }
        )
    finally:
        session.stop()

    events = _read_events(tmp_path, trajectory_id)
    types = [e["type"] for e in events]
    # pre-click keyframe, click, post-click keyframe (delay=0 → synchronous).
    assert types == ["keyframe", "click", "keyframe"]
    assert events[0]["payload"]["reason"] == "pre_click"
    assert events[1]["payload"]["button"] == "left"
    assert events[1]["payload"]["modifiers"] == ["cmd"]
    assert events[1]["target"] == target
    assert events[2]["payload"]["reason"] == "post_click"
    # AX resolver was called once with the click location.
    assert bag["ax_calls"] == [(150.0, 205.0)]
    # Monotonic seq.
    seqs = [e["seq"] for e in events]
    assert seqs == [1, 2, 3]
    # Keyframe screenshot_refs follow the zero-padded pattern.
    assert events[0]["screenshot_ref"] == "screenshots/0001.png"
    assert events[2]["screenshot_ref"] == "screenshots/0003.png"
    # Both screenshot files written.
    for seq in (1, 3):
        assert (tmp_path / trajectory_id / "screenshots" / f"{seq:04d}.png").is_file()


def test_ax_resolver_timeout_yields_null_target(tmp_path: Path) -> None:
    session, bag = _build_session(
        tmp_path,
        ax_target=None,
        keyframe_policy=_no_periodic_policy(),
    )
    trajectory_id = session.start("demo")
    try:
        bag["event_tap"].fire(
            {
                "cg_event_type": "left_mouse_down",
                "timestamp_ms": 111,
                "location_x": 10.0,
                "location_y": 20.0,
                "modifiers": [],
            }
        )
    finally:
        session.stop()

    events = _read_events(tmp_path, trajectory_id)
    click = next(e for e in events if e["type"] == "click")
    assert click["target"] is None


def test_ax_resolver_exception_is_swallowed(tmp_path: Path) -> None:
    session, bag = _build_session(
        tmp_path,
        ax_raises=True,
        keyframe_policy=_no_periodic_policy(),
    )
    trajectory_id = session.start("demo")
    try:
        bag["event_tap"].fire(
            {
                "cg_event_type": "left_mouse_down",
                "timestamp_ms": 111,
                "location_x": 10.0,
                "location_y": 20.0,
                "modifiers": [],
            }
        )
    finally:
        session.stop()

    events = _read_events(tmp_path, trajectory_id)
    click = next(e for e in events if e["type"] == "click")
    assert click["target"] is None


def test_screenshot_failure_records_event_without_ref(tmp_path: Path) -> None:
    session, bag = _build_session(
        tmp_path,
        screenshot_bytes=None,
        keyframe_policy=_no_periodic_policy(),
    )
    trajectory_id = session.start("demo")
    try:
        bag["event_tap"].fire(
            {
                "cg_event_type": "left_mouse_down",
                "timestamp_ms": 111,
                "location_x": 10.0,
                "location_y": 20.0,
                "modifiers": [],
            }
        )
    finally:
        session.stop()

    events = _read_events(tmp_path, trajectory_id)
    keyframes = [e for e in events if e["type"] == "keyframe"]
    assert len(keyframes) == 2
    for kf in keyframes:
        assert kf["screenshot_ref"] is None


def test_screenshot_capturer_exception_is_swallowed(tmp_path: Path) -> None:
    session, bag = _build_session(
        tmp_path,
        screenshot_raises=True,
        keyframe_policy=_no_periodic_policy(),
    )
    trajectory_id = session.start("demo")
    try:
        bag["event_tap"].fire(
            {
                "cg_event_type": "left_mouse_down",
                "timestamp_ms": 111,
                "location_x": 10.0,
                "location_y": 20.0,
                "modifiers": [],
            }
        )
    finally:
        session.stop()

    events = _read_events(tmp_path, trajectory_id)
    # Click event still emitted even though every keyframe capture failed.
    assert any(e["type"] == "click" for e in events)


def test_periodic_keyframes_fire(tmp_path: Path) -> None:
    policy = KeyframePolicy(periodic_interval_seconds=0.05, post_click_delay_seconds=0.0)
    session, _ = _build_session(tmp_path, keyframe_policy=policy)
    trajectory_id = session.start("demo")
    try:
        time.sleep(0.25)
    finally:
        session.stop()

    events = _read_events(tmp_path, trajectory_id)
    periodic = [e for e in events if e.get("payload", {}).get("reason") == "periodic"]
    assert len(periodic) >= 2, f"expected at least 2 periodic keyframes, got {len(periodic)}"


# ----------------------------------------------------- scroll / keypress / app_switch


def test_scroll_direction_selection(tmp_path: Path) -> None:
    session, bag = _build_session(tmp_path, keyframe_policy=_no_periodic_policy())
    trajectory_id = session.start("demo")
    try:
        bag["event_tap"].fire(
            {
                "cg_event_type": "scroll_wheel",
                "timestamp_ms": 1,
                "modifiers": [],
                "scroll_delta_y": -3,
                "scroll_delta_x": 0,
            }
        )
        bag["event_tap"].fire(
            {
                "cg_event_type": "scroll_wheel",
                "timestamp_ms": 2,
                "modifiers": [],
                "scroll_delta_y": 0,
                "scroll_delta_x": 5,
            }
        )
        # Zero-delta scroll: dropped.
        bag["event_tap"].fire(
            {
                "cg_event_type": "scroll_wheel",
                "timestamp_ms": 3,
                "modifiers": [],
                "scroll_delta_y": 0,
                "scroll_delta_x": 0,
            }
        )
    finally:
        session.stop()

    events = _read_events(tmp_path, trajectory_id)
    scrolls = [e for e in events if e["type"] == "scroll"]
    assert len(scrolls) == 2
    assert scrolls[0]["payload"] == {"direction": "down", "delta": 3.0}
    assert scrolls[1]["payload"] == {"direction": "right", "delta": 5.0}


def test_shortcut_keypress_emits_schema_event(tmp_path: Path) -> None:
    session, bag = _build_session(tmp_path, keyframe_policy=_no_periodic_policy())
    trajectory_id = session.start("demo")
    try:
        bag["event_tap"].fire(
            {
                "cg_event_type": "key_down",
                "timestamp_ms": 1,
                "modifiers": ["command"],
                "key_code": 6,
                "chars": "z",
            }
        )
    finally:
        session.stop()

    events = _read_events(tmp_path, trajectory_id)
    press = next(e for e in events if e["type"] == "keypress")
    assert press["payload"]["modifiers"] == ["cmd"]
    assert press["payload"]["keys"] == ["cmd", "z"]


def test_non_shortcut_key_forwards_to_text_aggregator(tmp_path: Path) -> None:
    session, bag = _build_session(tmp_path, keyframe_policy=_no_periodic_policy())
    trajectory_id = session.start("demo")
    try:
        bag["event_tap"].fire(
            {
                "cg_event_type": "key_down",
                "timestamp_ms": 1,
                "modifiers": [],
                "key_code": 4,
                "chars": "h",
            }
        )
        # Session emits text_input on the aggregator's behalf:
        bag["text_aggregator"].fire_text_input("hello", "Email", "com.google.Chrome")
    finally:
        session.stop()

    ta = bag["text_aggregator"]
    assert len(ta.key_events) == 1
    assert ta.key_events[0]["chars"] == "h"

    events = _read_events(tmp_path, trajectory_id)
    typed = [e for e in events if e["type"] == "text_input"]
    assert len(typed) == 1
    assert typed[0]["payload"] == {"text": "hello", "field_label": "Email"}
    assert typed[0]["app"]["bundle_id"] == "com.google.Chrome"


def test_empty_text_input_suppressed(tmp_path: Path) -> None:
    session, bag = _build_session(tmp_path, keyframe_policy=_no_periodic_policy())
    trajectory_id = session.start("demo")
    try:
        bag["text_aggregator"].fire_text_input("", None, "com.apple.Safari")
    finally:
        session.stop()
    events = _read_events(tmp_path, trajectory_id)
    assert not any(e["type"] == "text_input" for e in events)


def test_app_switch_writes_keyframe_and_clears_aggregator_focus(tmp_path: Path) -> None:
    session, bag = _build_session(tmp_path, keyframe_policy=_no_periodic_policy())
    trajectory_id = session.start("demo")
    try:
        bag["focus_tracker"].fire_app_switch(
            {
                "from_bundle_id": None,
                "to_bundle_id": "com.google.Chrome",
                "from_name": None,
                "to_name": "Google Chrome",
            }
        )
        bag["focus_tracker"].fire_app_switch(
            {
                "from_bundle_id": "com.google.Chrome",
                "to_bundle_id": "com.apple.Terminal",
                "from_name": "Google Chrome",
                "to_name": "Terminal",
            }
        )
    finally:
        session.stop()

    events = _read_events(tmp_path, trajectory_id)
    switches = [e for e in events if e["type"] == "app_switch"]
    keyframes = [e for e in events if e["type"] == "keyframe"]
    assert len(switches) == 2
    assert any(kf["payload"]["reason"] == "app_switch" for kf in keyframes)
    # from_bundle_id is omitted when null per session builder.
    assert "from_bundle_id" not in switches[0]["payload"]
    assert switches[1]["payload"]["from_bundle_id"] == "com.google.Chrome"
    # Aggregator focus was cleared on each app switch (bundle, None, None).
    focus_calls = bag["text_aggregator"].set_focus_calls
    assert any(call == ("com.google.Chrome", None, None) for call in focus_calls)
    assert any(call == ("com.apple.Terminal", None, None) for call in focus_calls)


def test_window_focus_empty_title_ignored(tmp_path: Path) -> None:
    session, bag = _build_session(tmp_path, keyframe_policy=_no_periodic_policy())
    trajectory_id = session.start("demo")
    try:
        bag["focus_tracker"].fire_window_focus(
            {"window_title": None, "app_bundle_id": "com.google.Chrome"}
        )
        bag["focus_tracker"].fire_window_focus(
            {"window_title": "Inbox", "app_bundle_id": "com.google.Chrome"}
        )
    finally:
        session.stop()

    events = _read_events(tmp_path, trajectory_id)
    wf = [e for e in events if e["type"] == "window_focus"]
    assert len(wf) == 1
    assert wf[0]["payload"] == {"window_title": "Inbox"}


def test_tap_reenabled_event_surfaces(tmp_path: Path) -> None:
    session, bag = _build_session(tmp_path, keyframe_policy=_no_periodic_policy())
    trajectory_id = session.start("demo")
    try:
        bag["event_tap"].fire(
            {
                "cg_event_type": "tap_reenabled",
                "reason": "timeout",
                "timestamp_ms": 5,
            }
        )
    finally:
        session.stop()

    events = _read_events(tmp_path, trajectory_id)
    tr = [e for e in events if e["type"] == "tap_reenabled"]
    assert len(tr) == 1
    assert tr[0]["payload"] == {"cause": "timeout"}


# --------------------------------------------------------------- metadata


def test_stop_records_duration_and_history(tmp_path: Path) -> None:
    session, bag = _build_session(tmp_path, keyframe_policy=_no_periodic_policy())
    trajectory_id = session.start("demo")
    try:
        bag["focus_tracker"].fire_app_switch(
            {
                "from_bundle_id": None,
                "to_bundle_id": "com.google.Chrome",
                "from_name": None,
                "to_name": "Google Chrome",
            }
        )
        time.sleep(0.02)
    finally:
        summary = session.stop()

    assert summary["trajectory_id"] == trajectory_id
    assert summary["event_count"] >= 1
    assert summary["duration_ms"] >= 10

    meta = _read_metadata(tmp_path, trajectory_id)
    assert meta["stopped_at"] is not None
    assert len(meta["app_focus_history"]) == 1
    entry = meta["app_focus_history"][0]
    assert entry["bundle_id"] == "com.google.Chrome"
    assert entry["exited_at"] is not None  # finalised by FocusTracker.stop()


def test_fallback_display_info_used_when_provider_returns_none(tmp_path: Path) -> None:
    session, _ = _build_session(
        tmp_path,
        display_info=None,  # default provider returns canonical dict
    )
    # Override with a provider that returns None explicitly.
    session._display_info_provider = lambda: None
    trajectory_id = session.start("demo")
    try:
        meta = _read_metadata(tmp_path, trajectory_id)
        assert meta["display_info"] == {"width": 1, "height": 1, "scale_factor": 1.0}
    finally:
        session.stop()


# ------------------------------------------------------------- concurrency


def test_seq_is_monotonic_under_concurrent_appends(tmp_path: Path) -> None:
    """Multiple threads firing events race through the single session lock.

    The AC requires a single lock guarding seq + writer.append_event so
    seq values never collide.  We simulate the event-tap thread and the
    text-aggregator thread by firing from several Python threads at once.
    """
    session, bag = _build_session(tmp_path, keyframe_policy=_no_periodic_policy())
    trajectory_id = session.start("demo")

    barrier = threading.Barrier(5)
    errors: list[BaseException] = []

    def worker(idx: int) -> None:
        try:
            barrier.wait()
            for n in range(20):
                bag["event_tap"].fire(
                    {
                        "cg_event_type": "scroll_wheel",
                        "timestamp_ms": n,
                        "modifiers": [],
                        "scroll_delta_y": -1,
                        "scroll_delta_x": 0,
                    }
                )
        except BaseException as exc:  # pragma: no cover — surface to test
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    session.stop()

    assert not errors
    events = _read_events(tmp_path, trajectory_id)
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))
    assert min(seqs) == 1


# ----------------------------------------------------------- callback safety


def test_callbacks_return_silently_after_stop(tmp_path: Path) -> None:
    session, bag = _build_session(tmp_path, keyframe_policy=_no_periodic_policy())
    session.start("demo")
    session.stop()
    # Late callbacks from threads that haven't noticed the stop yet must
    # not crash or leak events.
    bag["event_tap"].fire(
        {
            "cg_event_type": "left_mouse_down",
            "timestamp_ms": 1,
            "location_x": 0.0,
            "location_y": 0.0,
            "modifiers": [],
        }
    )
    bag["focus_tracker"].fire_app_switch(
        {
            "from_bundle_id": None,
            "to_bundle_id": "com.example.app",
            "from_name": None,
            "to_name": "App",
        }
    )
    bag["text_aggregator"].fire_text_input("late", None, "com.example.app")


def test_context_manager_stops_on_exit(tmp_path: Path) -> None:
    session, bag = _build_session(tmp_path, keyframe_policy=_no_periodic_policy())
    with session as s:
        s.start("demo")
        assert s.is_active() is True
    assert session.is_active() is False
    assert bag["event_tap"].stopped is True
