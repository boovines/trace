"""Recording session orchestrator.

The :class:`RecordingSession` is the integration point where the four
capture components (event tap, focus tracker, text aggregator, periodic
keyframe timer) are wired into a single :class:`~recorder.writer.TrajectoryWriter`.
It is responsible for:

* Translating raw component output (PyObjC framework events, modifier
  names in the long ``"command"`` form, etc.) into schema-valid trajectory
  events that pass :func:`recorder.schema.validate_event`.
* Owning the **single** monotonic ``seq`` counter shared between every
  capture thread.  A session-level :class:`threading.Lock` guards both
  ``seq`` assignment and the call into :meth:`TrajectoryWriter.append_event`,
  so the on-disk order of ``events.jsonl`` always matches ``seq`` order.
* Building the click "envelope" — pre-click keyframe → AX target
  resolution → click event → post-click keyframe (delayed) — atomically
  enough that the four events are never interleaved with another thread's
  emission.
* Driving a periodic keyframe thread that emits one ``keyframe`` event
  every :data:`~recorder.keyframe_policy.PERIODIC_INTERVAL_SECONDS` seconds
  while the session is active.
* Refusing concurrent sessions (only one :class:`RecordingSession` instance
  may be active at a time on a single instance) and supporting an
  idempotent :meth:`stop`.

Every external dependency (``EventTap``, ``FocusTracker``, ``TextAggregator``,
the AX resolver, the screenshot capturer, the permissions check, the
display-info reader, the keyframe policy) is injected via constructor
keyword arguments that default to the real recorder module functions.
Tests substitute hermetic fakes for the macOS-dependent pieces and drive
the session with synthetic event dicts.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TypedDict

from recorder.ax_resolver import ResolvedTarget, resolve_element_at
from recorder.event_tap import EventCallback, EventTap
from recorder.focus_tracker import (
    AppFocusHistoryEntry,
    AppSwitchPayload,
    FocusTracker,
    WindowFocusPayload,
)
from recorder.index_db import IndexDB
from recorder.keyframe_policy import KeyframePolicy
from recorder.permissions import (
    PermissionsError,
    get_missing_permissions_error,
)
from recorder.screenshot import (
    DisplayInfo,
    capture_main_display,
    get_main_display_info,
)
from recorder.text_aggregator import (
    TextAggregator,
    TextInputEmit,
    TextInputEvent,
    resolve_focused_field,
)
from recorder.writer import TrajectoryWriter

__all__ = [
    "PermissionsMissingError",
    "RecordingSession",
    "SessionAlreadyActiveError",
    "SessionNotActiveError",
    "SessionSummary",
]

logger = logging.getLogger(__name__)


class SessionSummary(TypedDict):
    """Return shape of :meth:`RecordingSession.stop`."""

    trajectory_id: str
    event_count: int
    duration_ms: int


class PermissionsMissingError(RuntimeError):
    """Raised by :meth:`RecordingSession.start` when permissions are missing.

    The structured error body (suitable for direct JSON serialisation by
    the HTTP layer) is exposed via :attr:`error`.
    """

    def __init__(self, error: PermissionsError) -> None:
        super().__init__(
            f"missing permissions: {', '.join(error['permissions'])}"
        )
        self.error: PermissionsError = error


class SessionAlreadyActiveError(RuntimeError):
    """Raised when :meth:`RecordingSession.start` is called twice in a row."""


class SessionNotActiveError(RuntimeError):
    """Raised when :meth:`RecordingSession.stop` is called before any start."""


# ---------------------------------------------------------- factory protocols

EventTapFactory = Callable[[EventCallback], "EventTap"]
FocusTrackerFactory = Callable[[], "FocusTracker"]
TextAggregatorFactory = Callable[[TextInputEmit], "TextAggregator"]
PermissionsCheckFn = Callable[[], "PermissionsError | None"]
AXResolverFn = Callable[[float, float], "ResolvedTarget | None"]
ScreenshotFn = Callable[[], "bytes | None"]
DisplayInfoFn = Callable[[], "DisplayInfo | None"]


# ----------------------------------------------------- module-level constants

# event_tap surfaces modifier names in PyObjC framework style.  The schema
# uses the short form.  We translate at the session boundary so every other
# layer can keep using the long names internally.
_MODIFIER_TRANSLATION: dict[str, str] = {
    "command": "cmd",
    "control": "ctrl",
    "option": "opt",
    "shift": "shift",
    "fn": "fn",
}

_CLICK_BUTTON_BY_TYPE: dict[str, str] = {
    "left_mouse_down": "left",
    "right_mouse_down": "right",
    "other_mouse_down": "other",
}

_KEY_DOWN_TYPES: frozenset[str] = frozenset({"key_down"})
_SHORTCUT_MODIFIERS: frozenset[str] = frozenset({"command", "control", "option"})

_FALLBACK_DISPLAY_INFO: DisplayInfo = {
    "width": 1,
    "height": 1,
    "scale_factor": 1.0,
}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _translate_modifiers(modifiers: list[str] | tuple[str, ...] | None) -> list[str]:
    """Translate event_tap modifier names into schema-enum values, deduped + ordered."""
    if not modifiers:
        return []
    seen: list[str] = []
    for raw in modifiers:
        translated = _MODIFIER_TRANSLATION.get(raw)
        if translated is not None and translated not in seen:
            seen.append(translated)
    return seen


def _empty_app() -> dict[str, Any]:
    """Schema-valid placeholder when no app is currently focused."""
    return {"bundle_id": "", "name": "", "pid": 0}


# --------------------------------------------------------------- the session


class RecordingSession:
    """Run a single recording from start to stop.

    A :class:`RecordingSession` instance is single-shot: once :meth:`stop`
    has run, construct a fresh instance to record again.  This matches the
    underlying components (:class:`recorder.event_tap.EventTap` is also
    single-use) and avoids subtle state-reset bugs.

    Constructor parameters of interest for tests (every macOS-dependent
    dependency has a hook):

    * ``event_tap_factory`` — callable ``(callback) -> EventTap``.
    * ``focus_tracker_factory`` — callable ``() -> FocusTracker``.
    * ``text_aggregator_factory`` — callable ``(emit) -> TextAggregator``.
    * ``permissions_check`` — callable ``() -> PermissionsError | None``.
    * ``ax_resolver`` — callable ``(x, y) -> ResolvedTarget | None``.
    * ``screenshot_capturer`` — callable ``() -> bytes | None``.
    * ``display_info_provider`` — callable ``() -> DisplayInfo | None``.
    * ``keyframe_policy`` — instance; tests may pass a tight interval to
      exercise the periodic-keyframe loop quickly.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        index_db: IndexDB | None = None,
        event_tap_factory: EventTapFactory | None = None,
        focus_tracker_factory: FocusTrackerFactory | None = None,
        text_aggregator_factory: TextAggregatorFactory | None = None,
        permissions_check: PermissionsCheckFn | None = None,
        ax_resolver: AXResolverFn | None = None,
        screenshot_capturer: ScreenshotFn | None = None,
        display_info_provider: DisplayInfoFn | None = None,
        keyframe_policy: KeyframePolicy | None = None,
    ) -> None:
        self.root: Path = Path(root)
        self._index_db: IndexDB | None = index_db

        self._event_tap_factory: EventTapFactory = (
            event_tap_factory if event_tap_factory is not None else EventTap
        )
        self._focus_tracker_factory: FocusTrackerFactory = (
            focus_tracker_factory if focus_tracker_factory is not None else FocusTracker
        )
        self._text_aggregator_factory: TextAggregatorFactory = (
            text_aggregator_factory
            if text_aggregator_factory is not None
            else TextAggregator
        )
        self._permissions_check: PermissionsCheckFn = (
            permissions_check
            if permissions_check is not None
            else get_missing_permissions_error
        )
        self._ax_resolver: AXResolverFn = (
            ax_resolver if ax_resolver is not None else resolve_element_at
        )
        self._screenshot_capturer: ScreenshotFn = (
            screenshot_capturer
            if screenshot_capturer is not None
            else capture_main_display
        )
        self._display_info_provider: DisplayInfoFn = (
            display_info_provider
            if display_info_provider is not None
            else get_main_display_info
        )
        self._policy: KeyframePolicy = (
            keyframe_policy if keyframe_policy is not None else KeyframePolicy()
        )

        # Lifecycle / lock layout:
        #   _start_stop_lock  → serialises start() + stop() bookkeeping
        #   _lock             → guards seq, writer.append_event, last keyframe time
        #
        # Capture-thread callbacks acquire _lock briefly, so a stop() that
        # holds _start_stop_lock + waits for capture threads to drain does
        # NOT also need _lock — once _active is False, callbacks bail early.
        self._start_stop_lock = threading.Lock()
        self._lock = threading.Lock()

        self._writer: TrajectoryWriter | None = None
        self._event_tap: EventTap | None = None
        self._focus_tracker: FocusTracker | None = None
        self._text_aggregator: TextAggregator | None = None
        self._keyframe_thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()

        self._active: bool = False
        self._summary: SessionSummary | None = None

        self._next_seq: int = 1
        self._started_monotonic: float = 0.0
        self._last_keyframe_monotonic: float = 0.0
        self._pending_timers: list[threading.Timer] = []

    # ------------------------------------------------------------- lifecycle

    def start(self, label: str) -> str:
        """Start a recording.  Returns the trajectory id (uuid string).

        Raises:
            SessionAlreadyActiveError: a session is already running on this
                instance.
            PermissionsMissingError: the macOS permissions check came back
                with one or more missing.  ``error.error`` carries the
                structured dict the HTTP layer can hand to the UI.
        """
        with self._start_stop_lock:
            if self._active:
                raise SessionAlreadyActiveError(
                    "RecordingSession.start() called while a session is active"
                )
            if self._summary is not None:
                raise SessionAlreadyActiveError(
                    "RecordingSession is single-shot; create a new instance"
                )

            permissions_error = self._permissions_check()
            if permissions_error is not None:
                raise PermissionsMissingError(permissions_error)

            display_info = self._display_info_provider() or _FALLBACK_DISPLAY_INFO

            writer = TrajectoryWriter(self.root, label, index_db=self._index_db)
            try:
                metadata: dict[str, Any] = {
                    "id": writer.id,
                    "label": label,
                    "started_at": _utc_now_iso(),
                    "stopped_at": None,
                    "display_info": dict(display_info),
                    "app_focus_history": [],
                }
                writer.write_metadata(metadata)
            except Exception:
                writer.close()
                raise

            text_aggregator = self._text_aggregator_factory(self._on_text_input)

            focus_tracker = self._focus_tracker_factory()
            focus_tracker.on_app_switch(self._on_app_switch)
            focus_tracker.on_window_focus_change(self._on_window_focus_change)

            event_tap = self._event_tap_factory(self._on_event_tap_event)

            # Reset per-session state under the same lock callbacks observe.
            with self._lock:
                self._writer = writer
                self._next_seq = 1
                self._last_keyframe_monotonic = time.monotonic()
                self._pending_timers = []

            # IMPORTANT: publish ``_text_aggregator`` and flip ``_active`` BEFORE
            # ``focus_tracker.start()``. ``start()`` synchronously fires the
            # initial app_switch callback (which seeds focus from
            # ``NSWorkspace.frontmostApplication()``). That callback goes through
            # ``_on_app_switch``, which (1) short-circuits on ``if not
            # self._active`` and (2) reads ``self._text_aggregator`` to call
            # ``set_focus`` on it. If either is still unset, the aggregator
            # never gets a focused field and every subsequent plain keystroke is
            # silently dropped for the lifetime of the session.
            self._text_aggregator = text_aggregator
            self._active = True
            try:
                focus_tracker.start()
            except Exception:
                self._active = False
                self._text_aggregator = None
                writer.close()
                self._writer = None
                raise
            try:
                event_tap.start()
            except Exception:
                self._active = False
                self._text_aggregator = None
                with self._lock:
                    self._writer = None
                try:
                    focus_tracker.stop()
                except Exception:
                    logger.debug("focus_tracker.stop after failed start raised", exc_info=True)
                writer.close()
                raise

            self._focus_tracker = focus_tracker
            self._event_tap = event_tap

            self._stop_event.clear()
            self._started_monotonic = time.monotonic()
            keyframe_thread = threading.Thread(
                target=self._periodic_keyframe_loop,
                name="recorder-session-keyframes",
                daemon=True,
            )
            keyframe_thread.start()
            self._keyframe_thread = keyframe_thread

            return writer.id

    def stop(self) -> SessionSummary:
        """Stop the recording and return a :class:`SessionSummary`.

        Idempotent: the second call returns the cached summary from the
        first.  Raises :class:`SessionNotActiveError` if called before
        :meth:`start` ever ran.
        """
        with self._start_stop_lock:
            if not self._active:
                if self._summary is not None:
                    return self._summary
                raise SessionNotActiveError(
                    "RecordingSession.stop() called before start()"
                )

            # Mark inactive first so capture-thread callbacks bail early.
            with self._lock:
                self._active = False

            self._stop_event.set()
            keyframe_thread = self._keyframe_thread
            self._keyframe_thread = None
            if keyframe_thread is not None:
                keyframe_thread.join(timeout=2.0)
                if keyframe_thread.is_alive():
                    logger.warning("periodic keyframe thread did not exit in 2s")

            with self._lock:
                pending = list(self._pending_timers)
                self._pending_timers.clear()
            for timer in pending:
                timer.cancel()

            event_tap = self._event_tap
            self._event_tap = None
            if event_tap is not None:
                try:
                    event_tap.stop()
                except Exception:
                    logger.exception("EventTap.stop raised during session stop")

            history: list[AppFocusHistoryEntry] = []
            focus_tracker = self._focus_tracker
            self._focus_tracker = None
            if focus_tracker is not None:
                try:
                    focus_tracker.stop()
                    history = focus_tracker.get_app_focus_history()
                except Exception:
                    logger.exception("FocusTracker.stop raised during session stop")

            text_aggregator = self._text_aggregator
            self._text_aggregator = None
            if text_aggregator is not None:
                try:
                    text_aggregator.stop()
                except Exception:
                    logger.exception("TextAggregator.stop raised during session stop")

            duration_ms = int((time.monotonic() - self._started_monotonic) * 1000)

            writer = self._writer
            self._writer = None

            event_count = self._next_seq - 1
            trajectory_id = writer.id if writer is not None else ""

            if writer is not None:
                try:
                    self._finalise_metadata(writer, history)
                except Exception:
                    logger.exception("finalising metadata raised; closing writer anyway")
                try:
                    writer.close()
                except Exception:
                    logger.exception("TrajectoryWriter.close raised during session stop")

            summary: SessionSummary = {
                "trajectory_id": trajectory_id,
                "event_count": event_count,
                "duration_ms": duration_ms,
            }
            self._summary = summary
            return summary

    # ----------------------------------------------------- introspection

    def is_active(self) -> bool:
        """Return ``True`` while a session is in flight."""
        return self._active

    @property
    def trajectory_id(self) -> str | None:
        """The id of the active or last-completed trajectory, or ``None``."""
        if self._writer is not None:
            return self._writer.id
        if self._summary is not None:
            return self._summary["trajectory_id"]
        return None

    # --------------------------------------------------- capture callbacks

    def _on_event_tap_event(self, event: dict[str, Any]) -> None:
        """Top-level dispatcher for the event-tap callback."""
        if not self._active:
            return
        cg_event_type = event.get("cg_event_type")
        if cg_event_type == "tap_reenabled":
            cause_raw = event.get("reason")
            cause = cause_raw if cause_raw in ("timeout", "user_input") else "unknown"
            self._emit_event(
                event_type="tap_reenabled",
                payload={"cause": cause},
                target=None,
                timestamp_ms=event.get("timestamp_ms"),
            )
            return
        if cg_event_type in _CLICK_BUTTON_BY_TYPE:
            self._handle_click(event)
            return
        if cg_event_type == "scroll_wheel":
            self._handle_scroll(event)
            return
        if cg_event_type in _KEY_DOWN_TYPES:
            self._handle_key_event(event)
            return
        # mouse_up / flags_changed / unknown — ignored at the session layer.

    def _on_app_switch(self, payload: AppSwitchPayload) -> None:
        if not self._active:
            return
        # Build the schema event payload — only include from_* if non-null
        # (the schema accepts null, but omitting keeps payloads tidy).
        schema_payload: dict[str, Any] = {"to_bundle_id": payload["to_bundle_id"]}
        if payload.get("from_bundle_id") is not None:
            schema_payload["from_bundle_id"] = payload["from_bundle_id"]
        if payload.get("from_name") is not None:
            schema_payload["from_name"] = payload["from_name"]
        if payload.get("to_name") is not None:
            schema_payload["to_name"] = payload["to_name"]
        self._emit_event(
            event_type="app_switch",
            payload=schema_payload,
            target=None,
        )
        # Switching apps means the previous field's text buffer is no
        # longer the right home for incoming keystrokes.  Clear focus so
        # the next keystroke either lands in a freshly-resolved field or
        # in the new app's fallback bucket.
        text_aggregator = self._text_aggregator
        if text_aggregator is not None:
            try:
                # Best-effort AX lookup of the focused field for this app so
                # ``text_input`` events carry a ``field_label`` (the smoke
                # checklist requires ≥1 labelled text_input per workflow).
                # ``resolve_focused_field`` is bounded by an internal
                # 200ms timeout and returns ``(None, None)`` when AX can't
                # resolve a text-like role — falling through gracefully.
                field_key, field_label = None, None
                tracker = self._focus_tracker
                current = tracker.get_current_app() if tracker is not None else None
                if current is not None:
                    try:
                        field_key, field_label = resolve_focused_field(
                            current["pid"]
                        )
                    except Exception:
                        logger.debug(
                            "resolve_focused_field raised", exc_info=True
                        )
                text_aggregator.set_focus(
                    payload["to_bundle_id"], field_key, field_label
                )
            except Exception:
                logger.exception("text_aggregator.set_focus raised on app switch")
        self._emit_keyframe("app_switch")

    def _on_window_focus_change(self, payload: WindowFocusPayload) -> None:
        if not self._active:
            return
        title = payload.get("window_title")
        if not title:
            return
        self._emit_event(
            event_type="window_focus",
            payload={"window_title": title},
            target=None,
        )

    def _on_text_input(self, payload: TextInputEvent) -> None:
        if not self._active:
            return
        text = payload["text"]
        if not text:
            return
        schema_payload: dict[str, Any] = {"text": text}
        field_label = payload.get("field_label")
        if field_label:
            schema_payload["field_label"] = field_label
        # text_input attributes to the app the buffer was bound to, not
        # whatever happens to be focused right now.
        app_override = self._app_dict_for_bundle(payload["app_bundle_id"])
        self._emit_event(
            event_type="text_input",
            payload=schema_payload,
            target=None,
            app=app_override,
        )

    # --------------------------------------------------------- click envelope

    def _handle_click(self, event: dict[str, Any]) -> None:
        cg_event_type = event["cg_event_type"]
        button = _CLICK_BUTTON_BY_TYPE.get(cg_event_type, "other")
        x = event.get("location_x")
        y = event.get("location_y")

        # 1. pre-click keyframe
        self._emit_keyframe("pre_click")

        # 2. AX target resolution (outside the lock — can take up to 200ms)
        target: ResolvedTarget | None = None
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            try:
                target = self._ax_resolver(float(x), float(y))
            except Exception:
                logger.exception("ax_resolver raised during click handling")
                target = None

        # 3. click event
        modifiers = _translate_modifiers(event.get("modifiers"))
        click_payload: dict[str, Any] = {"button": button}
        if modifiers:
            click_payload["modifiers"] = modifiers
        self._emit_event(
            event_type="click",
            payload=click_payload,
            target=target,
            timestamp_ms=event.get("timestamp_ms"),
        )

        # 4. post-click keyframe (delayed)
        self._schedule_post_click_keyframe()

    def _schedule_post_click_keyframe(self) -> None:
        delay = self._policy.post_click_delay_seconds
        if delay <= 0:
            self._emit_keyframe("post_click")
            return

        def _fire() -> None:
            try:
                if not self._active:
                    return
                self._emit_keyframe("post_click")
            except Exception:
                logger.exception("post-click keyframe emit raised")

        timer = threading.Timer(delay, _fire)
        timer.daemon = True
        timer.name = "recorder-session-post-click"
        with self._lock:
            if not self._active:
                return
            self._pending_timers.append(timer)
        timer.start()

    # ----------------------------------------------------------- key events

    def _handle_key_event(self, event: dict[str, Any]) -> None:
        modifiers_raw = event.get("modifiers") or []
        is_shortcut = any(m in _SHORTCUT_MODIFIERS for m in modifiers_raw)
        if is_shortcut:
            self._emit_keypress(event, modifiers_raw)
            return
        # Forward to the text aggregator; it owns the per-field buffering.
        # Note: ``recorder.event_tap`` does NOT populate ``chars`` today —
        # the orchestrator (or a future event_tap shim) is responsible for
        # adding it.  For now we hand over whatever is in the dict; the
        # aggregator silently ignores entries with no ``chars``.
        text_aggregator = self._text_aggregator
        if text_aggregator is None:
            return
        try:
            text_aggregator.handle_key_event(event)
        except Exception:
            logger.exception("text_aggregator.handle_key_event raised")

    def _emit_keypress(self, event: dict[str, Any], modifiers_raw: list[str]) -> None:
        modifiers = _translate_modifiers(modifiers_raw)
        chars = event.get("chars")
        key_code = event.get("key_code")
        keys: list[str] = list(modifiers)
        if isinstance(chars, str) and chars:
            keys.append(chars)
        elif key_code is not None:
            with contextlib.suppress(Exception):
                keys.append(f"key_{int(key_code)}")
        if not keys:
            keys = ["unknown"]
        payload: dict[str, Any] = {"keys": keys}
        if modifiers:
            payload["modifiers"] = modifiers
        self._emit_event(
            event_type="keypress",
            payload=payload,
            target=None,
            timestamp_ms=event.get("timestamp_ms"),
        )

    # --------------------------------------------------------- scroll events

    def _handle_scroll(self, event: dict[str, Any]) -> None:
        dy = event.get("scroll_delta_y") or 0
        dx = event.get("scroll_delta_x") or 0
        try:
            dy_i = int(dy)
            dx_i = int(dx)
        except Exception:
            return
        if dy_i == 0 and dx_i == 0:
            return
        if abs(dx_i) > abs(dy_i):
            direction = "right" if dx_i > 0 else "left"
            magnitude = abs(dx_i)
        else:
            direction = "up" if dy_i > 0 else "down"
            magnitude = abs(dy_i)
        self._emit_event(
            event_type="scroll",
            payload={"direction": direction, "delta": float(magnitude)},
            target=None,
            timestamp_ms=event.get("timestamp_ms"),
        )

    # --------------------------------------------------- periodic keyframes

    def _periodic_keyframe_loop(self) -> None:
        interval = self._policy.periodic_interval_seconds
        # Poll at half the interval (or 0.5s, whichever is shorter) so the
        # actual fire time stays close to the configured interval.
        poll = max(0.05, min(0.5, interval / 2))
        while not self._stop_event.wait(poll):
            if not self._active:
                break
            with self._lock:
                last = self._last_keyframe_monotonic
            elapsed = time.monotonic() - last
            if not self._policy.should_capture("tick", elapsed):
                continue
            try:
                self._emit_keyframe("periodic")
            except Exception:
                logger.exception("periodic keyframe emit raised")

    # ------------------------------------------------------------ emit core

    def _emit_keyframe(self, reason: str) -> None:
        """Capture a screenshot (best-effort) and append a keyframe event."""
        png: bytes | None = None
        try:
            png = self._screenshot_capturer()
        except Exception:
            logger.exception("screenshot capture raised; emitting keyframe without image")
            png = None

        with self._lock:
            if not self._active or self._writer is None:
                return
            seq = self._next_seq
            self._next_seq += 1
            screenshot_ref: str | None = None
            if png is not None:
                try:
                    self._writer.write_screenshot(seq, png)
                    screenshot_ref = f"screenshots/{seq:04d}.png"
                except Exception:
                    logger.exception("write_screenshot failed for keyframe")
                    screenshot_ref = None

            event: dict[str, Any] = {
                "seq": seq,
                "timestamp_ms": int(time.time() * 1000),
                "type": "keyframe",
                "screenshot_ref": screenshot_ref,
                "app": self._current_app_dict(),
                "target": None,
                "payload": {"reason": reason},
            }
            try:
                self._writer.append_event(event)
            except Exception:
                logger.exception("writer.append_event failed for keyframe")
                return
            self._last_keyframe_monotonic = time.monotonic()

    def _emit_event(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        target: ResolvedTarget | None,
        timestamp_ms: int | None = None,
        app: dict[str, Any] | None = None,
    ) -> None:
        """Append a non-keyframe schema event under the session lock."""
        with self._lock:
            if not self._active or self._writer is None:
                return
            seq = self._next_seq
            self._next_seq += 1
            ts_ms = int(timestamp_ms) if timestamp_ms is not None else int(time.time() * 1000)
            event: dict[str, Any] = {
                "seq": seq,
                "timestamp_ms": ts_ms,
                "type": event_type,
                "screenshot_ref": None,
                "app": app if app is not None else self._current_app_dict(),
                "target": dict(target) if target is not None else None,
                "payload": payload,
            }
            try:
                self._writer.append_event(event)
            except Exception:
                logger.exception("writer.append_event failed for %s", event_type)

    # ----------------------------------------------------------- helpers

    def _current_app_dict(self) -> dict[str, Any]:
        focus_tracker = self._focus_tracker
        if focus_tracker is None:
            return _empty_app()
        try:
            current = focus_tracker.get_current_app()
        except Exception:
            logger.debug("focus_tracker.get_current_app raised", exc_info=True)
            return _empty_app()
        if current is None:
            return _empty_app()
        return {
            "bundle_id": current["bundle_id"],
            "name": current["name"],
            "pid": int(current["pid"]),
        }

    def _app_dict_for_bundle(self, bundle_id: str) -> dict[str, Any]:
        """Return an ``app`` dict for ``bundle_id``, falling back to current app."""
        current = self._current_app_dict()
        if current["bundle_id"] == bundle_id:
            return current
        # Look the bundle id up in the focus history for name/pid.
        focus_tracker = self._focus_tracker
        if focus_tracker is not None:
            try:
                history = focus_tracker.get_app_focus_history()
            except Exception:
                history = []
            for entry in reversed(history):
                if entry["bundle_id"] == bundle_id:
                    return {
                        "bundle_id": bundle_id,
                        "name": entry["name"],
                        "pid": current["pid"],
                    }
        return {"bundle_id": bundle_id, "name": "", "pid": 0}

    def _finalise_metadata(
        self,
        writer: TrajectoryWriter,
        history: list[AppFocusHistoryEntry],
    ) -> None:
        """Write the final ``app_focus_history`` into ``metadata.json``.

        ``writer.close()`` will then read the file, set ``stopped_at``, and
        atomically rewrite — so history + stopped_at end up co-resident.
        """
        metadata_path = writer.dir / "metadata.json"
        if not metadata_path.is_file():
            return
        with metadata_path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        metadata["app_focus_history"] = [
            {
                "bundle_id": entry["bundle_id"],
                "name": entry["name"],
                "entered_at": entry["entered_at"],
                **(
                    {"exited_at": entry["exited_at"]}
                    if entry.get("exited_at") is not None
                    else {"exited_at": None}
                ),
            }
            for entry in history
        ]
        writer.write_metadata(metadata)

    # ------------------------------------------------------------- context

    def __enter__(self) -> RecordingSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._active:
            try:
                self.stop()
            except Exception:
                logger.exception("RecordingSession.stop raised from __exit__")


