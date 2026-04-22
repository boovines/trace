"""Aggregate individual keystrokes into per-field ``text_input`` events.

Raw key-down events would bloat ``events.jsonl`` and make downstream
synthesis prompts nearly unreadable (a 30-character sentence becomes 30
events).  :class:`TextAggregator` buffers typed characters per focused
text field and emits a single ``text_input`` payload when:

1. Focus leaves the field (new field, new app, or focus cleared).
2. ``idle_timeout`` seconds pass without a new keystroke.
3. :meth:`stop` is called (trajectory ended).

The aggregator does NOT talk to the keyboard hardware directly.  The
session orchestrator feeds it normalised key-event dicts (already
produced by :mod:`recorder.event_tap` for the raw event) plus the
per-keystroke Unicode string (``chars``).  Keeping the CGEvent → string
translation out of this module means tests do not need PyObjC — they
drive the aggregator with synthetic dicts.

Per the PRD, modifier-held combinations (cmd/ctrl/opt) are keyboard
shortcuts, not text input.  They are silently ignored here so the caller
can hand every key_down event to both :meth:`handle_key_event` and the
keypress emitter without worrying about double-counting.  Shift alone is
*not* suppressed — capitalised characters arrive through ``chars``.

The focused-field identity is supplied by the caller via
:meth:`set_focus`.  A small helper, :func:`resolve_focused_field`, uses
AX to derive a (key, label) pair from the current frontmost app's
``AXFocusedUIElement``; it is best-effort and returns ``(None, None)``
when AX is unavailable or the focused element is not a text field.  When
no AX-backed field is known the aggregator falls back to app-level
buffering under a synthetic ``app:<bundle_id>`` key.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

__all__ = [
    "BACKSPACE_KEY_CODE",
    "FORWARD_DELETE_KEY_CODE",
    "IDLE_TIMEOUT_SECONDS",
    "TextAggregator",
    "TextInputEmit",
    "TextInputEvent",
    "resolve_focused_field",
]

logger = logging.getLogger(__name__)

#: macOS virtual key code for the primary backspace / delete-back key.
BACKSPACE_KEY_CODE: int = 51
#: macOS virtual key code for the forward-delete (fn+delete) key.
FORWARD_DELETE_KEY_CODE: int = 117
#: Default idle duration before a buffered field is auto-flushed.
IDLE_TIMEOUT_SECONDS: float = 1.5

# Modifier names emitted by :mod:`recorder.event_tap`.  Pure shortcut keys
# hold command / control / option and must not enter the text buffer; shift
# and fn are fine — shifted characters arrive through ``chars`` already.
_SUPPRESSED_MODIFIERS: frozenset[str] = frozenset({"command", "control", "option"})

# AX roles that look like editable text.  ``AXComboBox`` is included because
# the focused combo box in many web apps has an editable text field child.
_TEXT_FIELD_ROLES: frozenset[str] = frozenset(
    {"AXTextField", "AXTextArea", "AXSearchField", "AXComboBox"}
)

_AX_ERROR_SUCCESS = 0


class TextInputEvent(TypedDict):
    """Payload handed to the :class:`TextAggregator` emit callback.

    Includes the owning ``app_bundle_id`` alongside the schema-required
    ``text`` / ``field_label`` so the session orchestrator can attach the
    right :class:`recorder.focus_tracker.AppInfo` when building the full
    ``text_input`` event for :mod:`recorder.writer`.
    """

    app_bundle_id: str
    text: str
    field_label: str | None


TextInputEmit = Callable[[TextInputEvent], None]


@dataclass
class _FieldBuffer:
    """Per-field buffer.  ``buffer`` holds one unicode string per keystroke."""

    app_bundle_id: str
    field_key: str
    field_label: str | None
    buffer: list[str] = field(default_factory=list)


class TextAggregator:
    """Aggregate normalised key events into per-field ``text_input`` payloads.

    Thread-safe.  All public methods acquire an internal re-entrant lock so
    the aggregator can be driven from the event-tap thread (keystrokes),
    the focus-tracker thread (focus changes), and the idle timer thread
    (auto-flush) without external synchronisation.
    """

    def __init__(
        self,
        emit: TextInputEmit,
        *,
        idle_timeout: float = IDLE_TIMEOUT_SECONDS,
    ) -> None:
        if idle_timeout <= 0:
            raise ValueError("idle_timeout must be positive")
        self._emit: TextInputEmit = emit
        self._idle_timeout: float = idle_timeout
        self._lock = threading.RLock()
        self._current: _FieldBuffer | None = None
        self._timer: threading.Timer | None = None
        self._stopped: bool = False

    # ------------------------------------------------------------- focus hooks

    def set_focus(
        self,
        app_bundle_id: str | None,
        field_key: str | None,
        field_label: str | None,
    ) -> None:
        """Tell the aggregator which field currently has keyboard focus.

        Passing ``None`` for ``app_bundle_id`` or ``field_key`` clears the
        active buffer after flushing whatever is in it.  This is the seam
        the session orchestrator uses when the user switches apps, clicks
        into a new field, or focus is dropped entirely.

        If ``app_bundle_id`` is supplied but ``field_key`` is ``None`` we
        fall back to an app-level buffer keyed on ``app:<bundle_id>`` — AX
        doesn't always expose a stable field identity (Electron apps pre
        the 2023 roll-up, in particular), and losing all text-capture in
        those apps would be worse than bundling everything under one
        bucket.
        """
        with self._lock:
            if self._stopped:
                return
            if app_bundle_id is None or not app_bundle_id:
                self._flush_locked()
                return
            effective_key = field_key if field_key else f"app:{app_bundle_id}"
            if (
                self._current is not None
                and self._current.app_bundle_id == app_bundle_id
                and self._current.field_key == effective_key
            ):
                # Same field re-focused — only refresh the label (AX may
                # resolve a richer label the second time around).
                if field_label is not None:
                    self._current.field_label = field_label
                return
            self._flush_locked()
            self._current = _FieldBuffer(
                app_bundle_id=app_bundle_id,
                field_key=effective_key,
                field_label=field_label,
            )

    # ---------------------------------------------------------- key dispatch

    def handle_key_event(self, event: dict[str, Any]) -> None:
        """Consume one normalised key event.

        ``event`` is the dict shape emitted by :mod:`recorder.event_tap`
        augmented with a ``chars`` field holding the Unicode string the
        keystroke would insert (empty / absent for non-printing keys).

        Events with cmd/ctrl/opt held are silently ignored — they are
        keyboard shortcuts and flow through the session as ``keypress``
        events via a different path.
        """
        with self._lock:
            if self._stopped or self._current is None:
                return

            modifiers = event.get("modifiers") or ()
            if any(m in _SUPPRESSED_MODIFIERS for m in modifiers):
                return

            key_code = event.get("key_code")
            if key_code in (BACKSPACE_KEY_CODE, FORWARD_DELETE_KEY_CODE):
                if self._current.buffer:
                    self._current.buffer.pop()
                self._reset_idle_timer_locked()
                return

            chars = event.get("chars")
            if not isinstance(chars, str) or not chars:
                return
            self._current.buffer.append(chars)
            self._reset_idle_timer_locked()

    # ------------------------------------------------------------- lifecycle

    def flush(self) -> None:
        """Emit the current buffer (if any) and clear focus."""
        with self._lock:
            self._flush_locked()

    def stop(self) -> None:
        """Flush in-flight buffer and refuse future events.  Idempotent."""
        with self._lock:
            if self._stopped:
                return
            self._flush_locked()
            self._stopped = True
            self._cancel_timer_locked()

    # --------------------------------------------------------- introspection

    def current_buffer(self) -> str:
        """Return the in-progress buffer text (for diagnostics / tests)."""
        with self._lock:
            if self._current is None:
                return ""
            return "".join(self._current.buffer)

    # --------------------------------------------------------------- internals

    def _flush_locked(self) -> None:
        self._cancel_timer_locked()
        current = self._current
        self._current = None
        if current is None:
            return
        text = "".join(current.buffer)
        if not text:
            # AC: do not emit empty text_input events.
            return
        payload: TextInputEvent = {
            "app_bundle_id": current.app_bundle_id,
            "text": text,
            "field_label": current.field_label,
        }
        try:
            self._emit(payload)
        except Exception:
            logger.exception("TextAggregator emit callback raised")

    def _reset_idle_timer_locked(self) -> None:
        self._cancel_timer_locked()
        if self._stopped or self._current is None:
            return
        timer = threading.Timer(self._idle_timeout, self._on_idle)
        timer.daemon = True
        timer.name = "recorder-text-aggregator-idle"
        timer.start()
        self._timer = timer

    def _cancel_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _on_idle(self) -> None:
        with self._lock:
            # A new keystroke could have slipped in between the Timer firing
            # and the lock being acquired; the timer reference would then
            # have been replaced.  Ignore stale firings.
            if self._stopped:
                return
            self._flush_locked()


# ---------------------------------------------------------------- AX helper


def resolve_focused_field(
    pid: int,
    *,
    timeout_seconds: float = 0.2,
) -> tuple[str | None, str | None]:
    """Return ``(field_key, field_label)`` for the app's focused text field.

    Best-effort.  Returns ``(None, None)`` when:

    * ``ApplicationServices`` is unavailable (non-darwin / no PyObjC),
    * the focused element is not a text-like role,
    * any AX call errors or the lookup exceeds ``timeout_seconds``.

    The lookup runs on a short-lived daemon thread gated by a
    :class:`threading.Event` so a stuck AX server cannot hang the caller
    — same pattern as :func:`recorder.ax_resolver.resolve_element_at`.
    """
    done = threading.Event()
    result: dict[str, str | None] = {"key": None, "label": None}

    def _worker() -> None:
        try:
            result["key"], result["label"] = _resolve_focused_field_sync(pid)
        except Exception:
            logger.debug("resolve_focused_field worker raised", exc_info=True)
        finally:
            done.set()

    thread = threading.Thread(
        target=_worker, name="recorder-focused-field", daemon=True
    )
    thread.start()
    if not done.wait(timeout_seconds):
        logger.warning(
            "resolve_focused_field timed out after %.3fs", timeout_seconds
        )
        return (None, None)
    return (result["key"], result["label"])


def _resolve_focused_field_sync(pid: int) -> tuple[str | None, str | None]:
    try:
        from ApplicationServices import (
            AXUIElementCopyAttributeValue,
            AXUIElementCreateApplication,
        )
    except ImportError:
        return (None, None)
    try:
        app_element = AXUIElementCreateApplication(pid)
    except Exception:
        logger.debug("AXUIElementCreateApplication raised", exc_info=True)
        return (None, None)
    if app_element is None:
        return (None, None)

    focused = _ax_copy(AXUIElementCopyAttributeValue, app_element, "AXFocusedUIElement")
    if focused is None:
        return (None, None)

    role_value = _ax_copy(AXUIElementCopyAttributeValue, focused, "AXRole")
    role = str(role_value) if role_value else None
    if role not in _TEXT_FIELD_ROLES:
        return (None, None)

    identifier_value = _ax_copy(AXUIElementCopyAttributeValue, focused, "AXIdentifier")
    identifier = str(identifier_value) if identifier_value else None

    title_value = _ax_copy(AXUIElementCopyAttributeValue, focused, "AXTitle")
    title = str(title_value) if title_value else None
    if not title:
        description_value = _ax_copy(
            AXUIElementCopyAttributeValue, focused, "AXDescription"
        )
        title = str(description_value) if description_value else None

    field_key = identifier or (f"role:{role}:label:{title}" if title else f"role:{role}")
    label = title or None
    return (field_key, label)


def _ax_copy(
    copy_attribute: Any,
    element: Any,
    attribute: str,
) -> Any:
    """Call ``AXUIElementCopyAttributeValue`` and return the value or ``None``.

    Hides the ``(error, value)`` return convention (and the three different
    error shapes PyObjC has surfaced over time) behind a single place, so
    callers can chain attribute reads without a try/except at every site.
    """
    try:
        err, value = copy_attribute(element, attribute, None)
    except Exception:
        logger.debug("AX copy(%s) raised", attribute, exc_info=True)
        return None
    if err != _AX_ERROR_SUCCESS or value is None:
        return None
    return value
