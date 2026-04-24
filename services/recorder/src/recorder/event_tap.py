"""Global mouse/keyboard/scroll capture via a passive ``CGEventTap``.

The :class:`EventTap` runs a dedicated background thread that owns a
``CFRunLoop`` with a ``CGEventTap`` source attached.  The tap is
*listen-only* (``kCGEventTapOptionListenOnly``) so it never swallows or
modifies user input — it just observes it.

Each observed event is normalised into a plain ``dict`` and handed to a
user-supplied callback.  The callback is invoked on the tap thread; callers
that need to do heavy work should queue the event elsewhere rather than
block the tap.  If the callback raises, the exception is logged and the tap
stays alive — one bad consumer cannot tear down capture.

macOS occasionally disables a long-running tap (``kCGEventTapDisabledBy*``).
When that happens, :class:`EventTap` re-enables the tap in place and emits a
synthetic ``{"cg_event_type": "tap_reenabled", ...}`` dict through the
callback so the session layer can write a ``tap_reenabled`` trajectory
event.  That self-healing behaviour is a hard requirement from the PRD —
without it, a CPU spike silently drops the rest of the recording.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

__all__ = ["EventCallback", "EventTap"]

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], None]

# Public event-type strings the normaliser emits. Downstream code (session
# orchestrator) maps these to the schema ``type`` values.
_TYPE_NAMES: dict[int, str] = {
    1: "left_mouse_down",
    2: "left_mouse_up",
    3: "right_mouse_down",
    4: "right_mouse_up",
    10: "key_down",
    11: "key_up",
    12: "flags_changed",
    22: "scroll_wheel",
    25: "other_mouse_down",
    26: "other_mouse_up",
}

# The ``type`` values CGEvent uses to tell us the tap was disabled. These are
# unsigned ``uint32`` constants in the CoreGraphics headers, but PyObjC
# surfaces them as small negative ints (-2 / -1) on some releases. Accept
# both forms.
_TAP_DISABLED_TIMEOUT: tuple[int, ...] = (0xFFFFFFFE, -2)
_TAP_DISABLED_USER_INPUT: tuple[int, ...] = (0xFFFFFFFF, -1)


def _event_mask_for_recorder() -> int:
    """Bitmask for mouse down/up (L/R/Other), key down+up, flags, scroll."""
    types = (1, 2, 3, 4, 10, 11, 12, 22, 25, 26)
    mask = 0
    for t in types:
        mask |= 1 << t
    return mask


def _modifiers_from_flags(flags: int) -> list[str]:
    """Decode a ``CGEventFlags`` bitmask into a sorted list of modifier names."""
    # Values from CoreGraphics/CGEventTypes.h.
    table = (
        (0x00020000, "shift"),
        (0x00040000, "control"),
        (0x00080000, "option"),
        (0x00100000, "command"),
        (0x00800000, "fn"),
        (0x00010000, "caps_lock"),
    )
    return [name for bit, name in table if flags & bit]


def _normalise_event(cg_event_type: int, event: Any, quartz: Any) -> dict[str, Any]:
    """Convert a ``CGEventRef`` into a JSON-friendly dict.

    ``quartz`` is the ``Quartz`` module, passed in so tests can inject a
    stub without having to install PyObjC.
    """
    payload: dict[str, Any] = {
        "cg_event_type": _TYPE_NAMES.get(cg_event_type, f"unknown_{cg_event_type}"),
        "cg_event_type_code": int(cg_event_type),
        "timestamp_ms": int(time.time() * 1000),
    }

    try:
        loc = quartz.CGEventGetLocation(event)
        # CGPoint has ``x`` and ``y`` attributes in PyObjC.
        payload["location_x"] = float(loc.x)
        payload["location_y"] = float(loc.y)
    except Exception:
        logger.debug("CGEventGetLocation failed", exc_info=True)
        payload["location_x"] = None
        payload["location_y"] = None

    try:
        flags = int(quartz.CGEventGetFlags(event))
    except Exception:
        logger.debug("CGEventGetFlags failed", exc_info=True)
        flags = 0
    payload["modifiers"] = _modifiers_from_flags(flags)
    payload["flags_raw"] = flags

    # Key events carry a key code.
    if cg_event_type in (10, 11, 12):
        try:
            key_code = int(
                quartz.CGEventGetIntegerValueField(
                    event, quartz.kCGKeyboardEventKeycode
                )
            )
        except Exception:
            logger.debug("CGEventGetIntegerValueField(keycode) failed", exc_info=True)
            key_code = None
        payload["key_code"] = key_code

        # Translate keycode → unicode string the keystroke would insert, so
        # ``text_aggregator.handle_key_event`` can buffer it.  Empty for
        # non-printing keys (arrows, function keys, etc.) — aggregator
        # ignores those silently.  Only meaningful for key_down (10); up
        # and flags_changed events don't carry typed text.
        chars: str | None = None
        if cg_event_type == 10:
            try:
                _, result = quartz.CGEventKeyboardGetUnicodeString(
                    event, 4, None, None
                )
                if isinstance(result, str):
                    chars = result
            except Exception:
                logger.debug(
                    "CGEventKeyboardGetUnicodeString failed", exc_info=True
                )
        payload["chars"] = chars

    # Scroll events carry per-axis deltas.
    if cg_event_type == 22:
        try:
            payload["scroll_delta_y"] = int(
                quartz.CGEventGetIntegerValueField(
                    event, quartz.kCGScrollWheelEventDeltaAxis1
                )
            )
        except Exception:
            logger.debug("CGEventGetIntegerValueField(scroll y) failed", exc_info=True)
            payload["scroll_delta_y"] = 0
        try:
            payload["scroll_delta_x"] = int(
                quartz.CGEventGetIntegerValueField(
                    event, quartz.kCGScrollWheelEventDeltaAxis2
                )
            )
        except Exception:
            logger.debug("CGEventGetIntegerValueField(scroll x) failed", exc_info=True)
            payload["scroll_delta_x"] = 0

    return payload


def _load_quartz() -> Any:
    """Import and return the ``Quartz`` module, or ``None`` on non-darwin.

    Factored out so callers can deal uniformly with a missing PyObjC without
    repeating ``try/except ImportError`` at each site.
    """
    try:
        import Quartz
    except ImportError:
        return None
    return Quartz


def _tap_reenabled_event(reason: str) -> dict[str, Any]:
    return {
        "cg_event_type": "tap_reenabled",
        "reason": reason,
        "timestamp_ms": int(time.time() * 1000),
    }


class EventTap:
    """Passive global event tap running on a dedicated thread.

    Usage::

        tap = EventTap(callback=lambda e: print(e))
        tap.start()
        ...
        tap.stop()

    :class:`EventTap` is not re-entrant; call :meth:`start` once per
    instance.  Construct a fresh tap if you need to restart after
    :meth:`stop`.
    """

    #: Join timeout when :meth:`stop` waits for the tap thread to exit.
    JOIN_TIMEOUT_SECONDS: float = 2.0

    def __init__(self, callback: EventCallback) -> None:
        self._callback: EventCallback = callback
        self._thread: threading.Thread | None = None
        self._started: threading.Event = threading.Event()
        self._start_error: BaseException | None = None
        self._run_loop: Any = None
        self._tap: Any = None
        self._run_loop_source: Any = None
        self._stopping: bool = False

    # ------------------------------------------------------------------ lifecycle

    def start(self, *, start_timeout: float = 5.0) -> None:
        """Start the tap thread and block until the tap is created.

        Raises :class:`RuntimeError` if the tap cannot be created (most
        commonly because Accessibility / Input Monitoring permission is not
        granted) or if the thread fails to initialise within
        ``start_timeout`` seconds.
        """
        if self._thread is not None:
            raise RuntimeError("EventTap.start() called more than once")

        self._thread = threading.Thread(
            target=self._run, name="recorder-event-tap", daemon=True
        )
        self._thread.start()

        if not self._started.wait(timeout=start_timeout):
            # The thread is still alive but never signalled readiness. Ask it
            # to give up so we do not leak a daemon thread.
            self._stopping = True
            raise RuntimeError(
                f"EventTap thread failed to initialise within {start_timeout}s"
            )
        if self._start_error is not None:
            raise RuntimeError(f"EventTap failed to start: {self._start_error}")

    def stop(self) -> None:
        """Stop the tap, exit the run loop, and join the thread.

        Idempotent — calling :meth:`stop` twice is safe.  Joins the thread
        with a hard :attr:`JOIN_TIMEOUT_SECONDS` second timeout so the
        caller (the HTTP ``/recorder/stop`` handler) is never blocked.
        """
        if self._thread is None:
            return
        self._stopping = True

        quartz = _load_quartz()

        if self._tap is not None and quartz is not None:
            try:
                quartz.CGEventTapEnable(self._tap, False)
            except Exception:
                logger.debug("CGEventTapEnable(False) during stop raised", exc_info=True)

        if self._run_loop is not None and quartz is not None:
            try:
                quartz.CFRunLoopStop(self._run_loop)
            except Exception:
                logger.debug("CFRunLoopStop raised", exc_info=True)

        self._thread.join(timeout=self.JOIN_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            logger.warning(
                "EventTap thread did not exit within %.1fs", self.JOIN_TIMEOUT_SECONDS
            )
        self._thread = None

    # -------------------------------------------------------------------- thread

    def _run(self) -> None:
        """Run-loop thread entry point. Creates the tap and enters CFRunLoop."""
        try:
            import Quartz
        except ImportError as exc:
            self._start_error = exc
            self._started.set()
            return

        try:
            tap = Quartz.CGEventTapCreate(
                # kCGSessionEventTap, not kCGHIDEventTap: HID-layer taps only
                # see system hotkeys for unprivileged processes (Cmd+Tab,
                # Cmd+Space, …); plain typing flows direct to the focused
                # app. Session-layer taps see every user keystroke provided
                # Accessibility + Input Monitoring are granted.
                Quartz.kCGSessionEventTap,
                Quartz.kCGHeadInsertEventTap,
                Quartz.kCGEventTapOptionListenOnly,
                _event_mask_for_recorder(),
                self._tap_callback,
                None,
            )
        except Exception as exc:
            self._start_error = exc
            self._started.set()
            return

        if tap is None:
            self._start_error = RuntimeError(
                "CGEventTapCreate returned NULL — Accessibility or Input "
                "Monitoring permission is likely not granted"
            )
            self._started.set()
            return

        self._tap = tap

        try:
            source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
            run_loop = Quartz.CFRunLoopGetCurrent()
            Quartz.CFRunLoopAddSource(run_loop, source, Quartz.kCFRunLoopCommonModes)
            Quartz.CGEventTapEnable(tap, True)
        except Exception as exc:
            self._start_error = exc
            self._started.set()
            return

        self._run_loop_source = source
        self._run_loop = run_loop
        self._started.set()

        try:
            Quartz.CFRunLoopRun()
        except Exception:
            logger.exception("CFRunLoopRun raised; tap thread exiting")
        finally:
            try:
                Quartz.CFRunLoopRemoveSource(
                    run_loop, source, Quartz.kCFRunLoopCommonModes
                )
            except Exception:
                logger.debug("CFRunLoopRemoveSource raised", exc_info=True)

    # ------------------------------------------------------------- tap callback

    def _tap_callback(
        self,
        proxy: Any,
        event_type: int,
        event: Any,
        refcon: Any,
    ) -> Any:
        """CGEventTap callback. Returns ``event`` unchanged (listen-only tap)."""
        del proxy, refcon  # required by CGEventTap signature, unused here
        try:
            self.handle_cg_event(int(event_type), event)
        except Exception:
            logger.exception("EventTap consumer callback raised")
        return event

    def handle_cg_event(self, event_type: int, event: Any) -> None:
        """Public hook that turns a raw CGEvent into a callback invocation.

        Split out from :meth:`_tap_callback` so unit tests can exercise the
        normalisation + tap-reenable path without needing a real
        ``CGEventTapProxy``.
        """
        if event_type in _TAP_DISABLED_TIMEOUT:
            self._reenable("timeout")
            return
        if event_type in _TAP_DISABLED_USER_INPUT:
            self._reenable("user_input")
            return

        try:
            import Quartz
        except ImportError:  # pragma: no cover — event tap only runs on darwin
            return
        payload = _normalise_event(event_type, event, Quartz)
        self._deliver(payload)

    def _reenable(self, reason: str) -> None:
        # When ``stop()`` calls ``CGEventTapEnable(False)`` the kernel posts a
        # synthetic ``kCGEventTapDisabledBy*`` event back through the tap
        # callback. That isn't a real "macOS suspended our tap" — it's the
        # disable we just requested, echoed. Don't log a scary WARNING for
        # it, and don't bother re-enabling a tap that's about to be
        # destroyed; the trajectory-level ``tap_reenabled`` event would also
        # be dropped by ``_deliver``'s ``_stopping`` guard anyway.
        if self._stopping:
            logger.debug(
                "Ignoring tap-disabled echo from our own stop() (%s)", reason
            )
            return
        logger.warning("CGEventTap disabled (%s); re-enabling", reason)
        if self._tap is not None:
            try:
                from Quartz import CGEventTapEnable

                CGEventTapEnable(self._tap, True)
            except Exception:
                logger.exception("CGEventTapEnable(True) during re-enable failed")
        self._deliver(_tap_reenabled_event(reason))

    def _deliver(self, payload: dict[str, Any]) -> None:
        if self._stopping:
            return
        try:
            self._callback(payload)
        except Exception:
            logger.exception("EventTap consumer callback raised")
