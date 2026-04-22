"""Track active application and focused window via NSWorkspace + AX.

The Recorder needs to attribute every captured event to the application that
owned it, and it needs ``app_switch`` / ``window_focus`` trajectory events so
cross-app workflows show up in the synthesised skill.  :class:`FocusTracker`
owns that state:

* Subscribes to ``NSWorkspaceDidActivateApplicationNotification`` on the
  shared ``NSWorkspace.notificationCenter()``.  The block handler converts
  the notification's ``NSRunningApplication`` into a plain ``AppInfo`` dict
  and feeds it through the state machine.
* Resolves the focused window title of the active app via
  ``AXUIElementCreateApplication(pid)`` + ``AXFocusedWindow`` + ``AXTitle``.
  This is only done on app-change — per PRD notes, window-title polling is
  expensive, so we never do it on a timer.
* Maintains an ``app_focus_history`` suitable for ``metadata.json``.

All PyObjC access happens inside helper methods that ``from X import …``
locally, so tests stub ``AppKit`` / ``ApplicationServices`` by installing a
``types.ModuleType`` into ``sys.modules``.  The core state machine
(:meth:`handle_app_activated`) can be driven synchronously from tests
without any framework in sight.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypedDict

__all__ = [
    "APP_ACTIVATED_NOTIFICATION",
    "AppFocusHistoryEntry",
    "AppInfo",
    "AppSwitchCallback",
    "AppSwitchPayload",
    "FocusTracker",
    "WindowFocusCallback",
    "WindowFocusPayload",
]

logger = logging.getLogger(__name__)

APP_ACTIVATED_NOTIFICATION = "NSWorkspaceDidActivateApplicationNotification"
_USER_INFO_APP_KEY = "NSWorkspaceApplicationKey"
_AX_ERROR_SUCCESS = 0


class AppInfo(TypedDict):
    """Identity of the frontmost application."""

    bundle_id: str
    name: str
    pid: int


class AppFocusHistoryEntry(TypedDict):
    """One row of ``metadata.json``'s ``app_focus_history`` array."""

    bundle_id: str
    name: str
    entered_at: str
    exited_at: str | None


class AppSwitchPayload(TypedDict):
    """Payload fired to :meth:`FocusTracker.on_app_switch` callbacks."""

    from_bundle_id: str | None
    to_bundle_id: str
    from_name: str | None
    to_name: str | None


class WindowFocusPayload(TypedDict):
    """Payload fired to :meth:`FocusTracker.on_window_focus_change` callbacks."""

    window_title: str | None
    app_bundle_id: str


AppSwitchCallback = Callable[[AppSwitchPayload], None]
WindowFocusCallback = Callable[[WindowFocusPayload], None]


class FocusTracker:
    """Observe app/window focus transitions and expose them to the session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_app: AppInfo | None = None
        self._current_window_title: str | None = None
        self._app_switch_callbacks: list[AppSwitchCallback] = []
        self._window_focus_callbacks: list[WindowFocusCallback] = []
        self._history: list[AppFocusHistoryEntry] = []
        self._observer_handle: Any = None
        self._started: bool = False

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Subscribe to NSWorkspace notifications and seed current state.

        Raises :class:`RuntimeError` if called more than once on the same
        instance — construct a fresh tracker to restart tracking.
        """
        if self._started:
            raise RuntimeError("FocusTracker.start() called more than once")
        self._started = True
        self._subscribe_workspace()
        current = self._query_frontmost_app()
        if current is not None:
            self._transition_to(current)

    def stop(self) -> None:
        """Unsubscribe and finalise the last history entry. Idempotent."""
        if not self._started:
            return
        self._started = False
        self._unsubscribe_workspace()
        with self._lock:
            if self._history and self._history[-1].get("exited_at") is None:
                self._history[-1]["exited_at"] = _now_iso()

    # -------------------------------------------------------- callback hooks

    def on_app_switch(self, callback: AppSwitchCallback) -> None:
        """Register a callback fired on every frontmost-app change."""
        self._app_switch_callbacks.append(callback)

    def on_window_focus_change(self, callback: WindowFocusCallback) -> None:
        """Register a callback fired when the focused window title changes."""
        self._window_focus_callbacks.append(callback)

    # --------------------------------------------------- public accessors

    def get_current_app(self) -> AppInfo | None:
        """Return a *copy* of the currently frontmost app, or ``None``."""
        with self._lock:
            if self._current_app is None:
                return None
            return AppInfo(
                bundle_id=self._current_app["bundle_id"],
                name=self._current_app["name"],
                pid=self._current_app["pid"],
            )

    def get_current_window_title(self) -> str | None:
        """Return the cached focused window title, or ``None`` if unknown."""
        with self._lock:
            return self._current_window_title

    def get_app_focus_history(self) -> list[AppFocusHistoryEntry]:
        """Return a copy of the focus history suitable for ``metadata.json``."""
        with self._lock:
            return [
                AppFocusHistoryEntry(
                    bundle_id=entry["bundle_id"],
                    name=entry["name"],
                    entered_at=entry["entered_at"],
                    exited_at=entry["exited_at"],
                )
                for entry in self._history
            ]

    # ----------------------------------------------------- driven from tests

    def handle_app_activated(self, app: AppInfo) -> None:
        """Drive the state machine directly with an app info dict.

        This is the seam tests (and the session orchestrator, if it ever
        needs to force a re-evaluation) use to push events in without going
        through the NSWorkspace notification plumbing.
        """
        self._transition_to(app)

    def refresh_window_title(self) -> None:
        """Re-query AX for the focused window title of the current app.

        Only ever fires the ``window_focus_change`` callback when the title
        actually changes, so it is cheap to call opportunistically — for
        example, on the start of every recording or after a known navigation
        event.  Consistent with the "never poll on a timer" note in the PRD.
        """
        with self._lock:
            app = (
                AppInfo(
                    bundle_id=self._current_app["bundle_id"],
                    name=self._current_app["name"],
                    pid=self._current_app["pid"],
                )
                if self._current_app is not None
                else None
            )
        if app is None:
            return
        title = self._query_focused_window_title(app["pid"])
        self._update_window_title(app, title)

    # ------------------------------------------------------ state transitions

    def _transition_to(self, new_app: AppInfo) -> None:
        """Update current app + history, fire callbacks, refresh window title."""
        with self._lock:
            prev = self._current_app
            same = (
                prev is not None
                and prev["bundle_id"] == new_app["bundle_id"]
                and prev["pid"] == new_app["pid"]
            )
            if same:
                return
            if self._history and self._history[-1].get("exited_at") is None:
                self._history[-1]["exited_at"] = _now_iso()
            snapshot = AppInfo(
                bundle_id=new_app["bundle_id"],
                name=new_app["name"],
                pid=new_app["pid"],
            )
            self._current_app = snapshot
            self._current_window_title = None
            self._history.append(
                AppFocusHistoryEntry(
                    bundle_id=snapshot["bundle_id"],
                    name=snapshot["name"],
                    entered_at=_now_iso(),
                    exited_at=None,
                )
            )
            payload = AppSwitchPayload(
                from_bundle_id=prev["bundle_id"] if prev is not None else None,
                to_bundle_id=snapshot["bundle_id"],
                from_name=prev["name"] if prev is not None else None,
                to_name=snapshot["name"],
            )

        self._dispatch_app_switch(payload)
        title = self._query_focused_window_title(new_app["pid"])
        self._update_window_title(new_app, title)

    def _update_window_title(self, app: AppInfo, title: str | None) -> None:
        with self._lock:
            if self._current_app is None or self._current_app["pid"] != app["pid"]:
                # Another transition beat us; drop this stale title update.
                return
            if self._current_window_title == title:
                return
            self._current_window_title = title
        payload = WindowFocusPayload(
            window_title=title,
            app_bundle_id=app["bundle_id"],
        )
        for cb in list(self._window_focus_callbacks):
            try:
                cb(payload)
            except Exception:
                logger.exception("FocusTracker window-focus callback raised")

    def _dispatch_app_switch(self, payload: AppSwitchPayload) -> None:
        for cb in list(self._app_switch_callbacks):
            try:
                cb(payload)
            except Exception:
                logger.exception("FocusTracker app-switch callback raised")

    # ------------------------------------------------ NSWorkspace integration

    def _subscribe_workspace(self) -> None:
        try:
            from AppKit import NSWorkspace
        except ImportError:
            logger.warning("AppKit unavailable; focus tracking disabled")
            return
        try:
            center = NSWorkspace.sharedWorkspace().notificationCenter()
        except Exception:
            logger.exception("NSWorkspace.notificationCenter() raised")
            return
        try:
            handle = center.addObserverForName_object_queue_usingBlock_(
                APP_ACTIVATED_NOTIFICATION,
                None,
                None,
                self._ns_workspace_block,
            )
        except Exception:
            logger.exception(
                "addObserverForName_object_queue_usingBlock_ raised"
            )
            return
        self._observer_handle = handle

    def _unsubscribe_workspace(self) -> None:
        if self._observer_handle is None:
            return
        handle = self._observer_handle
        self._observer_handle = None
        try:
            from AppKit import NSWorkspace
        except ImportError:
            return
        try:
            center = NSWorkspace.sharedWorkspace().notificationCenter()
            center.removeObserver_(handle)
        except Exception:
            logger.debug(
                "removeObserver_ during unsubscribe raised", exc_info=True
            )

    def _ns_workspace_block(self, notification: Any) -> None:
        """Invoked by NSWorkspace when an app activates."""
        app_info = _extract_app_info(notification)
        if app_info is None:
            return
        try:
            self._transition_to(app_info)
        except Exception:
            logger.exception("FocusTracker._transition_to raised from block")

    # ---------------------------------------------------- AX / NSWorkspace ops

    def _query_frontmost_app(self) -> AppInfo | None:
        try:
            from AppKit import NSWorkspace
        except ImportError:
            return None
        try:
            app = NSWorkspace.sharedWorkspace().frontmostApplication()
        except Exception:
            logger.debug(
                "NSWorkspace.frontmostApplication raised", exc_info=True
            )
            return None
        if app is None:
            return None
        return _ns_running_app_to_dict(app)

    def _query_focused_window_title(self, pid: int) -> str | None:
        try:
            from ApplicationServices import (
                AXUIElementCopyAttributeValue,
                AXUIElementCreateApplication,
            )
        except ImportError:
            return None
        try:
            app_element = AXUIElementCreateApplication(pid)
        except Exception:
            logger.debug(
                "AXUIElementCreateApplication raised", exc_info=True
            )
            return None
        if app_element is None:
            return None
        try:
            err, window = AXUIElementCopyAttributeValue(
                app_element, "AXFocusedWindow", None
            )
        except Exception:
            logger.debug(
                "AXUIElementCopyAttributeValue(AXFocusedWindow) raised",
                exc_info=True,
            )
            return None
        if err != _AX_ERROR_SUCCESS or window is None:
            return None
        try:
            err, title = AXUIElementCopyAttributeValue(window, "AXTitle", None)
        except Exception:
            logger.debug(
                "AXUIElementCopyAttributeValue(AXTitle) raised", exc_info=True
            )
            return None
        if err != _AX_ERROR_SUCCESS or title is None:
            return None
        try:
            text = str(title)
        except Exception:
            return None
        return text or None


# ----------------------------------------------------------- module helpers


def _now_iso() -> str:
    """ISO-8601 timestamp with millisecond precision, compatible with the schema."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _extract_app_info(notification: Any) -> AppInfo | None:
    """Pull an :class:`AppInfo` dict out of an NSWorkspace notification."""
    try:
        user_info = notification.userInfo()
    except Exception:
        logger.debug("notification.userInfo() raised", exc_info=True)
        return None
    if user_info is None:
        return None
    running_app: Any = None
    # NSDictionary supports both ``.get`` (via PyObjC bridge) and the ObjC
    # ``objectForKey_`` method.  Try the Python-friendly form first and fall
    # back to ObjC so tests can stub either shape.
    get = getattr(user_info, "get", None)
    if callable(get):
        try:
            running_app = get(_USER_INFO_APP_KEY)
        except Exception:
            running_app = None
    if running_app is None:
        object_for_key = getattr(user_info, "objectForKey_", None)
        if callable(object_for_key):
            try:
                running_app = object_for_key(_USER_INFO_APP_KEY)
            except Exception:
                running_app = None
    if running_app is None:
        return None
    return _ns_running_app_to_dict(running_app)


def _ns_running_app_to_dict(app: Any) -> AppInfo | None:
    """Convert an ``NSRunningApplication`` to an :class:`AppInfo` dict."""
    bundle_id: Any = None
    name: Any = None
    pid: Any = None
    try:
        bundle_id = app.bundleIdentifier()
    except Exception:
        logger.debug("bundleIdentifier() raised", exc_info=True)
    try:
        name = app.localizedName()
    except Exception:
        logger.debug("localizedName() raised", exc_info=True)
    try:
        pid = app.processIdentifier()
    except Exception:
        logger.debug("processIdentifier() raised", exc_info=True)
    if pid is None:
        return None
    try:
        pid_int = int(pid)
    except Exception:
        return None
    return AppInfo(
        bundle_id=str(bundle_id) if bundle_id else "",
        name=str(name) if name else "",
        pid=pid_int,
    )
