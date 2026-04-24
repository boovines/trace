"""macOS permission detection for the Recorder.

The Recorder needs three macOS privacy permissions to function:

* **Accessibility** — to create a passive ``CGEventTap`` and to query the
  Accessibility (AX) API for semantic targets of clicks.
* **Screen Recording** — to capture keyframe screenshots via
  ``CGWindowListCreateImage`` / ``CGDisplayCreateImage``.
* **Input Monitoring** — required on newer macOS versions for ``CGEventTap``
  to actually receive mouse and keyboard events.

This module only *inspects* the current status. It never prompts the user —
the Tauri UI owns the prompting UX and decides when to open the System
Settings pane. Missing permissions are reported back to the UI as a
structured error so the UI can guide the user.

All framework access is wrapped in narrow, individually mockable helpers so
tests can drive every combination of permission states without touching the
real macOS frameworks.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

__all__ = [
    "HOW_TO_GRANT",
    "PermissionName",
    "PermissionsError",
    "PermissionsStatus",
    "check_permissions",
    "get_missing_permissions_error",
]

logger = logging.getLogger(__name__)

PermissionName = Literal["accessibility", "screen_recording", "input_monitoring"]


class PermissionsStatus(TypedDict):
    """Boolean status for each required macOS permission."""

    accessibility: bool
    screen_recording: bool
    input_monitoring: bool


class PermissionsError(TypedDict):
    """Structured error the HTTP API returns when permissions are missing."""

    error: Literal["missing_permission"]
    permissions: list[str]
    how_to_grant: dict[str, str]


HOW_TO_GRANT: dict[str, str] = {
    "accessibility": (
        "Open System Settings → Privacy & Security → Accessibility and enable "
        "Trace. Required for the global event tap and Accessibility (AX) "
        "queries that resolve clicks to semantic targets."
    ),
    "screen_recording": (
        "Open System Settings → Privacy & Security → Screen Recording and "
        "enable Trace. Required for the keyframe screenshots that give the "
        "synthesizer visual context."
    ),
    "input_monitoring": (
        "Open System Settings → Privacy & Security → Input Monitoring and "
        "enable Trace. Required by CGEventTap on modern macOS to receive "
        "global mouse and keyboard events."
    ),
}


def _check_accessibility() -> bool:
    """Return True when the process is trusted for the Accessibility API.

    Uses ``AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: False})``
    so that this call is pure inspection — it never surfaces the system
    Accessibility prompt. The UI decides when (if ever) to prompt.
    """
    try:
        from ApplicationServices import (
            AXIsProcessTrustedWithOptions,
            kAXTrustedCheckOptionPrompt,
        )
    except ImportError:
        logger.warning(
            "ApplicationServices unavailable; reporting accessibility=False"
        )
        return False

    try:
        options = {kAXTrustedCheckOptionPrompt: False}
        return bool(AXIsProcessTrustedWithOptions(options))
    except Exception:
        logger.exception("AXIsProcessTrustedWithOptions raised")
        return False


def _check_screen_recording() -> bool:
    """Return True when the process holds Screen Recording permission.

    Prefers ``CGPreflightScreenCaptureAccess`` (which exists in recent
    PyObjC/Quartz builds). Falls back to attempting a 1x1 pixel capture and
    inspecting whether the returned ``CGImage`` is non-NULL, which is the
    documented workaround on older systems.
    """
    try:
        from Quartz import CGPreflightScreenCaptureAccess
    except ImportError:
        return _check_screen_recording_fallback()

    try:
        return bool(CGPreflightScreenCaptureAccess())
    except Exception:
        logger.exception("CGPreflightScreenCaptureAccess raised")
        return _check_screen_recording_fallback()


def _check_screen_recording_fallback() -> bool:
    """Try to grab a tiny screenshot — a non-NULL image means permission is OK."""
    try:
        from Quartz import (
            CGDisplayCreateImageForRect,
            CGMainDisplayID,
            CGRectMake,
        )
    except ImportError:
        logger.warning("Quartz unavailable; reporting screen_recording=False")
        return False

    try:
        image = CGDisplayCreateImageForRect(CGMainDisplayID(), CGRectMake(0, 0, 1, 1))
        return image is not None
    except Exception:
        logger.exception("Screen Recording fallback capture raised")
        return False


def _check_input_monitoring() -> bool:
    """Return True when a listen-only ``CGEventTap`` can be created.

    This is a best-effort check. ``CGEventTapCreate`` returns ``None`` when
    the caller lacks Input Monitoring permission on macOS 10.15+, which is
    exactly the signal we need. We immediately invalidate the tap afterwards
    so we do not leave a dangling run-loop source.
    """
    try:
        from Quartz import (
            CFMachPortInvalidate,
            CGEventTapCreate,
            kCGEventTapOptionListenOnly,
            kCGHeadInsertEventTap,
            kCGSessionEventTap,
        )
    except ImportError:
        logger.warning("Quartz unavailable; reporting input_monitoring=False")
        return False

    try:
        tap = CGEventTapCreate(
            kCGSessionEventTap,
            kCGHeadInsertEventTap,
            kCGEventTapOptionListenOnly,
            0,
            _noop_tap_callback,
            None,
        )
    except Exception:
        logger.exception("CGEventTapCreate raised")
        return False

    if tap is None:
        return False

    try:
        CFMachPortInvalidate(tap)
    except Exception:
        logger.debug("CFMachPortInvalidate raised during probe cleanup", exc_info=True)
    return True


def _noop_tap_callback(proxy: Any, event_type: Any, event: Any, refcon: Any) -> Any:
    """Callback used only for the Input Monitoring probe tap; never fires."""
    return event


def check_permissions() -> PermissionsStatus:
    """Return the current permission status for the three required permissions.

    Never prompts. Safe to call at any time, including from the HTTP service
    health check endpoint.
    """
    return PermissionsStatus(
        accessibility=_check_accessibility(),
        screen_recording=_check_screen_recording(),
        input_monitoring=_check_input_monitoring(),
    )


def get_missing_permissions_error() -> PermissionsError | None:
    """Return a structured error dict if any permission is missing, else None.

    The returned dict is shaped for direct JSON serialisation by the HTTP
    layer:

    ``{"error": "missing_permission", "permissions": [...], "how_to_grant": {...}}``

    ``permissions`` is the list of missing permission names in the canonical
    order ``accessibility``, ``screen_recording``, ``input_monitoring`` so
    consumers can rely on deterministic output.
    """
    status = check_permissions()
    order: tuple[PermissionName, ...] = (
        "accessibility",
        "screen_recording",
        "input_monitoring",
    )
    missing = [name for name in order if not status[name]]
    if not missing:
        return None
    return PermissionsError(
        error="missing_permission",
        permissions=list(missing),
        how_to_grant={name: HOW_TO_GRANT[name] for name in missing},
    )
