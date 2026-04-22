"""Main-display screenshot capture for trajectory keyframes.

The Recorder captures keyframe screenshots (not per-event) to give the
Synthesizer visual context without exploding storage.  This module owns
the low-level capture: it asks CoreGraphics for a ``CGImage`` of the
on-screen content of the main display and encodes it to PNG bytes that
:class:`~recorder.writer.TrajectoryWriter` can persist.

Two public entry points:

* :func:`capture_main_display` — returns PNG bytes or ``None`` on any
  framework failure.  Never raises past the module boundary so the event
  loop cannot be killed by a transient capture error.
* :func:`get_main_display_info` — returns ``{width, height, scale_factor}``
  for ``metadata.display_info``.  Screen-point width/height (Retina-
  independent); ``scale_factor`` tells consumers how to map to pixels.

All PyObjC access happens inside the function body with ``from X import …``
so tests can stub ``Quartz`` / ``AppKit`` by installing a
``types.ModuleType`` into :data:`sys.modules`.  No real macOS APIs are
touched during the default hermetic run.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

__all__ = [
    "DisplayInfo",
    "capture_main_display",
    "get_main_display_info",
]

logger = logging.getLogger(__name__)

# ``NSBitmapImageFileTypePNG`` raw value from AppKit. Named as a constant so
# tests can reference it by name instead of a bare literal.
_NS_BITMAP_FILE_TYPE_PNG: int = 4

# PNG magic bytes; exposed for callers that want to sanity-check bytes.
PNG_MAGIC: bytes = b"\x89PNG\r\n\x1a\n"


class DisplayInfo(TypedDict):
    """Shape of ``metadata.json``'s ``display_info`` object."""

    width: int
    height: int
    scale_factor: float


def capture_main_display() -> bytes | None:
    """Capture the main display's on-screen content as PNG bytes.

    Returns ``None`` when:

    * ``Quartz`` or ``AppKit`` cannot be imported (e.g. non-darwin host),
    * ``CGWindowListCreateImage`` returns ``None`` (Screen Recording
      permission denied, or the window server is transiently unavailable),
    * the ``NSBitmapImageRep`` round-trip fails to produce PNG bytes.

    The call is best-effort — every framework step is wrapped in
    ``try/except`` and logged at warning level so the session
    orchestrator can keep appending events when a single keyframe fails.
    """
    try:
        import Quartz
    except ImportError:
        logger.warning("Quartz unavailable; screenshot capture disabled")
        return None

    try:
        from AppKit import NSBitmapImageRep
    except ImportError:
        logger.warning("AppKit unavailable; screenshot capture disabled")
        return None

    cg_image = _create_main_display_image(Quartz)
    if cg_image is None:
        return None

    return _cg_image_to_png_bytes(cg_image, NSBitmapImageRep)


def get_main_display_info() -> DisplayInfo | None:
    """Return ``{width, height, scale_factor}`` for the main display.

    Width / height are in **screen points** (Retina-independent); the
    ``scale_factor`` is ``NSScreen.backingScaleFactor`` (typically ``1.0``
    on non-Retina, ``2.0`` on Retina).  Returns ``None`` if the info can
    not be read — the caller is expected to substitute a best-effort
    fallback or skip the recording.
    """
    try:
        from AppKit import NSScreen
    except ImportError:
        logger.warning("AppKit unavailable; cannot read display info")
        return None

    try:
        screen = NSScreen.mainScreen()
    except Exception:
        logger.exception("NSScreen.mainScreen() raised")
        return None
    if screen is None:
        return None

    try:
        frame = screen.frame()
        width = int(frame.size.width)
        height = int(frame.size.height)
        scale = float(screen.backingScaleFactor())
    except Exception:
        logger.exception("NSScreen frame/backingScaleFactor failed")
        return None

    if width <= 0 or height <= 0 or scale <= 0:
        return None

    return DisplayInfo(width=width, height=height, scale_factor=scale)


def _create_main_display_image(quartz: Any) -> Any | None:
    """Call ``CGWindowListCreateImage`` for the full main display.

    Uses ``kCGNullWindowID`` + ``kCGWindowListOptionOnScreenOnly`` +
    ``kCGWindowImageDefault`` — the canonical combination for "one
    screenshot of every on-screen window on the main display at their
    current composited state" (i.e. what the user sees).
    """
    try:
        rect = quartz.CGRectInfinite
        on_screen_only = quartz.kCGWindowListOptionOnScreenOnly
        null_window = quartz.kCGNullWindowID
        image_default = quartz.kCGWindowImageDefault
    except AttributeError:
        logger.warning("Quartz is missing CGWindowList* sentinels")
        return None

    try:
        cg_image = quartz.CGWindowListCreateImage(
            rect, on_screen_only, null_window, image_default
        )
    except Exception:
        logger.exception("CGWindowListCreateImage raised")
        return None

    if cg_image is None:
        logger.warning(
            "CGWindowListCreateImage returned None — Screen Recording "
            "permission may be missing",
        )
        return None
    return cg_image


def _cg_image_to_png_bytes(cg_image: Any, bitmap_cls: Any) -> bytes | None:
    """Convert a CGImageRef to PNG bytes via ``NSBitmapImageRep``."""
    try:
        bitmap = bitmap_cls.alloc().initWithCGImage_(cg_image)
    except Exception:
        logger.exception("NSBitmapImageRep.initWithCGImage_ raised")
        return None
    if bitmap is None:
        return None

    try:
        data = bitmap.representationUsingType_properties_(
            _NS_BITMAP_FILE_TYPE_PNG, None
        )
    except Exception:
        logger.exception("NSBitmapImageRep.representationUsingType_properties_ raised")
        return None
    if data is None:
        logger.warning("NSBitmapImageRep returned no PNG data")
        return None

    try:
        png_bytes = bytes(data)
    except Exception:
        logger.exception("Converting NSData to bytes failed")
        return None

    if not png_bytes.startswith(PNG_MAGIC):
        logger.warning("Captured bytes do not start with PNG magic header")
        return None
    return png_bytes
