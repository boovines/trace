"""Live macOS screen source backed by CGWindowListCreateImage.

This is the ONLY module in the runner that captures real pixels off the
user's display. Instantiating ``LiveScreenSource`` calls
``require_live_mode()`` first — construction fails loudly with
``LiveModeNotAllowed`` unless ``TRACE_ALLOW_LIVE=1`` is set in the
environment. Ralph iterations and automated tests must NOT set that flag.

The capture pipeline is:

1. ``CGWindowListCreateImage`` with ``CGRectInfinite`` / on-screen-only /
   default image option returns a ``CGImage`` of the main display.
2. ``NSBitmapImageRep.initWithCGImage_`` wraps the image, and
   ``representationUsingType_properties_(NSBitmapImageFileTypePNG, ...)``
   serialises it to PNG bytes.
3. Pillow decodes those bytes; we hand the resulting ``PIL.Image.Image`` to
   ``runner.coords.capture_and_normalize`` which performs the LANCZOS
   downscale to ``target_longest_edge`` and returns the ``ImageMapping``
   that lets the runner translate Claude-space coordinates back to display
   points.

Screen Recording permission is detected via
``CGPreflightScreenCaptureAccess`` — when capture returns ``None`` we
probe the permission and raise ``PermissionDeniedError`` with an
actionable message. On newer macOS versions a denied permission causes
capture to return ``None``; on some configurations it returns a blank
image, in which case the preflight check is the definitive answer.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Final

from AppKit import NSBitmapImageFileTypePNG, NSBitmapImageRep
from PIL import Image
from Quartz import (
    CGPreflightScreenCaptureAccess,
    CGRectInfinite,
    CGWindowListCreateImage,
    kCGNullWindowID,
    kCGWindowImageDefault,
    kCGWindowListOptionOnScreenOnly,
)

from runner.coords import (
    DisplayInfo,
    ImageMapping,
    capture_and_normalize,
    get_main_display_info,
)
from runner.safety import require_live_mode

logger = logging.getLogger(__name__)


CAPTURE_WARN_SECONDS: Final[float] = 0.500
DEFAULT_TARGET_LONGEST_EDGE: Final[int] = 1568


class PermissionDeniedError(RuntimeError):
    """Raised when Screen Recording permission is denied.

    The message tells the user exactly where in System Settings to grant
    the permission; callers should surface it verbatim in UI / logs.
    """


class ScreenCaptureError(RuntimeError):
    """Raised when a capture fails for a reason other than permission."""


class LiveScreenSource:
    """Capture real screenshots of the main display.

    Constructing this source requires ``TRACE_ALLOW_LIVE=1``; the
    ``require_live_mode()`` call runs first and raises
    ``LiveModeNotAllowed`` otherwise. The source snapshots a
    ``DisplayInfo`` at construction time so the scale factor is stable
    across the run — if the user plugs in an external monitor mid-run the
    mapping will not update automatically (same behaviour as
    ``LiveInputAdapter``).
    """

    def __init__(
        self,
        *,
        display_info: DisplayInfo | None = None,
        target_longest_edge: int = DEFAULT_TARGET_LONGEST_EDGE,
    ) -> None:
        require_live_mode()
        self._display_info = display_info or get_main_display_info()
        self._target_longest_edge = target_longest_edge

    @property
    def display_info(self) -> DisplayInfo:
        return self._display_info

    def capture(self) -> tuple[bytes, ImageMapping]:
        """Capture the main display and return Claude-ready PNG bytes + mapping.

        Raises ``PermissionDeniedError`` when Screen Recording permission is
        not granted. Emits a warning log when the capture takes longer than
        ``CAPTURE_WARN_SECONDS`` (still returns successfully).
        """
        start = time.perf_counter()
        cg_image = self._create_screenshot()
        elapsed = time.perf_counter() - start

        if cg_image is None:
            if not self._has_screen_recording_permission():
                raise PermissionDeniedError(
                    "Screen Recording permission is not granted. Open "
                    "System Settings → Privacy & Security → Screen & System "
                    "Audio Recording, enable Trace, and restart the app."
                )
            raise ScreenCaptureError(
                "CGWindowListCreateImage returned None despite Screen "
                "Recording permission being granted."
            )

        if elapsed > CAPTURE_WARN_SECONDS:
            logger.warning(
                "Screen capture took %.3fs (>%.3fs threshold); "
                "agent responsiveness may degrade.",
                elapsed,
                CAPTURE_WARN_SECONDS,
            )

        pil_image = self._cg_image_to_pil(cg_image)
        return capture_and_normalize(
            self._display_info,
            self._target_longest_edge,
            source_image=pil_image,
        )

    # ------------------------------------------------------------------ #
    # Internal plumbing — factored so tests can mock one seam at a time
    # ------------------------------------------------------------------ #

    def _create_screenshot(self) -> object | None:
        try:
            image: object | None = CGWindowListCreateImage(
                CGRectInfinite,
                kCGWindowListOptionOnScreenOnly,
                kCGNullWindowID,
                kCGWindowImageDefault,
            )
        except Exception as exc:
            raise ScreenCaptureError(
                f"CGWindowListCreateImage raised: {exc!r}"
            ) from exc
        return image

    def _has_screen_recording_permission(self) -> bool:
        try:
            return bool(CGPreflightScreenCaptureAccess())
        except Exception:  # pragma: no cover - extremely defensive
            return False

    def _cg_image_to_pil(self, cg_image: object) -> Image.Image:
        rep = NSBitmapImageRep.alloc().initWithCGImage_(cg_image)
        if rep is None:
            raise ScreenCaptureError(
                "NSBitmapImageRep.initWithCGImage_ returned nil."
            )
        data = rep.representationUsingType_properties_(
            NSBitmapImageFileTypePNG, None
        )
        if data is None:
            raise ScreenCaptureError(
                "NSBitmapImageRep produced no PNG representation."
            )
        return Image.open(io.BytesIO(bytes(data))).convert("RGB")


__all__ = [
    "CAPTURE_WARN_SECONDS",
    "DEFAULT_TARGET_LONGEST_EDGE",
    "LiveScreenSource",
    "PermissionDeniedError",
    "ScreenCaptureError",
]
