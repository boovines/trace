"""Retina-aware coordinate mapping and screenshot normalization.

The Runner captures screenshots in **pixels** but the macOS event system (and
the coordinates Claude is shown in the resized image) needs mapping back to
**points** on the main display. On a 2.0-scale Retina display the pixel/point
ratio is 2:1, so mis-mapping produces clicks that land at double or half the
intended location — the single most common source of mis-click bugs.

Design:
* ``DisplayInfo`` captures the backing scale factor plus width/height in both
  point and pixel space.
* ``ImageMapping`` captures the downscaling applied when preparing a screenshot
  for Claude (Anthropic recommends a longest edge of 1568 for computer use).
* ``capture_and_normalize`` is the glue that turns a captured image into the
  PNG bytes Claude should see plus the mapping needed to get back to points.

Live capture (``CGWindowListCreateImage`` + permission handling + timing) is
implemented by ``LiveScreenSource`` in X-009; see ``runner.live_screen``. The
default ``source_image=None`` path here deliberately raises so the call site is
forced to provide an image — the one exception is ``LiveScreenSource`` which
captures, converts to PIL, and passes ``source_image`` explicitly.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True, slots=True)
class DisplayInfo:
    """Logical (point) and physical (pixel) dimensions of a display."""

    width_points: float
    height_points: float
    scale_factor: float
    width_pixels: int
    height_pixels: int


@dataclass(frozen=True, slots=True)
class ImageMapping:
    """Captures the downscaling applied to a screenshot.

    ``scale_from_resized_to_points`` multiplies a coordinate in the resized
    image's pixel space to get the equivalent display-point coordinate. It
    combines the resized→original-pixels ratio with the pixels→points ratio
    (the display's backing scale factor).
    """

    original_pixels: tuple[int, int]
    resized_pixels: tuple[int, int]
    scale_from_resized_to_points: float


DryRunDisplayInfo: DisplayInfo = DisplayInfo(
    width_points=1440.0,
    height_points=900.0,
    scale_factor=2.0,
    width_pixels=2880,
    height_pixels=1800,
)


def get_main_display_info() -> DisplayInfo:
    """Query ``NSScreen.mainScreen()`` for logical size and scale factor."""

    from AppKit import NSScreen

    screen = NSScreen.mainScreen()
    if screen is None:
        raise RuntimeError(
            "No main screen available; is the process running headless or "
            "before the AppKit event loop is up?"
        )
    frame = screen.frame()
    scale = float(screen.backingScaleFactor())
    width_pt = float(frame.size.width)
    height_pt = float(frame.size.height)
    return DisplayInfo(
        width_points=width_pt,
        height_points=height_pt,
        scale_factor=scale,
        width_pixels=round(width_pt * scale),
        height_pixels=round(height_pt * scale),
    )


def points_to_pixels(x_pt: float, y_pt: float, info: DisplayInfo) -> tuple[int, int]:
    """Map a display-point coordinate to its backing pixel coordinate."""

    return (
        round(x_pt * info.scale_factor),
        round(y_pt * info.scale_factor),
    )


def pixels_to_points(x_px: int, y_px: int, info: DisplayInfo) -> tuple[float, float]:
    """Map a backing-pixel coordinate to the display-point coordinate."""

    return (x_px / info.scale_factor, y_px / info.scale_factor)


def resized_pixels_to_points(
    x: float, y: float, mapping: ImageMapping
) -> tuple[float, float]:
    """Map a coordinate in the resized image's pixel space to display points."""

    s = mapping.scale_from_resized_to_points
    return (x * s, y * s)


def capture_and_normalize(
    info: DisplayInfo,
    target_longest_edge: int = 1568,
    *,
    source_image: Image.Image | None = None,
) -> tuple[bytes, ImageMapping]:
    """Normalise a screenshot for Claude's computer-use input.

    Downscales ``source_image`` (if provided) so its longest edge equals
    ``target_longest_edge``, preserving aspect ratio via Pillow's LANCZOS
    filter. If the image is already at or under the target, no resize is
    applied and the bytes encode the original dimensions.

    Returns a ``(png_bytes, ImageMapping)`` tuple; the mapping lets a caller
    convert coordinates Claude produces (in the resized image's pixel space)
    back to the display-point space the input adapter expects.

    ``source_image=None`` is reserved for ``LiveScreenSource`` (X-009) which
    does its own capture. Test and dry-run callers must pass an image
    explicitly.
    """

    if source_image is None:
        raise NotImplementedError(
            "Live capture is owned by LiveScreenSource (X-009); pass "
            "source_image explicitly for tests, dry-run, and any non-live "
            "caller."
        )

    original_size: tuple[int, int] = (source_image.width, source_image.height)
    longest = max(original_size)

    if longest <= target_longest_edge:
        resized = source_image
        resized_size: tuple[int, int] = original_size
    else:
        ratio = target_longest_edge / longest
        new_w = round(source_image.width * ratio)
        new_h = round(source_image.height * ratio)
        resized = source_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized_size = (new_w, new_h)

    scale_from_resized_to_points = (
        original_size[0] / resized_size[0]
    ) / info.scale_factor

    buf = io.BytesIO()
    resized.save(buf, format="PNG")
    return (
        buf.getvalue(),
        ImageMapping(
            original_pixels=original_size,
            resized_pixels=resized_size,
            scale_from_resized_to_points=scale_from_resized_to_points,
        ),
    )
