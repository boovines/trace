"""Screen sources for the runner's computer-use loop.

Every ``capture()`` returns ``(png_bytes, ImageMapping)``, the same shape the
``LiveScreenSource`` (X-009) will return. The PNG is the resized image Claude
sees; the mapping lets the runner translate a coordinate Claude returns (in
the resized image's pixel space) back to display points for the input
adapter.

``TrajectoryScreenSource`` is the dry-run source: it reads the keyframes under
``<trajectories_root>/<trajectory_id>/screenshots/`` in order. If a given
keyframe is missing (e.g. the recorder dropped one) the source falls back to a
static blank canvas so the runner can still make progress in Ralph iterations.
The directory missing entirely is a real error — that means the caller is
pointing at the wrong trajectory and should fail loudly.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Final, Protocol

from PIL import Image, UnidentifiedImageError

from runner.coords import (
    DisplayInfo,
    DryRunDisplayInfo,
    ImageMapping,
    capture_and_normalize,
)

_SCREENSHOT_DIGITS: Final[int] = 4
_BLANK_CANVAS_SIZE: Final[tuple[int, int]] = (1440, 900)
_BLANK_CANVAS_COLOR: Final[tuple[int, int, int]] = (255, 255, 255)


class ScreenSource(Protocol):
    """Protocol every screen source must implement.

    Each call to ``capture()`` returns the screenshot for the *current* moment
    in the runner's timeline. Implementations advance their own notion of
    "current" on each call — trajectory replay walks through the keyframes;
    the live source grabs a fresh frame from the display.
    """

    def capture(self) -> tuple[bytes, ImageMapping]: ...


def _blank_canvas() -> Image.Image:
    """Return the fallback PNG used when a keyframe is missing."""

    return Image.new("RGB", _BLANK_CANVAS_SIZE, _BLANK_CANVAS_COLOR)


class TrajectoryScreenSource:
    """Replay a recorded trajectory's screenshots in order.

    Constructed with the trajectory id and the root containing trajectory
    directories. Each ``capture()`` advances to the next keyframe; once the
    keyframes are exhausted the source keeps returning the last seen frame
    (or, if it ran out without having read any, the blank canvas) so a
    longer-than-expected agent loop does not crash.

    ``display_info`` overrides the display used for the coordinate mapping;
    it defaults to ``DryRunDisplayInfo`` (the canonical 2880x1800 Retina
    panel the rest of the runner's dry-run uses).

    Does NOT call ``require_live_mode()`` — pure disk read, safe for Ralph.
    """

    def __init__(
        self,
        trajectory_id: str,
        *,
        trajectories_root: Path,
        display_info: DisplayInfo | None = None,
    ) -> None:
        self._trajectory_id = trajectory_id
        self._root = trajectories_root / trajectory_id
        self._screenshots_dir = self._root / "screenshots"
        self._display_info = display_info or DryRunDisplayInfo
        self._index = 0

        if not self._root.is_dir():
            raise FileNotFoundError(
                f"Trajectory directory not found: {self._root}. "
                f"Check that trajectory_id={trajectory_id!r} and "
                f"trajectories_root={trajectories_root} are correct."
            )

    @property
    def index(self) -> int:
        """Number of frames already returned. Zero before the first capture."""
        return self._index

    def capture(self) -> tuple[bytes, ImageMapping]:
        """Return the next keyframe (resized PNG + mapping).

        Missing or unreadable keyframes fall back to a blank canvas at the
        default dry-run dimensions so a replay never hard-fails on a dropped
        frame.
        """
        frame_path = self._frame_path(self._index)
        image = self._load_or_blank(frame_path)
        self._index += 1
        return capture_and_normalize(self._display_info, source_image=image)

    def _frame_path(self, index: int) -> Path:
        # Trajectories use 1-based, 4-digit zero-padded screenshot filenames
        # (``0001.png``) per the trajectory contract.
        return self._screenshots_dir / f"{index + 1:0{_SCREENSHOT_DIGITS}d}.png"

    def _load_or_blank(self, path: Path) -> Image.Image:
        if not path.is_file():
            return _blank_canvas()
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except (OSError, UnidentifiedImageError):
            return _blank_canvas()


def blank_canvas_png() -> bytes:
    """Return the raw PNG bytes of the fallback blank canvas.

    Exposed for tests and for any caller that wants to pre-seed a placeholder
    screenshot without instantiating a full ``TrajectoryScreenSource``.
    """
    buf = io.BytesIO()
    _blank_canvas().save(buf, format="PNG")
    return buf.getvalue()
