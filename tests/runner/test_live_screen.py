"""Tests for :mod:`runner.live_screen` — the CGWindowListCreateImage-backed source.

Like ``test_live_input``, this file uses mock-based tests by default (patching
the PyObjC seams at the module level) and exposes a small set of
``@pytest.mark.macos`` integration tests that require a real display and the
Screen Recording permission. The integration tests are skipped off-darwin.
"""

from __future__ import annotations

import io
import logging
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image
from runner import live_screen as live_screen_module
from runner.coords import DisplayInfo, ImageMapping
from runner.live_screen import (
    CAPTURE_WARN_SECONDS,
    DEFAULT_TARGET_LONGEST_EDGE,
    LiveScreenSource,
    PermissionDeniedError,
    ScreenCaptureError,
)
from runner.safety import LiveModeNotAllowed

_FAKE_DISPLAY = DisplayInfo(
    width_points=1440.0,
    height_points=900.0,
    scale_factor=2.0,
    width_pixels=2880,
    height_pixels=1800,
)


def _png_bytes_of_size(size: tuple[int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (12, 34, 56)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def patch_capture_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, MagicMock | object]:
    """Patch the Quartz/AppKit seams and the CGImage→PIL converter.

    ``CGWindowListCreateImage`` returns a sentinel ``_FakeCGImage`` so tests
    can track what flowed through the pipeline. ``_cg_image_to_pil`` is
    overridden (via monkeypatching the bound method on the class) to return
    a 2880x1800 PIL image — matching the ``_FAKE_DISPLAY`` constants — so
    ``capture_and_normalize`` produces a predictable 1568-wide resize.
    """
    fake_cg_image = object()

    cg_create = MagicMock(return_value=fake_cg_image)
    cg_preflight = MagicMock(return_value=True)
    monkeypatch.setattr(live_screen_module, "CGWindowListCreateImage", cg_create)
    monkeypatch.setattr(
        live_screen_module, "CGPreflightScreenCaptureAccess", cg_preflight
    )

    def fake_cg_to_pil(self: LiveScreenSource, cg_image: object) -> Image.Image:
        assert cg_image is fake_cg_image
        return Image.new("RGB", (2880, 1800), (255, 255, 255))

    monkeypatch.setattr(LiveScreenSource, "_cg_image_to_pil", fake_cg_to_pil)

    return {
        "cg_image": fake_cg_image,
        "CGWindowListCreateImage": cg_create,
        "CGPreflightScreenCaptureAccess": cg_preflight,
    }


# --------------------------------------------------------------------------- #
# Safety gate
# --------------------------------------------------------------------------- #


def test_init_raises_without_flag() -> None:
    with pytest.raises(LiveModeNotAllowed) as exc:
        LiveScreenSource(display_info=_FAKE_DISPLAY)
    assert "TRACE_ALLOW_LIVE" in str(exc.value)


def test_init_succeeds_with_flag(live_mode_allowed: None) -> None:
    _ = live_mode_allowed
    src = LiveScreenSource(display_info=_FAKE_DISPLAY)
    assert src.display_info is _FAKE_DISPLAY


def test_default_target_longest_edge_constant() -> None:
    assert DEFAULT_TARGET_LONGEST_EDGE == 1568


def test_capture_warn_threshold_constant() -> None:
    assert CAPTURE_WARN_SECONDS == 0.500


# --------------------------------------------------------------------------- #
# Capture pipeline
# --------------------------------------------------------------------------- #


def test_capture_returns_png_and_mapping(
    live_mode_allowed: None,
    patch_capture_pipeline: dict[str, Any],
) -> None:
    _ = live_mode_allowed
    src = LiveScreenSource(display_info=_FAKE_DISPLAY)

    png_bytes, mapping = src.capture()

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert isinstance(mapping, ImageMapping)
    assert mapping.original_pixels == (2880, 1800)
    assert mapping.resized_pixels[0] == 1568  # longest-edge downscale
    # scale_from_resized_to_points = (2880/1568) / 2.0
    assert mapping.scale_from_resized_to_points == pytest.approx(
        (2880 / 1568) / 2.0
    )

    cg_create = patch_capture_pipeline["CGWindowListCreateImage"]
    assert cg_create.call_count == 1
    args = cg_create.call_args.args
    # (screenBounds, listOption, windowID, imageOption)
    assert args[0] is live_screen_module.CGRectInfinite
    assert args[1] == live_screen_module.kCGWindowListOptionOnScreenOnly
    assert args[2] == live_screen_module.kCGNullWindowID
    assert args[3] == live_screen_module.kCGWindowImageDefault


def test_capture_honors_custom_target_longest_edge(
    live_mode_allowed: None,
    patch_capture_pipeline: dict[str, Any],
) -> None:
    _ = live_mode_allowed, patch_capture_pipeline
    src = LiveScreenSource(display_info=_FAKE_DISPLAY, target_longest_edge=800)

    _, mapping = src.capture()
    assert mapping.resized_pixels[0] == 800


def test_capture_does_not_preflight_on_success(
    live_mode_allowed: None,
    patch_capture_pipeline: dict[str, Any],
) -> None:
    _ = live_mode_allowed
    src = LiveScreenSource(display_info=_FAKE_DISPLAY)
    src.capture()
    # Preflight is only consulted when the image is None — a successful
    # capture must not needlessly pay for the permission check.
    assert patch_capture_pipeline["CGPreflightScreenCaptureAccess"].call_count == 0


# --------------------------------------------------------------------------- #
# Permission handling
# --------------------------------------------------------------------------- #


def test_permission_denied_raises_with_actionable_message(
    live_mode_allowed: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = live_mode_allowed
    monkeypatch.setattr(
        live_screen_module, "CGWindowListCreateImage", MagicMock(return_value=None)
    )
    monkeypatch.setattr(
        live_screen_module,
        "CGPreflightScreenCaptureAccess",
        MagicMock(return_value=False),
    )

    src = LiveScreenSource(display_info=_FAKE_DISPLAY)
    with pytest.raises(PermissionDeniedError) as exc:
        src.capture()

    message = str(exc.value)
    assert "System Settings" in message
    assert "Screen" in message


def test_capture_none_but_permission_granted_raises_screen_capture_error(
    live_mode_allowed: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = live_mode_allowed
    monkeypatch.setattr(
        live_screen_module, "CGWindowListCreateImage", MagicMock(return_value=None)
    )
    monkeypatch.setattr(
        live_screen_module,
        "CGPreflightScreenCaptureAccess",
        MagicMock(return_value=True),
    )

    src = LiveScreenSource(display_info=_FAKE_DISPLAY)
    with pytest.raises(ScreenCaptureError, match="returned None"):
        src.capture()


def test_cg_window_list_create_image_exception_becomes_screen_capture_error(
    live_mode_allowed: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = live_mode_allowed
    monkeypatch.setattr(
        live_screen_module,
        "CGWindowListCreateImage",
        MagicMock(side_effect=RuntimeError("kaboom")),
    )
    src = LiveScreenSource(display_info=_FAKE_DISPLAY)
    with pytest.raises(ScreenCaptureError, match="kaboom"):
        src.capture()


# --------------------------------------------------------------------------- #
# Slow-capture warning
# --------------------------------------------------------------------------- #


def test_slow_capture_emits_warning(
    live_mode_allowed: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    patch_capture_pipeline: dict[str, Any],
) -> None:
    _ = live_mode_allowed, patch_capture_pipeline
    # Fake a monotonic clock jump larger than the warn threshold by driving
    # time.perf_counter inside the module.
    ticks = iter([10.000, 10.000 + CAPTURE_WARN_SECONDS + 0.050])
    monkeypatch.setattr(live_screen_module.time, "perf_counter", lambda: next(ticks))

    src = LiveScreenSource(display_info=_FAKE_DISPLAY)
    caplog.set_level(logging.WARNING, logger=live_screen_module.logger.name)
    src.capture()

    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert warnings, "expected a slow-capture warning"
    assert "Screen capture took" in warnings[0].getMessage()


def test_fast_capture_does_not_warn(
    live_mode_allowed: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    patch_capture_pipeline: dict[str, Any],
) -> None:
    _ = live_mode_allowed, patch_capture_pipeline
    ticks = iter([0.000, 0.005])  # 5 ms — well under threshold
    monkeypatch.setattr(live_screen_module.time, "perf_counter", lambda: next(ticks))

    src = LiveScreenSource(display_info=_FAKE_DISPLAY)
    caplog.set_level(logging.WARNING, logger=live_screen_module.logger.name)
    src.capture()

    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert warnings == []


# --------------------------------------------------------------------------- #
# CGImage → PIL converter seam
# --------------------------------------------------------------------------- #


def test_cg_image_to_pil_calls_nsbitmap_round_trip(
    live_mode_allowed: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default converter wraps the CGImage in an NSBitmapImageRep and
    asks for a PNG representation. Verify the call sequence with mocks."""
    _ = live_mode_allowed

    png_payload = _png_bytes_of_size((2880, 1800))

    class FakeNSData:
        def __bytes__(self) -> bytes:
            return png_payload

    fake_rep = MagicMock(name="NSBitmapImageRep_instance")
    fake_rep.representationUsingType_properties_.return_value = FakeNSData()
    fake_alloc = MagicMock(name="NSBitmapImageRep_alloc")
    fake_alloc.initWithCGImage_.return_value = fake_rep
    fake_class = MagicMock(name="NSBitmapImageRep_class")
    fake_class.alloc.return_value = fake_alloc

    monkeypatch.setattr(live_screen_module, "NSBitmapImageRep", fake_class)
    monkeypatch.setattr(live_screen_module, "NSBitmapImageFileTypePNG", 4)

    src = LiveScreenSource(display_info=_FAKE_DISPLAY)
    sentinel_cg_image = object()
    pil_image = src._cg_image_to_pil(sentinel_cg_image)

    fake_alloc.initWithCGImage_.assert_called_once_with(sentinel_cg_image)
    fake_rep.representationUsingType_properties_.assert_called_once_with(4, None)
    assert pil_image.size == (2880, 1800)


def test_cg_image_to_pil_raises_if_rep_is_nil(
    live_mode_allowed: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = live_mode_allowed
    fake_alloc = MagicMock()
    fake_alloc.initWithCGImage_.return_value = None
    fake_class = MagicMock()
    fake_class.alloc.return_value = fake_alloc
    monkeypatch.setattr(live_screen_module, "NSBitmapImageRep", fake_class)

    src = LiveScreenSource(display_info=_FAKE_DISPLAY)
    with pytest.raises(ScreenCaptureError, match="NSBitmapImageRep"):
        src._cg_image_to_pil(object())


def test_cg_image_to_pil_raises_if_representation_is_nil(
    live_mode_allowed: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = live_mode_allowed
    fake_rep = MagicMock()
    fake_rep.representationUsingType_properties_.return_value = None
    fake_alloc = MagicMock()
    fake_alloc.initWithCGImage_.return_value = fake_rep
    fake_class = MagicMock()
    fake_class.alloc.return_value = fake_alloc
    monkeypatch.setattr(live_screen_module, "NSBitmapImageRep", fake_class)

    src = LiveScreenSource(display_info=_FAKE_DISPLAY)
    with pytest.raises(ScreenCaptureError, match="no PNG representation"):
        src._cg_image_to_pil(object())


# --------------------------------------------------------------------------- #
# Live integration — real capture; skipped off-darwin or without permission.
# --------------------------------------------------------------------------- #


@pytest.mark.macos
@pytest.mark.skipif(sys.platform != "darwin", reason="requires real macOS display")
def test_live_capture_returns_valid_png(
    live_mode_allowed: None,
) -> None:  # pragma: no cover - environment-specific
    _ = live_mode_allowed
    from runner.live_screen import CGPreflightScreenCaptureAccess

    if not CGPreflightScreenCaptureAccess():
        pytest.skip("Screen Recording permission not granted")

    src = LiveScreenSource()
    png_bytes, mapping = src.capture()

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(png_bytes)) as img:
        assert img.size == mapping.resized_pixels
    assert max(mapping.resized_pixels) <= DEFAULT_TARGET_LONGEST_EDGE
