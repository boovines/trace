"""Tests for :mod:`runner.coords` — retina coordinate mapping + normalisation.

The arithmetic here is the single most failure-prone part of the Runner (a
2x or 1/2x off-by-factor turns into mysterious misclicks), so the tests below
exercise explicit numeric invariants rather than round-trip-only assertions
wherever a specific Retina geometry is being verified.
"""

from __future__ import annotations

import io
import sys

import pytest
from PIL import Image

from runner.coords import (
    DisplayInfo,
    DryRunDisplayInfo,
    ImageMapping,
    capture_and_normalize,
    get_main_display_info,
    pixels_to_points,
    points_to_pixels,
    resized_pixels_to_points,
)

_RETINA = DisplayInfo(
    width_points=1440.0,
    height_points=900.0,
    scale_factor=2.0,
    width_pixels=2880,
    height_pixels=1800,
)
_NON_RETINA = DisplayInfo(
    width_points=1920.0,
    height_points=1080.0,
    scale_factor=1.0,
    width_pixels=1920,
    height_pixels=1080,
)


def _solid_image(w: int, h: int, color: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)


# --------------------------------------------------------------------------- #
# DryRunDisplayInfo + DisplayInfo sanity                                      #
# --------------------------------------------------------------------------- #


def test_dry_run_display_info_matches_common_retina() -> None:
    assert DryRunDisplayInfo.width_points == 1440.0
    assert DryRunDisplayInfo.height_points == 900.0
    assert DryRunDisplayInfo.scale_factor == 2.0
    assert DryRunDisplayInfo.width_pixels == 2880
    assert DryRunDisplayInfo.height_pixels == 1800


def test_display_info_is_frozen() -> None:
    with pytest.raises(AttributeError):
        DryRunDisplayInfo.scale_factor = 3.0  # type: ignore[misc]


def test_image_mapping_is_frozen() -> None:
    mapping = ImageMapping(
        original_pixels=(2880, 1800),
        resized_pixels=(1568, 980),
        scale_from_resized_to_points=0.9183673469387755,
    )
    with pytest.raises(AttributeError):
        mapping.scale_from_resized_to_points = 1.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# points ↔ pixels                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("x_pt", "y_pt"),
    [(0.0, 0.0), (1.0, 1.0), (720.0, 450.0), (1439.5, 899.5), (1440.0, 900.0)],
)
def test_retina_round_trip_points_to_pixels_and_back(x_pt: float, y_pt: float) -> None:
    px = points_to_pixels(x_pt, y_pt, _RETINA)
    assert px == (round(x_pt * 2.0), round(y_pt * 2.0))
    back = pixels_to_points(px[0], px[1], _RETINA)
    # With integer pixel rounding, round-trip is exact for half-point values at
    # scale 2.0 and within 0.5 of a point elsewhere.
    assert abs(back[0] - x_pt) <= 0.5
    assert abs(back[1] - y_pt) <= 0.5


@pytest.mark.parametrize(
    ("x_pt", "y_pt"),
    [(0.0, 0.0), (50.0, 50.0), (1919.0, 1079.0), (1920.0, 1080.0)],
)
def test_non_retina_round_trip_is_exact(x_pt: float, y_pt: float) -> None:
    px = points_to_pixels(x_pt, y_pt, _NON_RETINA)
    assert px == (int(x_pt), int(y_pt))
    back = pixels_to_points(px[0], px[1], _NON_RETINA)
    assert back == (float(px[0]), float(px[1]))


def test_points_to_pixels_rounds_to_nearest() -> None:
    # 10.49 pt * 2.0 = 20.98 → 21 px
    assert points_to_pixels(10.49, 10.49, _RETINA) == (21, 21)
    # 10.51 pt * 2.0 = 21.02 → 21 px
    assert points_to_pixels(10.51, 10.51, _RETINA) == (21, 21)


# --------------------------------------------------------------------------- #
# resized_pixels_to_points                                                    #
# --------------------------------------------------------------------------- #


def test_resized_pixels_to_points_for_1568x980_on_retina() -> None:
    """A 2880x1800 Retina screenshot resized to longest-edge 1568 (→1568x980).

    The center of the resized image (784, 490) must map back to the center of
    the display in points (720, 450).
    """

    img = _solid_image(2880, 1800)
    _png, mapping = capture_and_normalize(_RETINA, 1568, source_image=img)

    assert mapping.original_pixels == (2880, 1800)
    assert mapping.resized_pixels == (1568, 980)
    cx_pt, cy_pt = resized_pixels_to_points(784, 490, mapping)
    assert cx_pt == pytest.approx(720.0, abs=0.01)
    assert cy_pt == pytest.approx(450.0, abs=0.01)


def test_button_pixel_center_maps_back_to_button_center_in_points() -> None:
    """Off-by-one precision check for a 100x100 button at a known location.

    The button occupies points (200..300, 300..400) → pixels (400..600, 600..800)
    on a 2.0-scale display → resized pixels (ratio 1568/2880 ≈ 0.5444) at
    approximately (217.8..326.7, 326.7..435.6). The center of that resized
    rectangle must round-trip back to (250, 350) points (the button center) to
    sub-pixel precision.
    """

    img = _solid_image(2880, 1800)
    _png, mapping = capture_and_normalize(_RETINA, 1568, source_image=img)

    ratio = 1568 / 2880
    cx_resized = ((400 + 600) / 2) * ratio  # center of button in resized px
    cy_resized = ((600 + 800) / 2) * ratio
    cx_pt, cy_pt = resized_pixels_to_points(cx_resized, cy_resized, mapping)
    assert cx_pt == pytest.approx(250.0, abs=0.01)
    assert cy_pt == pytest.approx(350.0, abs=0.01)


def test_resized_pixels_to_points_on_non_retina_is_identity_in_pixel_space() -> None:
    """Non-Retina: resized==original and scale_factor==1.0 → mapping is identity."""

    img = _solid_image(1920, 1080)
    _png, mapping = capture_and_normalize(_NON_RETINA, 1568, source_image=img)
    # 1920 > 1568 so it does resize on a non-retina display too.
    assert mapping.resized_pixels[0] == 1568
    pt = resized_pixels_to_points(0, 0, mapping)
    assert pt == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# capture_and_normalize                                                       #
# --------------------------------------------------------------------------- #


def test_capture_and_normalize_returns_valid_png_bytes() -> None:
    img = _solid_image(2880, 1800, color=(10, 20, 30))
    png, _mapping = capture_and_normalize(_RETINA, 1568, source_image=img)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    reopened = Image.open(io.BytesIO(png))
    reopened.load()
    assert reopened.size == (1568, 980)
    assert reopened.format == "PNG"


def test_capture_and_normalize_preserves_aspect_ratio() -> None:
    # 3000x1500 → resize longest edge to 1500 → 1500x750 (ratio preserved)
    img = _solid_image(3000, 1500)
    _png, mapping = capture_and_normalize(_RETINA, 1500, source_image=img)
    rw, rh = mapping.resized_pixels
    assert rw == 1500
    assert rh == 750
    assert pytest.approx(rw / rh, rel=1e-9) == 3000 / 1500


def test_capture_and_normalize_tiny_image_not_upscaled() -> None:
    """800x600 with target 1568 — no upscaling, resized == original."""

    img = _solid_image(800, 600, color=(42, 42, 42))
    png, mapping = capture_and_normalize(_RETINA, 1568, source_image=img)

    assert mapping.original_pixels == (800, 600)
    assert mapping.resized_pixels == (800, 600)
    # scale_from_resized_to_points = (orig_w / resized_w) / scale_factor
    # = 1.0 / 2.0 = 0.5 for a 2.0-scale Retina display
    assert mapping.scale_from_resized_to_points == pytest.approx(0.5, rel=1e-12)

    reopened = Image.open(io.BytesIO(png))
    reopened.load()
    assert reopened.size == (800, 600)


def test_capture_and_normalize_exactly_at_target_is_not_resized() -> None:
    img = _solid_image(1568, 1000)
    _png, mapping = capture_and_normalize(_RETINA, 1568, source_image=img)
    assert mapping.resized_pixels == (1568, 1000)


def test_capture_and_normalize_source_image_required() -> None:
    """Default path is reserved for LiveScreenSource (X-009); None → NotImplementedError."""

    with pytest.raises(NotImplementedError) as exc:
        capture_and_normalize(_RETINA, 1568)
    assert "source_image" in str(exc.value)
    assert "X-009" in str(exc.value) or "LiveScreenSource" in str(exc.value)


def test_scale_from_resized_to_points_formula() -> None:
    """Explicit arithmetic check: on a 2.0 display, a 2880→1568 resize yields
    scale_from_resized_to_points = (2880/1568)/2.0 ≈ 0.918367...
    """

    img = _solid_image(2880, 1800)
    _png, mapping = capture_and_normalize(_RETINA, 1568, source_image=img)
    expected = (2880 / 1568) / 2.0
    assert mapping.scale_from_resized_to_points == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------- #
# Integration: real display (macOS-only, skipped elsewhere)                   #
# --------------------------------------------------------------------------- #


@pytest.mark.macos
@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS main display")
def test_get_main_display_info_matches_physical_display() -> None:
    info = get_main_display_info()
    assert info.width_points > 0
    assert info.height_points > 0
    assert info.scale_factor in {1.0, 2.0, 3.0}
    assert info.width_pixels == round(info.width_points * info.scale_factor)
    assert info.height_pixels == round(info.height_points * info.scale_factor)
