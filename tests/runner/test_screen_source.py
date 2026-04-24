"""Tests for :mod:`runner.screen_source` — dry-run trajectory screen source.

The live source (X-009) lives in a separate module. Here we verify only the
dry-run path: reading keyframes from a trajectory directory in order, falling
back to a blank canvas on a missing frame, and loudly rejecting a missing
trajectory directory.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from runner.coords import DisplayInfo, DryRunDisplayInfo, ImageMapping
from runner.screen_source import (
    ScreenSource,
    TrajectoryScreenSource,
    blank_canvas_png,
)


def _make_trajectory(
    root: Path,
    trajectory_id: str,
    frame_sizes: list[tuple[int, int] | None],
) -> Path:
    """Build a trajectory dir. ``None`` entries are skipped (missing keyframes)."""

    traj_dir = root / trajectory_id
    (traj_dir / "screenshots").mkdir(parents=True)
    for i, size in enumerate(frame_sizes, start=1):
        if size is None:
            continue
        frame = Image.new("RGB", size, (i * 10 % 256, 0, 0))
        frame.save(traj_dir / "screenshots" / f"{i:04d}.png")
    return traj_dir


def test_capture_returns_png_bytes_and_mapping(tmp_path: Path) -> None:
    _make_trajectory(tmp_path, "abc", [(2880, 1800)])
    src = TrajectoryScreenSource("abc", trajectories_root=tmp_path)

    png_bytes, mapping = src.capture()

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert isinstance(mapping, ImageMapping)
    # Full-size 2880x1800 retina frame resizes to 1568 on the long edge.
    assert mapping.original_pixels == (2880, 1800)
    assert mapping.resized_pixels[0] == 1568


def test_capture_advances_through_frames_in_order(tmp_path: Path) -> None:
    _make_trajectory(
        tmp_path,
        "seq",
        [(2880, 1800), (1440, 900), (2880, 1800)],
    )
    src = TrajectoryScreenSource("seq", trajectories_root=tmp_path)

    _, m0 = src.capture()
    _, m1 = src.capture()
    _, m2 = src.capture()

    assert m0.original_pixels == (2880, 1800)
    assert m1.original_pixels == (1440, 900)
    assert m2.original_pixels == (2880, 1800)
    assert src.index == 3


def test_missing_keyframe_falls_back_to_blank_canvas(tmp_path: Path) -> None:
    # Frame 1 exists, frame 2 is missing (None), frame 3 exists.
    _make_trajectory(tmp_path, "gap", [(2880, 1800), None, (2880, 1800)])
    src = TrajectoryScreenSource("gap", trajectories_root=tmp_path)

    _, m0 = src.capture()
    png1, m1 = src.capture()
    _, m2 = src.capture()

    assert m0.original_pixels == (2880, 1800)
    # Blank canvas is 1440x900 (already under target edge, so no resize).
    assert m1.original_pixels == (1440, 900)
    assert m1.resized_pixels == (1440, 900)
    assert png1.startswith(b"\x89PNG\r\n\x1a\n")
    # Decoding the returned bytes should produce a white 1440x900 image.
    with Image.open(io.BytesIO(png1)) as img:
        assert img.size == (1440, 900)
        assert img.convert("RGB").getpixel((0, 0)) == (255, 255, 255)
    assert m2.original_pixels == (2880, 1800)


def test_missing_trajectory_directory_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        TrajectoryScreenSource("nope", trajectories_root=tmp_path)
    message = str(exc.value)
    assert "nope" in message
    assert "Trajectory directory not found" in message


def test_index_is_zero_before_first_capture(tmp_path: Path) -> None:
    _make_trajectory(tmp_path, "empty-start", [(1440, 900)])
    src = TrajectoryScreenSource("empty-start", trajectories_root=tmp_path)
    assert src.index == 0
    src.capture()
    assert src.index == 1


def test_capture_uses_provided_display_info(tmp_path: Path) -> None:
    _make_trajectory(tmp_path, "custom", [(1920, 1080)])
    custom = DisplayInfo(
        width_points=1920.0,
        height_points=1080.0,
        scale_factor=1.0,
        width_pixels=1920,
        height_pixels=1080,
    )
    src = TrajectoryScreenSource(
        "custom",
        trajectories_root=tmp_path,
        display_info=custom,
    )
    _, mapping = src.capture()
    # Non-retina 1920x1080 source, scale 1.0 → s = (1920/1568)/1.0 ≈ 1.2245.
    expected = (1920 / mapping.resized_pixels[0]) / 1.0
    assert mapping.scale_from_resized_to_points == pytest.approx(expected)


def test_defaults_to_dry_run_display_info(tmp_path: Path) -> None:
    _make_trajectory(tmp_path, "default-display", [(2880, 1800)])
    src = TrajectoryScreenSource("default-display", trajectories_root=tmp_path)
    _, mapping = src.capture()
    # Retina defaults: (2880/1568) / 2.0
    expected = (2880 / mapping.resized_pixels[0]) / DryRunDisplayInfo.scale_factor
    assert mapping.scale_from_resized_to_points == pytest.approx(expected)


def test_ran_out_of_frames_returns_blank_canvas(tmp_path: Path) -> None:
    _make_trajectory(tmp_path, "short", [(1440, 900)])
    src = TrajectoryScreenSource("short", trajectories_root=tmp_path)
    src.capture()  # real frame
    png, mapping = src.capture()  # past the end → blank
    assert mapping.original_pixels == (1440, 900)
    with Image.open(io.BytesIO(png)) as img:
        assert img.convert("RGB").getpixel((0, 0)) == (255, 255, 255)


def test_corrupt_keyframe_falls_back_to_blank(tmp_path: Path) -> None:
    traj_dir = tmp_path / "corrupt"
    (traj_dir / "screenshots").mkdir(parents=True)
    # A .png file that is not actually a PNG.
    (traj_dir / "screenshots" / "0001.png").write_bytes(b"not a png")

    src = TrajectoryScreenSource("corrupt", trajectories_root=tmp_path)
    png, mapping = src.capture()

    assert mapping.original_pixels == (1440, 900)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_blank_canvas_helper_is_valid_png() -> None:
    payload = blank_canvas_png()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(payload)) as img:
        assert img.size == (1440, 900)


def test_trajectory_source_satisfies_protocol(tmp_path: Path) -> None:
    _make_trajectory(tmp_path, "proto", [(1440, 900)])
    src: ScreenSource = TrajectoryScreenSource("proto", trajectories_root=tmp_path)
    # ScreenSource is declared for static-typing only; duck-check the call site.
    assert callable(getattr(src, "capture", None))


def test_dry_run_sources_do_not_call_require_live_mode(tmp_path: Path) -> None:
    # No live_mode_allowed fixture: constructing must succeed anyway.
    _make_trajectory(tmp_path, "safe", [(1440, 900)])
    src = TrajectoryScreenSource("safe", trajectories_root=tmp_path)
    png, _ = src.capture()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
