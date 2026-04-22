"""Tests for ``recorder.screenshot``.

The hermetic tests stub ``Quartz`` and ``AppKit`` in ``sys.modules`` so we
never touch the real CoreGraphics/AppKit APIs on the default run. A small
valid PNG payload is crafted inline for the happy-path check — we do not
depend on Pillow here.

A single opt-in ``@pytest.mark.macos`` integration test captures the real
main display and asserts PNG validity + Retina-dimension sanity. Run it
via ``TRACE_RUN_MACOS_TESTS=1 pytest tests/recorder/test_screenshot.py``.
"""

from __future__ import annotations

import struct
import sys
import types
from collections.abc import Iterator
from typing import Any, ClassVar

import pytest

from recorder import screenshot

# ---------------------------------------------------------------------------
# Minimal valid PNG for the stubbed pipeline's output.
# ---------------------------------------------------------------------------


def _make_min_png(width: int = 4, height: int = 4) -> bytes:
    """Synthesize a tiny but structurally valid PNG (magic + IHDR + IDAT + IEND).

    We only need ``startswith(PNG_MAGIC)`` to pass and the dimension
    parse in the integration test to work. A full encoder is overkill —
    we emit a hand-rolled 4x4 RGB image with a zlib-compressed IDAT.
    """
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw_row = b"\x00" + b"\x00\x00\x00" * width  # filter byte + width RGB pixels
    raw = raw_row * height
    idat = zlib.compress(raw, 6)
    return (
        screenshot.PNG_MAGIC
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", idat)
        + chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# Fake Quartz / AppKit modules
# ---------------------------------------------------------------------------


class _FakeBitmap:
    def __init__(self, png_bytes: bytes | None = None) -> None:
        self.png_bytes = png_bytes
        self.rep_calls: list[tuple[int, Any]] = []

    def representationUsingType_properties_(self, file_type: int, props: Any) -> Any:
        self.rep_calls.append((file_type, props))
        return self.png_bytes


class _FakeBitmapImageRep:
    """Stand-in for ``AppKit.NSBitmapImageRep``.

    ``NSBitmapImageRep.alloc().initWithCGImage_(cg_image)`` must return a
    bitmap whose ``representationUsingType_properties_`` returns PNG bytes.
    We model the two-step ``alloc() -> init...`` idiom with a tiny helper.
    """

    _next_png: ClassVar[bytes | None] = None
    _init_should_fail: ClassVar[bool] = False
    last_cg_image: ClassVar[Any] = None
    rep_calls: ClassVar[list[tuple[int, Any]]] = []  # reset per-test via subclass

    @classmethod
    def alloc(cls) -> _FakeBitmapImageRep:
        return cls()

    def initWithCGImage_(self, cg_image: Any) -> Any:
        type(self).last_cg_image = cg_image
        if type(self)._init_should_fail:
            return None
        bitmap = _FakeBitmap(type(self)._next_png)
        # Record the call on the class-level list too so tests can assert it.
        orig_rep = bitmap.representationUsingType_properties_

        def _wrapped(file_type: int, props: Any) -> Any:
            type(self).rep_calls.append((file_type, props))
            return orig_rep(file_type, props)

        bitmap.representationUsingType_properties_ = _wrapped  # type: ignore[method-assign]
        return bitmap


def _make_fake_quartz(
    *,
    image_result: Any = "__sentinel_cg_image__",
    raise_on_create: bool = False,
    missing_sentinels: bool = False,
) -> types.ModuleType:
    module = types.ModuleType("Quartz")
    if not missing_sentinels:
        module.CGRectInfinite = ("rect-infinite",)  # type: ignore[attr-defined]
        module.kCGWindowListOptionOnScreenOnly = 1  # type: ignore[attr-defined]
        module.kCGNullWindowID = 0  # type: ignore[attr-defined]
        module.kCGWindowImageDefault = 0  # type: ignore[attr-defined]

    captured: dict[str, Any] = {}

    def _create_image(rect: Any, opts: Any, win_id: Any, img_opt: Any) -> Any:
        captured["args"] = (rect, opts, win_id, img_opt)
        if raise_on_create:
            raise RuntimeError("simulated framework failure")
        return image_result

    module.CGWindowListCreateImage = _create_image  # type: ignore[attr-defined]
    module._captured = captured  # type: ignore[attr-defined]
    return module


def _make_fake_appkit(
    *,
    png_bytes: bytes | None,
    init_fail: bool = False,
) -> types.ModuleType:
    module = types.ModuleType("AppKit")
    # Fresh subclass per module so class-level state doesn't bleed across tests.
    rep_cls: type[_FakeBitmapImageRep] = type(
        "NSBitmapImageRep",
        (_FakeBitmapImageRep,),
        {
            "_next_png": png_bytes,
            "_init_should_fail": init_fail,
            "rep_calls": [],
        },
    )
    module.NSBitmapImageRep = rep_cls  # type: ignore[attr-defined]
    return module


@pytest.fixture
def install_fake_frameworks(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Any]:
    """Factory for installing a configured Quartz+AppKit pair in sys.modules."""

    def _apply(
        *,
        image_result: Any = "__sentinel_cg_image__",
        png_bytes: bytes | None = None,
        raise_on_create: bool = False,
        init_fail: bool = False,
        missing_sentinels: bool = False,
    ) -> tuple[types.ModuleType, types.ModuleType]:
        quartz = _make_fake_quartz(
            image_result=image_result,
            raise_on_create=raise_on_create,
            missing_sentinels=missing_sentinels,
        )
        appkit = _make_fake_appkit(png_bytes=png_bytes, init_fail=init_fail)
        monkeypatch.setitem(sys.modules, "Quartz", quartz)
        monkeypatch.setitem(sys.modules, "AppKit", appkit)
        return quartz, appkit

    yield _apply


# ---------------------------------------------------------------------------
# capture_main_display — happy path and error branches
# ---------------------------------------------------------------------------


def test_capture_returns_png_bytes(install_fake_frameworks: Any) -> None:
    png = _make_min_png(width=8, height=8)
    quartz, appkit = install_fake_frameworks(png_bytes=png)

    result = screenshot.capture_main_display()

    assert result == png
    assert result is not None
    assert result.startswith(screenshot.PNG_MAGIC)
    # Verify the canonical CGWindowListCreateImage args were used.
    args = quartz._captured["args"]
    assert args[0] is quartz.CGRectInfinite
    assert args[1] == quartz.kCGWindowListOptionOnScreenOnly
    assert args[2] == quartz.kCGNullWindowID
    assert args[3] == quartz.kCGWindowImageDefault
    # And that we asked NSBitmapImageRep for the PNG file type (4).
    rep_calls = appkit.NSBitmapImageRep.rep_calls
    assert rep_calls == [(4, None)]


def test_capture_returns_none_when_quartz_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Install a Quartz module that doesn't even have __spec__ — the import
    # itself must fail. We model this via a meta-path hook that raises.
    monkeypatch.setitem(sys.modules, "Quartz", None)  # triggers ImportError
    assert screenshot.capture_main_display() is None


def test_capture_returns_none_when_appkit_missing(
    install_fake_frameworks: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_frameworks(png_bytes=_make_min_png())
    monkeypatch.setitem(sys.modules, "AppKit", None)  # force ImportError
    assert screenshot.capture_main_display() is None


def test_capture_returns_none_when_cg_window_list_returns_none(
    install_fake_frameworks: Any,
) -> None:
    install_fake_frameworks(image_result=None, png_bytes=_make_min_png())
    assert screenshot.capture_main_display() is None


def test_capture_returns_none_when_cg_window_list_raises(
    install_fake_frameworks: Any,
) -> None:
    install_fake_frameworks(raise_on_create=True, png_bytes=_make_min_png())
    assert screenshot.capture_main_display() is None


def test_capture_returns_none_when_sentinels_missing(
    install_fake_frameworks: Any,
) -> None:
    install_fake_frameworks(missing_sentinels=True, png_bytes=_make_min_png())
    assert screenshot.capture_main_display() is None


def test_capture_returns_none_when_bitmap_init_fails(
    install_fake_frameworks: Any,
) -> None:
    install_fake_frameworks(init_fail=True, png_bytes=_make_min_png())
    assert screenshot.capture_main_display() is None


def test_capture_returns_none_when_bitmap_returns_no_data(
    install_fake_frameworks: Any,
) -> None:
    install_fake_frameworks(png_bytes=None)
    assert screenshot.capture_main_display() is None


def test_capture_returns_none_when_bytes_lack_png_magic(
    install_fake_frameworks: Any,
) -> None:
    install_fake_frameworks(png_bytes=b"not-a-png-payload")
    assert screenshot.capture_main_display() is None


# ---------------------------------------------------------------------------
# get_main_display_info — happy path + failure branches
# ---------------------------------------------------------------------------


class _FakeFrame:
    def __init__(self, width: float, height: float) -> None:
        self.size = types.SimpleNamespace(width=width, height=height)


class _FakeScreen:
    def __init__(
        self,
        *,
        width: float = 1440.0,
        height: float = 900.0,
        scale: float = 2.0,
    ) -> None:
        self._frame = _FakeFrame(width, height)
        self._scale = scale

    def frame(self) -> _FakeFrame:
        return self._frame

    def backingScaleFactor(self) -> float:
        return self._scale


def _install_fake_appkit_with_screen(
    monkeypatch: pytest.MonkeyPatch,
    screen: Any,
) -> None:
    module = types.ModuleType("AppKit")

    class NSScreen:
        @staticmethod
        def mainScreen() -> Any:
            return screen

    module.NSScreen = NSScreen  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "AppKit", module)


def test_display_info_returns_points_and_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_appkit_with_screen(
        monkeypatch, _FakeScreen(width=1440.0, height=900.0, scale=2.0)
    )
    info = screenshot.get_main_display_info()
    assert info == {"width": 1440, "height": 900, "scale_factor": 2.0}


def test_display_info_returns_none_when_no_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_appkit_with_screen(monkeypatch, None)
    assert screenshot.get_main_display_info() is None


def test_display_info_returns_none_when_appkit_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "AppKit", None)
    assert screenshot.get_main_display_info() is None


def test_display_info_returns_none_when_frame_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadScreen:
        def frame(self) -> Any:
            raise RuntimeError("boom")

        def backingScaleFactor(self) -> float:
            return 2.0

    _install_fake_appkit_with_screen(monkeypatch, BadScreen())
    assert screenshot.get_main_display_info() is None


def test_display_info_returns_none_on_invalid_dims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_appkit_with_screen(
        monkeypatch, _FakeScreen(width=0.0, height=900.0, scale=2.0)
    )
    assert screenshot.get_main_display_info() is None


def test_display_info_returns_none_on_bad_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_appkit_with_screen(
        monkeypatch, _FakeScreen(width=1440.0, height=900.0, scale=0.0)
    )
    assert screenshot.get_main_display_info() is None


def test_display_info_non_retina_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_appkit_with_screen(
        monkeypatch, _FakeScreen(width=1920.0, height=1080.0, scale=1.0)
    )
    assert screenshot.get_main_display_info() == {
        "width": 1920,
        "height": 1080,
        "scale_factor": 1.0,
    }


# ---------------------------------------------------------------------------
# Opt-in real-macOS integration test
# ---------------------------------------------------------------------------


@pytest.mark.macos
def test_macos_real_capture_produces_valid_png() -> None:
    """Capture the real main display and assert PNG structure + Retina sanity.

    Skipped unless ``TRACE_RUN_MACOS_TESTS=1`` is set and the host has
    granted Screen Recording permission to the pytest process.
    """
    png = screenshot.capture_main_display()
    assert png is not None, "capture_main_display returned None — Screen Recording permission?"
    assert png.startswith(screenshot.PNG_MAGIC)

    # Parse the IHDR chunk to read pixel width/height.
    assert png[12:16] == b"IHDR", "Missing IHDR chunk at canonical offset"
    px_w, px_h = struct.unpack(">II", png[16:24])
    assert px_w > 0 and px_h > 0

    info = screenshot.get_main_display_info()
    assert info is not None
    expected_w = round(info["width"] * info["scale_factor"])
    expected_h = round(info["height"] * info["scale_factor"])
    # Retina sanity: captured pixel dimensions should match display points *
    # scale_factor (within a 1px rounding slack per axis).
    assert abs(px_w - expected_w) <= 1, f"{px_w} vs {expected_w}"
    assert abs(px_h - expected_h) <= 1, f"{px_h} vs {expected_h}"
