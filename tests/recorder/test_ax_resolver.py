"""Tests for ``recorder.ax_resolver``.

These tests are hermetic — the PyObjC ``ApplicationServices`` module is
stubbed via ``sys.modules`` so we never touch the real Accessibility API on
the default run. A single opt-in ``@pytest.mark.macos`` test at the bottom
exercises a real resolution against Finder on a developer Mac; see
``tests/recorder/README.md`` for how to run it.
"""

from __future__ import annotations

import sys
import time
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from recorder import ax_resolver

# ---------------------------------------------------------------------------
# Fake AX element + fake ApplicationServices module
# ---------------------------------------------------------------------------


@dataclass
class FakeElement:
    """In-memory stand-in for an ``AXUIElement`` used by unit tests."""

    role: str | None = None
    title: str | None = None
    description: str | None = None
    identifier: str | None = None
    position: tuple[float, float] | None = (10.0, 20.0)
    size: tuple[float, float] | None = (100.0, 40.0)
    parent: FakeElement | None = None
    # Additional attributes the resolver may not know about.
    extra: dict[str, Any] = field(default_factory=dict)


def _attr_value(elem: FakeElement, attr: str) -> Any:
    """Replicate ``AXUIElementCopyAttributeValue`` on a ``FakeElement``."""
    if attr == "AXRole":
        return elem.role
    if attr == "AXTitle":
        return elem.title
    if attr == "AXDescription":
        return elem.description
    if attr == "AXIdentifier":
        return elem.identifier
    if attr == "AXPosition":
        return elem.position
    if attr == "AXSize":
        return elem.size
    if attr == "AXParent":
        return elem.parent
    return elem.extra.get(attr)


def _make_fake_ax(
    hit_element: FakeElement | None,
    hit_error: int = 0,
) -> types.ModuleType:
    """Build a stub ``ApplicationServices`` module with the hit configured."""
    module = types.ModuleType("ApplicationServices")

    system_wide = object()
    module.AXUIElementCreateSystemWide = lambda: system_wide  # type: ignore[attr-defined]

    def copy_element_at_position(root: Any, x: float, y: float, _out: Any) -> Any:
        assert root is system_wide
        _ = (x, y)
        return (hit_error, hit_element)

    module.AXUIElementCopyElementAtPosition = copy_element_at_position  # type: ignore[attr-defined]

    def copy_attribute_value(elem: Any, attr: str, _out: Any) -> Any:
        if elem is None:
            return (-1, None)
        value = _attr_value(elem, attr)
        if value is None:
            # Emulate AXErrorAttributeUnsupported (-25205) — any non-zero code.
            return (-25205, None)
        return (0, value)

    module.AXUIElementCopyAttributeValue = copy_attribute_value  # type: ignore[attr-defined]
    # Sentinels used by the resolver's value-unpacking fallback path.
    module.kAXValueCGPointType = 1  # type: ignore[attr-defined]
    module.kAXValueCGSizeType = 2  # type: ignore[attr-defined]
    return module


@pytest.fixture
def install_fake_ax(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Factory fixture for installing a configured fake AX module."""

    def _apply(hit: FakeElement | None, hit_error: int = 0) -> types.ModuleType:
        fake = _make_fake_ax(hit, hit_error)
        monkeypatch.setitem(sys.modules, "ApplicationServices", fake)
        return fake

    yield _apply


# ---------------------------------------------------------------------------
# Happy path + basic field mapping
# ---------------------------------------------------------------------------


def test_resolve_element_returns_structured_target(install_fake_ax: Any) -> None:
    element = FakeElement(
        role="AXButton",
        title="Send",
        description="Send message",
        identifier="send-btn",
        position=(100.0, 200.0),
        size=(80.0, 32.0),
    )
    install_fake_ax(element)

    result = ax_resolver.resolve_element_at(120.0, 210.0)

    assert result == {
        "role": "AXButton",
        "label": "Send",
        "description": "Send message",
        "frame": {"x": 100.0, "y": 200.0, "w": 80.0, "h": 32.0},
        "ax_identifier": "send-btn",
    }


def test_resolve_element_returns_none_when_no_hit(install_fake_ax: Any) -> None:
    install_fake_ax(None)
    assert ax_resolver.resolve_element_at(0.0, 0.0) is None


def test_resolve_element_returns_none_on_ax_error(install_fake_ax: Any) -> None:
    # Non-zero error from AXUIElementCopyElementAtPosition — must be treated as miss.
    install_fake_ax(FakeElement(role="AXButton", title="x"), hit_error=-25200)
    assert ax_resolver.resolve_element_at(0.0, 0.0) is None


def test_resolve_element_returns_none_when_no_frame(install_fake_ax: Any) -> None:
    element = FakeElement(role="AXButton", title="Send", position=None, size=None)
    install_fake_ax(element)
    assert ax_resolver.resolve_element_at(0.0, 0.0) is None


def test_resolve_element_returns_none_when_application_services_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = types.ModuleType("ApplicationServices")  # no AX symbols
    monkeypatch.setitem(sys.modules, "ApplicationServices", broken)
    assert ax_resolver.resolve_element_at(0.0, 0.0) is None


def test_resolve_element_missing_optional_fields_become_none(
    install_fake_ax: Any,
) -> None:
    element = FakeElement(role="AXButton", title="Click me")
    install_fake_ax(element)
    result = ax_resolver.resolve_element_at(1.0, 2.0)
    assert result is not None
    assert result["description"] is None
    assert result["ax_identifier"] is None
    assert result["label"] == "Click me"
    assert result["frame"] == {"x": 10.0, "y": 20.0, "w": 100.0, "h": 40.0}


# ---------------------------------------------------------------------------
# Ancestor walk
# ---------------------------------------------------------------------------


def test_ancestor_walk_promotes_parent_label_when_hit_has_none(
    install_fake_ax: Any,
) -> None:
    # Electron-style: raw span with no label, container with the real name.
    root = FakeElement(role="AXGroup", title="Send Button", identifier="send-btn-root")
    hit = FakeElement(
        role="AXStaticText",
        title=None,
        description=None,
        position=(0.0, 0.0),
        size=(10.0, 10.0),
        parent=root,
    )
    install_fake_ax(hit)

    result = ax_resolver.resolve_element_at(5.0, 5.0)

    assert result is not None
    # Label came from the ancestor, but role/identifier/frame came from the hit.
    assert result["label"] == "Send Button"
    assert result["role"] == "AXStaticText"
    assert result["ax_identifier"] is None  # hit's own identifier was empty
    assert result["frame"] == {"x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0}


def test_ancestor_walk_stops_at_max_depth(install_fake_ax: Any) -> None:
    # Build a 5-level chain with the labelled parent at depth 4 (just past the limit).
    labelled = FakeElement(role="AXGroup", title="Deep Label")
    level3 = FakeElement(role="AXGroup", parent=labelled)
    level2 = FakeElement(role="AXGroup", parent=level3)
    level1 = FakeElement(role="AXGroup", parent=level2)
    hit = FakeElement(
        role="AXStaticText",
        position=(0.0, 0.0),
        size=(1.0, 1.0),
        parent=level1,
    )
    install_fake_ax(hit)

    result = ax_resolver.resolve_element_at(0.0, 0.0)

    assert result is not None
    assert result["label"] is None  # MAX_ANCESTOR_DEPTH=3 exhausted before reaching labelled


def test_ancestor_walk_finds_label_exactly_at_depth_limit(
    install_fake_ax: Any,
) -> None:
    labelled = FakeElement(role="AXWindow", title="Compose")
    level2 = FakeElement(role="AXGroup", parent=labelled)
    level1 = FakeElement(role="AXGroup", parent=level2)
    hit = FakeElement(
        role="AXStaticText",
        position=(0.0, 0.0),
        size=(1.0, 1.0),
        parent=level1,
    )
    install_fake_ax(hit)

    result = ax_resolver.resolve_element_at(0.0, 0.0)

    assert result is not None
    assert result["label"] == "Compose"


def test_ancestor_walk_skipped_when_hit_has_own_label(install_fake_ax: Any) -> None:
    parent = FakeElement(role="AXGroup", title="Parent Label")
    hit = FakeElement(role="AXButton", title="Hit Label", parent=parent)
    install_fake_ax(hit)

    result = ax_resolver.resolve_element_at(0.0, 0.0)

    assert result is not None
    assert result["label"] == "Hit Label"


def test_ancestor_walk_skipped_when_hit_has_description_only(
    install_fake_ax: Any,
) -> None:
    parent = FakeElement(role="AXGroup", title="Parent Label")
    hit = FakeElement(role="AXButton", description="only description", parent=parent)
    install_fake_ax(hit)

    result = ax_resolver.resolve_element_at(0.0, 0.0)

    assert result is not None
    # Description alone is enough to not walk — label stays None, description preserved.
    assert result["description"] == "only description"
    assert result["label"] is None


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def test_resolve_respects_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_seconds = 0.5
    module = types.ModuleType("ApplicationServices")

    def slow_system_wide() -> Any:
        time.sleep(sleep_seconds)
        return object()

    module.AXUIElementCreateSystemWide = slow_system_wide  # type: ignore[attr-defined]
    module.AXUIElementCopyElementAtPosition = (  # type: ignore[attr-defined]
        lambda *a, **k: (0, None)
    )
    module.AXUIElementCopyAttributeValue = (  # type: ignore[attr-defined]
        lambda *a, **k: (-1, None)
    )
    monkeypatch.setitem(sys.modules, "ApplicationServices", module)

    start = time.monotonic()
    result = ax_resolver.resolve_element_at(0.0, 0.0, timeout_seconds=0.05)
    elapsed = time.monotonic() - start

    assert result is None
    # Must return well before the slow framework call finishes.
    assert elapsed < sleep_seconds, (
        f"resolve_element_at did not honour its timeout (took {elapsed:.3f}s)"
    )


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


def test_framework_raises_are_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("ApplicationServices")

    def boom_system_wide() -> Any:
        raise RuntimeError("AX died")

    module.AXUIElementCreateSystemWide = boom_system_wide  # type: ignore[attr-defined]
    module.AXUIElementCopyElementAtPosition = (  # type: ignore[attr-defined]
        lambda *a, **k: (0, None)
    )
    module.AXUIElementCopyAttributeValue = (  # type: ignore[attr-defined]
        lambda *a, **k: (-1, None)
    )
    monkeypatch.setitem(sys.modules, "ApplicationServices", module)

    assert ax_resolver.resolve_element_at(0.0, 0.0) is None


def test_attribute_copy_raising_is_swallowed(install_fake_ax: Any) -> None:
    fake = install_fake_ax(FakeElement(role="AXButton", title="x"))

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("attr blew up")

    fake.AXUIElementCopyAttributeValue = boom  # type: ignore[attr-defined]

    # A failing attribute copy leaves every field unknown → no frame → None.
    assert ax_resolver.resolve_element_at(0.0, 0.0) is None


# ---------------------------------------------------------------------------
# Frame unpacking — cover the three value shapes the resolver tolerates
# ---------------------------------------------------------------------------


class _PointObject:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class _SizeObject:
    def __init__(self, w: float, h: float) -> None:
        self.width = w
        self.height = h


def test_frame_accepts_attribute_bearing_objects(install_fake_ax: Any) -> None:
    element = FakeElement(role="AXButton", title="hi")
    element.position = _PointObject(1.5, 2.5)  # type: ignore[assignment]
    element.size = _SizeObject(3.0, 4.0)  # type: ignore[assignment]
    install_fake_ax(element)

    result = ax_resolver.resolve_element_at(0.0, 0.0)

    assert result is not None
    assert result["frame"] == {"x": 1.5, "y": 2.5, "w": 3.0, "h": 4.0}


def test_frame_accepts_ax_value_ref_via_ax_value_get_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # AXValueRef style — the position/size come out as opaque sentinels and
    # must be unwrapped via AXValueGetValue (success, unpacked) tuple.
    position_ref = object()
    size_ref = object()

    element = FakeElement(
        role="AXButton",
        title="hi",
    )
    element.position = position_ref  # type: ignore[assignment]
    element.size = size_ref  # type: ignore[assignment]

    module = _make_fake_ax(element)

    def ax_value_get_value(value: Any, type_id: int, _out: Any) -> Any:
        if value is position_ref and type_id == module.kAXValueCGPointType:  # type: ignore[attr-defined]
            return (True, _PointObject(5.0, 6.0))
        if value is size_ref and type_id == module.kAXValueCGSizeType:  # type: ignore[attr-defined]
            return (True, _SizeObject(7.0, 8.0))
        return (False, None)

    module.AXValueGetValue = ax_value_get_value  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ApplicationServices", module)

    result = ax_resolver.resolve_element_at(0.0, 0.0)

    assert result is not None
    assert result["frame"] == {"x": 5.0, "y": 6.0, "w": 7.0, "h": 8.0}


# ---------------------------------------------------------------------------
# macOS integration (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.macos
def test_macos_real_resolve_hits_finder_menu_bar() -> None:
    """Resolve the macOS menu bar top-left (Apple menu) on a real Mac.

    Requires Accessibility permission. The top-left corner of the main
    display is almost always the Apple menu regardless of frontmost app, so
    this is the most stable real-framework assertion we can make without
    scripting an app into a specific state.
    """
    result = ax_resolver.resolve_element_at(2.0, 2.0)
    assert result is not None, "expected an AX element at screen (2, 2)"
    assert result["role"] is not None
    assert result["frame"]["w"] >= 1.0
    assert result["frame"]["h"] >= 1.0
