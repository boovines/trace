"""Accessibility (AX) target resolution at screen coordinates.

When the event tap reports a click at ``(x, y)``, the recorder wants more
than the raw coordinate — it wants the *semantic* target under the cursor so
a synthesized skill can later say "click the **Send** button" instead of
"click at (812, 394)". This module owns that resolution.

The public entry point is :func:`resolve_element_at`, which returns a
trajectory-schema-compatible ``target`` dict or ``None``.  It is designed to
be cheap and always bounded:

* A hard **200 ms** wall-clock timeout — slow AX servers (Electron apps
  with deep trees, apps during heavy loading) never block the event loop.
* Every AX framework call is wrapped in ``try/except`` and logged at warning
  level.  AX errors never escape this module.
* When the hit element has no label and no description of its own, we walk
  up to three ancestors looking for one — most Electron-style apps have
  their semantic name only on a parent container (the click lands on a raw
  text node or image inside).

All PyObjC access is done with ``from ApplicationServices import …`` inside
helper functions so tests can stub the framework by installing a
``types.ModuleType`` into ``sys.modules``; no real macOS APIs are touched
during unit testing.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, TypedDict

__all__ = [
    "AX_TIMEOUT_SECONDS",
    "MAX_ANCESTOR_DEPTH",
    "ResolvedTarget",
    "resolve_element_at",
]

logger = logging.getLogger(__name__)

#: Hard wall-clock timeout for a single :func:`resolve_element_at` call.
AX_TIMEOUT_SECONDS: float = 0.2

#: Maximum number of parent hops when walking for a meaningful label.
MAX_ANCESTOR_DEPTH: int = 3

# AX attribute names. Using the string form rather than the PyObjC symbol
# keeps the module testable without importing ApplicationServices at module
# scope — the real framework accepts these plain CFString constants.
_ATTR_ROLE = "AXRole"
_ATTR_TITLE = "AXTitle"
_ATTR_DESCRIPTION = "AXDescription"
_ATTR_IDENTIFIER = "AXIdentifier"
_ATTR_POSITION = "AXPosition"
_ATTR_SIZE = "AXSize"
_ATTR_PARENT = "AXParent"

_AX_ERROR_SUCCESS = 0


class ResolvedTarget(TypedDict):
    """Trajectory-schema compatible ``target`` dict.

    ``frame`` is always present (a target without a frame is reported as
    ``None`` instead); every other field may be ``None`` when the underlying
    AX element does not expose it.
    """

    role: str | None
    label: str | None
    description: str | None
    frame: dict[str, float]
    ax_identifier: str | None


def resolve_element_at(
    x: float,
    y: float,
    *,
    timeout_seconds: float = AX_TIMEOUT_SECONDS,
) -> ResolvedTarget | None:
    """Resolve the semantic target at screen coordinates ``(x, y)``.

    Returns the structured target dict, or ``None`` on any of:

    * the AX framework could not be imported,
    * the system-wide element reports no element at that point,
    * the element has no frame (position + size) information,
    * the overall call exceeds ``timeout_seconds`` seconds.

    This function is thread-safe: the actual AX work runs on a short-lived
    helper thread and the caller blocks on the join with a timeout, so a
    hung AX server can never hang the recorder's own event thread for more
    than ``timeout_seconds``.
    """
    result: list[ResolvedTarget | None] = [None]
    error_box: list[BaseException | None] = [None]
    done = threading.Event()

    def _worker() -> None:
        try:
            result[0] = _resolve_element_at_sync(x, y)
        except BaseException as exc:  # defensive: _resolve_element_at_sync swallows AXError
            error_box[0] = exc
        finally:
            done.set()

    worker = threading.Thread(
        target=_worker, name="recorder-ax-resolver", daemon=True
    )
    worker.start()
    if not done.wait(timeout=timeout_seconds):
        logger.warning(
            "AX resolve_element_at(%s, %s) timed out after %.3fs",
            x,
            y,
            timeout_seconds,
        )
        return None
    if error_box[0] is not None:
        logger.warning("AX resolve_element_at raised: %r", error_box[0])
        return None
    return result[0]


def _resolve_element_at_sync(x: float, y: float) -> ResolvedTarget | None:
    """Synchronous AX hit-test + target construction. May block on AX."""
    try:
        import ApplicationServices as ax
    except ImportError:
        logger.warning("ApplicationServices unavailable; AX resolution disabled")
        return None

    try:
        system_wide = ax.AXUIElementCreateSystemWide()
    except Exception:
        logger.exception("AXUIElementCreateSystemWide raised")
        return None
    if system_wide is None:
        return None

    try:
        err, element = ax.AXUIElementCopyElementAtPosition(
            system_wide, float(x), float(y), None
        )
    except Exception:
        logger.exception("AXUIElementCopyElementAtPosition raised")
        return None

    if err != _AX_ERROR_SUCCESS or element is None:
        return None

    return _build_target(element, ax)


def _build_target(element: Any, ax: Any) -> ResolvedTarget | None:
    """Assemble a :class:`ResolvedTarget` from an AXUIElement.

    Walks up to :data:`MAX_ANCESTOR_DEPTH` parents when the hit element has
    no label or description of its own. The hit element's own ``role``,
    ``ax_identifier`` and ``frame`` are always used — only the missing label
    / description are pulled from an ancestor. This preserves "where the
    click landed" while giving downstream code a usable name.
    """
    role = _get_string_attr(element, _ATTR_ROLE, ax)
    label = _get_string_attr(element, _ATTR_TITLE, ax)
    description = _get_string_attr(element, _ATTR_DESCRIPTION, ax)
    ax_identifier = _get_string_attr(element, _ATTR_IDENTIFIER, ax)
    frame = _get_frame(element, ax)

    if not label and not description:
        ancestor_label, ancestor_description = _walk_for_label(element, ax)
        if ancestor_label:
            label = ancestor_label
        if ancestor_description and not description:
            description = ancestor_description

    if frame is None:
        return None

    return ResolvedTarget(
        role=role or None,
        label=label or None,
        description=description or None,
        frame=frame,
        ax_identifier=ax_identifier or None,
    )


def _walk_for_label(element: Any, ax: Any) -> tuple[str | None, str | None]:
    """Walk up to :data:`MAX_ANCESTOR_DEPTH` parents looking for a label.

    Returns ``(label, description)`` from the first ancestor that has a
    non-empty label. Returns ``(None, None)`` if no ancestor inside the walk
    depth exposes a meaningful label. Description is only surfaced when it
    accompanies a found label; we do not treat a lone description as a
    substitute.
    """
    current = element
    for _ in range(MAX_ANCESTOR_DEPTH):
        parent = _get_object_attr(current, _ATTR_PARENT, ax)
        if parent is None:
            return None, None
        parent_label = _get_string_attr(parent, _ATTR_TITLE, ax)
        parent_description = _get_string_attr(parent, _ATTR_DESCRIPTION, ax)
        if parent_label:
            return parent_label, parent_description or None
        current = parent
    return None, None


def _get_string_attr(element: Any, attr: str, ax: Any) -> str | None:
    """Return a string AX attribute, or ``None`` on missing / error."""
    value = _copy_attribute_value(element, attr, ax)
    if value is None:
        return None
    try:
        text = str(value)
    except Exception:
        logger.debug("str() on AX attr %s raised", attr, exc_info=True)
        return None
    return text or None


def _get_object_attr(element: Any, attr: str, ax: Any) -> Any | None:
    """Return a non-string AX attribute (e.g. parent element), or ``None``."""
    return _copy_attribute_value(element, attr, ax)


def _copy_attribute_value(element: Any, attr: str, ax: Any) -> Any | None:
    """Wrap ``AXUIElementCopyAttributeValue`` into a single safe return.

    PyObjC surfaces the call as ``(error_code, value)`` because the C API
    has an out-parameter. We return ``None`` for every failure mode: the AX
    error codes, a missing value, or the framework raising.
    """
    try:
        err, value = ax.AXUIElementCopyAttributeValue(element, attr, None)
    except Exception:
        logger.debug("AXUIElementCopyAttributeValue(%s) raised", attr, exc_info=True)
        return None
    if err != _AX_ERROR_SUCCESS:
        return None
    return value


def _get_frame(element: Any, ax: Any) -> dict[str, float] | None:
    """Extract ``{x, y, w, h}`` from the element's position + size attrs.

    Returns ``None`` if either attribute is missing or cannot be unpacked
    into numeric values. Accepts three shapes for the unpacked value to
    stay compatible with PyObjC versions that do and do not bridge
    ``AXValueRef`` to Python tuples automatically:

    * a sequence ``(a, b)`` (tuples, lists, ``NSValue`` bridge),
    * an object with ``.x``/``.y`` (CGPoint) or ``.width``/``.height``
      (CGSize) attributes,
    * an ``AXValueRef`` unwrapped via ``AXValueGetValue`` when the module
      exposes that helper and the ``kAXValueCG*Type`` sentinels.
    """
    position = _copy_attribute_value(element, _ATTR_POSITION, ax)
    size = _copy_attribute_value(element, _ATTR_SIZE, ax)
    if position is None or size is None:
        return None

    pos = _unpack_ax_value(position, ("x", "y"), getattr(ax, "kAXValueCGPointType", 1), ax)
    sz = _unpack_ax_value(size, ("width", "height"), getattr(ax, "kAXValueCGSizeType", 2), ax)
    if pos is None or sz is None:
        return None

    px, py = pos
    w, h = sz
    return {"x": float(px), "y": float(py), "w": float(w), "h": float(h)}


def _unpack_ax_value(
    value: Any,
    attr_names: tuple[str, str],
    ax_value_type: int,
    ax: Any,
) -> tuple[float, float] | None:
    """Best-effort conversion of an AX position/size value to a 2-tuple."""
    # Shape 1: already a (a, b) sequence.
    try:
        a, b = value
        return float(a), float(b)
    except Exception:
        pass

    # Shape 2: attribute-bearing record (CGPoint / CGSize bridge).
    try:
        a = getattr(value, attr_names[0])
        b = getattr(value, attr_names[1])
        return float(a), float(b)
    except Exception:
        pass

    # Shape 3: AXValueRef — unwrap via AXValueGetValue if available.
    ax_get_value = getattr(ax, "AXValueGetValue", None)
    if ax_get_value is not None:
        try:
            result = ax_get_value(value, ax_value_type, None)
        except Exception:
            logger.debug("AXValueGetValue raised", exc_info=True)
            return None
        if isinstance(result, tuple) and len(result) >= 2 and result[0]:
            unpacked = result[1]
            try:
                a = getattr(unpacked, attr_names[0])
                b = getattr(unpacked, attr_names[1])
                return float(a), float(b)
            except Exception:
                try:
                    a, b = unpacked
                    return float(a), float(b)
                except Exception:
                    return None
    return None
