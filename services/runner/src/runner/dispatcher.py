"""Computer tool action dispatcher (X-013).

Translates Claude's ``computer_20250124`` tool_input into concrete
``InputAdapter`` + ``ScreenSource`` calls. The dispatcher is the bridge
between the parser's :class:`~runner.parser.ToolCallAction` and the
machine-action layer.

Design
------
Every side-effecting action that changes screen state (click, type, key,
scroll, wait) is followed by a fresh screenshot so Claude sees the result
before its next turn. ``mouse_move`` is the one exception — moving the
cursor does not affect screen content.

Coordinates in ``tool_input`` are in the **resized image's pixel space**
(what Claude sees). The dispatcher maps them back to **display points**
via ``runner.coords.resized_pixels_to_points`` using the provided
``ImageMapping``.

Error handling
--------------
The dispatcher never raises on a malformed ``tool_input``. Unknown
actions, missing/invalid coordinates, wrong-shape scroll parameters, and
non-string ``text`` fields all produce a ``ToolResult(is_error=True)``
carrying a human-readable message. Adapter or screen-source exceptions
DO propagate — the executor decides how to abort.

Safety
------
``wait`` is capped at :data:`MAX_WAIT_SECONDS` (10s) so an agent cannot
sleep the run into a week-long no-op.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Final, cast

from runner.coords import ImageMapping, resized_pixels_to_points
from runner.input_adapter import (
    InputAdapter,
    MouseButton,
    ScrollDirection,
)
from runner.parser import ToolCallAction
from runner.screen_source import ScreenSource

MAX_WAIT_SECONDS: Final[float] = 10.0

_CLICK_ACTIONS: Final[frozenset[str]] = frozenset(
    {"left_click", "right_click", "double_click", "middle_click"}
)

_CLICK_BUTTON: Final[dict[str, MouseButton]] = {
    "left_click": "left",
    "right_click": "right",
    "middle_click": "middle",
    "double_click": "left",
}

_SCROLL_DIRECTIONS: Final[frozenset[str]] = frozenset(
    {"up", "down", "left", "right"}
)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Outcome of one tool dispatch.

    ``content_blocks`` populate the ``tool_result`` block's ``content``
    array fed back to Claude on the next turn. ``new_image_mapping`` is
    set whenever the dispatcher captured a fresh screenshot — the
    executor should call ``runtime.set_image_mapping(new_image_mapping)``
    before the next turn so subsequent coordinates decode correctly.
    ``is_error`` signals a validation failure before any side effect; the
    executor forwards this to the ``tool_result`` block's ``is_error``.
    """

    content_blocks: list[dict[str, Any]]
    new_image_mapping: ImageMapping | None = None
    is_error: bool = False


def _error_result(message: str) -> ToolResult:
    return ToolResult(
        content_blocks=[{"type": "text", "text": message}],
        new_image_mapping=None,
        is_error=True,
    )


def _screenshot_block(
    screen_source: ScreenSource,
) -> tuple[dict[str, Any], ImageMapping]:
    png_bytes, mapping = screen_source.capture()
    encoded = base64.standard_b64encode(png_bytes).decode("ascii")
    return (
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": encoded,
            },
        },
        mapping,
    )


def _parse_coordinate(
    raw: Any, mapping: ImageMapping
) -> tuple[float, float] | str:
    """Validate a ``[x, y]`` coordinate and return display points.

    Returns a ``(x_pt, y_pt)`` tuple on success, or an error string on
    failure. Booleans are explicitly rejected even though ``bool`` is a
    subclass of ``int`` in Python — a coordinate of ``True`` is never a
    real intent.
    """

    if not isinstance(raw, list | tuple) or len(raw) != 2:
        return "coordinate must be a 2-element [x, y] list"
    x, y = raw
    if isinstance(x, bool) or isinstance(y, bool):
        return "coordinate values must be numeric, not bool"
    if not isinstance(x, int | float) or not isinstance(y, int | float):
        return "coordinate values must be numeric"
    if x < 0 or y < 0:
        return f"coordinate [{x}, {y}] has negative component"
    w, h = mapping.resized_pixels
    if x >= w or y >= h:
        return (
            f"coordinate [{x}, {y}] is out of bounds for resized image "
            f"{w}x{h}"
        )
    x_pt, y_pt = resized_pixels_to_points(float(x), float(y), mapping)
    return (x_pt, y_pt)


def parse_key_expression(text: str) -> list[str]:
    """Split a ``cmd+shift+s`` style expression into ordered components.

    A single named key (``Return``, ``Escape``) returns a one-element
    list. Empty segments are discarded, so ``"+"`` by itself yields
    ``[]`` and the caller treats that as invalid.
    """

    return [segment for segment in text.split("+") if segment]


def _dispatch_click(
    name: str,
    tool_input: dict[str, Any],
    input_adapter: InputAdapter,
    screen_source: ScreenSource,
    image_mapping: ImageMapping,
) -> ToolResult:
    coord = _parse_coordinate(tool_input.get("coordinate"), image_mapping)
    if isinstance(coord, str):
        return _error_result(coord)
    x_pt, y_pt = coord
    modifiers_raw = tool_input.get("text")
    modifiers: list[str] = (
        parse_key_expression(modifiers_raw)
        if isinstance(modifiers_raw, str)
        else []
    )
    button = _CLICK_BUTTON[name]
    clicks = 2 if name == "double_click" else 1
    for _ in range(clicks):
        input_adapter.click(x_pt, y_pt, button=button, modifiers=modifiers)
    block, new_mapping = _screenshot_block(screen_source)
    return ToolResult(content_blocks=[block], new_image_mapping=new_mapping)


def _dispatch_type(
    tool_input: dict[str, Any],
    input_adapter: InputAdapter,
    screen_source: ScreenSource,
) -> ToolResult:
    text = tool_input.get("text")
    if not isinstance(text, str):
        return _error_result("'type' action requires 'text' string")
    input_adapter.type_text(text)
    block, new_mapping = _screenshot_block(screen_source)
    return ToolResult(content_blocks=[block], new_image_mapping=new_mapping)


def _dispatch_key(
    tool_input: dict[str, Any],
    input_adapter: InputAdapter,
    screen_source: ScreenSource,
) -> ToolResult:
    text = tool_input.get("text")
    if not isinstance(text, str) or not text:
        return _error_result("'key' action requires non-empty 'text' string")
    keys = parse_key_expression(text)
    if not keys:
        return _error_result(f"'key' action produced no keys from {text!r}")
    input_adapter.key_press(keys)
    block, new_mapping = _screenshot_block(screen_source)
    return ToolResult(content_blocks=[block], new_image_mapping=new_mapping)


def _dispatch_scroll(
    tool_input: dict[str, Any],
    input_adapter: InputAdapter,
    screen_source: ScreenSource,
    image_mapping: ImageMapping,
) -> ToolResult:
    coord = _parse_coordinate(tool_input.get("coordinate"), image_mapping)
    if isinstance(coord, str):
        return _error_result(coord)
    direction_raw = tool_input.get("scroll_direction")
    if direction_raw not in _SCROLL_DIRECTIONS:
        return _error_result(
            "'scroll' requires scroll_direction in {up, down, left, right}"
        )
    amount_raw = tool_input.get("scroll_amount")
    if isinstance(amount_raw, bool) or not isinstance(amount_raw, int):
        return _error_result(
            "'scroll' requires scroll_amount as a non-negative integer"
        )
    if amount_raw < 0:
        return _error_result("'scroll' scroll_amount must be non-negative")
    direction = cast(ScrollDirection, direction_raw)
    x_pt, y_pt = coord
    input_adapter.scroll(x_pt, y_pt, direction, amount_raw)
    block, new_mapping = _screenshot_block(screen_source)
    return ToolResult(content_blocks=[block], new_image_mapping=new_mapping)


def _dispatch_mouse_move(
    tool_input: dict[str, Any],
    input_adapter: InputAdapter,
    image_mapping: ImageMapping,
) -> ToolResult:
    coord = _parse_coordinate(tool_input.get("coordinate"), image_mapping)
    if isinstance(coord, str):
        return _error_result(coord)
    x_pt, y_pt = coord
    input_adapter.move_mouse(x_pt, y_pt)
    return ToolResult(content_blocks=[], new_image_mapping=None)


def _dispatch_wait(
    tool_input: dict[str, Any],
    screen_source: ScreenSource,
) -> ToolResult:
    duration_raw = tool_input.get("duration")
    if isinstance(duration_raw, bool) or not isinstance(
        duration_raw, int | float
    ):
        return _error_result("'wait' requires numeric duration")
    if duration_raw < 0:
        return _error_result("'wait' duration must be non-negative")
    capped = min(float(duration_raw), MAX_WAIT_SECONDS)
    time.sleep(capped)
    block, new_mapping = _screenshot_block(screen_source)
    return ToolResult(content_blocks=[block], new_image_mapping=new_mapping)


def dispatch_tool_call(
    action: ToolCallAction,
    input_adapter: InputAdapter,
    screen_source: ScreenSource,
    image_mapping: ImageMapping,
) -> ToolResult:
    """Dispatch a single Claude ``computer`` tool call.

    ``image_mapping`` is the mapping produced by the most recent
    screenshot — it determines both the bounds for coordinate validation
    and the resized→points conversion factor. Actions that take a fresh
    screenshot return a ``ToolResult`` whose ``new_image_mapping`` the
    executor must propagate back to the runtime before the next turn.
    """

    tool_input = action.tool_input
    action_name_raw = tool_input.get("action")
    if not isinstance(action_name_raw, str):
        return _error_result(
            "Computer tool_input missing a string 'action' field"
        )
    name = action_name_raw

    if name == "screenshot":
        block, mapping = _screenshot_block(screen_source)
        return ToolResult(
            content_blocks=[block], new_image_mapping=mapping
        )
    if name in _CLICK_ACTIONS:
        return _dispatch_click(
            name, tool_input, input_adapter, screen_source, image_mapping
        )
    if name == "type":
        return _dispatch_type(tool_input, input_adapter, screen_source)
    if name == "key":
        return _dispatch_key(tool_input, input_adapter, screen_source)
    if name == "scroll":
        return _dispatch_scroll(
            tool_input, input_adapter, screen_source, image_mapping
        )
    if name == "mouse_move":
        return _dispatch_mouse_move(
            tool_input, input_adapter, image_mapping
        )
    if name == "wait":
        return _dispatch_wait(tool_input, screen_source)

    return _error_result(f"Unknown computer action: {name}")


__all__ = [
    "MAX_WAIT_SECONDS",
    "ToolResult",
    "dispatch_tool_call",
    "parse_key_expression",
]
