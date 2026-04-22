"""Live macOS input adapter backed by CGEventPost.

This is the ONLY module in the runner that drives real mouse and keyboard
input on the user's machine. Instantiating ``LiveInputAdapter`` calls
``require_live_mode()`` first — construction fails loudly with
``LiveModeNotAllowed`` unless ``TRACE_ALLOW_LIVE=1`` is set in the
environment. Ralph iterations and automated tests must NOT set that flag.

Coordinates are accepted in macOS **display points** (the same units the
event system natively uses). Callers that are working from a Claude-resized
screenshot should go through ``runner.coords.resized_pixels_to_points``
first.

All CGEventPost calls are wrapped so any PyObjC-level error surfaces as a
``LiveInputError`` with an ``<op> failed: <reason>`` message rather than a
cryptic framework traceback. A small inter-event delay (20 ms by default)
keeps the target application from dropping events under burst load.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Final

from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventCreateMouseEvent,
    CGEventCreateScrollWheelEvent,
    CGEventKeyboardSetUnicodeString,
    CGEventPost,
    CGEventSetFlags,
    kCGEventFlagMaskAlternate,
    kCGEventFlagMaskCommand,
    kCGEventFlagMaskControl,
    kCGEventFlagMaskShift,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGEventMouseMoved,
    kCGEventOtherMouseDown,
    kCGEventOtherMouseUp,
    kCGEventRightMouseDown,
    kCGEventRightMouseUp,
    kCGHIDEventTap,
    kCGMouseButtonCenter,
    kCGMouseButtonLeft,
    kCGMouseButtonRight,
    kCGScrollEventUnitPixel,
)

from runner.coords import DisplayInfo, get_main_display_info
from runner.input_adapter import MouseButton, ScrollDirection
from runner.safety import require_live_mode

logger = logging.getLogger(__name__)


DEFAULT_EVENT_DELAY_SECONDS: Final[float] = 0.020


class LiveInputError(RuntimeError):
    """Raised when a CGEventPost call fails or returns an unexpected result."""


_MODIFIER_FLAGS: Final[dict[str, int]] = {
    "cmd": int(kCGEventFlagMaskCommand),
    "command": int(kCGEventFlagMaskCommand),
    "shift": int(kCGEventFlagMaskShift),
    "option": int(kCGEventFlagMaskAlternate),
    "alt": int(kCGEventFlagMaskAlternate),
    "control": int(kCGEventFlagMaskControl),
    "ctrl": int(kCGEventFlagMaskControl),
}


_VIRTUAL_KEYCODES: Final[dict[str, int]] = {
    # US QWERTY letters
    "a": 0x00, "b": 0x0B, "c": 0x08, "d": 0x02, "e": 0x0E, "f": 0x03,
    "g": 0x05, "h": 0x04, "i": 0x22, "j": 0x26, "k": 0x28, "l": 0x25,
    "m": 0x2E, "n": 0x2D, "o": 0x1F, "p": 0x23, "q": 0x0C, "r": 0x0F,
    "s": 0x01, "t": 0x11, "u": 0x20, "v": 0x09, "w": 0x0D, "x": 0x07,
    "y": 0x10, "z": 0x06,
    # Digits
    "0": 0x1D, "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "5": 0x17,
    "6": 0x16, "7": 0x1A, "8": 0x1C, "9": 0x19,
    # Symbols
    "-": 0x1B, "=": 0x18, "[": 0x21, "]": 0x1E, "\\": 0x2A, ";": 0x29,
    "'": 0x27, ",": 0x2B, ".": 0x2F, "/": 0x2C, "`": 0x32,
    # Named keys
    "return": 0x24, "enter": 0x24, "tab": 0x30, "space": 0x31,
    "delete": 0x33, "backspace": 0x33, "escape": 0x35, "esc": 0x35,
    "right": 0x7C, "left": 0x7B, "down": 0x7D, "up": 0x7E,
    # Modifier keys as standalone keycodes (kVK_Command et al.)
    "cmd": 0x37, "command": 0x37,
    "shift": 0x38,
    "capslock": 0x39,
    "option": 0x3A, "alt": 0x3A,
    "control": 0x3B, "ctrl": 0x3B,
    "fn": 0x3F,
}


_MOUSE_BUTTON_DOWN: Final[dict[MouseButton, int]] = {
    "left": int(kCGEventLeftMouseDown),
    "right": int(kCGEventRightMouseDown),
    "middle": int(kCGEventOtherMouseDown),
}
_MOUSE_BUTTON_UP: Final[dict[MouseButton, int]] = {
    "left": int(kCGEventLeftMouseUp),
    "right": int(kCGEventRightMouseUp),
    "middle": int(kCGEventOtherMouseUp),
}
_MOUSE_BUTTON_CODE: Final[dict[MouseButton, int]] = {
    "left": int(kCGMouseButtonLeft),
    "right": int(kCGMouseButtonRight),
    "middle": int(kCGMouseButtonCenter),
}


def _flags_for(modifiers: Sequence[str]) -> int:
    mask = 0
    for name in modifiers:
        key = name.lower()
        if key not in _MODIFIER_FLAGS:
            raise LiveInputError(f"unknown modifier: {name!r}")
        mask |= _MODIFIER_FLAGS[key]
    return mask


def _keycode_for(key: str) -> int:
    code = _VIRTUAL_KEYCODES.get(key.lower())
    if code is None:
        raise LiveInputError(f"unknown key: {key!r}")
    return code


class LiveInputAdapter:
    """Post real mouse and keyboard events via CGEventPost.

    Constructing this adapter requires ``TRACE_ALLOW_LIVE=1``; the
    ``require_live_mode()`` call runs first and raises
    ``LiveModeNotAllowed`` otherwise. The adapter also captures a
    ``DisplayInfo`` snapshot at construction time so the scale factor is
    stable across the run.
    """

    def __init__(
        self,
        *,
        display_info: DisplayInfo | None = None,
        event_delay_seconds: float = DEFAULT_EVENT_DELAY_SECONDS,
    ) -> None:
        require_live_mode()
        self._display_info = display_info or get_main_display_info()
        self._event_delay_seconds = event_delay_seconds

    @property
    def display_info(self) -> DisplayInfo:
        return self._display_info

    # ------------------------------------------------------------------ #
    # InputAdapter protocol
    # ------------------------------------------------------------------ #

    def click(
        self,
        x: float,
        y: float,
        button: MouseButton = "left",
        modifiers: Sequence[str] = (),
    ) -> None:
        if button not in _MOUSE_BUTTON_DOWN:
            raise LiveInputError(f"unknown mouse button: {button!r}")
        flags = _flags_for(modifiers)
        position = (float(x), float(y))
        button_code = _MOUSE_BUTTON_CODE[button]

        down = self._create_mouse_event(
            _MOUSE_BUTTON_DOWN[button], position, button_code, operation="click"
        )
        if flags:
            CGEventSetFlags(down, flags)
        self._post(down, operation="click")

        up = self._create_mouse_event(
            _MOUSE_BUTTON_UP[button], position, button_code, operation="click"
        )
        if flags:
            CGEventSetFlags(up, flags)
        self._post(up, operation="click")

    def type_text(self, text: str) -> None:
        if not text:
            return
        down = self._create_keyboard_event(0, key_down=True, operation="type_text")
        CGEventKeyboardSetUnicodeString(down, len(text), text)
        self._post(down, operation="type_text")

        up = self._create_keyboard_event(0, key_down=False, operation="type_text")
        CGEventKeyboardSetUnicodeString(up, len(text), text)
        self._post(up, operation="type_text")

    def key_press(self, keys: Sequence[str]) -> None:
        if not keys:
            return
        codes = [_keycode_for(k) for k in keys]
        for code in codes:
            event = self._create_keyboard_event(
                code, key_down=True, operation="key_press"
            )
            self._post(event, operation="key_press")
        for code in reversed(codes):
            event = self._create_keyboard_event(
                code, key_down=False, operation="key_press"
            )
            self._post(event, operation="key_press")

    def scroll(
        self, x: float, y: float, direction: ScrollDirection, amount: int
    ) -> None:
        if amount < 0:
            raise LiveInputError(f"scroll amount must be non-negative, got {amount}")
        self.move_mouse(x, y)

        wheels: tuple[int, ...]
        if direction in ("up", "down"):
            delta_y = amount if direction == "up" else -amount
            wheel_count = 1
            wheels = (delta_y,)
        else:
            delta_x = amount if direction == "right" else -amount
            wheel_count = 2
            wheels = (0, delta_x)

        try:
            event = CGEventCreateScrollWheelEvent(
                None, int(kCGScrollEventUnitPixel), wheel_count, *wheels
            )
        except Exception as exc:
            raise LiveInputError(f"scroll failed: {exc!r}") from exc
        if event is None:
            raise LiveInputError("scroll failed: CGEventCreateScrollWheelEvent returned None")
        self._post(event, operation="scroll")

    def move_mouse(self, x: float, y: float) -> None:
        position = (float(x), float(y))
        event = self._create_mouse_event(
            int(kCGEventMouseMoved), position, int(kCGMouseButtonLeft),
            operation="move_mouse",
        )
        self._post(event, operation="move_mouse")

    # ------------------------------------------------------------------ #
    # Internal plumbing — factored so tests can mock one seam at a time
    # ------------------------------------------------------------------ #

    def _create_mouse_event(
        self,
        event_type: int,
        position: tuple[float, float],
        button_code: int,
        *,
        operation: str,
    ) -> object:
        try:
            event = CGEventCreateMouseEvent(None, event_type, position, button_code)
        except Exception as exc:
            raise LiveInputError(f"{operation} failed: {exc!r}") from exc
        if event is None:
            raise LiveInputError(
                f"{operation} failed: CGEventCreateMouseEvent returned None"
            )
        return event

    def _create_keyboard_event(
        self, keycode: int, *, key_down: bool, operation: str
    ) -> object:
        try:
            event = CGEventCreateKeyboardEvent(None, keycode, key_down)
        except Exception as exc:
            raise LiveInputError(f"{operation} failed: {exc!r}") from exc
        if event is None:
            raise LiveInputError(
                f"{operation} failed: CGEventCreateKeyboardEvent returned None"
            )
        return event

    def _post(self, event: object, *, operation: str) -> None:
        try:
            CGEventPost(kCGHIDEventTap, event)
        except Exception as exc:
            raise LiveInputError(f"{operation} failed: {exc!r}") from exc
        if self._event_delay_seconds > 0:
            time.sleep(self._event_delay_seconds)
        self._check_event_tap_state(operation)

    def _check_event_tap_state(self, operation: str) -> None:
        """Best-effort probe for event-tap throttling.

        We do not own a ``CFMachPort`` tap so ``CGEventTapIsEnabled`` is not
        directly queryable from the sender side; this method is a seam that
        tests can patch to verify the call path runs after every post, and
        that future work (loopback tap) can fill in. A subclass may override
        it to emit ``logger.warning(...)`` when throttling is detected.
        """
        return None
