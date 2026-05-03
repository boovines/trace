"""Input adapters for the runner's computer-use loop.

The runner dispatches Claude's tool calls (click, type, scroll, key, move) into
an ``InputAdapter``. Two implementations ship:

* ``DryRunInputAdapter`` (this module) — records every call into a list and
  does nothing else. Safe regardless of the ``TRACE_ALLOW_LIVE`` flag because
  it cannot drive the real machine. Used by Ralph iterations and every unit
  test.
* ``LiveInputAdapter`` (``runner.live_input``) — posts real mouse and keyboard
  events via CGEventPost. Its ``__init__`` calls ``require_live_mode()`` so
  construction fails loudly when the flag is not set. Re-exported from this
  module for backwards compatibility.

The adapter contract is declared as a ``typing.Protocol`` so real code only
depends on the method shape, not on an inheritance chain. Callers should type
their parameters as ``InputAdapter`` and pass whichever concrete instance
matches the mode.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

MouseButton = Literal["left", "right", "middle"]
ScrollDirection = Literal["up", "down", "left", "right"]


@runtime_checkable
class InputAdapter(Protocol):
    """Protocol every input adapter must implement.

    Coordinates are in **display points** (not pixels) — callers must map from
    Claude's resized-image pixel space to points via
    ``runner.coords.resized_pixels_to_points`` before calling these methods.
    """

    def click(
        self,
        x: float,
        y: float,
        button: MouseButton = "left",
        modifiers: Sequence[str] = (),
    ) -> None: ...

    def type_text(self, text: str) -> None: ...

    def key_press(self, keys: Sequence[str]) -> None: ...

    def scroll(
        self, x: float, y: float, direction: ScrollDirection, amount: int
    ) -> None: ...

    def move_mouse(self, x: float, y: float) -> None: ...


RecordedCall = tuple[str, tuple[object, ...], dict[str, object]]


class DryRunInputAdapter:
    """Records every call without posting any real events.

    The recorded-calls list is a list of ``(method_name, args, kwargs)`` triples
    in call order. Keyword-only arguments are normalised into ``kwargs``; the
    positional tuple matches the ``InputAdapter`` signature so tests can assert
    on positional args directly.

    Safe to use regardless of ``TRACE_ALLOW_LIVE`` — this adapter never touches
    the event system.
    """

    def __init__(self) -> None:
        self._calls: list[RecordedCall] = []

    def click(
        self,
        x: float,
        y: float,
        button: MouseButton = "left",
        modifiers: Sequence[str] = (),
    ) -> None:
        self._calls.append(
            ("click", (x, y, button, tuple(modifiers)), {})
        )

    def type_text(self, text: str) -> None:
        self._calls.append(("type_text", (text,), {}))

    def key_press(self, keys: Sequence[str]) -> None:
        self._calls.append(("key_press", (tuple(keys),), {}))

    def scroll(
        self, x: float, y: float, direction: ScrollDirection, amount: int
    ) -> None:
        self._calls.append(("scroll", (x, y, direction, amount), {}))

    def move_mouse(self, x: float, y: float) -> None:
        self._calls.append(("move_mouse", (x, y), {}))

    def get_recorded_calls(self) -> list[RecordedCall]:
        """Return a copy of the recorded-call log in order.

        Returning a copy prevents a caller from mutating our internal list.
        """
        return list(self._calls)

    def clear(self) -> None:
        """Drop every recorded call. Used to reuse the adapter between steps."""
        self._calls.clear()


__all__ = [
    "DryRunInputAdapter",
    "InputAdapter",
    "MouseButton",
    "RecordedCall",
    "ScrollDirection",
]
