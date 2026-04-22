"""Input adapters for the runner's computer-use loop.

The runner dispatches Claude's tool calls (click, type, scroll, key, move) into
an ``InputAdapter``. Two implementations live in this module:

* ``DryRunInputAdapter`` — records every call into a list and does nothing
  else. This is the adapter Ralph iterations and the unit tests use; it is
  safe regardless of the ``TRACE_ALLOW_LIVE`` flag because it cannot drive the
  real machine.
* ``LiveInputAdapter`` — the live-mode adapter. Its ``__init__`` calls
  ``require_live_mode()`` so construction fails loudly when the flag is not
  set. The actual CGEventPost plumbing lands in X-008 (``runner.live_input``);
  this module holds the constructor gate so X-007's integration tests can
  already assert on the safety behavior.

The adapter contract is declared as a ``typing.Protocol`` so real code only
depends on the method shape, not on an inheritance chain. Callers should type
their parameters as ``InputAdapter`` and pass whichever concrete instance
matches the mode.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from runner.safety import require_live_mode

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


class LiveInputAdapter:
    """Placeholder for the X-008 live adapter.

    The full implementation (CGEventPost plumbing, Retina coordinate mapping,
    modifier flags, Unicode text input, rate limiting) is authored on the
    X-008 story in ``runner.live_input``. This class exists in X-007 so the
    safety gate is wired up and testable today: constructing a ``LiveInputAdapter``
    without ``TRACE_ALLOW_LIVE=1`` raises ``LiveModeNotAllowed`` before any
    event-system code can run.
    """

    def __init__(self) -> None:
        require_live_mode()
        raise NotImplementedError(
            "LiveInputAdapter is implemented in X-008 (runner.live_input). "
            "Only the TRACE_ALLOW_LIVE gate is wired up in X-007."
        )

    def click(
        self,
        x: float,
        y: float,
        button: MouseButton = "left",
        modifiers: Sequence[str] = (),
    ) -> None:
        raise NotImplementedError

    def type_text(self, text: str) -> None:
        raise NotImplementedError

    def key_press(self, keys: Sequence[str]) -> None:
        raise NotImplementedError

    def scroll(
        self, x: float, y: float, direction: ScrollDirection, amount: int
    ) -> None:
        raise NotImplementedError

    def move_mouse(self, x: float, y: float) -> None:
        raise NotImplementedError
