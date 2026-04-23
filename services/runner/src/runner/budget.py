"""Per-run budgets and rate limiting (X-014).

A running agent can spiral into unbounded cost or action bursts if something
goes wrong — an infinite click loop, a prompt that elicits ever-growing
completions, or a stall that never returns control. ``BudgetTracker`` is the
third layer of the runner's safety architecture (after the skill's ⚠️ markers
and the harness-layer keyword gate) and caps:

* **Input tokens** and **output tokens** — cumulative over the full run.
* **Wall-clock seconds** — measured with ``time.monotonic`` from the
  tracker's construction.
* **Total actions** — a hard ceiling on tool calls for the entire run.
* **Actions per minute** — a *soft* sliding-window rate limit.

Hard vs. soft
-------------
The token, time, and total-action caps are *hard* limits: when tripped, the
tracker returns :class:`BudgetStatus` with ``kind=BUDGET_EXCEEDED`` and the
executor aborts the run.

The per-minute rate limit is a *soft* limit: a well-behaved agent may burst
during a short click → verify → click sequence. When tripped, ``check()``
returns ``kind=RATE_LIMITED`` with a ``wait_seconds`` hint. The executor is
expected to sleep for that long and then retry, NOT abort.

Testing
-------
``BudgetTracker`` accepts an optional ``time_source`` callable (defaulting to
``time.monotonic``) so tests can inject a fake clock without monkeypatching
the ``time`` module. The sliding window's timestamps use the same source.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Final

DEFAULT_MAX_INPUT_TOKENS: Final[int] = 500_000
DEFAULT_MAX_OUTPUT_TOKENS: Final[int] = 50_000
DEFAULT_MAX_WALL_CLOCK_SECONDS: Final[float] = 600.0
DEFAULT_MAX_ACTIONS_PER_MINUTE: Final[int] = 30
DEFAULT_MAX_TOTAL_ACTIONS: Final[int] = 100

RATE_LIMIT_WINDOW_SECONDS: Final[float] = 60.0


class BudgetStatusKind(StrEnum):
    """Outcome of a :meth:`BudgetTracker.check` call."""

    OK = "ok"
    BUDGET_EXCEEDED = "budget_exceeded"
    RATE_LIMITED = "rate_limited"


class BudgetReason(StrEnum):
    """Which specific hard limit tripped a ``BUDGET_EXCEEDED`` status."""

    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    WALL_CLOCK = "wall_clock"
    ACTIONS = "actions"


@dataclass(frozen=True)
class BudgetStatus:
    """Tracker check result.

    ``kind`` determines which of ``reason`` / ``wait_seconds`` is meaningful:

    * ``OK`` — neither is set.
    * ``BUDGET_EXCEEDED`` — ``reason`` names the tripped limit.
    * ``RATE_LIMITED`` — ``wait_seconds`` is the minimum duration the caller
      should sleep before retrying, always > 0.
    """

    kind: BudgetStatusKind
    reason: BudgetReason | None = None
    wait_seconds: float | None = None

    @classmethod
    def ok(cls) -> BudgetStatus:
        return cls(kind=BudgetStatusKind.OK)

    @classmethod
    def budget_exceeded(cls, reason: BudgetReason) -> BudgetStatus:
        return cls(kind=BudgetStatusKind.BUDGET_EXCEEDED, reason=reason)

    @classmethod
    def rate_limited(cls, wait_seconds: float) -> BudgetStatus:
        return cls(
            kind=BudgetStatusKind.RATE_LIMITED, wait_seconds=wait_seconds
        )


@dataclass(frozen=True)
class RunBudget:
    """Caps applied to a single run.

    All fields default to the module-level ``DEFAULT_*`` constants, which were
    chosen to comfortably exceed the five reference workflows while still
    boxing a runaway loop.
    """

    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_wall_clock_seconds: float = DEFAULT_MAX_WALL_CLOCK_SECONDS
    max_actions_per_minute: int = DEFAULT_MAX_ACTIONS_PER_MINUTE
    max_total_actions: int = DEFAULT_MAX_TOTAL_ACTIONS

    def __post_init__(self) -> None:
        for name in (
            "max_input_tokens",
            "max_output_tokens",
            "max_actions_per_minute",
            "max_total_actions",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    f"RunBudget.{name} must be a positive int, got {value!r}"
                )
        wall = self.max_wall_clock_seconds
        if isinstance(wall, bool) or not isinstance(wall, (int, float)) or wall <= 0:
            raise ValueError(
                "RunBudget.max_wall_clock_seconds must be a positive number, "
                f"got {wall!r}"
            )

    @classmethod
    def from_skill_meta(cls, meta: dict[str, Any]) -> RunBudget:
        """Build a RunBudget from an optional ``runtime_limits`` block.

        Missing fields (or a missing ``runtime_limits`` entirely) fall back to
        the module-level defaults so callers can layer partial overrides.
        """
        overrides = meta.get("runtime_limits")
        if overrides is None:
            return cls()
        if not isinstance(overrides, dict):
            raise TypeError(
                "skill.meta.json runtime_limits must be an object, got "
                f"{type(overrides).__name__}"
            )
        known = {
            "max_input_tokens",
            "max_output_tokens",
            "max_wall_clock_seconds",
            "max_actions_per_minute",
            "max_total_actions",
        }
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(
                f"Unknown runtime_limits fields: {sorted(unknown)}"
            )
        base = cls()
        return replace(base, **{k: overrides[k] for k in overrides})


@dataclass
class BudgetTracker:
    """Accumulates run-level counters and enforces :class:`RunBudget`.

    Not thread-safe; the executor calls ``record_turn`` / ``record_action`` /
    ``check`` from a single async task.
    """

    budget: RunBudget
    time_source: Callable[[], float] = time.monotonic

    input_tokens_used: int = field(init=False, default=0)
    output_tokens_used: int = field(init=False, default=0)
    total_actions: int = field(init=False, default=0)
    _start_monotonic: float = field(init=False)
    _action_timestamps: deque[float] = field(init=False, default_factory=deque)

    def __post_init__(self) -> None:
        self._start_monotonic = self.time_source()

    def record_turn(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate token usage from a completed agent turn."""
        if isinstance(input_tokens, bool) or not isinstance(input_tokens, int):
            raise TypeError("input_tokens must be a non-bool int")
        if isinstance(output_tokens, bool) or not isinstance(output_tokens, int):
            raise TypeError("output_tokens must be a non-bool int")
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        self.input_tokens_used += input_tokens
        self.output_tokens_used += output_tokens

    def record_action(self) -> None:
        """Register that the harness dispatched one tool action.

        Bumps the total and appends the current timestamp to the sliding
        rate-limit window. Expired timestamps (older than 60s) are trimmed
        lazily on ``check``.
        """
        self.total_actions += 1
        self._action_timestamps.append(self.time_source())

    def elapsed_seconds(self) -> float:
        """Seconds since the tracker was constructed."""
        return self.time_source() - self._start_monotonic

    def check(self) -> BudgetStatus:
        """Evaluate all caps and return the first tripping status.

        Order of precedence:

        1. Hard limits — input tokens, output tokens, wall clock, total
           actions. Any trip returns :attr:`BudgetStatusKind.BUDGET_EXCEEDED`.
        2. Soft limit — per-minute rate. Returns
           :attr:`BudgetStatusKind.RATE_LIMITED` with ``wait_seconds``.
        3. Otherwise :attr:`BudgetStatusKind.OK`.
        """
        if self.input_tokens_used >= self.budget.max_input_tokens:
            return BudgetStatus.budget_exceeded(BudgetReason.INPUT_TOKENS)
        if self.output_tokens_used >= self.budget.max_output_tokens:
            return BudgetStatus.budget_exceeded(BudgetReason.OUTPUT_TOKENS)
        if self.elapsed_seconds() >= self.budget.max_wall_clock_seconds:
            return BudgetStatus.budget_exceeded(BudgetReason.WALL_CLOCK)
        if self.total_actions >= self.budget.max_total_actions:
            return BudgetStatus.budget_exceeded(BudgetReason.ACTIONS)

        wait = self._rate_limit_wait()
        if wait is not None:
            return BudgetStatus.rate_limited(wait)

        return BudgetStatus.ok()

    def _rate_limit_wait(self) -> float | None:
        """Return the seconds until the sliding window has room, or None.

        Trims timestamps older than :data:`RATE_LIMIT_WINDOW_SECONDS` so the
        deque size represents actions-in-window. If the in-window count meets
        or exceeds the per-minute cap, the wait is the time until the oldest
        in-window timestamp falls off.
        """
        now = self.time_source()
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS
        while self._action_timestamps and self._action_timestamps[0] <= cutoff:
            self._action_timestamps.popleft()
        if len(self._action_timestamps) < self.budget.max_actions_per_minute:
            return None
        oldest = self._action_timestamps[0]
        wait = (oldest + RATE_LIMIT_WINDOW_SECONDS) - now
        if wait <= 0:
            return None
        return wait


__all__ = [
    "DEFAULT_MAX_ACTIONS_PER_MINUTE",
    "DEFAULT_MAX_INPUT_TOKENS",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_MAX_TOTAL_ACTIONS",
    "DEFAULT_MAX_WALL_CLOCK_SECONDS",
    "RATE_LIMIT_WINDOW_SECONDS",
    "BudgetReason",
    "BudgetStatus",
    "BudgetStatusKind",
    "BudgetTracker",
    "RunBudget",
]
