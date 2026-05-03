"""Tests for runner.budget (X-014).

Every BudgetTracker is constructed with an explicit ``time_source`` so we can
drive deterministic timelines without monkeypatching the ``time`` module.
"""

from __future__ import annotations

import pytest

from runner.budget import (
    DEFAULT_MAX_ACTIONS_PER_MINUTE,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_TOTAL_ACTIONS,
    DEFAULT_MAX_WALL_CLOCK_SECONDS,
    RATE_LIMIT_WINDOW_SECONDS,
    BudgetReason,
    BudgetStatus,
    BudgetStatusKind,
    BudgetTracker,
    RunBudget,
)


class FakeClock:
    """Deterministic monotonic clock used as ``time_source``."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _tracker(
    budget: RunBudget | None = None, start: float = 1000.0
) -> tuple[BudgetTracker, FakeClock]:
    clock = FakeClock(start)
    return BudgetTracker(budget=budget or RunBudget(), time_source=clock), clock


def test_default_budget_matches_spec() -> None:
    budget = RunBudget()
    assert budget.max_input_tokens == DEFAULT_MAX_INPUT_TOKENS == 500_000
    assert budget.max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS == 50_000
    assert budget.max_wall_clock_seconds == DEFAULT_MAX_WALL_CLOCK_SECONDS == 600.0
    assert budget.max_actions_per_minute == DEFAULT_MAX_ACTIONS_PER_MINUTE == 30
    assert budget.max_total_actions == DEFAULT_MAX_TOTAL_ACTIONS == 100


def test_fresh_tracker_returns_ok() -> None:
    tracker, _ = _tracker()
    assert tracker.check() == BudgetStatus.ok()
    assert tracker.input_tokens_used == 0
    assert tracker.output_tokens_used == 0
    assert tracker.total_actions == 0


@pytest.mark.parametrize(
    "kind",
    [BudgetStatusKind.OK, BudgetStatusKind.BUDGET_EXCEEDED, BudgetStatusKind.RATE_LIMITED],
)
def test_budget_status_kinds_are_strings(kind: BudgetStatusKind) -> None:
    # BudgetStatusKind inherits from (str, Enum) so equality works both ways.
    assert kind == kind.value


def test_record_turn_accumulates_tokens() -> None:
    tracker, _ = _tracker()
    tracker.record_turn(100, 50)
    tracker.record_turn(200, 75)
    assert tracker.input_tokens_used == 300
    assert tracker.output_tokens_used == 125


def test_input_token_limit_triggers_budget_exceeded() -> None:
    budget = RunBudget(max_input_tokens=1000)
    tracker, _ = _tracker(budget)
    tracker.record_turn(400, 0)
    tracker.record_turn(500, 0)
    assert tracker.check() == BudgetStatus.ok()
    tracker.record_turn(100, 0)
    status = tracker.check()
    assert status.kind is BudgetStatusKind.BUDGET_EXCEEDED
    assert status.reason is BudgetReason.INPUT_TOKENS


def test_output_token_limit_triggers_budget_exceeded() -> None:
    budget = RunBudget(max_output_tokens=100)
    tracker, _ = _tracker(budget)
    tracker.record_turn(0, 50)
    tracker.record_turn(0, 49)
    assert tracker.check() == BudgetStatus.ok()
    tracker.record_turn(0, 1)
    status = tracker.check()
    assert status.kind is BudgetStatusKind.BUDGET_EXCEEDED
    assert status.reason is BudgetReason.OUTPUT_TOKENS


def test_input_tokens_trip_before_output_tokens() -> None:
    budget = RunBudget(max_input_tokens=100, max_output_tokens=100)
    tracker, _ = _tracker(budget)
    tracker.record_turn(100, 100)
    status = tracker.check()
    assert status.reason is BudgetReason.INPUT_TOKENS


def test_wall_clock_limit_triggers_budget_exceeded() -> None:
    budget = RunBudget(max_wall_clock_seconds=30.0)
    tracker, clock = _tracker(budget)
    clock.advance(29.9)
    assert tracker.check() == BudgetStatus.ok()
    clock.advance(0.2)  # crosses 30s threshold
    status = tracker.check()
    assert status.kind is BudgetStatusKind.BUDGET_EXCEEDED
    assert status.reason is BudgetReason.WALL_CLOCK


def test_wall_clock_exact_boundary_trips() -> None:
    budget = RunBudget(max_wall_clock_seconds=30.0)
    tracker, clock = _tracker(budget)
    clock.advance(30.0)  # >= boundary
    status = tracker.check()
    assert status.kind is BudgetStatusKind.BUDGET_EXCEEDED
    assert status.reason is BudgetReason.WALL_CLOCK


def test_total_actions_limit_triggers_budget_exceeded() -> None:
    budget = RunBudget(max_total_actions=5, max_actions_per_minute=1000)
    tracker, clock = _tracker(budget)
    for _ in range(4):
        tracker.record_action()
        clock.advance(0.001)
    assert tracker.check() == BudgetStatus.ok()
    tracker.record_action()
    status = tracker.check()
    assert status.kind is BudgetStatusKind.BUDGET_EXCEEDED
    assert status.reason is BudgetReason.ACTIONS


def test_rate_limit_soft_with_wait_seconds() -> None:
    # 30 actions in quick succession — the 30th check should rate-limit.
    budget = RunBudget(
        max_total_actions=10_000, max_actions_per_minute=30
    )
    tracker, clock = _tracker(budget)
    # Record 29 actions across the first second.
    for _ in range(29):
        tracker.record_action()
        clock.advance(1.0 / 29.0)
    assert tracker.check() == BudgetStatus.ok()
    # 30th action still fits the cap (we hit the cap at 30).
    tracker.record_action()
    status = tracker.check()
    assert status.kind is BudgetStatusKind.RATE_LIMITED
    assert status.wait_seconds is not None
    assert status.wait_seconds > 0
    # Oldest action was at t=0 (relative), now is ~1s — wait is ~59s.
    assert 58.0 < status.wait_seconds <= 60.0


def test_rate_limit_wait_shrinks_as_time_passes() -> None:
    budget = RunBudget(max_actions_per_minute=3, max_total_actions=1000)
    tracker, clock = _tracker(budget)
    tracker.record_action()  # t=0 (relative)
    clock.advance(1.0)
    tracker.record_action()  # t=1
    clock.advance(1.0)
    tracker.record_action()  # t=2
    status = tracker.check()
    assert status.kind is BudgetStatusKind.RATE_LIMITED
    assert status.wait_seconds is not None
    assert status.wait_seconds == pytest.approx(58.0)  # 0 + 60 - 2

    clock.advance(30.0)
    status = tracker.check()
    assert status.kind is BudgetStatusKind.RATE_LIMITED
    assert status.wait_seconds == pytest.approx(28.0)  # 0 + 60 - 32


def test_rate_limit_sliding_window_releases_old_actions() -> None:
    """29 actions in one second, wait 61s, 1 more action → OK."""
    budget = RunBudget(max_actions_per_minute=30, max_total_actions=1000)
    tracker, clock = _tracker(budget)
    for _ in range(29):
        tracker.record_action()
        clock.advance(1.0 / 29.0)
    assert tracker.check() == BudgetStatus.ok()
    clock.advance(61.0)
    tracker.record_action()  # only action still in-window
    status = tracker.check()
    assert status == BudgetStatus.ok()


def test_rate_limit_window_edge_61_seconds_later() -> None:
    """An action exactly 61s after a full-window burst clears the window."""
    budget = RunBudget(max_actions_per_minute=2, max_total_actions=1000)
    tracker, clock = _tracker(budget)
    tracker.record_action()
    tracker.record_action()
    # At this moment we're at cap; a 3rd would rate-limit.
    clock.advance(61.0)
    tracker.record_action()
    assert tracker.check() == BudgetStatus.ok()


def test_hard_limits_take_precedence_over_rate_limit() -> None:
    budget = RunBudget(max_total_actions=5, max_actions_per_minute=2)
    tracker, clock = _tracker(budget)
    for _ in range(5):
        tracker.record_action()
        clock.advance(0.001)
    status = tracker.check()
    assert status.kind is BudgetStatusKind.BUDGET_EXCEEDED
    assert status.reason is BudgetReason.ACTIONS


def test_custom_budget_from_skill_meta_runtime_limits() -> None:
    meta = {
        "runtime_limits": {
            "max_input_tokens": 1000,
            "max_output_tokens": 500,
            "max_wall_clock_seconds": 120,
            "max_actions_per_minute": 10,
            "max_total_actions": 20,
        }
    }
    budget = RunBudget.from_skill_meta(meta)
    assert budget.max_input_tokens == 1000
    assert budget.max_output_tokens == 500
    assert budget.max_wall_clock_seconds == 120
    assert budget.max_actions_per_minute == 10
    assert budget.max_total_actions == 20


def test_partial_runtime_limits_overrides_fall_back_to_defaults() -> None:
    budget = RunBudget.from_skill_meta({"runtime_limits": {"max_total_actions": 7}})
    assert budget.max_total_actions == 7
    assert budget.max_input_tokens == DEFAULT_MAX_INPUT_TOKENS
    assert budget.max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS
    assert budget.max_wall_clock_seconds == DEFAULT_MAX_WALL_CLOCK_SECONDS
    assert budget.max_actions_per_minute == DEFAULT_MAX_ACTIONS_PER_MINUTE


def test_missing_runtime_limits_yields_default_budget() -> None:
    assert RunBudget.from_skill_meta({}) == RunBudget()
    assert RunBudget.from_skill_meta({"runtime_limits": None}) == RunBudget()


def test_runtime_limits_rejects_non_object() -> None:
    with pytest.raises(TypeError, match="runtime_limits"):
        RunBudget.from_skill_meta({"runtime_limits": [1, 2, 3]})


def test_runtime_limits_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unknown runtime_limits fields"):
        RunBudget.from_skill_meta(
            {"runtime_limits": {"max_input_tokens": 10, "max_foo": 1}}
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_input_tokens", 0),
        ("max_input_tokens", -1),
        ("max_output_tokens", 0),
        ("max_actions_per_minute", 0),
        ("max_total_actions", 0),
        ("max_wall_clock_seconds", 0),
        ("max_wall_clock_seconds", -1.5),
    ],
)
def test_run_budget_rejects_non_positive(field: str, value: int | float) -> None:
    with pytest.raises(ValueError, match=field):
        RunBudget(**{field: value})  # type: ignore[arg-type]


def test_run_budget_rejects_bool_as_int() -> None:
    # bool subclasses int; guard explicitly.
    with pytest.raises(ValueError):
        RunBudget(max_input_tokens=True)  # type: ignore[arg-type]


def test_record_turn_rejects_bool_and_negative() -> None:
    tracker, _ = _tracker()
    with pytest.raises(TypeError):
        tracker.record_turn(True, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tracker.record_turn(0, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tracker.record_turn(-1, 0)
    with pytest.raises(ValueError):
        tracker.record_turn(0, -1)


def test_elapsed_seconds_tracks_time_source() -> None:
    tracker, clock = _tracker()
    assert tracker.elapsed_seconds() == 0
    clock.advance(5.5)
    assert tracker.elapsed_seconds() == 5.5


def test_budget_status_rate_limited_wait_seconds_positive() -> None:
    status = BudgetStatus.rate_limited(12.5)
    assert status.kind is BudgetStatusKind.RATE_LIMITED
    assert status.wait_seconds == 12.5
    assert status.reason is None


def test_budget_status_budget_exceeded_carries_reason() -> None:
    status = BudgetStatus.budget_exceeded(BudgetReason.WALL_CLOCK)
    assert status.kind is BudgetStatusKind.BUDGET_EXCEEDED
    assert status.reason is BudgetReason.WALL_CLOCK
    assert status.wait_seconds is None


def test_window_constant_is_60_seconds() -> None:
    assert RATE_LIMIT_WINDOW_SECONDS == 60.0
