"""Tests for the harness-layer destructive gate (X-015)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from runner.coords import ImageMapping
from runner.parser import ToolCallAction
from runner.pre_action_gate import (
    ACTIONABLE_AX_ROLES,
    AllowAction,
    AXTarget,
    RequireConfirmation,
    Unknown,
    apply_gate_to_tool_call,
    inspect_click_target,
)


@dataclass
class FakeAXResolver:
    """Test double: returns a pre-programmed target at any coordinate.

    ``call_log`` records every (x_pt, y_pt) pair the gate queried so tests
    can assert the coordinate mapping.
    """

    target: AXTarget | None = None
    raise_on_resolve: Exception | None = None

    def __post_init__(self) -> None:
        self.call_log: list[tuple[float, float]] = []

    def resolve_at(self, x_pt: float, y_pt: float) -> AXTarget | None:
        self.call_log.append((x_pt, y_pt))
        if self.raise_on_resolve is not None:
            raise self.raise_on_resolve
        return self.target


def _mapping(scale: float = 1.0) -> ImageMapping:
    """1568x980 resized image matching the DryRunDisplayInfo 1440x900 display.

    ``scale_from_resized_to_points`` defaults to 1.0 so coordinates pass
    through unchanged — tests can then assert the resolver received the
    same (x, y) Claude sent.
    """

    return ImageMapping(
        original_pixels=(2880, 1800),
        resized_pixels=(1568, 980),
        scale_from_resized_to_points=scale,
    )


def _click_action(action: str = "left_click", coordinate: object = (100, 200)) -> ToolCallAction:
    return ToolCallAction(
        tool_name="computer",
        tool_input={"action": action, "coordinate": coordinate},
        tool_use_id="toolu_test",
    )


# ---------- inspect_click_target -------------------------------------------------


def test_send_button_requires_confirmation() -> None:
    resolver = FakeAXResolver(target=AXTarget(role="AXButton", label="Send"))
    decision = inspect_click_target(100.0, 200.0, resolver)
    assert isinstance(decision, RequireConfirmation)
    assert decision.label == "Send"
    assert "AXButton" in decision.reason
    assert "Send" in decision.reason


def test_cancel_button_is_allowed() -> None:
    resolver = FakeAXResolver(target=AXTarget(role="AXButton", label="Cancel"))
    decision = inspect_click_target(100.0, 200.0, resolver)
    assert isinstance(decision, AllowAction)


def test_resolver_returns_none_is_unknown() -> None:
    resolver = FakeAXResolver(target=None)
    decision = inspect_click_target(100.0, 200.0, resolver)
    assert isinstance(decision, Unknown)


def test_resolver_exception_is_unknown() -> None:
    """The gate degrades open: an AX failure must not crash the executor."""

    resolver = FakeAXResolver(raise_on_resolve=TimeoutError("AX timed out"))
    decision = inspect_click_target(100.0, 200.0, resolver)
    assert isinstance(decision, Unknown)


def test_destructive_keyword_on_static_text_is_allowed() -> None:
    """AXStaticText is not actionable — a click through it shouldn't gate."""

    resolver = FakeAXResolver(target=AXTarget(role="AXStaticText", label="Delete everything"))
    decision = inspect_click_target(100.0, 200.0, resolver)
    assert isinstance(decision, AllowAction)


def test_delete_account_link_requires_confirmation() -> None:
    resolver = FakeAXResolver(target=AXTarget(role="AXLink", label="Delete Account"))
    decision = inspect_click_target(100.0, 200.0, resolver)
    assert isinstance(decision, RequireConfirmation)
    assert decision.label == "Delete Account"


def test_menu_item_with_destructive_keyword_requires_confirmation() -> None:
    resolver = FakeAXResolver(target=AXTarget(role="AXMenuItem", label="Remove from list"))
    decision = inspect_click_target(100.0, 200.0, resolver)
    assert isinstance(decision, RequireConfirmation)


def test_checkbox_with_destructive_keyword_requires_confirmation() -> None:
    resolver = FakeAXResolver(target=AXTarget(role="AXCheckBox", label="Authorize charge"))
    decision = inspect_click_target(100.0, 200.0, resolver)
    assert isinstance(decision, RequireConfirmation)


@pytest.mark.parametrize("role", sorted(ACTIONABLE_AX_ROLES))
def test_all_actionable_roles_gate_destructive_labels(role: str) -> None:
    resolver = FakeAXResolver(target=AXTarget(role=role, label="Send"))
    decision = inspect_click_target(1.0, 2.0, resolver)
    assert isinstance(decision, RequireConfirmation)


@pytest.mark.parametrize(
    "role",
    ["AXStaticText", "AXGroup", "AXScrollArea", "AXImage", "AXUnknown"],
)
def test_non_actionable_roles_never_gate(role: str) -> None:
    resolver = FakeAXResolver(target=AXTarget(role=role, label="Send"))
    decision = inspect_click_target(1.0, 2.0, resolver)
    assert isinstance(decision, AllowAction)


def test_substring_match_does_not_gate() -> None:
    """'sender' contains 'send' — word boundary must prevent this match."""

    resolver = FakeAXResolver(target=AXTarget(role="AXButton", label="Sender info"))
    decision = inspect_click_target(1.0, 2.0, resolver)
    assert isinstance(decision, AllowAction)


def test_case_insensitive_match() -> None:
    resolver = FakeAXResolver(target=AXTarget(role="AXButton", label="DELETE FILE"))
    decision = inspect_click_target(1.0, 2.0, resolver)
    assert isinstance(decision, RequireConfirmation)


def test_delete_in_mixed_context_matches() -> None:
    """'Delete' embedded in a longer English label matches (v1 is English-only)."""

    resolver = FakeAXResolver(
        target=AXTarget(role="AXButton", label="Please delete my subscription")
    )
    decision = inspect_click_target(1.0, 2.0, resolver)
    assert isinstance(decision, RequireConfirmation)


# ---------- apply_gate_to_tool_call ---------------------------------------------


def test_type_action_is_never_gated() -> None:
    resolver = FakeAXResolver(target=AXTarget(role="AXButton", label="Send"))
    action = ToolCallAction(
        tool_name="computer",
        tool_input={"action": "type", "text": "hello"},
        tool_use_id="t1",
    )
    decision = apply_gate_to_tool_call(action, _mapping(), resolver, mode="execute")
    assert isinstance(decision, AllowAction)
    assert resolver.call_log == []


@pytest.mark.parametrize(
    "action_name",
    ["scroll", "mouse_move", "wait", "key", "screenshot"],
)
def test_non_click_actions_are_never_gated(action_name: str) -> None:
    resolver = FakeAXResolver(target=AXTarget(role="AXButton", label="Send"))
    action = ToolCallAction(
        tool_name="computer",
        tool_input={"action": action_name, "coordinate": (10, 20)},
        tool_use_id="t1",
    )
    decision = apply_gate_to_tool_call(action, _mapping(), resolver, mode="execute")
    assert isinstance(decision, AllowAction)
    assert resolver.call_log == []


def test_dry_run_mode_always_allows_even_for_destructive_target() -> None:
    resolver = FakeAXResolver(target=AXTarget(role="AXButton", label="Send"))
    action = _click_action()
    decision = apply_gate_to_tool_call(action, _mapping(), resolver, mode="dry_run")
    assert isinstance(decision, AllowAction)
    # Resolver must never be consulted in dry-run — the AX tree is undefined
    # against a trajectory screenshot.
    assert resolver.call_log == []


@pytest.mark.parametrize(
    "click_name",
    sorted({"left_click", "right_click", "double_click", "middle_click"}),
)
def test_all_click_variants_are_gated_in_execute_mode(click_name: str) -> None:
    resolver = FakeAXResolver(target=AXTarget(role="AXButton", label="Send"))
    action = _click_action(action=click_name)
    decision = apply_gate_to_tool_call(action, _mapping(), resolver, mode="execute")
    assert isinstance(decision, RequireConfirmation)


def test_coordinate_mapping_scales_to_points() -> None:
    """A resized-pixel coordinate must be scaled by the mapping before AX query."""

    resolver = FakeAXResolver(target=AXTarget(role="AXButton", label="OK"))
    # scale=2.0 means resized-pixel 100 → 200 display-points.
    mapping = _mapping(scale=2.0)
    action = _click_action(coordinate=(100, 300))
    apply_gate_to_tool_call(action, mapping, resolver, mode="execute")
    assert resolver.call_log == [(200.0, 600.0)]


@pytest.mark.parametrize(
    "bad_coord",
    [
        None,
        "100,200",
        (100,),
        (100, 200, 300),
        (True, 200),
        (100, False),
        (100, "y"),
    ],
)
def test_malformed_coordinates_degrade_to_allow(bad_coord: object) -> None:
    """The gate mirrors the dispatcher's tolerance — bad input is not a confirmation trigger."""

    resolver = FakeAXResolver(target=AXTarget(role="AXButton", label="Send"))
    action = _click_action(coordinate=bad_coord)
    decision = apply_gate_to_tool_call(action, _mapping(), resolver, mode="execute")
    assert isinstance(decision, AllowAction)
    assert resolver.call_log == []


def test_missing_action_name_degrades_to_allow() -> None:
    resolver = FakeAXResolver(target=AXTarget(role="AXButton", label="Send"))
    action = ToolCallAction(
        tool_name="computer",
        tool_input={"coordinate": (10, 20)},
        tool_use_id="t1",
    )
    decision = apply_gate_to_tool_call(action, _mapping(), resolver, mode="execute")
    assert isinstance(decision, AllowAction)


def test_unknown_action_name_degrades_to_allow() -> None:
    resolver = FakeAXResolver(target=AXTarget(role="AXButton", label="Send"))
    action = ToolCallAction(
        tool_name="computer",
        tool_input={"action": "teleport", "coordinate": (10, 20)},
        tool_use_id="t1",
    )
    decision = apply_gate_to_tool_call(action, _mapping(), resolver, mode="execute")
    assert isinstance(decision, AllowAction)
    assert resolver.call_log == []


def test_resolver_returning_none_in_execute_surfaces_unknown() -> None:
    resolver = FakeAXResolver(target=None)
    action = _click_action()
    decision = apply_gate_to_tool_call(action, _mapping(), resolver, mode="execute")
    assert isinstance(decision, Unknown)
