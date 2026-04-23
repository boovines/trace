"""Tests for the computer tool action dispatcher (X-013)."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from runner.coords import ImageMapping
from runner.dispatcher import (
    MAX_WAIT_SECONDS,
    ToolResult,
    dispatch_tool_call,
    parse_key_expression,
)
from runner.input_adapter import DryRunInputAdapter
from runner.parser import ToolCallAction
from runner.screen_source import TrajectoryScreenSource, blank_canvas_png


@pytest.fixture
def trajectory_root(tmp_path: Path) -> tuple[Path, str]:
    traj_id = "dispatcher_test"
    screenshots = tmp_path / traj_id / "screenshots"
    screenshots.mkdir(parents=True)
    png = blank_canvas_png()
    for i in range(1, 11):
        (screenshots / f"{i:04d}.png").write_bytes(png)
    return (tmp_path, traj_id)


@pytest.fixture
def screen_source(
    trajectory_root: tuple[Path, str],
) -> TrajectoryScreenSource:
    root, traj_id = trajectory_root
    return TrajectoryScreenSource(traj_id, trajectories_root=root)


@pytest.fixture
def image_mapping() -> ImageMapping:
    return ImageMapping(
        original_pixels=(2880, 1800),
        resized_pixels=(1568, 980),
        scale_from_resized_to_points=(2880 / 1568) / 2.0,
    )


@pytest.fixture
def input_adapter() -> DryRunInputAdapter:
    return DryRunInputAdapter()


def _action(tool_input: dict[str, object]) -> ToolCallAction:
    return ToolCallAction(
        tool_name="computer",
        tool_input=tool_input,
        tool_use_id="toolu_test",
    )


def test_screenshot_returns_image_block_and_mapping(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    result = dispatch_tool_call(
        _action({"action": "screenshot"}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert isinstance(result, ToolResult)
    assert result.is_error is False
    assert result.new_image_mapping is not None
    assert len(result.content_blocks) == 1
    block = result.content_blocks[0]
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"
    # Base64 decodes into real PNG bytes starting with the PNG signature.
    decoded = base64.standard_b64decode(block["source"]["data"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
    # Screenshot advanced the screen source by one frame.
    assert screen_source.index == 1
    # No adapter interaction for a pure screenshot.
    assert input_adapter.get_recorded_calls() == []


def test_left_click_dispatches_to_click_and_maps_coordinate(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    result = dispatch_tool_call(
        _action({"action": "left_click", "coordinate": [100, 100]}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert result.is_error is False
    calls = input_adapter.get_recorded_calls()
    assert len(calls) == 1
    name, args, _kwargs = calls[0]
    assert name == "click"
    x_pt, y_pt, button, modifiers = args
    assert button == "left"
    assert modifiers == ()
    # Mapping factor ≈ 0.9184; center (100,100) ≈ (91.8, 91.8).
    assert isinstance(x_pt, float)
    assert isinstance(y_pt, float)
    assert abs(x_pt - 91.8) < 0.1
    assert abs(y_pt - 91.8) < 0.1
    # Post-click screenshot taken.
    assert result.new_image_mapping is not None
    assert screen_source.index == 1


def test_left_click_with_modifiers_passes_them_to_adapter(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    dispatch_tool_call(
        _action(
            {
                "action": "left_click",
                "coordinate": [50, 50],
                "text": "cmd+shift",
            }
        ),
        input_adapter,
        screen_source,
        image_mapping,
    )
    calls = input_adapter.get_recorded_calls()
    assert len(calls) == 1
    _, args, _ = calls[0]
    _, _, button, modifiers = args
    assert button == "left"
    assert modifiers == ("cmd", "shift")


@pytest.mark.parametrize(
    ("action_name", "expected_button"),
    [("right_click", "right"), ("middle_click", "middle")],
)
def test_other_single_click_buttons(
    action_name: str,
    expected_button: str,
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    dispatch_tool_call(
        _action({"action": action_name, "coordinate": [200, 200]}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    calls = input_adapter.get_recorded_calls()
    assert len(calls) == 1
    _, args, _ = calls[0]
    assert args[2] == expected_button


def test_double_click_emits_two_left_click_calls(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    dispatch_tool_call(
        _action({"action": "double_click", "coordinate": [300, 300]}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    calls = input_adapter.get_recorded_calls()
    assert len(calls) == 2
    for name, args, _ in calls:
        assert name == "click"
        assert args[2] == "left"
    # One screenshot regardless of click count.
    assert screen_source.index == 1


def test_type_dispatches_to_type_text(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    result = dispatch_tool_call(
        _action({"action": "type", "text": "hello world"}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert result.is_error is False
    calls = input_adapter.get_recorded_calls()
    assert calls == [("type_text", ("hello world",), {})]
    assert result.new_image_mapping is not None


def test_key_cmd_shift_s_parses_into_ordered_list(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    dispatch_tool_call(
        _action({"action": "key", "text": "cmd+shift+s"}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    calls = input_adapter.get_recorded_calls()
    assert len(calls) == 1
    name, args, _ = calls[0]
    assert name == "key_press"
    (keys,) = args
    assert keys == ("cmd", "shift", "s")


def test_key_single_named_key(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    dispatch_tool_call(
        _action({"action": "key", "text": "Return"}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    calls = input_adapter.get_recorded_calls()
    assert calls[0][1][0] == ("Return",)


def test_parse_key_expression_helper() -> None:
    assert parse_key_expression("cmd+shift+s") == ["cmd", "shift", "s"]
    assert parse_key_expression("Return") == ["Return"]
    assert parse_key_expression("") == []
    assert parse_key_expression("+") == []


def test_scroll_dispatches_with_direction_and_amount(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    dispatch_tool_call(
        _action(
            {
                "action": "scroll",
                "coordinate": [400, 400],
                "scroll_direction": "down",
                "scroll_amount": 5,
            }
        ),
        input_adapter,
        screen_source,
        image_mapping,
    )
    calls = input_adapter.get_recorded_calls()
    assert len(calls) == 1
    name, args, _ = calls[0]
    assert name == "scroll"
    x_pt, _y_pt, direction, amount = args
    assert direction == "down"
    assert amount == 5
    assert abs(x_pt - 400 * (2880 / 1568) / 2.0) < 0.01


def test_mouse_move_does_not_trigger_screenshot(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    result = dispatch_tool_call(
        _action({"action": "mouse_move", "coordinate": [250, 250]}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert result.is_error is False
    assert result.content_blocks == []
    assert result.new_image_mapping is None
    assert screen_source.index == 0
    calls = input_adapter.get_recorded_calls()
    assert calls[0][0] == "move_mouse"


def test_wait_is_capped_at_max_seconds(
    monkeypatch: pytest.MonkeyPatch,
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    import runner.dispatcher as dispatcher_module

    observed: list[float] = []
    monkeypatch.setattr(
        dispatcher_module.time, "sleep", lambda s: observed.append(s)
    )

    dispatch_tool_call(
        _action({"action": "wait", "duration": 60}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert observed == [MAX_WAIT_SECONDS]


def test_wait_short_duration_is_not_capped(
    monkeypatch: pytest.MonkeyPatch,
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    import runner.dispatcher as dispatcher_module

    observed: list[float] = []
    monkeypatch.setattr(
        dispatcher_module.time, "sleep", lambda s: observed.append(s)
    )

    dispatch_tool_call(
        _action({"action": "wait", "duration": 0.5}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert observed == [0.5]


def test_unknown_action_returns_error_without_adapter_call(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    result = dispatch_tool_call(
        _action({"action": "teleport", "coordinate": [1, 1]}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert result.is_error is True
    assert result.new_image_mapping is None
    assert "Unknown computer action: teleport" in result.content_blocks[0]["text"]
    assert input_adapter.get_recorded_calls() == []
    assert screen_source.index == 0


def test_missing_action_field_returns_error(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    result = dispatch_tool_call(
        _action({"coordinate": [1, 1]}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert result.is_error is True
    assert input_adapter.get_recorded_calls() == []


def test_click_missing_coordinate_returns_error(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    result = dispatch_tool_call(
        _action({"action": "left_click"}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert result.is_error is True
    assert input_adapter.get_recorded_calls() == []


def test_click_negative_coordinate_returns_error(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    result = dispatch_tool_call(
        _action({"action": "left_click", "coordinate": [-10, 50]}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert result.is_error is True
    assert "negative" in result.content_blocks[0]["text"]
    assert input_adapter.get_recorded_calls() == []


def test_click_out_of_bounds_returns_error(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    result = dispatch_tool_call(
        _action({"action": "left_click", "coordinate": [2000, 500]}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert result.is_error is True
    assert "out of bounds" in result.content_blocks[0]["text"]
    assert input_adapter.get_recorded_calls() == []


def test_click_bad_coordinate_shape_returns_error(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    result = dispatch_tool_call(
        _action({"action": "left_click", "coordinate": [1, 2, 3]}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert result.is_error is True
    assert input_adapter.get_recorded_calls() == []


def test_type_without_text_returns_error(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    result = dispatch_tool_call(
        _action({"action": "type"}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert result.is_error is True
    assert input_adapter.get_recorded_calls() == []


def test_key_empty_text_returns_error(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    result = dispatch_tool_call(
        _action({"action": "key", "text": ""}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert result.is_error is True
    assert input_adapter.get_recorded_calls() == []


def test_scroll_invalid_direction_returns_error(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    result = dispatch_tool_call(
        _action(
            {
                "action": "scroll",
                "coordinate": [100, 100],
                "scroll_direction": "diagonal",
                "scroll_amount": 3,
            }
        ),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert result.is_error is True
    assert input_adapter.get_recorded_calls() == []


def test_scroll_negative_amount_returns_error(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    result = dispatch_tool_call(
        _action(
            {
                "action": "scroll",
                "coordinate": [100, 100],
                "scroll_direction": "up",
                "scroll_amount": -1,
            }
        ),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert result.is_error is True
    assert input_adapter.get_recorded_calls() == []


def test_wait_negative_duration_returns_error(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    result = dispatch_tool_call(
        _action({"action": "wait", "duration": -1}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert result.is_error is True


def test_wait_missing_duration_returns_error(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    result = dispatch_tool_call(
        _action({"action": "wait"}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    assert result.is_error is True


def test_screenshot_advances_screen_source_across_multiple_calls(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    for _ in range(3):
        dispatch_tool_call(
            _action({"action": "screenshot"}),
            input_adapter,
            screen_source,
            image_mapping,
        )
    assert screen_source.index == 3


def test_coordinate_mapping_exact_formula(
    input_adapter: DryRunInputAdapter,
    screen_source: TrajectoryScreenSource,
    image_mapping: ImageMapping,
) -> None:
    """100x100 resized → (100 * 2880/1568) / 2.0 ≈ 91.8367."""
    dispatch_tool_call(
        _action({"action": "left_click", "coordinate": [100, 100]}),
        input_adapter,
        screen_source,
        image_mapping,
    )
    _, args, _ = input_adapter.get_recorded_calls()[0]
    x_pt, y_pt, *_ = args
    expected = (100 * 2880 / 1568) / 2.0
    assert abs(x_pt - expected) < 1e-6
    assert abs(y_pt - expected) < 1e-6
