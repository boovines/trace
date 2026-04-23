"""Tests for :mod:`runner.parser` (X-012)."""

from __future__ import annotations

from typing import Any

import pytest
from runner.agent_runtime import AgentResponse
from runner.parser import (
    ConfirmationRequest,
    ToolCallAction,
    UnknownAction,
    WorkflowComplete,
    WorkflowFailed,
    parse_agent_response,
)


def _response(
    blocks: list[dict[str, Any]],
    stop_reason: str = "end_turn",
    *,
    turn_number: int = 1,
) -> AgentResponse:
    return AgentResponse(
        content_blocks=list(blocks),
        stop_reason=stop_reason,
        input_tokens=100,
        output_tokens=20,
        turn_number=turn_number,
    )


def test_tool_use_block_returns_tool_call_action() -> None:
    block = {
        "type": "tool_use",
        "id": "toolu_abc",
        "name": "computer",
        "input": {"action": "screenshot"},
    }
    result = parse_agent_response(_response([block], stop_reason="tool_use"))
    assert isinstance(result, ToolCallAction)
    assert result.tool_name == "computer"
    assert result.tool_input == {"action": "screenshot"}
    assert result.tool_use_id == "toolu_abc"


def test_tool_use_with_coordinate_payload() -> None:
    block = {
        "type": "tool_use",
        "id": "toolu_xyz",
        "name": "computer",
        "input": {"action": "left_click", "coordinate": [512, 96]},
    }
    result = parse_agent_response(_response([block], stop_reason="tool_use"))
    assert isinstance(result, ToolCallAction)
    assert result.tool_input["action"] == "left_click"
    assert result.tool_input["coordinate"] == [512, 96]


def test_needs_confirmation_tag_returns_confirmation_request() -> None:
    block = {
        "type": "text",
        "text": 'Reply is composed. <needs_confirmation step="5"/>',
    }
    result = parse_agent_response(_response([block]))
    assert result == ConfirmationRequest(step_number=5)


def test_workflow_complete_tag_returns_workflow_complete() -> None:
    block = {"type": "text", "text": "All done. <workflow_complete/>"}
    result = parse_agent_response(_response([block]))
    assert isinstance(result, WorkflowComplete)


def test_workflow_failed_tag_returns_workflow_failed() -> None:
    block = {
        "type": "text",
        "text": 'Cannot proceed. <workflow_failed reason="app not found"/>',
    }
    result = parse_agent_response(_response([block]))
    assert result == WorkflowFailed(reason="app not found")


def test_workflow_failed_reason_with_punctuation() -> None:
    block = {
        "type": "text",
        "text": (
            'Issue detected. <workflow_failed reason="screen '
            'blank, no UI visible"/>'
        ),
    }
    result = parse_agent_response(_response([block]))
    assert result == WorkflowFailed(reason="screen blank, no UI visible")


def test_both_tool_use_and_confirmation_returns_confirmation() -> None:
    text_block = {
        "type": "text",
        "text": 'Destructive step ahead. <needs_confirmation step="7"/>',
    }
    tool_block = {
        "type": "tool_use",
        "id": "toolu_send",
        "name": "computer",
        "input": {"action": "left_click", "coordinate": [900, 200]},
    }
    result = parse_agent_response(
        _response([text_block, tool_block], stop_reason="tool_use")
    )
    assert result == ConfirmationRequest(step_number=7)


def test_both_tool_use_and_confirmation_blocks_in_reverse_order() -> None:
    tool_block = {
        "type": "tool_use",
        "id": "toolu_send",
        "name": "computer",
        "input": {"action": "left_click", "coordinate": [900, 200]},
    }
    text_block = {
        "type": "text",
        "text": '<needs_confirmation step="3"/>',
    }
    result = parse_agent_response(
        _response([tool_block, text_block], stop_reason="tool_use")
    )
    assert result == ConfirmationRequest(step_number=3)


@pytest.mark.parametrize(
    "text",
    [
        "<needs_confirmation step=5/>",  # missing quotes
        '<needs_confirmation step="abc"/>',  # non-integer step
        '<needs_confirmation step="3" extra="x"/>',  # extra attributes
        "<needs_confirmation/>",  # missing step attribute
        "<needs_confirmation>",  # not self-closing
        '<needs_confirmation step=""/>',  # empty step value
    ],
)
def test_malformed_confirmation_returns_unknown_action(text: str) -> None:
    block = {"type": "text", "text": text}
    result = parse_agent_response(_response([block]))
    assert isinstance(result, UnknownAction)
    assert result.raw_text == text


def test_malformed_confirmation_drops_concurrent_tool_use() -> None:
    """Safety: a malformed confirmation must not let a tool_use through."""
    text_block = {
        "type": "text",
        "text": "<needs_confirmation step=5/>",
    }
    tool_block = {
        "type": "tool_use",
        "id": "toolu_send",
        "name": "computer",
        "input": {"action": "left_click", "coordinate": [900, 200]},
    }
    result = parse_agent_response(
        _response([text_block, tool_block], stop_reason="tool_use")
    )
    assert isinstance(result, UnknownAction)


@pytest.mark.parametrize("step_value", ["0", "-1", "-42"])
def test_non_positive_step_returns_unknown_action(step_value: str) -> None:
    text = f'<needs_confirmation step="{step_value}"/>'
    block = {"type": "text", "text": text}
    result = parse_agent_response(_response([block]))
    assert isinstance(result, UnknownAction)


@pytest.mark.parametrize(
    "text",
    [
        '< needs_confirmation step="3" />',  # leading and trailing spaces
        '<needs_confirmation  step="3"/>',  # extra whitespace between attrs
        '<needs_confirmation step = "3"/>',  # spaces around the equals sign
        '< needs_confirmation  step = "3" / >',  # all of the above
    ],
)
def test_whitespace_tolerance_in_confirmation_tag(text: str) -> None:
    block = {"type": "text", "text": text}
    result = parse_agent_response(_response([block]))
    assert result == ConfirmationRequest(step_number=3)


def test_whitespace_tolerance_in_workflow_complete() -> None:
    block = {"type": "text", "text": "< workflow_complete / >"}
    result = parse_agent_response(_response([block]))
    assert isinstance(result, WorkflowComplete)


def test_whitespace_tolerance_in_workflow_failed() -> None:
    block = {
        "type": "text",
        "text": '< workflow_failed  reason = "boom" / >',
    }
    result = parse_agent_response(_response([block]))
    assert result == WorkflowFailed(reason="boom")


def test_plain_text_no_tags_returns_unknown_action() -> None:
    block = {"type": "text", "text": "Just some commentary, no tags."}
    result = parse_agent_response(_response([block]))
    assert isinstance(result, UnknownAction)
    assert result.raw_text == "Just some commentary, no tags."


def test_end_turn_with_no_blocks_returns_unknown_action() -> None:
    result = parse_agent_response(_response([]))
    assert isinstance(result, UnknownAction)
    assert result.raw_text == ""


def test_multiple_text_blocks_joined_for_tag_detection() -> None:
    blocks = [
        {"type": "text", "text": "Preparing..."},
        {"type": "text", "text": '<needs_confirmation step="2"/>'},
    ]
    result = parse_agent_response(_response(blocks))
    assert result == ConfirmationRequest(step_number=2)


def test_large_step_number() -> None:
    block = {"type": "text", "text": '<needs_confirmation step="99"/>'}
    result = parse_agent_response(_response([block]))
    assert result == ConfirmationRequest(step_number=99)


def test_unknown_action_preserves_raw_text() -> None:
    raw = "The model rambled without emitting any tags."
    block = {"type": "text", "text": raw}
    result = parse_agent_response(_response([block]))
    assert isinstance(result, UnknownAction)
    assert result.raw_text == raw


def test_tool_use_with_non_computer_name_still_returns_tool_call() -> None:
    block = {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "str_replace_editor",
        "input": {"command": "view"},
    }
    result = parse_agent_response(_response([block], stop_reason="tool_use"))
    assert isinstance(result, ToolCallAction)
    assert result.tool_name == "str_replace_editor"


def test_workflow_complete_trumps_concurrent_tool_use() -> None:
    text_block = {"type": "text", "text": "Done. <workflow_complete/>"}
    tool_block = {
        "type": "tool_use",
        "id": "toolu_stale",
        "name": "computer",
        "input": {"action": "screenshot"},
    }
    result = parse_agent_response(
        _response([text_block, tool_block], stop_reason="tool_use")
    )
    assert isinstance(result, WorkflowComplete)


def test_workflow_failed_trumps_concurrent_tool_use() -> None:
    text_block = {
        "type": "text",
        "text": '<workflow_failed reason="network down"/>',
    }
    tool_block = {
        "type": "tool_use",
        "id": "toolu_stale",
        "name": "computer",
        "input": {"action": "screenshot"},
    }
    result = parse_agent_response(
        _response([text_block, tool_block], stop_reason="tool_use")
    )
    assert result == WorkflowFailed(reason="network down")


def test_non_dict_block_is_ignored() -> None:
    blocks: list[Any] = [
        "not a dict",
        42,
        {"type": "text", "text": "<workflow_complete/>"},
    ]
    result = parse_agent_response(_response(blocks))
    assert isinstance(result, WorkflowComplete)


def test_text_block_without_text_field_is_ignored() -> None:
    blocks: list[dict[str, Any]] = [
        {"type": "text"},
        {"type": "text", "text": "<workflow_complete/>"},
    ]
    result = parse_agent_response(_response(blocks))
    assert isinstance(result, WorkflowComplete)


def test_parser_never_raises_on_weird_input() -> None:
    """The contract: the parser NEVER raises. Throw odd shapes at it."""
    weird_blocks: list[Any] = [
        {"type": "text", "text": "<<<<>>>>"},
        {"type": "tool_use"},
        {"type": "tool_use", "name": 123, "input": "not a dict"},
        {"type": "unknown_block"},
        {},
    ]
    for block in weird_blocks:
        parse_agent_response(_response([block]))


def test_workflow_failed_with_empty_reason() -> None:
    block = {"type": "text", "text": '<workflow_failed reason=""/>'}
    result = parse_agent_response(_response([block]))
    assert result == WorkflowFailed(reason="")


def test_fixture_gmail_turn_3_is_confirmation() -> None:
    """Pin the fake-mode fixture: gmail_reply turn 3 → ConfirmationRequest(7)."""
    block = {
        "type": "text",
        "text": 'Reply is composed and ready to send. <needs_confirmation step="7"/>',
    }
    result = parse_agent_response(_response([block]))
    assert result == ConfirmationRequest(step_number=7)


def test_fixture_gmail_turn_4_is_workflow_complete() -> None:
    block = {
        "type": "text",
        "text": "Send confirmed and delivered. <workflow_complete/>",
    }
    result = parse_agent_response(_response([block]))
    assert isinstance(result, WorkflowComplete)
