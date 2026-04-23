"""Tool call parser and confirmation tag detector (X-012).

Parses an :class:`~runner.agent_runtime.AgentResponse` into a structured
:class:`ParsedAction` for the executor. The parser duck-types content
blocks as dicts (see the X-010 learnings: ``ClaudeRuntime`` converts
typed SDK blocks via ``model_dump()`` before they ever reach this module,
and fake-mode fixtures ship dicts natively).

Safety rule
-----------
When the response contains BOTH a ``tool_use`` block and a
``<needs_confirmation step="N"/>`` tag, the pending tool call is dropped
and :class:`ConfirmationRequest` is returned. The confirmation signal is
always honored — a destructive step must never slip past the prompt-layer
gate just because the model also emitted a concrete action in the same
turn. The harness-layer keyword matcher (X-015) is the belt-and-suspenders
backup; this parser is the first line of defence.

Tag priority
------------
``<workflow_complete/>`` and ``<workflow_failed .../>`` also trump a
``tool_use`` block in the same turn — a terminal signal means the agent
intended to stop. Only :class:`ConfirmationRequest` trumps a terminal
tag; in practice the model never emits both.

Malformed tags (missing quotes, non-integer step, extra attributes, step
<= 0) are returned as :class:`UnknownAction` with the original text.
The parser NEVER raises on malformed input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

from runner.agent_runtime import AgentResponse

# Strict, well-formed tag patterns. Whitespace tolerance is intentional —
# acceptance criterion: `< needs_confirmation step="3" />` must match.
_CONFIRMATION_RE: Final[re.Pattern[str]] = re.compile(
    r'<\s*needs_confirmation\s+step\s*=\s*"(?P<step>[^"]*)"\s*/\s*>'
)
_COMPLETE_RE: Final[re.Pattern[str]] = re.compile(
    r"<\s*workflow_complete\s*/\s*>"
)
_FAILED_RE: Final[re.Pattern[str]] = re.compile(
    r'<\s*workflow_failed\s+reason\s*=\s*"(?P<reason>[^"]*)"\s*/\s*>'
)

# Looser "attempt" pattern: matches even when the attributes are malformed.
# Used to decide that the model tried to signal a confirmation but got it
# wrong — we return UnknownAction rather than silently falling through to
# a tool_use block that would then execute the destructive action.
_CONFIRMATION_ATTEMPT_RE: Final[re.Pattern[str]] = re.compile(
    r"<\s*needs_confirmation\b"
)


@dataclass(frozen=True, slots=True)
class ToolCallAction:
    """The agent emitted a ``tool_use`` block for the computer tool."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str


@dataclass(frozen=True, slots=True)
class ConfirmationRequest:
    """The agent requested confirmation for a destructive step."""

    step_number: int


@dataclass(frozen=True, slots=True)
class WorkflowComplete:
    """The agent signaled the workflow finished successfully."""


@dataclass(frozen=True, slots=True)
class WorkflowFailed:
    """The agent gave up on the workflow."""

    reason: str


@dataclass(frozen=True, slots=True)
class UnknownAction:
    """Fallback: unrecognized, malformed, or bare end-of-turn text."""

    raw_text: str


ParsedAction = (
    ToolCallAction
    | ConfirmationRequest
    | WorkflowComplete
    | WorkflowFailed
    | UnknownAction
)


def _collect_text(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _find_tool_use(blocks: list[Any]) -> dict[str, Any] | None:
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            return block
    return None


def parse_agent_response(response: AgentResponse) -> ParsedAction:
    """Parse one agent turn into a :class:`ParsedAction`.

    Never raises. See module docstring for priority and safety rules.
    """
    blocks = response.content_blocks
    text = _collect_text(blocks)

    confirmation_match = _CONFIRMATION_RE.search(text)
    if confirmation_match is not None:
        raw_step = confirmation_match.group("step")
        try:
            step_num = int(raw_step)
        except ValueError:
            return UnknownAction(raw_text=text)
        if step_num > 0:
            return ConfirmationRequest(step_number=step_num)
        return UnknownAction(raw_text=text)

    # The model tried to emit a needs_confirmation but the attributes were
    # malformed (missing quotes, extra attributes). Treat as UnknownAction
    # so a concurrent tool_use block doesn't silently execute.
    if _CONFIRMATION_ATTEMPT_RE.search(text) is not None:
        return UnknownAction(raw_text=text)

    if _COMPLETE_RE.search(text) is not None:
        return WorkflowComplete()

    failed_match = _FAILED_RE.search(text)
    if failed_match is not None:
        return WorkflowFailed(reason=failed_match.group("reason"))

    tool_use = _find_tool_use(blocks)
    if tool_use is not None:
        name = tool_use.get("name", "")
        tool_input = tool_use.get("input", {})
        tool_use_id = tool_use.get("id", "")
        if (
            isinstance(name, str)
            and isinstance(tool_use_id, str)
            and isinstance(tool_input, dict)
        ):
            return ToolCallAction(
                tool_name=name,
                tool_input=dict(tool_input),
                tool_use_id=tool_use_id,
            )

    return UnknownAction(raw_text=text)


__all__ = [
    "ConfirmationRequest",
    "ParsedAction",
    "ToolCallAction",
    "UnknownAction",
    "WorkflowComplete",
    "WorkflowFailed",
    "parse_agent_response",
]
