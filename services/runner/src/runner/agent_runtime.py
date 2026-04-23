"""Agent runtime abstraction for the runner.

The runner talks to its LLM through the ``AgentRuntime`` protocol so the
concrete runtime can be swapped without touching the executor. The shipped
implementation is :class:`runner.claude_runtime.ClaudeRuntime`; an
``OpenAIRuntime`` stub lives alongside it to document the swap seam.

Design notes:

* ``Message`` is re-exported from ``anthropic.types.MessageParam`` — the
  executor already builds message dicts in that shape, so keeping the
  protocol coupled to the Anthropic wire format avoids a pointless
  intermediate representation.
* ``AgentResponse.content_blocks`` is typed as ``list[Any]`` because the
  anthropic SDK's content-block union is large and the downstream parser
  (X-012) duck-types each block by its ``type`` field anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from anthropic.types import MessageParam

Message = MessageParam


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """One model turn: content blocks + token usage + stop reason."""

    content_blocks: list[Any]
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    turn_number: int


@runtime_checkable
class AgentRuntime(Protocol):
    """Abstracts the LLM runtime so OpenAI/Hermes can be dropped in later."""

    async def run_turn(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 4096,
    ) -> AgentResponse:
        """Send one turn to the model and return the structured response."""


__all__ = ["AgentResponse", "AgentRuntime", "Message"]
