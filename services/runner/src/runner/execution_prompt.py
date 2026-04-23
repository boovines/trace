"""Build the system prompt that tells Claude how to execute a workflow.

The prompt is load-bearing for safety: the executor (``runner.tool_parser``,
landing in X-012) looks for the literal tag ``<needs_confirmation step="N"/>``
when deciding whether a destructive step should be confirmed. If this prompt
drifts and the model emits a slightly different tag, the parser will miss the
signal and a destructive action could slip past the prompt-layer gate. The
harness-layer keyword matcher (X-015) is the belt-and-suspenders backup, but
the prompt is the first line of defence.

``build_execution_prompt`` is intentionally deterministic — calling it twice
with the same ``LoadedSkill`` returns byte-identical output. This matters for
prompt caching on the Anthropic side and for the fake-mode fixture selector
in :mod:`runner.claude_runtime`, which matches by substring of the system
prompt.
"""

from __future__ import annotations

from typing import Final, Literal

from synthesizer.skill_doc import ParsedSkill, SkillStep

from runner.skill_loader import LoadedSkill

Mode = Literal["dry_run", "live"]

DESTRUCTIVE_PROTOCOL: Final[str] = (
    'For any step marked ⚠️ [DESTRUCTIVE], DO NOT use the computer tool to '
    "execute it. Instead, emit EXACTLY this text: "
    '<needs_confirmation step="N"/> where N is the step number. Then stop '
    "your turn. When confirmed, you will receive a user message allowing "
    "you to proceed."
)

COMPLETION_PROTOCOL: Final[str] = (
    "When all steps are done, emit <workflow_complete/> and stop."
)

ERROR_PROTOCOL: Final[str] = (
    "If you cannot proceed (app not in expected state, UI missing, screen "
    'unreadable), emit <workflow_failed reason="..."/> and stop.'
)

VERIFICATION_PROTOCOL: Final[str] = (
    "After each click/type action, take a screenshot with the computer "
    "tool before moving on."
)

DRY_RUN_NOTICE: Final[str] = (
    "This is a DRY RUN. Your tool calls will be received but no real "
    "input will be posted."
)

PREAMBLE: Final[str] = (
    "You are a macOS computer-use agent executing a saved workflow. "
    "Perform the steps below in order, one at a time. After each action, "
    "verify the screen state before moving on. Do not skip steps, do not "
    "improvise, and do not run anything outside the listed steps."
)


def _render_skill_markdown(parsed: ParsedSkill) -> str:
    """Render a ``ParsedSkill`` back into the canonical SKILL.md shape.

    The parser strips the ⚠️ marker from destructive steps and exposes the
    flag separately. When re-rendering for the prompt we re-attach an
    explicit ``⚠️ [DESTRUCTIVE]`` marker so the instruction text in
    ``DESTRUCTIVE_PROTOCOL`` matches what the model actually sees.
    """
    lines: list[str] = [f"# {parsed.title}", "", "## Steps", ""]
    for step in parsed.steps:
        lines.append(_render_step_line(step))
    lines.append("")
    return "\n".join(lines)


def _render_step_line(step: SkillStep) -> str:
    prefix = "⚠️ [DESTRUCTIVE] " if step.destructive else ""
    return f"{step.number}. {prefix}{step.text}"


def build_execution_prompt(loaded_skill: LoadedSkill, mode: Mode) -> str:
    """Build the system prompt for a runner execution turn.

    Args:
        loaded_skill: the fully-resolved skill (parameters already
            substituted via :func:`runner.skill_loader.substitute_parameters`).
        mode: ``"dry_run"`` or ``"live"``. Dry-run adds a single informational
            line; the protocol itself is identical across modes.

    Returns:
        A deterministic plain-text prompt. Same input → byte-identical output.
    """
    if mode not in ("dry_run", "live"):
        raise ValueError(
            f"mode must be 'dry_run' or 'live', got {mode!r}"
        )

    skill_markdown = _render_skill_markdown(loaded_skill.parsed_skill)

    sections: list[str] = [PREAMBLE]
    if mode == "dry_run":
        sections.append(DRY_RUN_NOTICE)
    sections.extend(
        [
            "## Workflow",
            "```markdown",
            skill_markdown.rstrip("\n"),
            "```",
            "## Verification protocol",
            VERIFICATION_PROTOCOL,
            "## Destructive-step protocol",
            DESTRUCTIVE_PROTOCOL,
            "## Completion protocol",
            COMPLETION_PROTOCOL,
            "## Error protocol",
            ERROR_PROTOCOL,
        ]
    )

    return "\n\n".join(sections) + "\n"


__all__ = [
    "COMPLETION_PROTOCOL",
    "DESTRUCTIVE_PROTOCOL",
    "DRY_RUN_NOTICE",
    "ERROR_PROTOCOL",
    "PREAMBLE",
    "VERIFICATION_PROTOCOL",
    "Mode",
    "build_execution_prompt",
]
