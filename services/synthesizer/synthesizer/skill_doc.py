"""SKILL.md parser.

SKILL.md is the human-readable, machine-parsed on-disk contract between the
Synthesizer (producer) and the Runner (consumer). This module owns the parser.

Format (strict — the Runner trusts it):

    # <Title>

    <optional prose / parameter docs>

    ## Steps

    1. First step text.
    2. Second step text.
    3. ⚠️ Destructive step text (the marker is part of the contract, not
       decorative: the destructive flag is derived from it).

Steps are numbered 1-based. A step whose text begins with the U+26A0 warning
sign followed by a space is marked ``destructive=True`` and the marker is
stripped from the stored text. Everything before ``## Steps`` is ignored by
the parser (but preserved on disk for humans).

This module intentionally lives under the synthesizer package — the runner
consumes SKILL.md but does NOT duplicate the parser. Changing the format here
is a cross-branch contract change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace


class SkillMarkdownError(ValueError):
    """Raised when a SKILL.md file cannot be parsed."""


@dataclass(frozen=True)
class SkillStep:
    """One step of a parsed SKILL.md.

    ``number`` is 1-based to match ``skill.meta.json``'s ``destructive_steps``
    indexing. ``text`` has the ⚠️ prefix stripped — ``destructive`` is the
    canonical flag.
    """

    number: int
    text: str
    destructive: bool


@dataclass(frozen=True)
class ParsedSkill:
    """Result of ``parse_skill_md`` — title plus ordered step list."""

    title: str
    steps: tuple[SkillStep, ...]


_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_STEPS_HEADING_RE = re.compile(r"^##\s+Steps\s*$", re.MULTILINE)
_STEP_LINE_RE = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$")
_WARNING_PREFIX = "⚠️ "  # ⚠️ + space (fully qualified emoji)
_WARNING_PREFIX_BARE = "⚠ "  # ⚠ without variation selector


def parse_skill_md(content: str) -> ParsedSkill:
    """Parse SKILL.md content into a ``ParsedSkill``.

    Raises ``SkillMarkdownError`` if the file has no top-level title, no
    ``## Steps`` section, no step lines under the heading, or step numbers
    that do not start at 1 and increment by 1.
    """
    title_match = _TITLE_RE.search(content)
    if title_match is None:
        raise SkillMarkdownError("SKILL.md must start with a '# <Title>' heading")
    title = title_match.group(1).strip()

    steps_match = _STEPS_HEADING_RE.search(content)
    if steps_match is None:
        raise SkillMarkdownError("SKILL.md must contain a '## Steps' section")

    steps_body = content[steps_match.end() :]
    # Stop at the next ## heading so later sections don't bleed in.
    next_heading = re.search(r"^##\s+\S", steps_body, re.MULTILINE)
    if next_heading is not None:
        steps_body = steps_body[: next_heading.start()]

    steps: list[SkillStep] = []
    for raw_line in steps_body.splitlines():
        line_match = _STEP_LINE_RE.match(raw_line)
        if line_match is None:
            continue
        number = int(line_match.group(1))
        text = line_match.group(2)
        destructive = False
        if text.startswith(_WARNING_PREFIX):
            destructive = True
            text = text[len(_WARNING_PREFIX) :].lstrip()
        elif text.startswith(_WARNING_PREFIX_BARE):
            destructive = True
            text = text[len(_WARNING_PREFIX_BARE) :].lstrip()
        steps.append(SkillStep(number=number, text=text, destructive=destructive))

    if not steps:
        raise SkillMarkdownError("SKILL.md '## Steps' section has no numbered steps")

    for index, step in enumerate(steps, start=1):
        if step.number != index:
            raise SkillMarkdownError(
                f"SKILL.md step numbering must be 1-based and contiguous; "
                f"expected {index} but got {step.number}"
            )

    return ParsedSkill(title=title, steps=tuple(steps))


def replace_step_text(step: SkillStep, new_text: str) -> SkillStep:
    """Return a copy of ``step`` with ``text`` replaced.

    Exposed as a helper because ``SkillStep`` is frozen — callers performing
    parameter substitution should use this rather than mutating in place.
    """
    return replace(step, text=new_text)


__all__ = [
    "ParsedSkill",
    "SkillMarkdownError",
    "SkillStep",
    "parse_skill_md",
    "replace_step_text",
]
