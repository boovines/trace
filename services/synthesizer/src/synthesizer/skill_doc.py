"""Parser and renderer for the strict SKILL.md format.

The SKILL.md file is a locked contract between the synthesizer (writer) and the
runner (reader). To avoid regex-hacking markdown in multiple places, this module
is the single source of truth for:

* :class:`ParsedSkill` — the structured in-memory representation.
* :func:`parse_skill_md` — markdown → :class:`ParsedSkill` with strict validation.
* :func:`render_skill_md` — :class:`ParsedSkill` → canonical markdown.
* :func:`extract_parameter_refs` — set of ``{name}`` references in markdown.

The round-trip property — ``parse_skill_md(render_skill_md(p)) == p`` — is the
strongest single test in this module. If it ever regresses, the runner will
start producing mystery bugs three weeks later.

Section order is STRICT and REQUIRED (except ``## Notes`` which is optional)::

    # Title

    Description paragraph.

    ## Parameters
    ## Preconditions
    ## Steps
    ## Expected outcome
    ## Notes     # optional

A hand-written parser is used on purpose — a generic markdown library
(commonmark, mistune) aggressively normalizes whitespace and would defeat
byte-stable round-tripping.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DESTRUCTIVE_MARKER",
    "Parameter",
    "ParsedSkill",
    "SkillParseError",
    "Step",
    "extract_parameter_refs",
    "parse_skill_md",
    "render_skill_md",
]


DESTRUCTIVE_MARKER = "⚠️ [DESTRUCTIVE]"
"""The exact prefix that marks a destructive step. Must agree with the
synthesizer's draft prompt and with the schema cross-check in ``schema.py``.
"""

_REQUIRED_SECTIONS: tuple[str, ...] = (
    "Parameters",
    "Preconditions",
    "Steps",
    "Expected outcome",
)
_OPTIONAL_SECTIONS: tuple[str, ...] = ("Notes",)
_ALL_SECTIONS: tuple[str, ...] = _REQUIRED_SECTIONS + _OPTIONAL_SECTIONS

_EMPTY_BULLET_MARKER = "_None._"
_PARAM_NAME_PATTERN = r"[a-z][a-z0-9_]{0,29}"

# Bullet line for ## Parameters.
#   - `name` (string, required)
#   - `name` (integer, optional, default: 42) — description text
#   - `name` (boolean, optional)
_PARAM_LINE_RE = re.compile(
    r"^- `(?P<name>" + _PARAM_NAME_PATTERN + r")` "
    r"\((?P<type>string|integer|boolean), "
    r"(?P<required>required|optional)"
    r"(?:, default: (?P<default>.+?))?\)"
    r"(?:\s*—\s*(?P<desc>.+))?$"
)

# Numbered step line under ## Steps: "1. text" or "12. text"
_STEP_LINE_RE = re.compile(r"^(?P<num>\d+)\.\s+(?P<rest>.*)$")

# Parameter reference in markdown. Requires a non-escaped `{` not preceded by
# another `{`, and a closing `}` not preceded by `\` and not followed by `}`.
_PARAM_REF_RE = re.compile(
    r"(?<!\\)(?<!\{)\{(" + _PARAM_NAME_PATTERN + r")(?<!\\)\}(?!\})"
)

_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


# --- Data models ------------------------------------------------------------


class Parameter(BaseModel):
    """A single parameter declaration in a SKILL.md."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,29}$")
    type: str = Field(pattern=r"^(string|integer|boolean)$")
    required: bool
    default: str | int | bool | None = None
    description: str | None = None


class Step(BaseModel):
    """A single numbered step in a SKILL.md."""

    model_config = ConfigDict(frozen=True)

    number: int = Field(ge=1)
    text: str
    destructive: bool = False


class ParsedSkill(BaseModel):
    """Structured representation of a SKILL.md file.

    Equality and hashing use the full field set, so round-trip tests can use
    ``==`` directly.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    description: str
    parameters: list[Parameter] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    expected_outcome: str
    notes: str | None = None


class SkillParseError(Exception):
    """Raised by :func:`parse_skill_md` when the markdown is malformed.

    Attributes mirror the PRD contract: ``section``, ``reason``, ``line``.
    The ``line`` is a 1-indexed line number into the input markdown.
    """

    def __init__(self, section: str, reason: str, line: int) -> None:
        super().__init__(f"[{section} L{line}] {reason}")
        self.section = section
        self.reason = reason
        self.line = line


# --- Parser -----------------------------------------------------------------


def parse_skill_md(markdown: str) -> ParsedSkill:
    """Parse a SKILL.md string into a :class:`ParsedSkill`.

    Raises :class:`SkillParseError` with a structured ``(section, reason, line)``
    on any malformed input.
    """
    lines = markdown.splitlines()
    cursor = 0

    # --- Title (H1) ---------------------------------------------------------
    cursor = _skip_blank(lines, cursor)
    if cursor >= len(lines):
        raise SkillParseError("title", "markdown is empty", cursor + 1)
    title_line = lines[cursor]
    if title_line.startswith("## "):
        raise SkillParseError(
            "title",
            "first heading must be an H1 ('# '), got an H2",
            cursor + 1,
        )
    if not title_line.startswith("# "):
        raise SkillParseError(
            "title",
            f"expected H1 title on the first non-blank line, got {title_line!r}",
            cursor + 1,
        )
    title = title_line[2:].strip()
    if not title:
        raise SkillParseError("title", "H1 title is empty", cursor + 1)
    cursor += 1

    # --- Description (non-empty lines between title and first ## heading) ---
    cursor = _skip_blank(lines, cursor)
    desc_start_line = cursor
    desc_lines: list[str] = []
    while cursor < len(lines) and not lines[cursor].startswith("## "):
        line = lines[cursor]
        if line.startswith("# "):
            raise SkillParseError(
                "description",
                "unexpected second H1 heading before ## Parameters",
                cursor + 1,
            )
        desc_lines.append(line.rstrip())
        cursor += 1
    while desc_lines and not desc_lines[-1].strip():
        desc_lines.pop()
    if not desc_lines:
        raise SkillParseError(
            "description",
            "missing description between the H1 title and ## Parameters",
            desc_start_line + 1,
        )
    description = "\n".join(desc_lines).strip()

    # --- Section walk -------------------------------------------------------
    # We collect section bodies keyed by header name; the enforcing of order
    # happens inline against ``expected_order``.
    expected_order: list[str] = list(_ALL_SECTIONS)
    sections: dict[str, tuple[int, list[str]]] = {}
    seen: list[str] = []

    while cursor < len(lines):
        line = lines[cursor]
        if line.startswith("## "):
            header = line[3:].strip()
            # Determine the next permitted section. Missing optional sections
            # can be skipped, but headers must arrive in the canonical order.
            remaining = [s for s in expected_order if s not in seen]
            permitted: list[str] = []
            for s in remaining:
                permitted.append(s)
                if s in _REQUIRED_SECTIONS:
                    break
            if header not in permitted:
                raise SkillParseError(
                    "section_order",
                    f"unexpected section '## {header}'; "
                    f"expected one of {permitted} next",
                    cursor + 1,
                )
            # Skip any required sections that were leapfrogged — they're now
            # permanently missing. We detect this after the walk.
            seen.append(header)
            body_start = cursor + 1
            cursor += 1
            body_lines: list[str] = []
            while cursor < len(lines) and not lines[cursor].startswith("## "):
                if lines[cursor].startswith("# "):
                    raise SkillParseError(
                        "section",
                        "unexpected H1 mid-document",
                        cursor + 1,
                    )
                body_lines.append(lines[cursor])
                cursor += 1
            sections[header] = (body_start, body_lines)
        elif line.startswith("# "):
            raise SkillParseError(
                "section",
                "unexpected H1 mid-document",
                cursor + 1,
            )
        else:
            if line.strip():
                raise SkillParseError(
                    "section",
                    f"text outside of any section: {line!r}",
                    cursor + 1,
                )
            cursor += 1

    # Required-section presence check.
    for required in _REQUIRED_SECTIONS:
        if required not in sections:
            raise SkillParseError(
                required.lower().replace(" ", "_"),
                f"missing required section '## {required}'",
                max(1, len(lines)),
            )

    param_start, param_body = sections["Parameters"]
    pre_start, pre_body = sections["Preconditions"]
    step_start, step_body = sections["Steps"]
    out_start, out_body = sections["Expected outcome"]

    parameters = _parse_parameters(param_start, param_body)
    preconditions = _parse_preconditions(pre_start, pre_body)
    steps = _parse_steps(step_start, step_body)
    expected_outcome = _parse_paragraph(
        out_start, out_body, section_name="expected_outcome", allow_empty=False
    )

    notes: str | None = None
    if "Notes" in sections:
        notes_start, notes_body = sections["Notes"]
        paragraph = _parse_paragraph(
            notes_start, notes_body, section_name="notes", allow_empty=True
        )
        notes = paragraph or None

    return ParsedSkill(
        title=title,
        description=description,
        parameters=parameters,
        preconditions=preconditions,
        steps=steps,
        expected_outcome=expected_outcome,
        notes=notes,
    )


def _skip_blank(lines: list[str], cursor: int) -> int:
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    return cursor


def _parse_parameters(start_line: int, body_lines: list[str]) -> list[Parameter]:
    params: list[Parameter] = []
    names_seen: set[str] = set()
    for offset, raw in enumerate(body_lines):
        line_no = start_line + offset + 1
        stripped = raw.strip()
        if not stripped or stripped == _EMPTY_BULLET_MARKER:
            continue
        match = _PARAM_LINE_RE.match(raw.rstrip())
        if match is None:
            raise SkillParseError(
                "parameters",
                f"malformed parameter line: {raw!r}",
                line_no,
            )
        name = match.group("name")
        if name in names_seen:
            raise SkillParseError(
                "parameters",
                f"duplicate parameter name {name!r}",
                line_no,
            )
        names_seen.add(name)
        type_ = match.group("type")
        required = match.group("required") == "required"
        default_raw = match.group("default")
        default: str | int | bool | None = None
        if default_raw is not None:
            default = _parse_default(default_raw, type_, line_no)
        params.append(
            Parameter(
                name=name,
                type=type_,
                required=required,
                default=default,
                description=match.group("desc"),
            )
        )
    return params


def _parse_default(raw: str, type_: str, line_no: int) -> str | int | bool:
    stripped = raw.strip()
    if type_ == "string":
        if len(stripped) < 2 or stripped[0] != '"' or stripped[-1] != '"':
            raise SkillParseError(
                "parameters",
                f"string default must be double-quoted: {stripped!r}",
                line_no,
            )
        return stripped[1:-1]
    if type_ == "integer":
        try:
            return int(stripped)
        except ValueError as e:
            raise SkillParseError(
                "parameters",
                f"integer default is not a valid int: {stripped!r}",
                line_no,
            ) from e
    # boolean
    if stripped == "true":
        return True
    if stripped == "false":
        return False
    raise SkillParseError(
        "parameters",
        f"boolean default must be 'true' or 'false': {stripped!r}",
        line_no,
    )


def _parse_preconditions(start_line: int, body_lines: list[str]) -> list[str]:
    out: list[str] = []
    for offset, raw in enumerate(body_lines):
        line_no = start_line + offset + 1
        stripped = raw.strip()
        if not stripped or stripped == _EMPTY_BULLET_MARKER:
            continue
        if not raw.startswith("- "):
            raise SkillParseError(
                "preconditions",
                f"precondition must be a '- ' bullet line: {raw!r}",
                line_no,
            )
        out.append(raw[2:].rstrip())
    return out


def _parse_steps(start_line: int, body_lines: list[str]) -> list[Step]:
    steps: list[Step] = []
    for offset, raw in enumerate(body_lines):
        line_no = start_line + offset + 1
        stripped = raw.strip()
        if not stripped:
            continue
        match = _STEP_LINE_RE.match(raw.rstrip())
        if match is None:
            raise SkillParseError(
                "steps",
                f"step line must start with '<n>. ': {raw!r}",
                line_no,
            )
        number = int(match.group("num"))
        rest = match.group("rest")
        destructive = False
        if rest.startswith(DESTRUCTIVE_MARKER):
            destructive = True
            rest = rest[len(DESTRUCTIVE_MARKER) :].lstrip()
        if "⚠️" in rest:
            raise SkillParseError(
                "steps",
                f"step {number} has a stray ⚠️ outside the canonical "
                f"'{DESTRUCTIVE_MARKER}' prefix: {raw!r}",
                line_no,
            )
        steps.append(Step(number=number, text=rest, destructive=destructive))

    if not steps:
        raise SkillParseError(
            "steps",
            "## Steps section has no numbered steps",
            max(1, start_line),
        )
    expected_numbers = list(range(1, len(steps) + 1))
    actual_numbers = [s.number for s in steps]
    if actual_numbers != expected_numbers:
        raise SkillParseError(
            "steps",
            "step numbers must be 1-indexed and sequential; "
            f"expected {expected_numbers}, got {actual_numbers}",
            max(1, start_line),
        )
    return steps


def _parse_paragraph(
    start_line: int,
    body_lines: list[str],
    *,
    section_name: str,
    allow_empty: bool,
) -> str:
    trimmed = [line.rstrip() for line in body_lines]
    while trimmed and not trimmed[0].strip():
        trimmed.pop(0)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    if not trimmed:
        if allow_empty:
            return ""
        raise SkillParseError(
            section_name,
            f"'## {section_name.replace('_', ' ')}' section has no content",
            max(1, start_line),
        )
    return "\n".join(trimmed).strip()


# --- Renderer ---------------------------------------------------------------


def render_skill_md(parsed: ParsedSkill) -> str:
    """Render a :class:`ParsedSkill` back to a canonical SKILL.md string.

    Output invariants:

    * one blank line between sections,
    * no trailing whitespace on any line,
    * exactly one trailing newline at end of file,
    * steps numbered 1..N, destructive prefixed with ``⚠️ [DESTRUCTIVE]``.
    """
    out: list[str] = []
    out.append(f"# {parsed.title}")
    out.append("")
    out.append(parsed.description)

    out.append("")
    out.append("## Parameters")
    out.append("")
    if parsed.parameters:
        for p in parsed.parameters:
            out.append(_render_parameter(p))
    else:
        out.append(_EMPTY_BULLET_MARKER)

    out.append("")
    out.append("## Preconditions")
    out.append("")
    if parsed.preconditions:
        for precondition in parsed.preconditions:
            out.append(f"- {precondition}")
    else:
        out.append(_EMPTY_BULLET_MARKER)

    out.append("")
    out.append("## Steps")
    out.append("")
    for step in parsed.steps:
        prefix = f"{DESTRUCTIVE_MARKER} " if step.destructive else ""
        out.append(f"{step.number}. {prefix}{step.text}")

    out.append("")
    out.append("## Expected outcome")
    out.append("")
    out.append(parsed.expected_outcome)

    if parsed.notes is not None and parsed.notes.strip():
        out.append("")
        out.append("## Notes")
        out.append("")
        out.append(parsed.notes)

    out.append("")  # trailing newline
    return "\n".join(out)


def _render_parameter(p: Parameter) -> str:
    req = "required" if p.required else "optional"
    line = f"- `{p.name}` ({p.type}, {req}"
    if p.default is not None:
        line += f", default: {_render_default(p.default, p.type)}"
    line += ")"
    if p.description:
        line += f" — {p.description}"
    return line


def _render_default(value: Any, type_: str) -> str:
    if type_ == "string":
        return f'"{value}"'
    if type_ == "integer":
        return str(int(value))
    if type_ == "boolean":
        return "true" if value else "false"
    return str(value)


# --- Parameter-ref extractor ------------------------------------------------


def extract_parameter_refs(markdown: str) -> set[str]:
    """Return the set of ``{name}`` references found in ``markdown``.

    The following are NOT counted as references:

    * tokens inside fenced code blocks (``` ... ```),
    * tokens inside inline code spans (``` `...` ```),
    * escaped braces (``\\{foo\\}``),
    * double-brace literals (``{{foo}}`` — common in template syntax).
    """
    without_fenced = _FENCED_CODE_RE.sub("", markdown)
    without_inline = _INLINE_CODE_RE.sub("", without_fenced)
    return set(_PARAM_REF_RE.findall(without_inline))
