"""Load a skill from disk and substitute user parameters into step text.

The runner consumes skills produced by the synthesizer:

    <skills_root>/<slug>/
    ├── SKILL.md           ← parsed via synthesizer.skill_doc
    └── skill.meta.json    ← validated against contracts/skill-meta.schema.json

``load_skill`` applies three layers of validation before returning:

1. Directory and file existence (``SkillNotFoundError``).
2. ``skill.meta.json`` passes the JSON schema (``SkillIntegrityError``).
3. ``meta.destructive_steps`` matches the ⚠️ markers in SKILL.md
   (``SkillIntegrityError``) — any drift here would undermine the
   destructive-step gate.

``substitute_parameters`` then folds a caller-supplied ``{name: value}`` map
into step text. Required vs optional is determined by the presence of a
``default`` in ``meta.parameters``. Unknown parameters are rejected loudly so
callers cannot silently pass typos.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JSONSchemaValidationError

# Synth's API drifted between when the runner branch was written and the
# canonical synth merged into main: ``SkillStep`` was renamed to ``Step``,
# ``SkillMarkdownError`` was renamed to ``SkillParseError``,
# ``SkillMetaMismatchError`` collapsed into ``jsonschema.ValidationError``,
# the helper ``replace_step_text`` went away in favour of pydantic v2's
# native ``model_copy(update=...)``, and ``validate_meta_against_markdown``
# now takes a markdown string instead of a ``ParsedSkill``. Map back to
# this module's local names so the rest of the file reads unchanged.
from synthesizer.schema import (
    ValidationError as SkillMetaMismatchError,
)
from synthesizer.schema import (
    validate_meta_against_markdown as _validate_meta_against_markdown_str,
)
from synthesizer.skill_doc import (
    ParsedSkill,
    parse_skill_md,
    render_skill_md,
)
from synthesizer.skill_doc import SkillParseError as SkillMarkdownError
from synthesizer.skill_doc import Step as SkillStep


def replace_step_text(step: SkillStep, new_text: str) -> SkillStep:
    """Return a copy of ``step`` with ``text`` replaced.

    Thin wrapper over pydantic v2's ``model_copy`` to keep the call sites
    in this file readable. The canonical synth ``Step`` is frozen.
    """
    return step.model_copy(update={"text": new_text})


def validate_meta_against_markdown(meta: dict[str, Any], parsed: ParsedSkill) -> None:
    """Adapter so callers can pass a parsed skill directly.

    Canonical synth's validator expects the raw markdown; we render the
    parsed skill back so the caller can keep its ``parsed_skill`` reference.
    """
    _validate_meta_against_markdown_str(meta, render_skill_md(parsed))

_SKILL_META_SCHEMA_PATH: Final[Path] = (
    Path(__file__).resolve().parents[4] / "contracts" / "skill-meta.schema.json"
)

# Sentinel used to temporarily stand in for escaped ``\{`` / ``\}`` during
# substitution. Chosen as a pair of ASCII control codes that cannot appear in
# a SKILL.md step text we control.
_ESCAPED_LBRACE: Final[str] = "\x00TRACE_LBRACE\x00"
_ESCAPED_RBRACE: Final[str] = "\x00TRACE_RBRACE\x00"

_PARAM_REF_RE: Final[re.Pattern[str]] = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


class SkillNotFoundError(FileNotFoundError):
    """Raised when the skill directory (or a required file) is missing."""


class SkillIntegrityError(ValueError):
    """Raised when skill.meta.json or SKILL.md fail validation."""


class MissingParameterError(ValueError):
    """Raised when a required parameter has no supplied value."""

    def __init__(self, parameter_name: str) -> None:
        super().__init__(
            f"Missing required parameter: {parameter_name!r}"
        )
        self.parameter_name = parameter_name


class UnknownParameterError(ValueError):
    """Raised when the caller supplies a parameter not declared in meta."""

    def __init__(self, parameter_name: str) -> None:
        super().__init__(
            f"Unknown parameter: {parameter_name!r}"
        )
        self.parameter_name = parameter_name


@dataclass(frozen=True)
class LoadedSkill:
    """A validated skill ready to be run.

    ``meta`` is kept as a raw dict rather than a model because the runner only
    uses a handful of fields (``parameters``, ``destructive_steps``, ``slug``)
    and the schema already guards the shape.
    """

    parsed_skill: ParsedSkill
    meta: dict[str, Any]
    skill_path: Path


@lru_cache(maxsize=1)
def _skill_meta_schema() -> dict[str, Any]:
    text = _SKILL_META_SCHEMA_PATH.read_text(encoding="utf-8")
    schema: dict[str, Any] = json.loads(text)
    return schema


@lru_cache(maxsize=1)
def _skill_meta_validator() -> Draft202012Validator:
    return Draft202012Validator(
        _skill_meta_schema(),
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )


def load_skill(slug: str, skills_root: Path) -> LoadedSkill:
    """Load, parse, and validate the skill at ``skills_root/<slug>/``.

    Raises:
        SkillNotFoundError: directory or a required sibling file is missing.
        SkillIntegrityError: meta fails JSON schema OR markdown+meta disagree.
    """
    skill_path = skills_root / slug
    if not skill_path.is_dir():
        raise SkillNotFoundError(f"Skill directory not found: {skill_path}")

    md_path = skill_path / "SKILL.md"
    meta_path = skill_path / "skill.meta.json"
    if not md_path.is_file():
        raise SkillNotFoundError(f"SKILL.md missing at {md_path}")
    if not meta_path.is_file():
        raise SkillNotFoundError(f"skill.meta.json missing at {meta_path}")

    try:
        meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SkillIntegrityError(
            f"skill.meta.json is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(meta_raw, dict):
        raise SkillIntegrityError(
            "skill.meta.json must be a JSON object at the top level"
        )
    meta: dict[str, Any] = meta_raw

    try:
        _skill_meta_validator().validate(meta)
    except JSONSchemaValidationError as exc:
        raise SkillIntegrityError(
            f"skill.meta.json failed schema validation: {exc.message}"
        ) from exc

    try:
        parsed = parse_skill_md(md_path.read_text(encoding="utf-8"))
    except SkillMarkdownError as exc:
        raise SkillIntegrityError(f"SKILL.md parse error: {exc}") from exc

    try:
        validate_meta_against_markdown(meta, parsed)
    except SkillMetaMismatchError as exc:
        raise SkillIntegrityError(
            f"skill.meta.json and SKILL.md disagree: {exc}"
        ) from exc

    return LoadedSkill(parsed_skill=parsed, meta=meta, skill_path=skill_path)


def _declared_parameters(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in meta.get("parameters", []) or []:
        # JSON schema already guarantees ``name`` exists and is a string.
        out[str(entry["name"])] = entry
    return out


def _substitute_one(text: str, values: dict[str, str]) -> str:
    """Substitute ``{name}`` references in ``text`` using ``values``.

    Escaped braces (``\\{`` / ``\\}``) are preserved as literal ``{``/``}``.
    Unknown references (declared neither with a default nor supplied) would
    have been caught before this helper runs, so any missed substitution here
    is a programmer error.
    """
    escaped = text.replace(r"\{", _ESCAPED_LBRACE).replace(r"\}", _ESCAPED_RBRACE)

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in values:
            return values[name]
        # Leave unknown placeholders untouched — the caller has already been
        # told they supplied no value for this, so preserving the reference
        # makes downstream failures easier to debug.
        return match.group(0)

    substituted = _PARAM_REF_RE.sub(_replace, escaped)
    return substituted.replace(_ESCAPED_LBRACE, "{").replace(_ESCAPED_RBRACE, "}")


def substitute_parameters(
    skill: LoadedSkill, params: dict[str, str]
) -> LoadedSkill:
    """Return a new ``LoadedSkill`` with ``{param}`` references filled in.

    Raises:
        UnknownParameterError: a key in ``params`` is not declared in meta.
        MissingParameterError: a declared parameter without a default was
            not supplied.

    Defaults come from ``meta.parameters[*].default`` — a parameter is
    "required" iff it has no ``default`` key at all (``default: null`` is
    treated as "no default", matching JSON intuition).
    """
    declared = _declared_parameters(skill.meta)

    for supplied_name in params:
        if supplied_name not in declared:
            raise UnknownParameterError(supplied_name)

    resolved: dict[str, str] = {}
    for name, spec in declared.items():
        if name in params:
            resolved[name] = params[name]
            continue
        if "default" in spec and spec["default"] is not None:
            resolved[name] = str(spec["default"])
            continue
        raise MissingParameterError(name)

    new_steps: list[SkillStep] = [
        replace_step_text(step, _substitute_one(step.text, resolved))
        for step in skill.parsed_skill.steps
    ]
    # Canonical ``ParsedSkill.steps`` is ``list[Step]`` (was ``tuple`` in the
    # earlier shape this branch was written against), and ``ParsedSkill`` is
    # a pydantic BaseModel — use ``model_copy`` instead of ``dataclasses.replace``.
    new_parsed = skill.parsed_skill.model_copy(update={"steps": new_steps})
    return replace(skill, parsed_skill=new_parsed)


__all__ = [
    "LoadedSkill",
    "MissingParameterError",
    "SkillIntegrityError",
    "SkillNotFoundError",
    "UnknownParameterError",
    "load_skill",
    "substitute_parameters",
]
