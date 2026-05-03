"""Tests for runner.skill_loader.

Covers the three validation layers in ``load_skill`` (existence, schema,
markdown/meta cross-consistency), parameter substitution (required, optional
with default, unknown, escape), and golden-fixture round-tripping against the
5 reference workflows under ``fixtures/skills/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from runner.skill_loader import (
    LoadedSkill,
    MissingParameterError,
    SkillIntegrityError,
    SkillNotFoundError,
    UnknownParameterError,
    load_skill,
    substitute_parameters,
)

_FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "skills"
_GOLDEN_SLUGS = (
    "gmail_reply",
    "calendar_block",
    "finder_organize",
    "slack_status",
    "notes_daily",
)


# ---------- load_skill ----------


@pytest.mark.parametrize("slug", _GOLDEN_SLUGS)
def test_load_skill_accepts_every_golden_fixture(slug: str) -> None:
    loaded = load_skill(slug, _FIXTURES_ROOT)
    assert isinstance(loaded, LoadedSkill)
    assert loaded.meta["slug"] == slug
    assert loaded.skill_path == _FIXTURES_ROOT / slug
    assert len(loaded.parsed_skill.steps) >= 1
    # Every destructive step in meta must be flagged in the parsed markdown.
    destructive_in_md = {s.number for s in loaded.parsed_skill.steps if s.destructive}
    assert set(loaded.meta["destructive_steps"]) == destructive_in_md


def test_load_skill_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(SkillNotFoundError) as exc_info:
        load_skill("does_not_exist", tmp_path)
    assert "does_not_exist" in str(exc_info.value)


def _write_skill(
    tmp_path: Path,
    slug: str,
    *,
    md: str,
    meta: dict[str, Any] | None,
) -> Path:
    skill_dir = tmp_path / slug
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(md, encoding="utf-8")
    if meta is not None:
        (skill_dir / "skill.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return skill_dir


_VALID_MD = (
    "# Test Skill\n\n"
    "## Steps\n\n"
    "1. First step.\n"
    "2. ⚠️ Second step.\n"
)


def _valid_meta(slug: str = "test_skill") -> dict[str, Any]:
    # Canonical skill-meta.schema.json (post-merge) requires:
    #   slug, name, trajectory_id, created_at, parameters, destructive_steps,
    #   preconditions, step_count.
    # ``version`` was dropped; ``trajectory_ref`` was renamed to ``trajectory_id``.
    return {
        "slug": slug,
        "name": "Test Skill",
        "trajectory_id": "00000000-0000-0000-0000-000000000000",
        "created_at": "2026-04-22T00:00:00Z",
        "parameters": [],
        "destructive_steps": [2],
        "preconditions": [],
        "step_count": 2,
    }


def test_load_skill_missing_meta_file(tmp_path: Path) -> None:
    _write_skill(tmp_path, "test_skill", md=_VALID_MD, meta=None)
    with pytest.raises(SkillNotFoundError) as exc_info:
        load_skill("test_skill", tmp_path)
    assert "skill.meta.json" in str(exc_info.value)


def test_load_skill_missing_markdown_file(tmp_path: Path) -> None:
    skill_dir = tmp_path / "test_skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.meta.json").write_text(json.dumps(_valid_meta()), encoding="utf-8")
    with pytest.raises(SkillNotFoundError) as exc_info:
        load_skill("test_skill", tmp_path)
    assert "SKILL.md" in str(exc_info.value)


def test_load_skill_malformed_meta_json(tmp_path: Path) -> None:
    skill_dir = tmp_path / "test_skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_VALID_MD, encoding="utf-8")
    (skill_dir / "skill.meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SkillIntegrityError):
        load_skill("test_skill", tmp_path)


def test_load_skill_meta_missing_required_field(tmp_path: Path) -> None:
    meta = _valid_meta()
    # Was ``trajectory_ref`` pre-merge; canonical schema renames it to
    # ``trajectory_id``. Either deletion still proves the schema validator
    # fires when a required key is missing.
    del meta["trajectory_id"]
    _write_skill(tmp_path, "test_skill", md=_VALID_MD, meta=meta)
    with pytest.raises(SkillIntegrityError):
        load_skill("test_skill", tmp_path)


def test_load_skill_meta_bad_slug_pattern(tmp_path: Path) -> None:
    meta = _valid_meta(slug="Not-A-Valid-Slug")
    _write_skill(tmp_path, "test_skill", md=_VALID_MD, meta=meta)
    with pytest.raises(SkillIntegrityError):
        load_skill("test_skill", tmp_path)


def test_load_skill_rejects_meta_that_is_array(tmp_path: Path) -> None:
    skill_dir = tmp_path / "test_skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_VALID_MD, encoding="utf-8")
    (skill_dir / "skill.meta.json").write_text("[]", encoding="utf-8")
    with pytest.raises(SkillIntegrityError):
        load_skill("test_skill", tmp_path)


def test_load_skill_destructive_marker_in_md_but_not_meta(tmp_path: Path) -> None:
    # Canonical synth requires the full 4-section SKILL.md plus a description.
    # Use a fully-shaped doc so the parser succeeds; the test is about
    # catching drift between the meta's ``destructive_steps`` and the
    # ``⚠️ [DESTRUCTIVE]`` markers in the markdown — that mismatch should
    # still bubble up as a SkillIntegrityError (now via the canonical
    # parser's drift check rather than runner's old hand-rolled one).
    md = (
        "# Test Skill\n\n"
        "A test skill used by runner unit tests.\n\n"
        "## Parameters\n\n_None._\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Steps\n\n"
        "1. First step.\n"
        "2. ⚠️ [DESTRUCTIVE] Second step (marked in md only).\n\n"
        "## Expected outcome\n\nDone.\n"
    )
    meta = _valid_meta()
    meta["destructive_steps"] = []  # drift: md says 2 is destructive, meta says none
    _write_skill(tmp_path, "test_skill", md=md, meta=meta)
    with pytest.raises(SkillIntegrityError) as exc_info:
        load_skill("test_skill", tmp_path)
    msg = str(exc_info.value).lower()
    # Either the historical "disagree" wording or any reference to
    # destructive-step drift / the marker character is acceptable.
    assert "disagree" in msg or "destructive" in msg or "⚠" in str(exc_info.value)


def test_load_skill_destructive_in_meta_but_not_md(tmp_path: Path) -> None:
    md = (
        "# Test Skill\n\n"
        "## Steps\n\n"
        "1. First step.\n"
        "2. Second step (no marker).\n"
    )
    meta = _valid_meta()
    meta["destructive_steps"] = [2]  # drift: meta says 2 is destructive, md doesn't
    _write_skill(tmp_path, "test_skill", md=md, meta=meta)
    with pytest.raises(SkillIntegrityError):
        load_skill("test_skill", tmp_path)


def test_load_skill_markdown_missing_title(tmp_path: Path) -> None:
    md = "## Steps\n\n1. First step.\n"
    _write_skill(tmp_path, "test_skill", md=md, meta=_valid_meta() | {"destructive_steps": []})
    with pytest.raises(SkillIntegrityError):
        load_skill("test_skill", tmp_path)


def test_load_skill_markdown_missing_steps(tmp_path: Path) -> None:
    md = "# Title\n\nSome prose but no steps section.\n"
    meta = _valid_meta()
    meta["destructive_steps"] = []
    _write_skill(tmp_path, "test_skill", md=md, meta=meta)
    with pytest.raises(SkillIntegrityError):
        load_skill("test_skill", tmp_path)


def test_load_skill_markdown_non_contiguous_numbering(tmp_path: Path) -> None:
    md = (
        "# Test\n\n"
        "## Steps\n\n"
        "1. First.\n"
        "3. Third (gap).\n"
    )
    meta = _valid_meta()
    meta["destructive_steps"] = []
    _write_skill(tmp_path, "test_skill", md=md, meta=meta)
    with pytest.raises(SkillIntegrityError):
        load_skill("test_skill", tmp_path)


# ---------- substitute_parameters ----------


def _render_param_bullet(p: dict[str, Any]) -> str:
    """Format one parameter as canonical synth's `## Parameters` bullet.

    Bullet format (from synth.skill_doc._PARAM_LINE_RE):
        - `name` (type, required|optional[, default: ...]) [— description]

    Synth's default-parser is strict: string defaults must be double-quoted,
    integer/boolean defaults are bare. JSON-encode by type so both shapes
    round-trip cleanly.
    """
    name = p["name"]
    type_ = p.get("type", "string")
    # Canonical synth's parameter type enum is {string, integer, boolean};
    # the pre-merge schema accepted ``number``. Map it to the canonical type
    # for parser compatibility.
    if type_ == "number":
        type_ = "integer"
    required = "required"
    default_clause = ""
    if "default" in p and p["default"] is not None:
        required = "optional"
        raw_default = p["default"]
        if type_ == "string":
            default_str = f'"{raw_default}"'
        elif type_ == "boolean":
            default_str = "true" if raw_default else "false"
        else:
            default_str = str(raw_default)
        default_clause = f", default: {default_str}"
    elif p.get("required") is False:
        required = "optional"
    return f"- `{name}` ({type_}, {required}{default_clause})"


def _extract_title_and_steps_block(md: str) -> tuple[str, str]:
    """Return ``(title_line, steps_body)`` extracted from a pre-merge
    minimal SKILL.md.

    The legacy fixtures look like::

        # Title

        ## Steps

        1. step one
        2. step two

    Anything outside the title and the Steps body is discarded. Bare
    ``⚠️`` markers are normalised to canonical ``⚠️ [DESTRUCTIVE]`` so
    the strict parser accepts them.
    """
    import re

    title = "# Inline Test Skill"
    steps_body = ""
    if md.startswith("# "):
        first_newline = md.find("\n")
        if first_newline != -1:
            title = md[:first_newline].strip()
    if "## Steps" in md:
        _, _, after = md.partition("## Steps")
        # Trim until the next H2 or EOF.
        next_h2 = after.find("\n## ")
        steps_body = after[:next_h2].strip() if next_h2 != -1 else after.strip()
    # Normalise the legacy bare-marker form. Match ``\d+. ⚠️ `` not already
    # followed by ``[DESTRUCTIVE]``.
    steps_body = re.sub(
        r"(\d+\. )⚠️ (?!\[DESTRUCTIVE\])",
        r"\1⚠️ [DESTRUCTIVE] ",
        steps_body,
    )
    return title, steps_body


def _inflate_to_canonical_md(md: str, parameters: list[dict[str, Any]]) -> str:
    """Build a fully-compliant SKILL.md around the test's steps body."""
    title, steps_body = _extract_title_and_steps_block(md)
    if parameters:
        params_section = "\n".join(_render_param_bullet(p) for p in parameters)
    else:
        params_section = "_None._"
    return (
        f"{title}\n\n"
        "A test skill used by runner unit tests.\n\n"
        "## Parameters\n\n"
        f"{params_section}\n\n"
        "## Preconditions\n\n"
        "_None._\n\n"
        "## Steps\n\n"
        f"{steps_body}\n\n"
        "## Expected outcome\n\n"
        "Test passes.\n"
    )


def _skill_with_params(parameters: list[dict[str, Any]], md: str) -> LoadedSkill:
    # Build a LoadedSkill in-memory without touching disk.
    from synthesizer.skill_doc import parse_skill_md

    # Canonical synth's SKILL.md parser is strict: it requires a description
    # paragraph and four named sections (Parameters, Preconditions, Steps,
    # Expected outcome). Runner's tests only care about the Steps body; they
    # used to pass minimalist markdown that the pre-merge synth accepted.
    # Synthesize a fully-compliant document by extracting the title + steps
    # from the supplied ``md`` and rendering the surrounding sections from
    # the parameters list.
    md = _inflate_to_canonical_md(md, parameters)
    parsed = parse_skill_md(md)
    meta: dict[str, Any] = {
        "slug": "inline_test",
        "name": "Inline Test Skill",
        "trajectory_id": "00000000-0000-0000-0000-000000000000",
        "created_at": "2026-04-22T00:00:00Z",
        "parameters": parameters,
        "destructive_steps": [s.number for s in parsed.steps if s.destructive],
        "preconditions": [],
        "step_count": len(parsed.steps),
    }
    return LoadedSkill(parsed_skill=parsed, meta=meta, skill_path=Path("/tmp/inline"))


def test_substitute_replaces_all_references() -> None:
    skill = _skill_with_params(
        parameters=[
            {"name": "sender", "type": "string"},
            {"name": "template", "type": "string"},
        ],
        md=(
            "# Gmail Reply\n\n## Steps\n\n"
            "1. Search for {sender}.\n"
            "2. Type {template} into compose.\n"
        ),
    )
    out = substitute_parameters(
        skill, {"sender": "alice@example.com", "template": "Thanks!"}
    )
    texts = [s.text for s in out.parsed_skill.steps]
    assert texts == [
        "Search for alice@example.com.",
        "Type Thanks! into compose.",
    ]
    for text in texts:
        assert "{" not in text or "\\{" in text  # no unsubstituted refs


def test_substitute_missing_required_parameter() -> None:
    skill = _skill_with_params(
        parameters=[{"name": "sender", "type": "string"}],
        md="# T\n\n## Steps\n\n1. Search for {sender}.\n",
    )
    with pytest.raises(MissingParameterError) as exc_info:
        substitute_parameters(skill, {})
    assert exc_info.value.parameter_name == "sender"
    assert "sender" in str(exc_info.value)


def test_substitute_unknown_parameter() -> None:
    skill = _skill_with_params(
        parameters=[{"name": "sender", "type": "string"}],
        md="# T\n\n## Steps\n\n1. Search for {sender}.\n",
    )
    with pytest.raises(UnknownParameterError) as exc_info:
        substitute_parameters(skill, {"sender": "a", "bogus": "b"})
    assert exc_info.value.parameter_name == "bogus"


def test_substitute_uses_defaults_for_optional_params() -> None:
    skill = _skill_with_params(
        parameters=[
            {"name": "emoji", "type": "string", "default": ":dart:"},
            {"name": "message", "type": "string", "default": "heads down"},
        ],
        md=(
            "# Slack\n\n## Steps\n\n"
            "1. Set {emoji} {message}.\n"
        ),
    )
    out = substitute_parameters(skill, {})
    assert out.parsed_skill.steps[0].text == "Set :dart: heads down."


def test_substitute_override_optional_default() -> None:
    skill = _skill_with_params(
        parameters=[{"name": "n", "type": "number", "default": 7}],
        md="# T\n\n## Steps\n\n1. Use {n} days.\n",
    )
    out = substitute_parameters(skill, {"n": "14"})
    assert out.parsed_skill.steps[0].text == "Use 14 days."


def test_substitute_coerces_default_number_to_string() -> None:
    skill = _skill_with_params(
        parameters=[{"name": "n", "type": "number", "default": 30}],
        md="# T\n\n## Steps\n\n1. Use {n} minutes.\n",
    )
    out = substitute_parameters(skill, {})
    assert out.parsed_skill.steps[0].text == "Use 30 minutes."


def test_substitute_escaped_braces_are_preserved() -> None:
    skill = _skill_with_params(
        parameters=[{"name": "name", "type": "string"}],
        md=(
            "# T\n\n## Steps\n\n"
            "1. Greet {name}, use literal \\{literal} and \\{name} too.\n"
        ),
    )
    out = substitute_parameters(skill, {"name": "Ada"})
    assert out.parsed_skill.steps[0].text == (
        "Greet Ada, use literal {literal} and {name} too."
    )


def test_substitute_preserves_step_numbers_and_destructive_flags() -> None:
    skill = _skill_with_params(
        parameters=[{"name": "x", "type": "string"}],
        md=(
            "# T\n\n## Steps\n\n"
            "1. Do {x}.\n"
            "2. ⚠️ Confirm {x}.\n"
        ),
    )
    out = substitute_parameters(skill, {"x": "thing"})
    assert [s.number for s in out.parsed_skill.steps] == [1, 2]
    assert [s.destructive for s in out.parsed_skill.steps] == [False, True]
    assert out.meta is skill.meta  # meta is shared (immutable dict pointer)


def test_substitute_default_null_counts_as_required() -> None:
    # A meta entry with ``"default": null`` is treated as NOT having a default
    # so that JSON round-tripping from tools that emit null cannot silently
    # change required-ness.
    skill = _skill_with_params(
        parameters=[{"name": "x", "type": "string", "default": None}],
        md="# T\n\n## Steps\n\n1. Use {x}.\n",
    )
    with pytest.raises(MissingParameterError):
        substitute_parameters(skill, {})


def test_substitute_on_golden_fixture_with_defaults() -> None:
    # calendar_block and slack_status have defaults for every parameter.
    loaded = load_skill("calendar_block", _FIXTURES_ROOT)
    out = substitute_parameters(loaded, {})
    for step in out.parsed_skill.steps:
        assert "{start_time}" not in step.text
        assert "{duration_minutes}" not in step.text


def test_substitute_golden_fixture_all_params_supplied() -> None:
    # Canonical gmail_reply fixture declares ``recipient_name`` and
    # ``reply_body`` (required) plus ``reply_subject_prefix`` (optional, default
    # ``"Re:"``). Pre-merge it had ``sender``/``template``.
    loaded = load_skill("gmail_reply", _FIXTURES_ROOT)
    out = substitute_parameters(
        loaded,
        {"recipient_name": "Alice", "reply_body": "Thanks!"},
    )
    for step in out.parsed_skill.steps:
        assert "{recipient_name}" not in step.text
        assert "{reply_body}" not in step.text
