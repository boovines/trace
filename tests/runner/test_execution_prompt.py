"""Tests for :mod:`runner.execution_prompt`."""

from __future__ import annotations

from pathlib import Path

import pytest

from runner.execution_prompt import (
    COMPLETION_PROTOCOL,
    DESTRUCTIVE_PROTOCOL,
    DRY_RUN_NOTICE,
    ERROR_PROTOCOL,
    VERIFICATION_PROTOCOL,
    build_execution_prompt,
)
from runner.skill_loader import LoadedSkill, load_skill, substitute_parameters

FIXTURES_ROOT: Path = Path(__file__).resolve().parents[2] / "fixtures" / "skills"

ALL_SLUGS = (
    "gmail_reply",
    "calendar_block",
    "finder_organize",
    "slack_status",
    "notes_daily",
)

# Canonical gmail_reply fixture (post-merge) declares ``recipient_name`` /
# ``reply_body`` / ``reply_subject_prefix``. Pre-merge it had
# ``sender``/``template``.
GMAIL_PARAMS: dict[str, str] = {
    "recipient_name": "Alice",
    "reply_body": "Got it — will reply by EOD.",
}

# notes_daily now declares ``note_template`` as a required (no-default) param.
# Other notes_daily params have defaults so we don't need to supply them.
NOTES_PARAMS: dict[str, str] = {"note_template": "- [ ] focus block\n"}

_PER_SLUG_PARAMS: dict[str, dict[str, str]] = {
    "gmail_reply": GMAIL_PARAMS,
    "notes_daily": NOTES_PARAMS,
}


def _resolved(slug: str) -> LoadedSkill:
    skill = load_skill(slug, FIXTURES_ROOT)
    return substitute_parameters(skill, _PER_SLUG_PARAMS.get(slug, {}))


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_prompt_contains_all_sections(slug: str) -> None:
    prompt = build_execution_prompt(_resolved(slug), "dry_run")

    assert "## Workflow" in prompt
    assert "## Verification protocol" in prompt
    assert "## Destructive-step protocol" in prompt
    assert "## Completion protocol" in prompt
    assert "## Error protocol" in prompt
    assert "```markdown" in prompt

    # Each protocol string appears verbatim.
    assert VERIFICATION_PROTOCOL in prompt
    assert DESTRUCTIVE_PROTOCOL in prompt
    assert COMPLETION_PROTOCOL in prompt
    assert ERROR_PROTOCOL in prompt


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_prompt_includes_skill_title_and_steps(slug: str) -> None:
    skill = _resolved(slug)
    prompt = build_execution_prompt(skill, "dry_run")

    assert f"# {skill.parsed_skill.title}" in prompt
    for step in skill.parsed_skill.steps:
        # Step number + text must be present (regardless of destructive
        # marker position).
        assert f"{step.number}. " in prompt
        assert step.text in prompt


def test_destructive_step_rendered_with_marker() -> None:
    prompt = build_execution_prompt(_resolved("gmail_reply"), "dry_run")
    # Canonical gmail_reply marks step 5 (Send) as destructive; pre-merge
    # fixture had a different step layout.
    assert "5. ⚠️ [DESTRUCTIVE] " in prompt


def test_non_destructive_skill_has_no_marker() -> None:
    prompt = build_execution_prompt(_resolved("notes_daily"), "dry_run")
    # notes_daily has no destructive steps, so the marker must not appear in
    # the rendered markdown (the protocol text *does* include it — we only
    # check the fenced workflow block).
    workflow_start = prompt.index("```markdown")
    workflow_end = prompt.index("```", workflow_start + len("```markdown"))
    workflow = prompt[workflow_start:workflow_end]
    assert "⚠️" not in workflow
    assert "[DESTRUCTIVE]" not in workflow


def test_parameters_substituted_in_prompt() -> None:
    prompt = build_execution_prompt(_resolved("gmail_reply"), "dry_run")
    assert "Alice" in prompt
    assert "Got it — will reply by EOD." in prompt
    # Raw placeholders should no longer appear.
    assert "{recipient_name}" not in prompt
    assert "{reply_body}" not in prompt


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_prompt_is_deterministic(slug: str) -> None:
    skill_a = _resolved(slug)
    skill_b = _resolved(slug)
    assert build_execution_prompt(skill_a, "dry_run") == build_execution_prompt(
        skill_b, "dry_run"
    )
    assert build_execution_prompt(skill_a, "live") == build_execution_prompt(
        skill_b, "live"
    )


def test_dry_run_notice_present_only_in_dry_run() -> None:
    skill = _resolved("gmail_reply")
    dry = build_execution_prompt(skill, "dry_run")
    live = build_execution_prompt(skill, "live")

    assert DRY_RUN_NOTICE in dry
    assert DRY_RUN_NOTICE not in live


def test_dry_run_and_live_share_protocol_sections() -> None:
    skill = _resolved("gmail_reply")
    dry = build_execution_prompt(skill, "dry_run")
    live = build_execution_prompt(skill, "live")

    for section in (
        DESTRUCTIVE_PROTOCOL,
        COMPLETION_PROTOCOL,
        ERROR_PROTOCOL,
        VERIFICATION_PROTOCOL,
    ):
        assert section in dry
        assert section in live


def test_destructive_protocol_has_exact_tag_literal() -> None:
    # The parser (X-012) matches this tag literally. If the wording drifts
    # and the model emits a different tag, destructive steps will slip past
    # the prompt-layer gate. Pin the exact string.
    assert '<needs_confirmation step="N"/>' in DESTRUCTIVE_PROTOCOL


def test_completion_protocol_tag() -> None:
    assert "<workflow_complete/>" in COMPLETION_PROTOCOL


def test_error_protocol_tag() -> None:
    assert '<workflow_failed reason="..."/>' in ERROR_PROTOCOL


def test_invalid_mode_rejected() -> None:
    skill = _resolved("gmail_reply")
    with pytest.raises(ValueError, match="mode must be"):
        build_execution_prompt(skill, "nope")  # type: ignore[arg-type]


def test_prompt_ends_with_newline() -> None:
    prompt = build_execution_prompt(_resolved("gmail_reply"), "dry_run")
    assert prompt.endswith("\n")


def test_prompt_is_pure_text() -> None:
    # Should be a plain string (not bytes, not a list).
    prompt = build_execution_prompt(_resolved("gmail_reply"), "live")
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_substitute_then_build_uses_resolved_text() -> None:
    raw = load_skill("gmail_reply", FIXTURES_ROOT)
    # Without substitution the prompt still builds but contains placeholders.
    prompt_raw = build_execution_prompt(raw, "dry_run")
    assert "{recipient_name}" in prompt_raw
    # After substitution they are gone.
    prompt_resolved = build_execution_prompt(
        substitute_parameters(raw, GMAIL_PARAMS), "dry_run"
    )
    assert "{recipient_name}" not in prompt_resolved
    assert "Alice" in prompt_resolved
