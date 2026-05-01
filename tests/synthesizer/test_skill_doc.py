"""Tests for the SKILL.md parser, renderer, and parameter-ref extractor.

The single most important guarantee this module provides is the round-trip
property::

    parse_skill_md(render_skill_md(p)) == p

If that ever fails, the Runner will start reporting mystery bugs days later.
"""

from __future__ import annotations

import pytest

from synthesizer.skill_doc import (
    DESTRUCTIVE_MARKER,
    Parameter,
    ParsedSkill,
    SkillParseError,
    Step,
    extract_parameter_refs,
    parse_skill_md,
    render_skill_md,
)

# --- Reference ParsedSkill examples -----------------------------------------
#
# These five skills mirror the five reference workflows in CLAUDE.md. They are
# hand-authored here (not loaded from fixtures/skills/) so that S-003 does not
# take a dependency on S-016's on-disk golden fixtures.


def _gmail_reply() -> ParsedSkill:
    return ParsedSkill(
        title="Gmail Reply",
        description="Reply to the most recent unread email from a sender with a template.",
        parameters=[
            Parameter(
                name="sender_email",
                type="string",
                required=True,
                description="The sender to find the most recent unread email from.",
            ),
            Parameter(
                name="reply_template",
                type="string",
                required=False,
                default="Thanks — will get back to you shortly.",
                description="The body of the reply.",
            ),
        ],
        preconditions=[
            "Google Chrome is the default browser.",
            "The Gmail account is already signed in.",
        ],
        steps=[
            Step(number=1, text="Open Chrome and navigate to gmail.com."),
            Step(
                number=2,
                text="Find the most recent unread email from {sender_email}.",
            ),
            Step(number=3, text="Click Reply and paste {reply_template}."),
            Step(
                number=4,
                text="Click the Send button.",
                destructive=True,
            ),
        ],
        expected_outcome="The reply appears in Sent Mail within a few seconds.",
        notes="If the inbox is empty the skill exits without sending.",
    )


def _calendar_block() -> ParsedSkill:
    return ParsedSkill(
        title="Calendar Focus Block",
        description="Create a focus block on Google Calendar for tomorrow afternoon.",
        parameters=[
            Parameter(
                name="duration_minutes",
                type="integer",
                required=False,
                default=30,
                description="The length of the focus block.",
            ),
        ],
        preconditions=["Google Calendar is accessible in Chrome."],
        steps=[
            Step(number=1, text="Open Chrome and navigate to calendar.google.com."),
            Step(number=2, text="Click the Create button."),
            Step(
                number=3,
                text="Set duration to {duration_minutes} and title to Focus Block.",
            ),
            Step(number=4, text="Click Save."),
        ],
        expected_outcome="A focus block of the requested duration is created.",
    )


def _finder_organize() -> ParsedSkill:
    return ParsedSkill(
        title="Finder Organize PDFs",
        description="Move PDFs older than a threshold from Downloads to an archive.",
        parameters=[
            Parameter(
                name="age_days",
                type="integer",
                required=False,
                default=7,
            ),
        ],
        preconditions=["The ~/Documents/Archive directory exists."],
        steps=[
            Step(number=1, text="Open Finder and navigate to ~/Downloads."),
            Step(
                number=2,
                text="Filter for PDF files older than {age_days} days.",
            ),
            Step(
                number=3,
                text="Move the filtered files to ~/Documents/Archive.",
                destructive=True,
            ),
        ],
        expected_outcome="Old PDFs live under ~/Documents/Archive and Downloads is tidy.",
    )


def _slack_status() -> ParsedSkill:
    return ParsedSkill(
        title="Slack Focus Status",
        description="Set the Slack status to a focus-mode emoji with a short expiry.",
        parameters=[
            Parameter(
                name="status_text",
                type="string",
                required=False,
                default="heads down",
            ),
            Parameter(
                name="clear_after_hours",
                type="integer",
                required=False,
                default=2,
            ),
        ],
        preconditions=["Slack desktop is running and logged in."],
        steps=[
            Step(number=1, text="Open Slack and click the profile avatar."),
            Step(number=2, text="Choose Update status."),
            Step(
                number=3,
                text="Set emoji to 🎯 and text to {status_text}.",
            ),
            Step(
                number=4,
                text="Set clear-after to {clear_after_hours} hours.",
            ),
            Step(number=5, text="Click Save.", destructive=True),
        ],
        expected_outcome="The new status is visible on the user's profile.",
        notes="The status auto-clears at the configured hour.",
    )


def _notes_daily() -> ParsedSkill:
    return ParsedSkill(
        title="Notes Daily Journal",
        description="Create a new Apple Notes entry titled with today's date.",
        parameters=[],
        preconditions=["Apple Notes is installed on the machine."],
        steps=[
            Step(number=1, text="Open Apple Notes."),
            Step(number=2, text="Create a new note."),
            Step(
                number=3,
                text="Paste the daily template into the body.",
            ),
        ],
        expected_outcome="A new note exists with today's date as the title.",
    )


ALL_REFERENCE_SKILLS: list[ParsedSkill] = [
    _gmail_reply(),
    _calendar_block(),
    _finder_organize(),
    _slack_status(),
    _notes_daily(),
]


# --- Round-trip -------------------------------------------------------------


@pytest.mark.parametrize("skill", ALL_REFERENCE_SKILLS, ids=lambda s: s.title)
def test_round_trip(skill: ParsedSkill) -> None:
    rendered = render_skill_md(skill)
    parsed = parse_skill_md(rendered)
    assert parsed == skill


def test_render_output_is_stable_byte_for_byte() -> None:
    skill = _gmail_reply()
    first = render_skill_md(skill)
    second = render_skill_md(skill)
    assert first == second


def test_render_has_no_trailing_whitespace_on_any_line() -> None:
    for skill in ALL_REFERENCE_SKILLS:
        rendered = render_skill_md(skill)
        for lineno, line in enumerate(rendered.splitlines(), start=1):
            assert line == line.rstrip(), (
                f"trailing whitespace on line {lineno} of rendered {skill.title!r}"
            )


def test_render_ends_with_exactly_one_newline() -> None:
    rendered = render_skill_md(_gmail_reply())
    assert rendered.endswith("\n")
    assert not rendered.endswith("\n\n")


def test_round_trip_empty_parameters_and_notes() -> None:
    skill = ParsedSkill(
        title="Minimal",
        description="Smallest legal skill.",
        parameters=[],
        preconditions=[],
        steps=[Step(number=1, text="Do the thing.")],
        expected_outcome="The thing is done.",
        notes=None,
    )
    parsed = parse_skill_md(render_skill_md(skill))
    assert parsed == skill


def test_destructive_marker_round_trip() -> None:
    skill = ParsedSkill(
        title="Destructive",
        description="All steps are destructive.",
        parameters=[],
        preconditions=[],
        steps=[
            Step(number=1, text="Delete the file.", destructive=True),
            Step(number=2, text="Confirm deletion.", destructive=True),
        ],
        expected_outcome="The file is gone.",
    )
    rendered = render_skill_md(skill)
    assert f"1. {DESTRUCTIVE_MARKER} Delete the file." in rendered
    assert f"2. {DESTRUCTIVE_MARKER} Confirm deletion." in rendered
    assert parse_skill_md(rendered) == skill


# --- Failure-mode tests -----------------------------------------------------


def test_missing_h1_title() -> None:
    md = "Nothing here.\n\n## Parameters\n\n_None._\n"
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "title"


def test_h2_before_h1_is_rejected() -> None:
    md = "## Parameters\n\n_None._\n"
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "title"


def test_empty_title_string_rejected() -> None:
    md = "# \n\nDesc.\n\n## Parameters\n\n_None._\n"
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "title"


def test_missing_description_raises() -> None:
    md = "# Title\n\n## Parameters\n\n_None._\n\n## Preconditions\n\n_None._\n"
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "description"


def test_missing_parameters_section() -> None:
    md = (
        "# Title\n\nDesc.\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Steps\n\n1. Do thing.\n\n"
        "## Expected outcome\n\nDone.\n"
    )
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "section_order"


def test_missing_steps_section() -> None:
    md = (
        "# Title\n\nDesc.\n\n"
        "## Parameters\n\n_None._\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Expected outcome\n\nDone.\n"
    )
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    # Either section_order (steps header missing so Expected comes "too early")
    # is acceptable here.
    assert excinfo.value.section in {"section_order", "steps"}


def test_missing_expected_outcome_section() -> None:
    md = (
        "# Title\n\nDesc.\n\n"
        "## Parameters\n\n_None._\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Steps\n\n1. Do thing.\n"
    )
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "expected_outcome"


def test_sections_out_of_order() -> None:
    md = (
        "# Title\n\nDesc.\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Parameters\n\n_None._\n\n"
        "## Steps\n\n1. Do.\n\n"
        "## Expected outcome\n\nDone.\n"
    )
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "section_order"


def test_steps_not_one_indexed() -> None:
    md = (
        "# Title\n\nDesc.\n\n"
        "## Parameters\n\n_None._\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Steps\n\n2. Do first.\n3. Do second.\n\n"
        "## Expected outcome\n\nDone.\n"
    )
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "steps"


def test_steps_non_sequential() -> None:
    md = (
        "# Title\n\nDesc.\n\n"
        "## Parameters\n\n_None._\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Steps\n\n1. First.\n2. Second.\n4. Fourth.\n\n"
        "## Expected outcome\n\nDone.\n"
    )
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "steps"


def test_empty_steps_section() -> None:
    md = (
        "# Title\n\nDesc.\n\n"
        "## Parameters\n\n_None._\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Steps\n\n\n"
        "## Expected outcome\n\nDone.\n"
    )
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "steps"


def test_stray_destructive_marker_in_step_text() -> None:
    md = (
        "# Title\n\nDesc.\n\n"
        "## Parameters\n\n_None._\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Steps\n\n1. Click ⚠️ the thing.\n\n"
        "## Expected outcome\n\nDone.\n"
    )
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "steps"


def test_malformed_step_line() -> None:
    md = (
        "# Title\n\nDesc.\n\n"
        "## Parameters\n\n_None._\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Steps\n\nDo the thing without a number.\n\n"
        "## Expected outcome\n\nDone.\n"
    )
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "steps"


def test_malformed_parameter_line() -> None:
    md = (
        "# Title\n\nDesc.\n\n"
        "## Parameters\n\n- nope, not valid\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Steps\n\n1. Do.\n\n"
        "## Expected outcome\n\nDone.\n"
    )
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "parameters"


def test_duplicate_parameter_name() -> None:
    md = (
        "# Title\n\nDesc.\n\n"
        "## Parameters\n\n"
        "- `foo` (string, required)\n"
        "- `foo` (string, optional)\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Steps\n\n1. Use {foo}.\n\n"
        "## Expected outcome\n\nDone.\n"
    )
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "parameters"
    assert "duplicate" in excinfo.value.reason.lower()


def test_invalid_integer_default() -> None:
    md = (
        "# Title\n\nDesc.\n\n"
        "## Parameters\n\n- `n` (integer, optional, default: not_a_number)\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Steps\n\n1. Use {n}.\n\n"
        "## Expected outcome\n\nDone.\n"
    )
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "parameters"


def test_invalid_boolean_default() -> None:
    md = (
        "# Title\n\nDesc.\n\n"
        "## Parameters\n\n- `flag` (boolean, optional, default: maybe)\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Steps\n\n1. Use {flag}.\n\n"
        "## Expected outcome\n\nDone.\n"
    )
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "parameters"


def test_unquoted_string_default() -> None:
    md = (
        "# Title\n\nDesc.\n\n"
        "## Parameters\n\n- `s` (string, optional, default: bare_value)\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Steps\n\n1. Use {s}.\n\n"
        "## Expected outcome\n\nDone.\n"
    )
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "parameters"


def test_unknown_section_rejected() -> None:
    md = (
        "# Title\n\nDesc.\n\n"
        "## Parameters\n\n_None._\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Foobar\n\nbody\n\n"
        "## Steps\n\n1. Do.\n\n"
        "## Expected outcome\n\nDone.\n"
    )
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    assert excinfo.value.section == "section_order"


def test_line_number_reported_on_error() -> None:
    md = (
        "# Title\n\nDesc.\n\n"
        "## Parameters\n\n"
        "- `good` (string, required)\n"
        "- not-a-valid-line\n\n"
        "## Preconditions\n\n_None._\n\n"
        "## Steps\n\n1. Use {good}.\n\n"
        "## Expected outcome\n\nDone.\n"
    )
    with pytest.raises(SkillParseError) as excinfo:
        parse_skill_md(md)
    # The bad line is at line 7 (1-indexed): 1 title, 2 blank, 3 desc, 4 blank,
    # 5 ## Parameters, 6 blank, 7 `good`, 8 `not-a-valid-line`.
    assert excinfo.value.line == 8


# --- extract_parameter_refs -------------------------------------------------


def test_extract_single_reference() -> None:
    assert extract_parameter_refs("Use {foo} here.") == {"foo"}


def test_extract_multiple_references_same_line() -> None:
    assert extract_parameter_refs("From {sender} to {recipient}.") == {
        "sender",
        "recipient",
    }


def test_extract_repeated_reference() -> None:
    assert extract_parameter_refs("{foo} and {foo} again.") == {"foo"}


def test_extract_ignores_fenced_code() -> None:
    md = "Before {real}.\n\n```\n{fake}\n```\n\nAfter."
    assert extract_parameter_refs(md) == {"real"}


def test_extract_ignores_inline_code() -> None:
    md = "Some `{fake}` and then {real}."
    assert extract_parameter_refs(md) == {"real"}


def test_extract_ignores_escaped_braces() -> None:
    md = r"Escaped: \{not_a_param\} and real {yes}."
    assert extract_parameter_refs(md) == {"yes"}


def test_extract_ignores_double_braces() -> None:
    md = "Template literal: {{not_a_param}} and real {yes}."
    assert extract_parameter_refs(md) == {"yes"}


def test_extract_ignores_invalid_names() -> None:
    # Uppercase / starts-with-digit / hyphen are not valid identifiers.
    md = "{Foo} {1bar} {has-hyphen} {ok_name}"
    assert extract_parameter_refs(md) == {"ok_name"}


def test_extract_empty_markdown() -> None:
    assert extract_parameter_refs("") == set()


def test_extract_no_references() -> None:
    assert extract_parameter_refs("Nothing to see here.") == set()
