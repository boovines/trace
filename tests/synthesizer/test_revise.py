"""Tests for :mod:`synthesizer.revise` — phase-2 revision generation.

All tests run in fake mode (via the ``_force_fake_mode`` autouse fixture in
conftest). Each test builds the exact message sequence ``generate_revision``
will send, computes the request hash via
:func:`synthesizer.llm_client.compute_request_hash`, and registers a canned
response at that hash using :func:`synthesizer.llm_client.save_fake_response`.

The pattern mirrors ``test_draft.py`` — in particular, retry tests extend
the message chain with ``{assistant, bad_text} + {user, feedback}`` turns
and register a second canned response at the follow-up hash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from synthesizer.draft import (
    DraftGenerationError,
    DraftResult,
    Question,
    _parse_response_json,
    _ResponseValidationError,
    _validate_full_response,
)
from synthesizer.llm_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    LLMClient,
    compute_request_hash,
    save_fake_response,
)
from synthesizer.revise import (
    MAX_REVISION_LLM_CALLS,
    build_revision_user_content,
    generate_revision,
)
from synthesizer.revise_prompt import REVISE_SYSTEM_PROMPT
from synthesizer.skill_doc import parse_skill_md
from synthesizer.trajectory_reader import TrajectoryReader

# --- repo-root locator (mirrors test_draft) --------------------------------


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "fixtures" / "trajectories").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate repo root with fixtures/")


FIXTURES_ROOT: Path = _repo_root() / "fixtures" / "trajectories"


def _reader_for(slug: str) -> TrajectoryReader:
    return TrajectoryReader(FIXTURES_ROOT / slug)


# --- shared fixtures (mirror test_draft) -----------------------------------


@pytest.fixture
def isolated_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("TRACE_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def isolated_fakes_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    fakes = tmp_path / "fakes"
    fakes.mkdir()
    monkeypatch.setenv("TRACE_FAKE_RESPONSES_DIR", str(fakes))
    return fakes


# --- helpers ---------------------------------------------------------------


def _markdown_for(
    slug: str,
    *,
    steps: list[str],
    parameters: list[dict[str, Any]] | None = None,
    expected: str = "The workflow completes successfully.",
    title: str | None = None,
    description: str | None = None,
) -> str:
    """Build a SKILL.md string that parses cleanly.

    ``steps`` entries must already include any ``⚠️ [DESTRUCTIVE]`` prefix
    but NOT the leading ``N. `` numbering (this helper adds that).
    """
    parameters = parameters or []
    title = title or slug.replace("_", " ").title()
    description = description or f"Automated workflow derived from the {slug} recording."

    if parameters:
        param_lines = [
            f"- `{p['name']}` ({p['type']}, "
            f"{'required' if p.get('required', True) else 'optional'})"
            for p in parameters
        ]
        param_block = "\n".join(param_lines)
    else:
        param_block = "_None._"

    step_lines = [f"{i}. {text}" for i, text in enumerate(steps, start=1)]

    return (
        f"# {title}\n"
        "\n"
        f"{description}\n"
        "\n"
        "## Parameters\n"
        "\n"
        f"{param_block}\n"
        "\n"
        "## Preconditions\n"
        "\n"
        "_None._\n"
        "\n"
        "## Steps\n"
        "\n"
        + "\n".join(step_lines)
        + "\n"
        "\n"
        "## Expected outcome\n"
        "\n"
        f"{expected}\n"
    )


def _meta_for(
    slug: str,
    *,
    parameters: list[dict[str, Any]] | None = None,
    destructive_steps: list[int] | None = None,
    step_count: int,
    name: str | None = None,
) -> dict[str, Any]:
    return {
        "slug": slug,
        "name": name if name is not None else slug.replace("_", " ").title(),
        "trajectory_id": "00000000-0000-0000-0000-000000000000",
        "created_at": "2026-04-22T00:00:00+00:00",
        "parameters": parameters or [],
        "destructive_steps": sorted(destructive_steps or []),
        "preconditions": [],
        "step_count": step_count,
    }


def _build_current_draft(
    slug: str,
    *,
    steps: list[str],
    parameters: list[dict[str, Any]] | None = None,
    destructive_steps: list[int] | None = None,
    questions: list[Question] | None = None,
    prior_cost: float = 0.0,
    prior_calls: int = 1,
    name: str | None = None,
) -> DraftResult:
    markdown = _markdown_for(
        slug, steps=steps, parameters=parameters, title=name
    )
    meta = _meta_for(
        slug,
        parameters=parameters,
        destructive_steps=destructive_steps,
        step_count=len(steps),
        name=name,
    )
    return DraftResult(
        markdown=markdown,
        parsed=parse_skill_md(markdown),
        meta=meta,
        questions=list(questions or []),
        llm_calls=prior_calls,
        total_cost_usd=prior_cost,
    )


def _register(
    *,
    messages: list[dict[str, Any]],
    text: str,
    fakes_dir: Path,
    input_tokens: int = 800,
    output_tokens: int = 300,
) -> str:
    request_hash = compute_request_hash(
        system=REVISE_SYSTEM_PROMPT,
        messages=messages,
        model=DEFAULT_MODEL,
        max_tokens=DEFAULT_MAX_TOKENS,
        tools=None,
    )
    save_fake_response(
        request_hash=request_hash,
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        directory=fakes_dir,
    )
    return request_hash


# --- happy-path: each of the 5 question categories -------------------------


def test_revise_parameterization_category(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """Category=parameterization: user confirms the typed body should be a param."""
    slug = "gmail_reply"
    reader = _reader_for(slug)
    current = _build_current_draft(
        slug,
        steps=[
            "Open Chrome.",
            'Click the Reply button.',
            'Type "Hi - thanks for the note" into the reply field.',
            "⚠️ [DESTRUCTIVE] Click the Send button.",
        ],
        destructive_steps=[4],
        questions=[
            Question(
                id="q1",
                category="parameterization",
                text="Should the reply body be a parameter?",
            ),
        ],
    )
    question = current.questions[0]

    revised_md = _markdown_for(
        slug,
        steps=[
            "Open Chrome.",
            "Click the Reply button.",
            "Type {reply_body} into the reply field.",
            "⚠️ [DESTRUCTIVE] Click the Send button.",
        ],
        parameters=[{"name": "reply_body", "type": "string", "required": True}],
    )
    revised_meta = _meta_for(
        slug,
        parameters=[{"name": "reply_body", "type": "string", "required": True}],
        destructive_steps=[4],
        step_count=4,
    )
    response_text = json.dumps(
        {"markdown": revised_md, "meta": revised_meta, "questions": []}
    )

    messages = [
        {
            "role": "user",
            "content": build_revision_user_content(current, question, "Yes, parameterize it"),
        }
    ]
    _register(messages=messages, text=response_text, fakes_dir=isolated_fakes_dir)

    client = LLMClient()
    result = generate_revision(
        current_draft=current,
        question=question,
        answer="Yes, parameterize it",
        client=client,
        reader=reader,
    )

    assert {p["name"] for p in result.meta["parameters"]} == {"reply_body"}
    assert "{reply_body}" in result.markdown
    assert result.questions == []


def test_revise_intent_category(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """Category=intent: user clarifies what a click was actually for."""
    slug = "calendar_block"
    reader = _reader_for(slug)
    current = _build_current_draft(
        slug,
        steps=[
            "Open Google Calendar.",
            "Click the Create button.",
        ],
        questions=[
            Question(
                id="q1",
                category="intent",
                text="Is this creating a focus block, an event, or a task?",
            ),
        ],
    )
    question = current.questions[0]

    revised_md = _markdown_for(
        slug,
        steps=[
            "Open Google Calendar.",
            "Click the Create button to start a new focus block.",
        ],
    )
    revised_meta = _meta_for(slug, step_count=2)
    response_text = json.dumps(
        {"markdown": revised_md, "meta": revised_meta, "questions": []}
    )

    messages = [
        {
            "role": "user",
            "content": build_revision_user_content(
                current, question, "A focus block on my calendar"
            ),
        }
    ]
    _register(messages=messages, text=response_text, fakes_dir=isolated_fakes_dir)

    client = LLMClient()
    result = generate_revision(
        current_draft=current,
        question=question,
        answer="A focus block on my calendar",
        client=client,
        reader=reader,
    )

    assert "focus block" in result.markdown.lower()


def test_revise_destructive_category_confirms_flag(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """Category=destructive: user confirms 'always ask before sending' → flag preserved."""
    slug = "gmail_reply"
    reader = _reader_for(slug)
    current = _build_current_draft(
        slug,
        steps=[
            "Open Chrome.",
            "Click the Reply button.",
            'Type "reply text" into the reply field.',
            "⚠️ [DESTRUCTIVE] Click the Send button.",
        ],
        destructive_steps=[4],
        questions=[
            Question(
                id="q1",
                category="destructive",
                text="Should the Send step always pause for confirmation?",
            ),
        ],
    )
    question = current.questions[0]

    # Revision confirms the flag — markdown + meta unchanged on destructive.
    revised_md = current.markdown
    revised_meta = dict(current.meta)
    response_text = json.dumps(
        {"markdown": revised_md, "meta": revised_meta, "questions": []}
    )

    messages = [
        {
            "role": "user",
            "content": build_revision_user_content(
                current, question, "Yes, always ask before sending"
            ),
        }
    ]
    _register(messages=messages, text=response_text, fakes_dir=isolated_fakes_dir)

    client = LLMClient()
    result = generate_revision(
        current_draft=current,
        question=question,
        answer="Yes, always ask before sending",
        client=client,
        reader=reader,
    )

    assert result.meta["destructive_steps"] == [4]
    destructive = [s for s in result.parsed.steps if s.destructive]
    assert len(destructive) == 1
    assert destructive[0].number == 4


def test_revise_precondition_category(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """Category=precondition: user adds a precondition string."""
    slug = "slack_status"
    reader = _reader_for(slug)
    current = _build_current_draft(
        slug,
        steps=[
            "Open Slack.",
            "Set the status via the profile menu.",
        ],
        questions=[
            Question(
                id="q1",
                category="precondition",
                text="Does Slack need to already be open?",
            ),
        ],
    )
    question = current.questions[0]

    revised_md = (
        f"# {slug.replace('_', ' ').title()}\n"
        "\n"
        f"Automated workflow derived from the {slug} recording.\n"
        "\n"
        "## Parameters\n"
        "\n"
        "_None._\n"
        "\n"
        "## Preconditions\n"
        "\n"
        "- Slack is already signed in on this machine.\n"
        "\n"
        "## Steps\n"
        "\n"
        "1. Open Slack.\n"
        "2. Set the status via the profile menu.\n"
        "\n"
        "## Expected outcome\n"
        "\n"
        "The workflow completes successfully.\n"
    )
    revised_meta = _meta_for(slug, step_count=2)
    revised_meta["preconditions"] = ["Slack is already signed in on this machine."]
    response_text = json.dumps(
        {"markdown": revised_md, "meta": revised_meta, "questions": []}
    )

    messages = [
        {
            "role": "user",
            "content": build_revision_user_content(
                current, question, "Yes, Slack must be signed in first"
            ),
        }
    ]
    _register(messages=messages, text=response_text, fakes_dir=isolated_fakes_dir)

    client = LLMClient()
    result = generate_revision(
        current_draft=current,
        question=question,
        answer="Yes, Slack must be signed in first",
        client=client,
        reader=reader,
    )

    assert result.meta["preconditions"] == [
        "Slack is already signed in on this machine."
    ]


def test_revise_naming_category_updates_meta_name_not_slug(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """Category=naming: user renames the skill → meta.name changes, slug stays."""
    slug = "notes_daily"
    reader = _reader_for(slug)
    current = _build_current_draft(
        slug,
        steps=[
            "Open Apple Notes.",
            "Create a new note with today's date.",
        ],
        questions=[
            Question(
                id="q1",
                category="naming",
                text="What should this skill be called?",
            ),
        ],
    )
    question = current.questions[0]

    revised_md = _markdown_for(
        slug,
        steps=[
            "Open Apple Notes.",
            "Create a new note with today's date.",
        ],
        title="Daily Journal Entry",
    )
    revised_meta = _meta_for(
        slug,
        step_count=2,
        name="Daily Journal Entry",
    )
    response_text = json.dumps(
        {"markdown": revised_md, "meta": revised_meta, "questions": []}
    )

    messages = [
        {
            "role": "user",
            "content": build_revision_user_content(
                current, question, "Daily Journal Entry"
            ),
        }
    ]
    _register(messages=messages, text=response_text, fakes_dir=isolated_fakes_dir)

    client = LLMClient()
    result = generate_revision(
        current_draft=current,
        question=question,
        answer="Daily Journal Entry",
        client=client,
        reader=reader,
    )

    assert result.meta["name"] == "Daily Journal Entry"
    # Slug is NOT regenerated during revision — finalized at write time.
    assert result.meta["slug"] == slug


# --- parameter removal -----------------------------------------------------


def test_revise_answer_removes_parameter(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """User says 'always send to jake@example.com' → literal replaces parameter."""
    slug = "gmail_reply"
    reader = _reader_for(slug)
    current = _build_current_draft(
        slug,
        steps=[
            "Open Chrome.",
            "Type {recipient_email} into the To field.",
            "⚠️ [DESTRUCTIVE] Click the Send button.",
        ],
        parameters=[
            {"name": "recipient_email", "type": "string", "required": True}
        ],
        destructive_steps=[3],
        questions=[
            Question(
                id="q1",
                category="parameterization",
                text="Is the recipient always the same, or should it vary?",
            ),
        ],
    )
    question = current.questions[0]

    revised_md = _markdown_for(
        slug,
        steps=[
            "Open Chrome.",
            "Type jake@example.com into the To field.",
            "⚠️ [DESTRUCTIVE] Click the Send button.",
        ],
    )
    revised_meta = _meta_for(slug, step_count=3, destructive_steps=[3])
    response_text = json.dumps(
        {"markdown": revised_md, "meta": revised_meta, "questions": []}
    )

    messages = [
        {
            "role": "user",
            "content": build_revision_user_content(
                current, question, "Always send to jake@example.com"
            ),
        }
    ]
    _register(messages=messages, text=response_text, fakes_dir=isolated_fakes_dir)

    client = LLMClient()
    result = generate_revision(
        current_draft=current,
        question=question,
        answer="Always send to jake@example.com",
        client=client,
        reader=reader,
    )

    assert result.meta["parameters"] == []
    assert "{recipient_email}" not in result.markdown
    assert "jake@example.com" in result.markdown


# --- invalid meta retry ----------------------------------------------------


def test_revise_retry_on_invalid_meta(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """First response has markdown/meta disagreement; revision retries and succeeds."""
    slug = "calendar_block"
    reader = _reader_for(slug)
    current = _build_current_draft(
        slug,
        steps=[
            "Open Google Calendar.",
            "Click the Create button.",
        ],
        questions=[Question(id="q1", category="intent", text="What event type?")],
    )
    question = current.questions[0]

    # Bad: markdown has 2 steps but meta.step_count says 5 (cross-check fails).
    bad_md = _markdown_for(
        slug,
        steps=["Open Google Calendar.", "Click the Create button."],
    )
    bad_meta = _meta_for(slug, step_count=5)
    bad_text = json.dumps(
        {"markdown": bad_md, "meta": bad_meta, "questions": []}
    )

    messages_1 = [
        {
            "role": "user",
            "content": build_revision_user_content(current, question, "A focus block"),
        }
    ]
    _register(messages=messages_1, text=bad_text, fakes_dir=isolated_fakes_dir)

    # Capture the validator's feedback string so our follow-up message matches
    # exactly what generate_revision will send on retry.
    try:
        _validate_full_response_from_text(bad_text)
    except _ResponseValidationError as err:
        feedback_1 = err.feedback
    else:
        pytest.fail("expected validation error on bad meta")

    # Good follow-up.
    good_md = _markdown_for(
        slug,
        steps=["Open Google Calendar.", "Click the Create button."],
    )
    good_meta = _meta_for(slug, step_count=2)
    good_text = json.dumps(
        {"markdown": good_md, "meta": good_meta, "questions": []}
    )

    messages_2 = [
        *messages_1,
        {"role": "assistant", "content": bad_text},
        {"role": "user", "content": feedback_1},
    ]
    _register(messages=messages_2, text=good_text, fakes_dir=isolated_fakes_dir)

    client = LLMClient()
    result = generate_revision(
        current_draft=current,
        question=question,
        answer="A focus block",
        client=client,
        reader=reader,
    )

    # The revised draft's llm_calls is the cumulative total: prior_calls (1) + 2 attempts.
    assert result.llm_calls == current.llm_calls + 2
    assert result.meta["step_count"] == 2


def _validate_full_response_from_text(text: str) -> None:
    """Run the draft validation chain against a raw text response.

    Raises :class:`_ResponseValidationError` at the first gate that fails —
    used by tests to capture the exact feedback string
    :func:`generate_revision` will inject on retry.
    """
    # Mirror draft.generate_draft's validation path against a fake
    # LLMResponse — we only need the ``text`` field because the other
    # LLMResponse fields are not read by the validators.
    class _FakeResp:
        def __init__(self, t: str) -> None:
            self.text = t

    _validate_full_response(_FakeResp(text))  # type: ignore[arg-type]


# --- hard cap: three malformed responses raise -----------------------------


def test_three_malformed_responses_raise_draft_generation_error(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    slug = "gmail_reply"
    reader = _reader_for(slug)
    current = _build_current_draft(
        slug,
        steps=[
            "Open Chrome.",
            "⚠️ [DESTRUCTIVE] Click the Send button.",
        ],
        destructive_steps=[2],
        questions=[Question(id="q1", category="intent", text="What?")],
    )
    question = current.questions[0]

    bad_text = "not json at all"

    def _feedback_for(text: str) -> str:
        try:
            _parse_response_json(text)
        except _ResponseValidationError as err:
            return err.feedback
        raise RuntimeError("expected validation failure")

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": build_revision_user_content(current, question, "idk"),
        }
    ]
    for _ in range(MAX_REVISION_LLM_CALLS):
        _register(messages=messages, text=bad_text, fakes_dir=isolated_fakes_dir)
        messages = [
            *messages,
            {"role": "assistant", "content": bad_text},
            {"role": "user", "content": _feedback_for(bad_text)},
        ]

    client = LLMClient()
    with pytest.raises(DraftGenerationError) as exc_info:
        generate_revision(
            current_draft=current,
            question=question,
            answer="idk",
            client=client,
            reader=reader,
        )

    assert len(exc_info.value.attempts) == MAX_REVISION_LLM_CALLS
    assert "not valid JSON" in exc_info.value.last_error


# --- destructive matcher re-applied ----------------------------------------


def test_destructive_matcher_reapplied_after_revision(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """Revision LLM accidentally drops the Send ⚠️ marker; matcher restores it.

    The gmail_reply fixture trajectory has a ``Send`` click; when the revision
    response omits the flag the secondary matcher must re-add it so the final
    :class:`DraftResult` agrees with the source trajectory.
    """
    slug = "gmail_reply"
    reader = _reader_for(slug)
    current = _build_current_draft(
        slug,
        steps=[
            "Open Chrome.",
            "Click the Reply button.",
            'Type "reply text" into the reply field.',
            "⚠️ [DESTRUCTIVE] Click the Send button.",
        ],
        destructive_steps=[4],
        questions=[Question(id="q1", category="intent", text="Anything to add?")],
    )
    question = current.questions[0]

    # Response DROPS the destructive flag; matcher must re-add it.
    revised_md = _markdown_for(
        slug,
        steps=[
            "Open Chrome.",
            "Click the Reply button.",
            'Type "reply text" into the reply field.',
            "Click the Send button.",
        ],
    )
    revised_meta = _meta_for(slug, step_count=4, destructive_steps=[])
    response_text = json.dumps(
        {"markdown": revised_md, "meta": revised_meta, "questions": []}
    )

    messages = [
        {
            "role": "user",
            "content": build_revision_user_content(current, question, "nope"),
        }
    ]
    _register(messages=messages, text=response_text, fakes_dir=isolated_fakes_dir)

    client = LLMClient()
    result = generate_revision(
        current_draft=current,
        question=question,
        answer="nope",
        client=client,
        reader=reader,
    )

    # Matcher should have re-added the flag on step 4 (the Send click).
    assert 4 in result.meta["destructive_steps"]
    assert "⚠️ [DESTRUCTIVE] Click the Send button." in result.markdown
    assert result.matcher_report.added_flags == [4]


# --- empty answer rejected before any LLM call ----------------------------


def test_empty_answer_raises_validation_error(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    slug = "gmail_reply"
    reader = _reader_for(slug)
    current = _build_current_draft(
        slug,
        steps=["Open Chrome.", "⚠️ [DESTRUCTIVE] Click Send."],
        destructive_steps=[2],
        questions=[Question(id="q1", category="intent", text="?")],
    )
    question = current.questions[0]
    client = LLMClient()

    with pytest.raises(ValueError, match="non-empty"):
        generate_revision(
            current_draft=current,
            question=question,
            answer="",
            client=client,
            reader=reader,
        )

    # Whitespace-only answers are also rejected.
    with pytest.raises(ValueError, match="non-empty"):
        generate_revision(
            current_draft=current,
            question=question,
            answer="   \t\n  ",
            client=client,
            reader=reader,
        )

    # No fake responses were consumed — a real LLM call would have failed
    # the empty-response-file test via FakeResponseNotFound.
    assert not any(isolated_fakes_dir.glob("*.json"))


# --- cost accumulation -----------------------------------------------------


def test_cost_accumulates_across_draft_and_revision(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """``DraftResult.total_cost_usd`` on a revision includes the prior draft's cost."""
    slug = "gmail_reply"
    reader = _reader_for(slug)
    # Seed the "current" draft with a non-zero prior cost as if it had been a
    # real-mode call. In fake mode the revision itself contributes 0.0, so
    # the sum should equal the prior cost exactly.
    current = _build_current_draft(
        slug,
        steps=[
            "Open Chrome.",
            "⚠️ [DESTRUCTIVE] Click the Send button.",
        ],
        destructive_steps=[2],
        questions=[Question(id="q1", category="intent", text="?")],
        prior_cost=0.0123,
        prior_calls=2,
    )
    question = current.questions[0]

    revised_md = current.markdown
    revised_meta = dict(current.meta)
    response_text = json.dumps(
        {"markdown": revised_md, "meta": revised_meta, "questions": []}
    )

    messages = [
        {
            "role": "user",
            "content": build_revision_user_content(current, question, "fine"),
        }
    ]
    _register(messages=messages, text=response_text, fakes_dir=isolated_fakes_dir)

    client = LLMClient()
    result = generate_revision(
        current_draft=current,
        question=question,
        answer="fine",
        client=client,
        reader=reader,
    )

    assert result.total_cost_usd == pytest.approx(0.0123)
    assert result.llm_calls == current.llm_calls + 1


# --- build_revision_user_content: structural checks ------------------------


def test_build_revision_user_content_shape() -> None:
    current = _build_current_draft(
        "gmail_reply",
        steps=["Open Chrome.", "⚠️ [DESTRUCTIVE] Click Send."],
        destructive_steps=[2],
        questions=[
            Question(id="q1", category="intent", text="Q1?"),
            Question(id="q2", category="naming", text="Q2?"),
        ],
    )
    question = current.questions[0]
    blocks = build_revision_user_content(current, question, "my answer")

    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    body = blocks[0]["text"]
    assert "CURRENT_MARKDOWN" in body
    assert "CURRENT_META" in body
    assert "ANSWERED_QUESTION" in body
    assert "REMAINING_QUESTIONS" in body
    assert "my answer" in body
    # q1 is being answered → only q2 appears in remaining.
    assert '"id": "q2"' in body
    assert '"id": "q1"' not in body.split("REMAINING_QUESTIONS")[1]


# --- prompt content sanity -------------------------------------------------


def test_revise_system_prompt_mentions_output_keys_and_rules() -> None:
    for key in ("markdown", "meta", "questions"):
        assert key in REVISE_SYSTEM_PROMPT
    assert "CURRENT_MARKDOWN" in REVISE_SYSTEM_PROMPT
    assert "CURRENT_META" in REVISE_SYSTEM_PROMPT
    assert "ANSWERED_QUESTION" in REVISE_SYSTEM_PROMPT
    assert "REMAINING_QUESTIONS" in REVISE_SYSTEM_PROMPT
    assert "⚠️ [DESTRUCTIVE]" in REVISE_SYSTEM_PROMPT
    # slug must be preserved during revision — the prompt says so explicitly.
    assert "slug" in REVISE_SYSTEM_PROMPT


# --- forwards-compat: DraftResult is immutable -----------------------------


def test_revision_does_not_mutate_current_draft(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """``DraftResult`` is a frozen dataclass — the caller's ``current_draft``
    is never touched, even when the revision succeeds. We verify the
    object-level invariant by round-tripping through a successful revision.
    """
    slug = "gmail_reply"
    reader = _reader_for(slug)
    current = _build_current_draft(
        slug,
        steps=["Open Chrome.", "⚠️ [DESTRUCTIVE] Click the Send button."],
        destructive_steps=[2],
        questions=[Question(id="q1", category="intent", text="?")],
    )
    question = current.questions[0]

    revised_md = _markdown_for(
        slug,
        steps=["Open Chrome.", "⚠️ [DESTRUCTIVE] Click the Send button."],
    )
    revised_meta = _meta_for(slug, step_count=2, destructive_steps=[2])
    response_text = json.dumps(
        {"markdown": revised_md, "meta": revised_meta, "questions": []}
    )

    messages = [
        {
            "role": "user",
            "content": build_revision_user_content(current, question, "ok"),
        }
    ]
    _register(messages=messages, text=response_text, fakes_dir=isolated_fakes_dir)

    snapshot_before = (
        current.markdown,
        dict(current.meta),
        tuple(current.questions),
        current.llm_calls,
        current.total_cost_usd,
    )

    client = LLMClient()
    revised = generate_revision(
        current_draft=current,
        question=question,
        answer="ok",
        client=client,
        reader=reader,
    )

    snapshot_after = (
        current.markdown,
        dict(current.meta),
        tuple(current.questions),
        current.llm_calls,
        current.total_cost_usd,
    )
    assert snapshot_before == snapshot_after
    assert revised is not current
    # Frozen dataclass: direct attribute writes raise.
    with pytest.raises((AttributeError, Exception)):
        current.llm_calls = 999  # type: ignore[misc]
