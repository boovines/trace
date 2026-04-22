"""Tests for :mod:`synthesizer.draft` — phase-1 draft generation.

All tests run in fake mode (via the ``_force_fake_mode`` autouse fixture in
conftest). Each test builds the exact message sequence ``generate_draft``
will send, computes the request hash, and registers a canned response at
that hash using :func:`synthesizer.llm_client.save_fake_response`. Retry
tests extend the message chain with a corrective user turn and register a
second canned response at the follow-up hash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from synthesizer.draft import (
    MAX_LLM_CALLS,
    DraftGenerationError,
    DraftResult,
    _parse_response_json,
    _ResponseValidationError,
    _validate_markdown,
    build_user_content,
    generate_draft,
)
from synthesizer.draft_prompt import DRAFT_OUTPUT_KEYS, DRAFT_SYSTEM_PROMPT
from synthesizer.llm_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    LLMClient,
    compute_request_hash,
    save_fake_response,
)
from synthesizer.preprocess import PreprocessedTrajectory, preprocess_trajectory
from synthesizer.trajectory_reader import TrajectoryReader

REFERENCE_SLUGS: tuple[str, ...] = (
    "gmail_reply",
    "calendar_block",
    "finder_organize",
    "slack_status",
    "notes_daily",
)


# --- locate fixtures --------------------------------------------------------


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "fixtures" / "trajectories").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate repo root with fixtures/")


FIXTURES_ROOT: Path = _repo_root() / "fixtures" / "trajectories"


def _reader_for(slug: str) -> TrajectoryReader:
    return TrajectoryReader(FIXTURES_ROOT / slug)


# --- shared fixtures --------------------------------------------------------


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


# --- canned-response helpers -----------------------------------------------


def _good_response_json(
    slug: str,
    preprocessed: PreprocessedTrajectory,
    *,
    extra_questions: list[dict[str, str]] | None = None,
    force_destructive: list[int] | None = None,
    parameters: list[dict[str, Any]] | None = None,
    markdown_override: str | None = None,
) -> str:
    """Build a valid draft JSON response for a given slug/digest.

    Produces a SKILL.md whose step count equals the number of non-synthetic
    digest entries (scroll_run / idle are not steps). Each click or text
    input in the digest becomes a step; app_switch becomes the first step;
    keyboard shortcuts are rendered as steps too.
    """
    step_entries = [
        e for e in preprocessed.digest if e.kind not in ("scroll_run", "idle")
    ]
    if not step_entries:
        step_entries = list(preprocessed.digest)

    destructive_numbers = set(force_destructive or [])
    param_list = parameters or []
    param_names = {p["name"] for p in param_list}

    if markdown_override is None:
        steps_md: list[str] = []
        for idx, entry in enumerate(step_entries, start=1):
            prefix = "⚠️ [DESTRUCTIVE] " if idx in destructive_numbers else ""
            # Sanitize embedded newlines in text_input summaries so the
            # step renders on a single line.
            summary = entry.summary_text.replace("\n", " ").replace("\r", " ")
            steps_md.append(f"{idx}. {prefix}{summary}.")

        # Inject every parameter as a literal {name} reference inside the
        # expected-outcome paragraph so the cross-check sees each param.
        if param_names:
            outcome_refs = " ".join(f"{{{n}}}" for n in sorted(param_names))
            expected = f"The workflow completes successfully for {outcome_refs}."
        else:
            expected = "The workflow completes successfully."

        param_lines: list[str] = []
        for p in param_list:
            req = "required" if p.get("required", True) else "optional"
            param_lines.append(f"- `{p['name']}` ({p['type']}, {req})")
        param_block = "\n".join(param_lines) if param_lines else "_None._"

        markdown = (
            f"# {slug.replace('_', ' ').title()}\n"
            "\n"
            f"Automated workflow derived from the {slug} recording.\n"
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
            + "\n".join(steps_md)
            + "\n"
            "\n"
            "## Expected outcome\n"
            "\n"
            f"{expected}\n"
        )
    else:
        markdown = markdown_override

    meta = {
        "slug": slug,
        "name": slug.replace("_", " ").title(),
        "trajectory_id": "00000000-0000-0000-0000-000000000000",
        "created_at": "2026-04-22T00:00:00+00:00",
        "parameters": param_list,
        "destructive_steps": sorted(destructive_numbers),
        "preconditions": [],
        "step_count": len(step_entries),
    }

    questions = extra_questions if extra_questions is not None else []

    return json.dumps(
        {"markdown": markdown, "meta": meta, "questions": questions}
    )


def _register(
    *,
    messages: list[dict[str, Any]],
    text: str,
    fakes_dir: Path,
    input_tokens: int = 1200,
    output_tokens: int = 400,
) -> str:
    request_hash = compute_request_hash(
        system=DRAFT_SYSTEM_PROMPT,
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


# --- acceptance: draft for each reference trajectory ------------------------


@pytest.mark.parametrize("slug", REFERENCE_SLUGS)
def test_draft_for_each_reference_trajectory(
    slug: str,
    isolated_data_dir: Path,
    isolated_fakes_dir: Path,
) -> None:
    reader = _reader_for(slug)
    preprocessed = preprocess_trajectory(reader)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": build_user_content(preprocessed, reader)}
    ]
    response_text = _good_response_json(slug, preprocessed)
    _register(messages=messages, text=response_text, fakes_dir=isolated_fakes_dir)

    client = LLMClient()
    result = generate_draft(preprocessed, client, reader=reader)

    assert isinstance(result, DraftResult)
    assert result.llm_calls == 1
    assert result.total_cost_usd == 0.0  # fake-mode pricing
    assert result.parsed.title.lower().startswith(slug.split("_")[0].lower())
    assert result.meta["slug"] == slug


# --- retry: JSON parse failure ---------------------------------------------


def test_retry_on_malformed_json(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    slug = "gmail_reply"
    reader = _reader_for(slug)
    preprocessed = preprocess_trajectory(reader)
    user_content = build_user_content(preprocessed, reader)

    # First attempt: invalid JSON (missing closing brace).
    bad_text = '{"markdown": "# Nope", "meta": {'
    messages_1: list[dict[str, Any]] = [
        {"role": "user", "content": user_content}
    ]
    _register(messages=messages_1, text=bad_text, fakes_dir=isolated_fakes_dir)

    # The feedback string that generate_draft will inject on retry matches
    # the one from draft._parse_response_json. We capture it by running the
    # same validator and reading its message, rather than hardcoding prose.
    try:
        _parse_response_json(bad_text)
    except _ResponseValidationError as err:
        feedback_1 = err.feedback
    else:
        pytest.fail("expected validation error on bad JSON")

    messages_2: list[dict[str, Any]] = [
        *messages_1,
        {"role": "assistant", "content": bad_text},
        {"role": "user", "content": feedback_1},
    ]
    good_text = _good_response_json(slug, preprocessed)
    _register(messages=messages_2, text=good_text, fakes_dir=isolated_fakes_dir)

    client = LLMClient()
    result = generate_draft(preprocessed, client, reader=reader)

    assert result.llm_calls == 2
    assert result.meta["slug"] == slug


# --- retry: markdown parse failure -----------------------------------------


def test_retry_on_malformed_markdown(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    slug = "calendar_block"
    reader = _reader_for(slug)
    preprocessed = preprocess_trajectory(reader)
    user_content = build_user_content(preprocessed, reader)

    # First attempt: valid JSON but malformed markdown (missing H1 title).
    bad_response = {
        "markdown": "No H1 here\n\n## Parameters\n\n_None._\n",
        "meta": {
            "slug": "calendar_block",
            "name": "Calendar Block",
            "trajectory_id": "00000000-0000-0000-0000-000000000000",
            "created_at": "2026-04-22T00:00:00+00:00",
            "parameters": [],
            "destructive_steps": [],
            "preconditions": [],
            "step_count": 0,
        },
        "questions": [],
    }
    bad_text = json.dumps(bad_response)

    messages_1: list[dict[str, Any]] = [
        {"role": "user", "content": user_content}
    ]
    _register(messages=messages_1, text=bad_text, fakes_dir=isolated_fakes_dir)

    try:
        obj = _parse_response_json(bad_text)
        _validate_markdown(obj)
    except _ResponseValidationError as err:
        feedback_1 = err.feedback
    else:
        pytest.fail("expected validation error on bad markdown")

    messages_2: list[dict[str, Any]] = [
        *messages_1,
        {"role": "assistant", "content": bad_text},
        {"role": "user", "content": feedback_1},
    ]
    good_text = _good_response_json(slug, preprocessed)
    _register(messages=messages_2, text=good_text, fakes_dir=isolated_fakes_dir)

    client = LLMClient()
    result = generate_draft(preprocessed, client, reader=reader)

    assert result.llm_calls == 2


# --- hard cap: three malformed responses raise -----------------------------


def test_three_malformed_responses_raise_draft_generation_error(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    slug = "slack_status"
    reader = _reader_for(slug)
    preprocessed = preprocess_trajectory(reader)
    user_content = build_user_content(preprocessed, reader)

    # Register the same "invalid JSON" text at three successive hashes.
    bad_text = "not json at all"

    def _feedback_for(text: str) -> str:
        try:
            _parse_response_json(text)
        except _ResponseValidationError as err:
            return err.feedback
        raise RuntimeError("expected validation failure")

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_content}
    ]
    for _ in range(MAX_LLM_CALLS):
        _register(messages=messages, text=bad_text, fakes_dir=isolated_fakes_dir)
        messages = [
            *messages,
            {"role": "assistant", "content": bad_text},
            {"role": "user", "content": _feedback_for(bad_text)},
        ]

    client = LLMClient()
    with pytest.raises(DraftGenerationError) as exc_info:
        generate_draft(preprocessed, client, reader=reader)

    assert len(exc_info.value.attempts) == MAX_LLM_CALLS
    assert "not valid JSON" in exc_info.value.last_error


# --- parameter extraction --------------------------------------------------


def test_parameter_extraction_either_param_or_question(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """A text_input that looks like an email address should end up either as
    a parameter in the draft OR as a follow-up question about parameterizing
    it — both are acceptable per the PRD."""
    slug = "gmail_reply"
    reader = _reader_for(slug)
    preprocessed = preprocess_trajectory(reader)
    user_content = build_user_content(preprocessed, reader)

    # Canned response: promote the typed text to a parameter.
    response_text = _good_response_json(
        slug,
        preprocessed,
        parameters=[{"name": "reply_body", "type": "string", "required": True}],
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_content}
    ]
    _register(messages=messages, text=response_text, fakes_dir=isolated_fakes_dir)

    client = LLMClient()
    result = generate_draft(preprocessed, client, reader=reader)

    param_names = {p["name"] for p in result.meta["parameters"]}
    question_text = " ".join(q.text for q in result.questions).lower()
    assert "reply_body" in param_names or "parameter" in question_text


# --- destructive flagging --------------------------------------------------


def test_destructive_flagging_send_step(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """The gmail_reply trajectory contains a 'Send' click; when the canned
    response flags the step as destructive, meta.destructive_steps and the
    markdown ⚠️ marker must agree (the cross-check enforces this)."""
    slug = "gmail_reply"
    reader = _reader_for(slug)
    preprocessed = preprocess_trajectory(reader)
    user_content = build_user_content(preprocessed, reader)

    # Identify the step number of the 'Send' click in the digest.
    step_entries = [
        e for e in preprocessed.digest if e.kind not in ("scroll_run", "idle")
    ]
    send_idx = next(
        i
        for i, e in enumerate(step_entries, start=1)
        if "Send" in e.summary_text
    )

    response_text = _good_response_json(
        slug, preprocessed, force_destructive=[send_idx]
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_content}
    ]
    _register(messages=messages, text=response_text, fakes_dir=isolated_fakes_dir)

    client = LLMClient()
    result = generate_draft(preprocessed, client, reader=reader)

    assert send_idx in result.meta["destructive_steps"]
    destructive_steps = [s for s in result.parsed.steps if s.destructive]
    assert len(destructive_steps) == 1
    assert destructive_steps[0].number == send_idx


# --- build_user_content: structural checks --------------------------------


def test_build_user_content_has_text_block_and_image_blocks(
    isolated_data_dir: Path,
) -> None:
    reader = _reader_for("gmail_reply")
    preprocessed = preprocess_trajectory(reader)
    blocks = build_user_content(preprocessed, reader)

    assert blocks[0]["type"] == "text"
    assert "digest" in blocks[0]["text"].lower()
    image_blocks = [b for b in blocks if b["type"] == "image"]
    assert 1 <= len(image_blocks) <= 20
    for b in image_blocks:
        assert b["source"]["type"] == "base64"
        assert b["source"]["media_type"] == "image/png"
        assert isinstance(b["source"]["data"], str) and b["source"]["data"]


# --- system prompt: contains required keys --------------------------------


def test_system_prompt_mentions_required_keys() -> None:
    for key in DRAFT_OUTPUT_KEYS:
        assert key in DRAFT_SYSTEM_PROMPT
    assert "⚠️ [DESTRUCTIVE]" in DRAFT_SYSTEM_PROMPT
    # Both few-shot examples must be embedded verbatim.
    assert "### Example 1" in DRAFT_SYSTEM_PROMPT
    assert "### Example 2" in DRAFT_SYSTEM_PROMPT
