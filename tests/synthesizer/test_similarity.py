"""Tests for :mod:`synthesizer.similarity` — snapshot similarity scoring.

All tests run in fake mode (via the ``_force_fake_mode`` autouse fixture in
conftest). LLM-touching tests register a canned Haiku response at the exact
request hash :func:`~synthesizer.similarity.score_skill_similarity` will
produce; structural-only tests (parameter_match, destructive_match) exercise
the local scoring helpers without any LLM call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from synthesizer.llm_client import (
    DEFAULT_MAX_TOKENS,
    PRICING_USD_PER_MTOK,
    LLMClient,
    compute_request_hash,
    costs_log_path,
    estimate_cost_usd,
    save_fake_response,
)
from synthesizer.similarity import (
    DEFAULT_SIMILARITY_MAX_TOKENS,
    SIMILARITY_MODEL,
    SIMILARITY_SYSTEM_PROMPT,
    SimilarityScore,
    SimilarityScoringError,
    build_similarity_user_content,
    score_skill_similarity,
)
from synthesizer.skill_doc import Parameter, ParsedSkill, Step, parse_skill_md

# --- Fixtures --------------------------------------------------------------


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


def _make_skill(
    *,
    title: str = "Send Gmail Reply",
    description: str = "Open Gmail and reply to the most recent unread message.",
    parameters: list[Parameter] | None = None,
    preconditions: list[str] | None = None,
    steps: list[Step] | None = None,
    expected_outcome: str = "The reply is sent successfully.",
    notes: str | None = None,
) -> ParsedSkill:
    return ParsedSkill(
        title=title,
        description=description,
        parameters=parameters or [],
        preconditions=preconditions or [],
        steps=steps
        or [
            Step(number=1, text="Open Chrome."),
            Step(number=2, text="Click the Reply button."),
            Step(number=3, text="Type the reply body."),
            Step(number=4, text="Click the Send button.", destructive=True),
        ],
        expected_outcome=expected_outcome,
        notes=notes,
    )


def _register_rubric_response(
    *,
    generated: ParsedSkill,
    golden: ParsedSkill,
    step_coverage: float,
    overall: float,
    reasoning: str,
    fakes_dir: Path,
    input_tokens: int = 800,
    output_tokens: int = 200,
) -> str:
    content = build_similarity_user_content(generated, golden)
    messages = [{"role": "user", "content": content}]
    request_hash = compute_request_hash(
        system=SIMILARITY_SYSTEM_PROMPT,
        messages=messages,
        model=SIMILARITY_MODEL,
        max_tokens=DEFAULT_SIMILARITY_MAX_TOKENS,
        tools=None,
    )
    payload = {
        "step_coverage": step_coverage,
        "overall": overall,
        "reasoning": reasoning,
    }
    save_fake_response(
        request_hash=request_hash,
        text=json.dumps(payload),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        directory=fakes_dir,
    )
    return request_hash


# --- AC tests --------------------------------------------------------------


def test_identical_skills_score_at_least_0_95_overall(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """AC: identical skills score ≥ 0.95 on overall."""
    skill = _make_skill()
    _register_rubric_response(
        generated=skill,
        golden=skill,
        step_coverage=1.0,
        overall=0.98,
        reasoning="Skills are identical in content and structure.",
        fakes_dir=isolated_fakes_dir,
    )

    client = LLMClient()
    score = score_skill_similarity(skill, skill, client)

    assert score.overall >= 0.95
    assert score.step_coverage == 1.0
    assert score.parameter_match == 1.0
    assert score.destructive_match == 1.0
    assert "identical" in score.reasoning.lower()


def test_wording_differences_score_at_least_0_85_overall(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """AC: skills differing only in wording score ≥ 0.85."""
    generated = _make_skill(
        description="Reply to the most recent unread email in Gmail.",
        steps=[
            Step(number=1, text="Launch Chrome."),
            Step(number=2, text="Press the Reply button."),
            Step(number=3, text="Enter the reply message."),
            Step(number=4, text="Press Send.", destructive=True),
        ],
    )
    golden = _make_skill()  # canonical wording
    _register_rubric_response(
        generated=generated,
        golden=golden,
        step_coverage=0.95,
        overall=0.88,
        reasoning="Same workflow and same destructive steps; only surface wording differs.",
        fakes_dir=isolated_fakes_dir,
    )

    client = LLMClient()
    score = score_skill_similarity(generated, golden, client)

    assert score.overall >= 0.85
    assert score.step_coverage >= 0.85
    assert score.parameter_match == 1.0  # empty == empty
    assert score.destructive_match == 1.0  # both flag step 4


def test_destructive_step_mismatch_scores_destructive_match_below_1(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """AC: skills differing in destructive step count → destructive_match < 1.0.

    The destructive_match dimension is computed structurally — no LLM call
    is needed to decide a binary fact — but the overall/step_coverage pair
    still requires a rubric response.
    """
    generated = _make_skill(
        steps=[
            Step(number=1, text="Open Chrome."),
            Step(number=2, text="Click the Reply button."),
            Step(number=3, text="Type the reply body."),
            Step(number=4, text="Click Send.", destructive=True),
        ],
    )
    # Golden flags BOTH step 3 and step 4 destructive; generated only step 4.
    golden = _make_skill(
        steps=[
            Step(number=1, text="Open Chrome."),
            Step(number=2, text="Click the Reply button."),
            Step(number=3, text="Confirm recipient.", destructive=True),
            Step(number=4, text="Click Send.", destructive=True),
        ],
    )
    _register_rubric_response(
        generated=generated,
        golden=golden,
        step_coverage=0.8,
        overall=0.7,
        reasoning="Generated skill is missing a destructive confirmation step.",
        fakes_dir=isolated_fakes_dir,
    )

    client = LLMClient()
    score = score_skill_similarity(generated, golden, client)

    assert score.destructive_match < 1.0
    assert score.destructive_match == 0.0


def test_missing_step_scores_step_coverage_below_0_8(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """AC: skills missing a step score step_coverage < 0.8."""
    generated = _make_skill(
        steps=[
            Step(number=1, text="Open Chrome."),
            Step(number=2, text="Click the Reply button."),
            Step(number=3, text="Click Send.", destructive=True),
        ],
    )
    golden = _make_skill()  # has the typing step; generated does not
    _register_rubric_response(
        generated=generated,
        golden=golden,
        step_coverage=0.65,
        overall=0.72,
        reasoning="Generated skill is missing the 'type reply body' step from golden.",
        fakes_dir=isolated_fakes_dir,
    )

    client = LLMClient()
    score = score_skill_similarity(generated, golden, client)

    assert score.step_coverage < 0.8


def test_parameter_match_exact_name_type_required(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """Structural: parameter_match is 1.0 iff (name, type, required) triples match exactly."""
    params = [
        Parameter(name="recipient", type="string", required=True),
        Parameter(name="reply_body", type="string", required=True),
    ]
    generated = _make_skill(parameters=list(params))
    golden = _make_skill(parameters=list(params))
    _register_rubric_response(
        generated=generated,
        golden=golden,
        step_coverage=1.0,
        overall=1.0,
        reasoning="Identical.",
        fakes_dir=isolated_fakes_dir,
    )

    client = LLMClient()
    score = score_skill_similarity(generated, golden, client)
    assert score.parameter_match == 1.0


def test_parameter_match_required_mismatch_scores_zero(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """Structural: flipping required vs. optional is a parameter mismatch."""
    generated = _make_skill(
        parameters=[Parameter(name="recipient", type="string", required=True)]
    )
    golden = _make_skill(
        parameters=[Parameter(name="recipient", type="string", required=False)]
    )
    _register_rubric_response(
        generated=generated,
        golden=golden,
        step_coverage=0.9,
        overall=0.85,
        reasoning="Same steps; parameter required-ness differs.",
        fakes_dir=isolated_fakes_dir,
    )

    client = LLMClient()
    score = score_skill_similarity(generated, golden, client)
    assert score.parameter_match == 0.0


def test_parameter_match_missing_parameter_scores_zero(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    generated = _make_skill(parameters=[])
    golden = _make_skill(
        parameters=[Parameter(name="recipient", type="string", required=True)]
    )
    _register_rubric_response(
        generated=generated,
        golden=golden,
        step_coverage=0.9,
        overall=0.75,
        reasoning="Generated skill omits recipient parameter.",
        fakes_dir=isolated_fakes_dir,
    )

    client = LLMClient()
    score = score_skill_similarity(generated, golden, client)
    assert score.parameter_match == 0.0


# --- Cost guard -----------------------------------------------------------


def test_cost_logged_and_under_two_cents(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """AC: cost per call < $0.02 (Haiku is cheap).

    Fake mode logs ``cost_estimate_usd=0.0`` which is trivially under the
    cap, but the meaningful check is that the Haiku pricing table gives a
    sub-$0.02 figure for a realistic token mix. We verify both: the logged
    cost line for the fake call AND the Haiku-pricing estimate for a
    typical 2000-input / 500-output call (well under $0.02).
    """
    generated = _make_skill()
    golden = _make_skill()
    _register_rubric_response(
        generated=generated,
        golden=golden,
        step_coverage=1.0,
        overall=1.0,
        reasoning="Identical.",
        fakes_dir=isolated_fakes_dir,
        input_tokens=1800,
        output_tokens=220,
    )

    client = LLMClient()
    score_skill_similarity(generated, golden, client)

    # costs.jsonl gained a line for this call; fake mode logs 0.0.
    log = costs_log_path()
    lines = log.read_text().strip().splitlines()
    assert lines, "similarity call did not append to costs.jsonl"
    last = json.loads(lines[-1])
    assert last["module"] == "synthesizer"
    assert last["cost_estimate_usd"] < 0.02
    assert last["context_label"] == "synthesizer:similarity"

    # At Haiku pricing a realistic rubric call is well under the cap.
    assert "claude-haiku-4-5" in PRICING_USD_PER_MTOK
    real_world_cost = estimate_cost_usd(
        model=SIMILARITY_MODEL, input_tokens=2000, output_tokens=500
    )
    assert real_world_cost < 0.02


# --- Response parsing ------------------------------------------------------


def test_invalid_json_response_raises_scoring_error(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    generated = _make_skill()
    golden = _make_skill()
    content = build_similarity_user_content(generated, golden)
    messages = [{"role": "user", "content": content}]
    request_hash = compute_request_hash(
        system=SIMILARITY_SYSTEM_PROMPT,
        messages=messages,
        model=SIMILARITY_MODEL,
        max_tokens=DEFAULT_SIMILARITY_MAX_TOKENS,
        tools=None,
    )
    save_fake_response(
        request_hash=request_hash,
        text="not json at all",
        directory=isolated_fakes_dir,
    )

    client = LLMClient()
    with pytest.raises(SimilarityScoringError, match="not valid JSON"):
        score_skill_similarity(generated, golden, client)


def test_response_missing_required_key_raises(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    generated = _make_skill()
    golden = _make_skill()
    _register_rubric_response(
        generated=generated,
        golden=golden,
        step_coverage=1.0,
        overall=1.0,
        reasoning="All good.",
        fakes_dir=isolated_fakes_dir,
    )
    # Overwrite the canned response with one missing `reasoning`.
    content = build_similarity_user_content(generated, golden)
    messages = [{"role": "user", "content": content}]
    request_hash = compute_request_hash(
        system=SIMILARITY_SYSTEM_PROMPT,
        messages=messages,
        model=SIMILARITY_MODEL,
        max_tokens=DEFAULT_SIMILARITY_MAX_TOKENS,
        tools=None,
    )
    save_fake_response(
        request_hash=request_hash,
        text=json.dumps({"step_coverage": 1.0, "overall": 1.0}),
        directory=isolated_fakes_dir,
    )
    client = LLMClient()
    with pytest.raises(SimilarityScoringError, match="reasoning"):
        score_skill_similarity(generated, golden, client)


def test_response_out_of_range_score_raises(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    generated = _make_skill()
    golden = _make_skill()
    content = build_similarity_user_content(generated, golden)
    messages = [{"role": "user", "content": content}]
    request_hash = compute_request_hash(
        system=SIMILARITY_SYSTEM_PROMPT,
        messages=messages,
        model=SIMILARITY_MODEL,
        max_tokens=DEFAULT_SIMILARITY_MAX_TOKENS,
        tools=None,
    )
    save_fake_response(
        request_hash=request_hash,
        text=json.dumps(
            {"step_coverage": 1.5, "overall": 0.9, "reasoning": "oops"}
        ),
        directory=isolated_fakes_dir,
    )
    client = LLMClient()
    with pytest.raises(SimilarityScoringError, match=r"\[0\.0, 1\.0\]"):
        score_skill_similarity(generated, golden, client)


def test_response_with_code_fence_is_tolerated(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """The model occasionally wraps JSON in a ``` fence despite instructions."""
    generated = _make_skill()
    golden = _make_skill()
    content = build_similarity_user_content(generated, golden)
    messages = [{"role": "user", "content": content}]
    request_hash = compute_request_hash(
        system=SIMILARITY_SYSTEM_PROMPT,
        messages=messages,
        model=SIMILARITY_MODEL,
        max_tokens=DEFAULT_SIMILARITY_MAX_TOKENS,
        tools=None,
    )
    payload = {"step_coverage": 1.0, "overall": 1.0, "reasoning": "ok"}
    save_fake_response(
        request_hash=request_hash,
        text=f"```json\n{json.dumps(payload)}\n```",
        directory=isolated_fakes_dir,
    )
    client = LLMClient()
    score = score_skill_similarity(generated, golden, client)
    assert score.overall == 1.0


# --- Message shape --------------------------------------------------------


def test_build_similarity_user_content_shape() -> None:
    generated = _make_skill(title="Generated")
    golden = _make_skill(title="Golden")
    blocks = build_similarity_user_content(generated, golden)

    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    body = blocks[0]["text"]
    assert "GENERATED:" in body
    assert "GOLDEN:" in body
    # Each skill's title appears exactly once inside the rendered markdown.
    assert body.count("# Generated") == 1
    assert body.count("# Golden") == 1
    # The GENERATED block precedes the GOLDEN block.
    assert body.index("GENERATED:") < body.index("GOLDEN:")


def test_system_prompt_instructs_no_code_fence_and_listed_keys() -> None:
    for key in ("step_coverage", "overall", "reasoning"):
        assert key in SIMILARITY_SYSTEM_PROMPT
    assert "code fence" in SIMILARITY_SYSTEM_PROMPT.lower()
    # Structural dimensions are NOT scored by the LLM per the PRD.
    assert "structural" in SIMILARITY_SYSTEM_PROMPT.lower()


def test_similarity_model_is_haiku_and_priced() -> None:
    assert SIMILARITY_MODEL == "claude-haiku-4-5"
    assert SIMILARITY_MODEL in PRICING_USD_PER_MTOK


def test_similarity_score_is_frozen() -> None:
    score = SimilarityScore(
        overall=1.0,
        step_coverage=1.0,
        parameter_match=1.0,
        destructive_match=1.0,
        reasoning="test",
    )
    with pytest.raises(Exception):  # noqa: B017 — pydantic v2 raises ValidationError
        score.overall = 0.5  # type: ignore[misc]


def test_similarity_scorer_fixture_returns_callable(
    similarity_scorer: Any,
) -> None:
    """S-014 requires the `similarity_scorer` fixture be importable for S-017."""
    assert callable(similarity_scorer)
    assert similarity_scorer.__name__ == "score_skill_similarity"


# --- Golden-round-trip integration ----------------------------------------


def test_score_uses_rendered_canonical_markdown(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """The scorer normalizes both skills through ``render_skill_md`` so the
    request hash is determined by canonical output, not the input markdown.

    Concretely: two ``ParsedSkill`` inputs that serialize identically must
    produce the same request hash. We re-build the parsed skill from its
    rendered markdown and assert the hash is stable.
    """
    generated = _make_skill()
    # Round-trip through the renderer and parser so we have an equivalent
    # but independently-constructed ParsedSkill instance.
    from synthesizer.skill_doc import render_skill_md

    roundtripped = parse_skill_md(render_skill_md(generated))
    assert roundtripped == generated

    content_a = build_similarity_user_content(generated, generated)
    content_b = build_similarity_user_content(roundtripped, roundtripped)
    assert content_a == content_b


# Sanity: DEFAULT_MAX_TOKENS is the draft default; similarity uses its own
# (smaller) default — make sure they do not collide by accident.
def test_similarity_token_default_is_independent() -> None:
    assert DEFAULT_SIMILARITY_MAX_TOKENS != DEFAULT_MAX_TOKENS
    assert DEFAULT_SIMILARITY_MAX_TOKENS <= DEFAULT_MAX_TOKENS
