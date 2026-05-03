"""End-to-end fake-mode integration tests for the synthesis pipeline (S-015).

Drives the whole pipeline for each of the five reference trajectories —
trajectory on disk → preprocess → draft → destructive matcher → 2 Q&A
revisions → approve → written skill — using canned fake-mode LLM responses
registered via :func:`synthesizer.llm_client.save_fake_response`.

No real API calls happen: ``TRACE_LLM_FAKE_MODE=1`` is autoused from
``conftest.py`` and the ``anthropic_mock`` respx router belts-and-suspenders
any attempt to hit ``api.anthropic.com``. Real-mode verification is S-017,
gated behind ``TRACE_REAL_API_TESTS=1``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import respx
from PIL import Image

from synthesizer.destructive_matcher import apply_destructive_matcher
from synthesizer.draft import (
    MAX_LLM_CALLS,
    DraftResult,
    Question,
    _parse_response_json,
    _ResponseValidationError,
    build_user_content,
    generate_draft,
)
from synthesizer.draft_prompt import DRAFT_SYSTEM_PROMPT
from synthesizer.llm_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    LLMClient,
    compute_request_hash,
    costs_log_path,
    save_fake_response,
)
from synthesizer.preprocess import PreprocessedTrajectory, preprocess_trajectory
from synthesizer.revise import build_revision_user_content, generate_revision
from synthesizer.revise_prompt import REVISE_SYSTEM_PROMPT
from synthesizer.schema import validate_meta, validate_meta_against_markdown
from synthesizer.session import (
    SessionState,
    SynthesisSession,
    session_dir,
)
from synthesizer.skill_doc import parse_skill_md
from synthesizer.trajectory_reader import TrajectoryReader
from synthesizer.writer import index_db_path

REFERENCE_SLUGS: tuple[str, ...] = (
    "gmail_reply",
    "calendar_block",
    "finder_organize",
    "slack_status",
    "notes_daily",
)


# --- fixture locators ------------------------------------------------------


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "fixtures" / "trajectories").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate repo root with fixtures/")


FIXTURES_ROOT: Path = _repo_root() / "fixtures" / "trajectories"


def _reader_for(slug: str) -> TrajectoryReader:
    return TrajectoryReader(FIXTURES_ROOT / slug)


# --- per-test isolation ----------------------------------------------------


@pytest.fixture
def isolated_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data_dir = tmp_path / "trace_data"
    data_dir.mkdir()
    monkeypatch.setenv("TRACE_DATA_DIR", str(data_dir))
    return data_dir


@pytest.fixture
def isolated_fakes_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    fakes = tmp_path / "fakes"
    fakes.mkdir()
    monkeypatch.setenv("TRACE_FAKE_RESPONSES_DIR", str(fakes))
    return fakes


# --- canned-response construction ------------------------------------------


def _step_entries(preprocessed: PreprocessedTrajectory) -> list[Any]:
    entries = [
        e for e in preprocessed.digest if e.kind not in ("scroll_run", "idle")
    ]
    return entries or list(preprocessed.digest)


def _build_skill_markdown(
    slug: str,
    preprocessed: PreprocessedTrajectory,
    *,
    destructive_nums: list[int],
    parameters: list[dict[str, Any]] | None = None,
    preconditions: list[str] | None = None,
    title_override: str | None = None,
) -> str:
    """Build a SKILL.md that parses cleanly and matches the trajectory's destructive clicks."""
    entries = _step_entries(preprocessed)
    destructive_set = set(destructive_nums)
    parameters = parameters or []
    preconditions = preconditions or []

    steps_md: list[str] = []
    for idx, entry in enumerate(entries, start=1):
        prefix = "⚠️ [DESTRUCTIVE] " if idx in destructive_set else ""
        # Multi-line text_input summaries must collapse to one line.
        summary = entry.summary_text.replace("\n", " ").replace("\r", " ")
        steps_md.append(f"{idx}. {prefix}{summary}.")

    param_names = {p["name"] for p in parameters}
    if param_names:
        refs = " ".join(f"{{{n}}}" for n in sorted(param_names))
        expected = f"The workflow completes successfully for {refs}."
    else:
        expected = "The workflow completes successfully."

    if parameters:
        param_lines = [
            f"- `{p['name']}` ({p['type']}, "
            f"{'required' if p.get('required', True) else 'optional'})"
            for p in parameters
        ]
        param_block = "\n".join(param_lines)
    else:
        param_block = "_None._"

    precond_block = (
        "\n".join(f"- {s}" for s in preconditions) if preconditions else "_None._"
    )

    title = title_override or slug.replace("_", " ").title()

    return (
        f"# {title}\n"
        "\n"
        f"Automated workflow derived from the {slug} recording.\n"
        "\n"
        "## Parameters\n"
        "\n"
        f"{param_block}\n"
        "\n"
        "## Preconditions\n"
        "\n"
        f"{precond_block}\n"
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


def _build_meta(
    slug: str,
    trajectory_id: str,
    step_count: int,
    *,
    destructive_nums: list[int],
    parameters: list[dict[str, Any]] | None = None,
    preconditions: list[str] | None = None,
    name_override: str | None = None,
) -> dict[str, Any]:
    return {
        "slug": slug,
        "name": name_override or slug.replace("_", " ").title(),
        "trajectory_id": trajectory_id,
        "created_at": "2026-04-22T00:00:00+00:00",
        "parameters": parameters or [],
        "destructive_steps": sorted(destructive_nums),
        "preconditions": preconditions or [],
        "step_count": step_count,
    }


def _discover_destructive_steps(
    slug: str, preprocessed: PreprocessedTrajectory, reader: TrajectoryReader
) -> list[int]:
    """Run the real destructive matcher over an un-flagged skeleton to discover
    which steps will end up destructive. Baking the result into the canned
    response makes :func:`~synthesizer.draft.generate_draft`'s matcher pass a
    no-op — the DraftResult's markdown equals the canned markdown verbatim.
    """
    skeleton = _build_skill_markdown(slug, preprocessed, destructive_nums=[])
    parsed = parse_skill_md(skeleton)
    result = apply_destructive_matcher(parsed, reader)
    return sorted(s.number for s in result.parsed.steps if s.destructive)


def _build_draft_result_from_canned(
    markdown: str, meta: dict[str, Any], questions: list[Question]
) -> DraftResult:
    return DraftResult(
        markdown=markdown,
        parsed=parse_skill_md(markdown),
        meta=dict(meta),
        questions=list(questions),
        llm_calls=1,
        total_cost_usd=0.0,
    )


def _register_response(
    *,
    system: str,
    messages: list[dict[str, Any]],
    text: str,
    fakes_dir: Path,
) -> None:
    request_hash = compute_request_hash(
        system=system,
        messages=messages,
        model=DEFAULT_MODEL,
        max_tokens=DEFAULT_MAX_TOKENS,
        tools=None,
    )
    save_fake_response(
        request_hash=request_hash,
        text=text,
        input_tokens=1000,
        output_tokens=400,
        directory=fakes_dir,
    )


# --- parametrized full pipeline --------------------------------------------


@pytest.mark.parametrize("slug", REFERENCE_SLUGS)
def test_full_synthesis_pipeline_fake_mode(
    slug: str,
    isolated_data_dir: Path,
    isolated_fakes_dir: Path,
    anthropic_mock: respx.MockRouter,
) -> None:
    reader = _reader_for(slug)
    trajectory_id = str(reader.metadata["id"])
    preprocessed = preprocess_trajectory(reader)
    step_entries = _step_entries(preprocessed)
    step_count = len(step_entries)
    destructive_nums = _discover_destructive_steps(slug, preprocessed, reader)

    # --- initial draft canned response -----------------------------------
    draft_markdown = _build_skill_markdown(
        slug, preprocessed, destructive_nums=destructive_nums
    )
    draft_meta = _build_meta(
        slug, trajectory_id, step_count, destructive_nums=destructive_nums
    )
    draft_questions = [
        Question(id="q1", category="intent", text="Is this the intended flow?"),
        Question(id="q2", category="naming", text="Preferred name for this skill?"),
    ]
    draft_response_text = json.dumps(
        {
            "markdown": draft_markdown,
            "meta": dict(draft_meta),
            "questions": [q.model_dump() for q in draft_questions],
        }
    )
    draft_user_content = build_user_content(preprocessed, reader)
    _register_response(
        system=DRAFT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": draft_user_content}],
        text=draft_response_text,
        fakes_dir=isolated_fakes_dir,
    )

    initial_draft = _build_draft_result_from_canned(
        draft_markdown, draft_meta, draft_questions
    )

    # --- revision 1: add a precondition, prune q1 ------------------------
    precond = "The target application is already open."
    rev1_md = _build_skill_markdown(
        slug,
        preprocessed,
        destructive_nums=destructive_nums,
        preconditions=[precond],
    )
    rev1_meta = _build_meta(
        slug,
        trajectory_id,
        step_count,
        destructive_nums=destructive_nums,
        preconditions=[precond],
    )
    rev1_questions = [draft_questions[1]]
    rev1_response_text = json.dumps(
        {
            "markdown": rev1_md,
            "meta": dict(rev1_meta),
            "questions": [q.model_dump() for q in rev1_questions],
        }
    )
    answer_1 = "Yes, this reflects the intended flow."
    rev1_user_content = build_revision_user_content(
        initial_draft, draft_questions[0], answer_1
    )
    _register_response(
        system=REVISE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": rev1_user_content}],
        text=rev1_response_text,
        fakes_dir=isolated_fakes_dir,
    )
    revised_draft_1 = _build_draft_result_from_canned(
        rev1_md, rev1_meta, rev1_questions
    )

    # --- revision 2: rename, drop all questions --------------------------
    new_name = "Renamed Workflow"
    rev2_md = _build_skill_markdown(
        slug,
        preprocessed,
        destructive_nums=destructive_nums,
        preconditions=[precond],
        title_override=new_name,
    )
    rev2_meta = _build_meta(
        slug,
        trajectory_id,
        step_count,
        destructive_nums=destructive_nums,
        preconditions=[precond],
        name_override=new_name,
    )
    rev2_response_text = json.dumps(
        {
            "markdown": rev2_md,
            "meta": dict(rev2_meta),
            "questions": [],
        }
    )
    rev2_user_content = build_revision_user_content(
        revised_draft_1, draft_questions[1], new_name
    )
    _register_response(
        system=REVISE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": rev2_user_content}],
        text=rev2_response_text,
        fakes_dir=isolated_fakes_dir,
    )

    # --- drive the session end-to-end ------------------------------------
    skills_root = isolated_data_dir / "skills"
    client = LLMClient()
    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=reader,
        client=client,
        skills_root=skills_root,
        draft_fn=generate_draft,
        revise_fn=generate_revision,
    )

    assert session.state == SessionState.GENERATING_DRAFT
    session.start_draft()
    assert session.state == SessionState.AWAITING_ANSWER
    assert [q.id for q in session.questions] == ["q1", "q2"]

    session.answer_question("q1", answer_1)
    assert session.state == SessionState.AWAITING_ANSWER
    session.answer_question("q2", new_name)
    assert session.state == SessionState.AWAITING_APPROVAL

    written = session.approve()
    assert session.state == SessionState.COMPLETED
    assert written.slug == slug

    # --- on-disk skill verification --------------------------------------
    assert written.path.is_dir()
    skill_md_path = written.path / "SKILL.md"
    meta_path = written.path / "skill.meta.json"
    assert skill_md_path.is_file()
    assert meta_path.is_file()

    on_disk_md = skill_md_path.read_text(encoding="utf-8")
    parsed_disk = parse_skill_md(on_disk_md)
    assert session.draft is not None
    # Written SKILL.md parses back to the same ParsedSkill as the final draft.
    assert parsed_disk == session.draft.parsed

    on_disk_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    validate_meta(on_disk_meta)
    validate_meta_against_markdown(on_disk_meta, on_disk_md)
    assert on_disk_meta["name"] == new_name
    assert on_disk_meta["slug"] == slug
    assert on_disk_meta["trajectory_id"] == trajectory_id
    assert on_disk_meta["destructive_steps"] == destructive_nums

    # --- preview screenshots are valid PNGs (Pillow can open) ------------
    preview_dir = written.path / "preview"
    if preview_dir.exists():
        pngs = sorted(preview_dir.glob("*.png"))
        assert 0 < len(pngs) <= 5
        for png in pngs:
            with Image.open(png) as img:
                img.verify()

    # --- cost log has ≥3 fake-mode entries -------------------------------
    costs_path = costs_log_path()
    assert costs_path.is_file()
    cost_lines = [
        json.loads(line)
        for line in costs_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(cost_lines) >= 3
    for entry in cost_lines:
        assert entry["module"] == "synthesizer"
        assert entry["model"] == "fake"

    # --- no real API calls (belt-and-suspenders respx assertion) --------
    assert len(anthropic_mock.calls) == 0

    # --- SQLite index shows the new skill --------------------------------
    db_path = index_db_path()
    assert db_path.is_file()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT slug, name, trajectory_id, step_count, destructive_step_count "
            "FROM skills WHERE slug = ?",
            (written.slug,),
        ).fetchone()
    assert row is not None
    assert row[0] == slug
    assert row[1] == new_name
    assert row[2] == trajectory_id
    assert row[3] == step_count
    assert row[4] == len(destructive_nums)

    # --- session persistence dir cleaned up after COMPLETED --------------
    assert not session_dir(session.session_id).exists()


# --- cancellation cleanup --------------------------------------------------


def test_cancel_in_progress_removes_session_persistence(
    isolated_data_dir: Path, isolated_fakes_dir: Path
) -> None:
    """Cancelling an in-flight session wipes ``synthesis_sessions/<sid>/``."""
    slug = "gmail_reply"
    reader = _reader_for(slug)
    trajectory_id = str(reader.metadata["id"])
    preprocessed = preprocess_trajectory(reader)
    destructive_nums = _discover_destructive_steps(slug, preprocessed, reader)
    step_count = len(_step_entries(preprocessed))

    draft_md = _build_skill_markdown(
        slug, preprocessed, destructive_nums=destructive_nums
    )
    draft_meta = _build_meta(
        slug, trajectory_id, step_count, destructive_nums=destructive_nums
    )
    draft_questions = [Question(id="q1", category="intent", text="?")]
    draft_response_text = json.dumps(
        {
            "markdown": draft_md,
            "meta": dict(draft_meta),
            "questions": [q.model_dump() for q in draft_questions],
        }
    )
    _register_response(
        system=DRAFT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_content(preprocessed, reader)}],
        text=draft_response_text,
        fakes_dir=isolated_fakes_dir,
    )

    skills_root = isolated_data_dir / "skills"
    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=reader,
        client=LLMClient(),
        skills_root=skills_root,
        draft_fn=generate_draft,
        revise_fn=generate_revision,
    )
    session.start_draft()
    assert session.state == SessionState.AWAITING_ANSWER
    persist = session_dir(session.session_id)
    assert persist.exists()

    session.cancel()
    assert session.state == SessionState.CANCELLED
    assert not persist.exists()
    # No skill directory was written.
    assert not skills_root.exists() or not any(skills_root.iterdir())


# --- error path: three malformed draft responses ---------------------------


def test_error_path_malformed_responses_end_in_errored(
    isolated_data_dir: Path,
    isolated_fakes_dir: Path,
    anthropic_mock: respx.MockRouter,
) -> None:
    """Three consecutive malformed draft responses → session ERRORED, no skill,
    no index entry. Exercises the ``DraftGenerationError`` → ``_fail`` path in
    :class:`SynthesisSession`."""
    slug = "gmail_reply"
    reader = _reader_for(slug)
    trajectory_id = str(reader.metadata["id"])
    preprocessed = preprocess_trajectory(reader)

    bad_text = "definitely not json"

    def _feedback_for(text: str) -> str:
        try:
            _parse_response_json(text)
        except _ResponseValidationError as err:
            return err.feedback
        raise RuntimeError("expected validation failure")

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": build_user_content(preprocessed, reader)}
    ]
    for _ in range(MAX_LLM_CALLS):
        _register_response(
            system=DRAFT_SYSTEM_PROMPT,
            messages=messages,
            text=bad_text,
            fakes_dir=isolated_fakes_dir,
        )
        messages = [
            *messages,
            {"role": "assistant", "content": bad_text},
            {"role": "user", "content": _feedback_for(bad_text)},
        ]

    skills_root = isolated_data_dir / "skills"
    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=reader,
        client=LLMClient(),
        skills_root=skills_root,
        draft_fn=generate_draft,
        revise_fn=generate_revision,
    )
    session.start_draft()

    assert session.state == SessionState.ERRORED
    assert session.error is not None
    assert "draft generation failed" in session.error

    # No skill on disk.
    assert not skills_root.exists() or not any(skills_root.iterdir())

    # No index entry — either the db doesn't exist yet, or it has no rows.
    db_path = index_db_path()
    if db_path.is_file():
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT slug FROM skills").fetchall()
        assert rows == []

    # No real API calls.
    assert len(anthropic_mock.calls) == 0
