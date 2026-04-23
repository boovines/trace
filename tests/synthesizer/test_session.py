"""Tests for :mod:`synthesizer.session` — S-011 synthesis session state machine.

The session orchestrates ``generate_draft`` + (future) ``generate_revision``
+ :class:`~synthesizer.writer.SkillWriter` behind a small state machine.
These tests run every state transition using injected fake draft / revise
callables — no real LLM calls, no dependency on the S-012 revise module.
"""

from __future__ import annotations

import json
import struct
import uuid
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from synthesizer.draft import DraftGenerationError, DraftResult, Question
from synthesizer.llm_client import LLMClient
from synthesizer.session import (
    DEFAULT_SESSION_TTL_SECONDS,
    IllegalStateTransition,
    SessionState,
    SessionStore,
    SynthesisSession,
    session_dir,
)
from synthesizer.skill_doc import Parameter, ParsedSkill, Step
from synthesizer.slug import SlugError
from synthesizer.trajectory_reader import TrajectoryReader

# --- shared helpers --------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data_dir = tmp_path / "trace_data"
    data_dir.mkdir()
    monkeypatch.setenv("TRACE_DATA_DIR", str(data_dir))
    return data_dir


def _iso(seconds_offset: float = 0.0) -> str:
    base = datetime(2026, 4, 22, 14, 0, 0, tzinfo=UTC)
    return (base + timedelta(seconds=seconds_offset)).isoformat().replace(
        "+00:00", "Z"
    )


def _one_pixel_png() -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"

    def _chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00", 9)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _make_trajectory(tmp_path: Path) -> TrajectoryReader:
    trajectory_id = str(uuid.uuid4())
    traj_dir = tmp_path / f"traj-{uuid.uuid4().hex[:8]}"
    traj_dir.mkdir()
    ss_dir = traj_dir / "screenshots"
    ss_dir.mkdir()
    (ss_dir / "0001.png").write_bytes(_one_pixel_png())
    (ss_dir / "0002.png").write_bytes(_one_pixel_png())
    events: list[dict[str, Any]] = [
        {
            "seq": 1,
            "t": _iso(0.0),
            "kind": "app_switch",
            "bundle_id": "com.google.Chrome",
            "screenshot_ref": "screenshots/0001.png",
        },
        {
            "seq": 2,
            "t": _iso(1.0),
            "kind": "click",
            "x": 100,
            "y": 200,
            "button": "left",
            "bundle_id": "com.google.Chrome",
            "target": {
                "label": "Send",
                "role": "button",
                "bundle_id": "com.google.Chrome",
            },
            "screenshot_ref": "screenshots/0002.png",
        },
    ]
    metadata = {
        "id": trajectory_id,
        "started_at": _iso(0.0),
        "stopped_at": _iso(2.0),
        "label": "test",
        "display_info": {"width": 2560, "height": 1440, "scale": 2.0},
        "app_focus_history": [
            {"at": _iso(0.0), "bundle_id": "com.google.Chrome", "title": "Test"}
        ],
    }
    (traj_dir / "metadata.json").write_text(json.dumps(metadata))
    with (traj_dir / "events.jsonl").open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    return TrajectoryReader(traj_dir)


def _make_parsed_skill() -> ParsedSkill:
    return ParsedSkill(
        title="Reply to Gmail",
        description="Reply to the newest unread Gmail message.",
        parameters=[
            Parameter(
                name="message_body",
                type="string",
                required=True,
                default=None,
                description="Body text of the reply.",
            )
        ],
        preconditions=["Chrome is open"],
        steps=[
            Step(
                number=1,
                text="Type {message_body} into the reply composer.",
                destructive=False,
            ),
            Step(
                number=2,
                text='Click "Send" to deliver the reply.',
                destructive=True,
            ),
        ],
        expected_outcome="The reply is sent successfully.",
        notes=None,
    )


def _make_draft_result(
    trajectory_id: str,
    *,
    slug: str = "gmail_reply",
    questions: list[Question] | None = None,
    total_cost_usd: float = 0.0,
) -> DraftResult:
    parsed = _make_parsed_skill()
    from synthesizer.skill_doc import render_skill_md

    markdown = render_skill_md(parsed)
    meta: dict[str, Any] = {
        "slug": slug,
        "name": "Reply to Gmail",
        "trajectory_id": trajectory_id,
        "created_at": _iso(0.0).replace("Z", "+00:00"),
        "parameters": [
            {"name": "message_body", "type": "string", "required": True}
        ],
        "destructive_steps": [2],
        "preconditions": ["Chrome is open"],
        "step_count": 2,
    }
    return DraftResult(
        markdown=markdown,
        parsed=parsed,
        meta=meta,
        questions=list(questions) if questions is not None else [],
        llm_calls=1,
        total_cost_usd=total_cost_usd,
    )


def _q(qid: str, text: str, category: str = "parameterization") -> Question:
    return Question(id=qid, category=category, text=text)


@pytest.fixture
def trajectory(tmp_path: Path) -> TrajectoryReader:
    return _make_trajectory(tmp_path)


@pytest.fixture
def client() -> LLMClient:
    return LLMClient()


@pytest.fixture
def skills_root(tmp_path: Path) -> Path:
    return tmp_path / "skills"


# --- happy path ------------------------------------------------------------


def test_full_happy_path_three_questions_then_approve(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    trajectory_id = trajectory.metadata["id"]
    questions = [_q("q1", "first?"), _q("q2", "second?"), _q("q3", "third?")]
    initial_draft = _make_draft_result(trajectory_id, questions=questions)

    # Each revision prunes one question (simulating the LLM removing
    # answered questions) and bumps cumulative cost.
    revision_calls: list[tuple[str, str]] = []

    def _fake_revise(
        *,
        current_draft: DraftResult,
        question: Question,
        answer: str,
        client: LLMClient,
        reader: TrajectoryReader,
    ) -> DraftResult:
        revision_calls.append((question.id, answer))
        remaining = [q for q in current_draft.questions if q.id != question.id]
        return _make_draft_result(
            trajectory_id,
            questions=remaining,
            total_cost_usd=current_draft.total_cost_usd + 0.01,
        )

    def _fake_draft(
        preprocessed: Any, client: LLMClient, *, reader: TrajectoryReader
    ) -> DraftResult:
        return initial_draft

    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        draft_fn=_fake_draft,
        revise_fn=_fake_revise,
    )

    assert session.state == SessionState.GENERATING_DRAFT
    session.start_draft()
    assert session.state == SessionState.AWAITING_ANSWER
    assert [q.id for q in session.questions] == ["q1", "q2", "q3"]
    assert session.next_question() is not None
    assert session.next_question().id == "q1"  # type: ignore[union-attr]

    session.answer_question("q1", "ans1")
    assert session.state == SessionState.AWAITING_ANSWER
    session.answer_question("q2", "ans2")
    assert session.state == SessionState.AWAITING_ANSWER
    session.answer_question("q3", "ans3")
    assert session.state == SessionState.AWAITING_APPROVAL
    assert revision_calls == [("q1", "ans1"), ("q2", "ans2"), ("q3", "ans3")]
    assert session.costs_so_far_usd == pytest.approx(0.03)

    written = session.approve()
    assert session.state == SessionState.COMPLETED
    assert written.slug == "gmail_reply"
    assert written.path.is_dir()
    assert (written.path / "SKILL.md").is_file()
    # Persistence dir is cleaned up on completion.
    assert not session_dir(session.session_id).exists()


# --- no questions → skip straight to awaiting_approval ---------------------


def test_draft_with_no_questions_skips_awaiting_answer(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    trajectory_id = trajectory.metadata["id"]
    draft = _make_draft_result(trajectory_id, questions=[])

    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        draft_fn=lambda preprocessed, client, *, reader: draft,
        revise_fn=None,
    )
    session.start_draft()
    assert session.state == SessionState.AWAITING_APPROVAL


# --- illegal transitions ---------------------------------------------------


def test_answer_question_during_generating_draft_raises(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    session = SynthesisSession(
        trajectory_id=trajectory.metadata["id"],
        reader=trajectory,
        client=client,
        skills_root=skills_root,
    )
    assert session.state == SessionState.GENERATING_DRAFT
    with pytest.raises(IllegalStateTransition):
        session.answer_question("q1", "whatever")


def test_approve_before_draft_raises(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    session = SynthesisSession(
        trajectory_id=trajectory.metadata["id"],
        reader=trajectory,
        client=client,
        skills_root=skills_root,
    )
    with pytest.raises(IllegalStateTransition):
        session.approve()


def test_start_draft_after_completion_raises(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    trajectory_id = trajectory.metadata["id"]
    draft = _make_draft_result(trajectory_id, questions=[])
    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        draft_fn=lambda preprocessed, client, *, reader: draft,
    )
    session.start_draft()
    session.approve()
    with pytest.raises(IllegalStateTransition):
        session.start_draft()


def test_cancel_from_terminal_state_raises(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    trajectory_id = trajectory.metadata["id"]
    draft = _make_draft_result(trajectory_id, questions=[])
    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        draft_fn=lambda preprocessed, client, *, reader: draft,
    )
    session.start_draft()
    session.approve()
    with pytest.raises(IllegalStateTransition):
        session.cancel()


# --- cancel paths ----------------------------------------------------------


def test_cancel_from_generating_draft(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    session = SynthesisSession(
        trajectory_id=trajectory.metadata["id"],
        reader=trajectory,
        client=client,
        skills_root=skills_root,
    )
    session.cancel()
    assert session.state == SessionState.CANCELLED
    assert not skills_root.exists() or not any(skills_root.iterdir())


def test_cancel_from_awaiting_answer_cleans_persistence(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    trajectory_id = trajectory.metadata["id"]
    draft = _make_draft_result(trajectory_id, questions=[_q("q1", "?")])
    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        draft_fn=lambda preprocessed, client, *, reader: draft,
    )
    session.start_draft()
    persist_dir = session_dir(session.session_id)
    assert persist_dir.exists()
    session.cancel()
    assert session.state == SessionState.CANCELLED
    assert not persist_dir.exists()


def test_cancel_from_awaiting_approval_writes_no_skill(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    trajectory_id = trajectory.metadata["id"]
    draft = _make_draft_result(trajectory_id, questions=[])
    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        draft_fn=lambda preprocessed, client, *, reader: draft,
    )
    session.start_draft()
    assert session.state == SessionState.AWAITING_APPROVAL
    session.cancel()
    assert session.state == SessionState.CANCELLED
    # No skill directory was written for this slug.
    assert not (skills_root / "gmail_reply").exists()


# --- error paths -----------------------------------------------------------


def test_draft_generation_error_transitions_to_errored(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    def _bad_draft(
        preprocessed: Any, client: LLMClient, *, reader: TrajectoryReader
    ) -> DraftResult:
        raise DraftGenerationError(
            "simulated failure", attempts=[], last_error="bad"
        )

    session = SynthesisSession(
        trajectory_id=trajectory.metadata["id"],
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        draft_fn=_bad_draft,
    )
    session.start_draft()
    assert session.state == SessionState.ERRORED
    assert session.error is not None
    assert "simulated failure" in session.error


def test_revision_error_transitions_to_errored(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    trajectory_id = trajectory.metadata["id"]
    draft = _make_draft_result(trajectory_id, questions=[_q("q1", "?")])

    def _bad_revise(
        *,
        current_draft: DraftResult,
        question: Question,
        answer: str,
        client: LLMClient,
        reader: TrajectoryReader,
    ) -> DraftResult:
        raise DraftGenerationError(
            "revision LLM failed", attempts=[], last_error="nope"
        )

    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        draft_fn=lambda preprocessed, client, *, reader: draft,
        revise_fn=_bad_revise,
    )
    session.start_draft()
    session.answer_question("q1", "ans")
    assert session.state == SessionState.ERRORED
    assert session.error is not None
    assert "revision" in session.error


# --- approve variants ------------------------------------------------------


def test_approve_with_colliding_slug_renames_via_resolve_unique(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    """LLM-suggested slug that collides gets a numeric suffix silently."""
    trajectory_id = trajectory.metadata["id"]
    draft = _make_draft_result(trajectory_id, questions=[])

    skills_root.mkdir(parents=True)
    (skills_root / "gmail_reply").mkdir()

    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        draft_fn=lambda preprocessed, client, *, reader: draft,
    )
    session.start_draft()
    written = session.approve()
    assert written.slug == "gmail_reply_2"


def test_approve_with_user_slug_colliding_raises_slug_error(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    """User-provided override that collides fails hard — no silent rename."""
    trajectory_id = trajectory.metadata["id"]
    draft = _make_draft_result(trajectory_id, questions=[])
    skills_root.mkdir(parents=True)
    (skills_root / "my_slug").mkdir()

    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        draft_fn=lambda preprocessed, client, *, reader: draft,
    )
    session.start_draft()
    with pytest.raises(SlugError):
        session.approve(slug="my_slug")
    assert session.state == SessionState.ERRORED


def test_approve_with_valid_user_slug(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    trajectory_id = trajectory.metadata["id"]
    draft = _make_draft_result(trajectory_id, questions=[])

    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        draft_fn=lambda preprocessed, client, *, reader: draft,
    )
    session.start_draft()
    written = session.approve(slug="custom_name")
    assert written.slug == "custom_name"
    assert written.path == skills_root / "custom_name"


# --- answer validation -----------------------------------------------------


def test_empty_answer_raises_value_error(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    trajectory_id = trajectory.metadata["id"]
    draft = _make_draft_result(trajectory_id, questions=[_q("q1", "?")])

    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        draft_fn=lambda preprocessed, client, *, reader: draft,
        revise_fn=lambda **kw: draft,  # should never be called
    )
    session.start_draft()
    with pytest.raises(ValueError):
        session.answer_question("q1", "   ")
    # State is unchanged on reject.
    assert session.state == SessionState.AWAITING_ANSWER


def test_unknown_question_id_raises(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    trajectory_id = trajectory.metadata["id"]
    draft = _make_draft_result(trajectory_id, questions=[_q("q1", "?")])

    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        draft_fn=lambda preprocessed, client, *, reader: draft,
        revise_fn=lambda **kw: draft,
    )
    session.start_draft()
    with pytest.raises(ValueError):
        session.answer_question("does_not_exist", "ans")


# --- persistence -----------------------------------------------------------


def test_draft_persistence_file_written_after_start_and_revision(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    trajectory_id = trajectory.metadata["id"]
    draft1 = _make_draft_result(trajectory_id, questions=[_q("q1", "?")])
    draft2 = _make_draft_result(
        trajectory_id, questions=[], total_cost_usd=0.02
    )

    def _revise(**kwargs: Any) -> DraftResult:
        return draft2

    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        draft_fn=lambda preprocessed, client, *, reader: draft1,
        revise_fn=_revise,
    )
    session.start_draft()
    path = session_dir(session.session_id) / "draft.json"
    assert path.is_file()
    payload1 = json.loads(path.read_text())
    assert payload1["state"] == SessionState.AWAITING_ANSWER.value
    assert payload1["trajectory_id"] == trajectory_id
    assert payload1["questions"][0]["id"] == "q1"

    session.answer_question("q1", "answered")
    assert path.is_file()
    payload2 = json.loads(path.read_text())
    assert payload2["state"] == SessionState.AWAITING_APPROVAL.value
    assert payload2["answered"] == [{"question_id": "q1", "answer": "answered"}]
    assert payload2["costs_so_far_usd"] == pytest.approx(0.02)


# --- SessionStore ----------------------------------------------------------


def test_session_store_add_get_remove(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    store = SessionStore()
    session = SynthesisSession(
        trajectory_id=trajectory.metadata["id"],
        reader=trajectory,
        client=client,
        skills_root=skills_root,
    )
    store.add(session)
    assert session.session_id in store
    assert store.get(session.session_id) is session
    assert len(store) == 1
    assert store.remove(session.session_id) is session
    assert store.get(session.session_id) is None
    assert len(store) == 0


def test_session_store_rejects_duplicate_add(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    store = SessionStore()
    session = SynthesisSession(
        trajectory_id=trajectory.metadata["id"],
        reader=trajectory,
        client=client,
        skills_root=skills_root,
    )
    store.add(session)
    with pytest.raises(ValueError):
        store.add(session)


def test_session_store_evicts_stale_session_past_ttl(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    """Drive a fake clock past the 30-minute TTL and confirm eviction."""
    now = [1000.0]

    def _clock() -> float:
        return now[0]

    store = SessionStore(clock=_clock)
    session = SynthesisSession(
        trajectory_id=trajectory.metadata["id"],
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        clock=_clock,
    )
    store.add(session)

    # Not yet stale.
    evicted = store.evict_stale()
    assert evicted == []
    assert session.session_id in store

    # Advance past TTL.
    now[0] += DEFAULT_SESSION_TTL_SECONDS + 1
    evicted = store.evict_stale()
    assert session.session_id in evicted
    assert session.session_id not in store


def test_session_store_evict_stale_prunes_terminal_sessions(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    trajectory_id = trajectory.metadata["id"]
    draft = _make_draft_result(trajectory_id, questions=[])

    store = SessionStore()
    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        draft_fn=lambda preprocessed, client, *, reader: draft,
    )
    session.start_draft()
    session.cancel()
    store.add(session)
    evicted = store.evict_stale()
    assert session.session_id in evicted
    assert session.session_id not in store


def test_session_store_cleanup_cancellable(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    store = SessionStore()
    assert not store.cleanup_cancelled()
    store.cancel_cleanup()
    assert store.cleanup_cancelled()


# --- snapshot --------------------------------------------------------------


def test_snapshot_shape_progresses_with_state(
    trajectory: TrajectoryReader, client: LLMClient, skills_root: Path
) -> None:
    trajectory_id = trajectory.metadata["id"]
    draft = _make_draft_result(trajectory_id, questions=[_q("q1", "first?")])
    session = SynthesisSession(
        trajectory_id=trajectory_id,
        reader=trajectory,
        client=client,
        skills_root=skills_root,
        draft_fn=lambda preprocessed, client, *, reader: draft,
    )
    snap0 = session.snapshot()
    assert snap0["state"] == SessionState.GENERATING_DRAFT.value
    assert snap0["draft"] is None
    assert snap0["next_question"] is None

    session.start_draft()
    snap1 = session.snapshot()
    assert snap1["state"] == SessionState.AWAITING_ANSWER.value
    assert snap1["draft"] is not None
    assert snap1["next_question"] is not None
    assert snap1["next_question"]["id"] == "q1"
    assert snap1["error"] is None
