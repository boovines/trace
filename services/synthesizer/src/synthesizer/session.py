"""In-memory synthesis session state machine.

A :class:`SynthesisSession` tracks the state of a single in-progress
synthesis from trajectory → draft → Q&A → approval → written skill. The HTTP
API (S-013) owns a :class:`SessionStore` that holds live sessions by id so
the UI can poll status or resume a Q&A round across multiple HTTP requests.

State transitions are explicit and enforced — an out-of-order call (e.g.
``answer_question`` while the draft is still generating) raises
:class:`IllegalStateTransition` rather than silently corrupting session
state. The :class:`SessionStore` evicts non-terminal sessions older than
its TTL so a crashed UI never leaks memory indefinitely.

Persistence is intentionally one-directional: each successful draft /
revision writes ``draft.json`` under
``~/Library/Application Support/Trace[-dev]/synthesis_sessions/<session_id>/``
so a synthesizer-service crash leaves the user's work recoverable by hand.
We do NOT rehydrate sessions on startup in v1 — that's a documented PRD
limitation.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from synthesizer.draft import (
    DraftGenerationError,
    DraftResult,
    Question,
    generate_draft,
)
from synthesizer.llm_client import LLMClient, trace_data_dir
from synthesizer.preprocess import preprocess_trajectory
from synthesizer.slug import SlugError, resolve_unique_slug, validate_user_slug
from synthesizer.trajectory_reader import TrajectoryReader
from synthesizer.writer import SkillWriter, WrittenSkill

__all__ = [
    "DEFAULT_SESSION_TTL_SECONDS",
    "DraftFn",
    "IllegalStateTransition",
    "ReviseFn",
    "SessionState",
    "SessionStore",
    "SynthesisSession",
    "session_dir",
    "sessions_root",
]

LOGGER = logging.getLogger(__name__)

DEFAULT_SESSION_TTL_SECONDS: float = 30 * 60
"""Non-terminal sessions older than this are evicted by :meth:`SessionStore.evict_stale`."""


class SessionState(StrEnum):
    """Lifecycle states a :class:`SynthesisSession` can occupy."""

    GENERATING_DRAFT = "generating_draft"
    AWAITING_ANSWER = "awaiting_answer"
    GENERATING_REVISION = "generating_revision"
    AWAITING_APPROVAL = "awaiting_approval"
    SAVING = "saving"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERRORED = "errored"


TERMINAL_STATES: frozenset[SessionState] = frozenset(
    {SessionState.COMPLETED, SessionState.CANCELLED, SessionState.ERRORED}
)

# Map of (from_state) -> set of legal (to_state) transitions. Every session
# mutation funnels through :meth:`SynthesisSession._transition` which consults
# this table; an invalid move raises IllegalStateTransition. Keeping the table
# declarative makes new states easy to add and mistakes hard to miss.
_LEGAL_TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    SessionState.GENERATING_DRAFT: frozenset(
        {
            SessionState.AWAITING_ANSWER,
            SessionState.AWAITING_APPROVAL,
            SessionState.ERRORED,
            SessionState.CANCELLED,
        }
    ),
    SessionState.AWAITING_ANSWER: frozenset(
        {
            SessionState.GENERATING_REVISION,
            SessionState.ERRORED,
            SessionState.CANCELLED,
        }
    ),
    SessionState.GENERATING_REVISION: frozenset(
        {
            SessionState.AWAITING_ANSWER,
            SessionState.AWAITING_APPROVAL,
            SessionState.ERRORED,
            SessionState.CANCELLED,
        }
    ),
    SessionState.AWAITING_APPROVAL: frozenset(
        {
            SessionState.SAVING,
            SessionState.ERRORED,
            SessionState.CANCELLED,
        }
    ),
    SessionState.SAVING: frozenset(
        {SessionState.COMPLETED, SessionState.ERRORED}
    ),
    SessionState.COMPLETED: frozenset(),
    SessionState.CANCELLED: frozenset(),
    SessionState.ERRORED: frozenset(),
}


class IllegalStateTransition(RuntimeError):
    """Raised when a session method is called in a state that forbids it."""


class DraftFn(Protocol):
    """Callable shape for draft generation — matches :func:`synthesizer.draft.generate_draft`."""

    def __call__(
        self,
        preprocessed: Any,
        client: LLMClient,
        *,
        reader: TrajectoryReader,
    ) -> DraftResult: ...


class ReviseFn(Protocol):
    """Callable shape for revision generation.

    S-012 will supply :func:`synthesizer.revise.generate_revision`; until
    then the session accepts any callable matching this protocol so tests
    can inject deterministic behavior without the real LLM path.
    """

    def __call__(
        self,
        *,
        current_draft: DraftResult,
        question: Question,
        answer: str,
        client: LLMClient,
        reader: TrajectoryReader,
    ) -> DraftResult: ...


# --- Persistence path helpers ----------------------------------------------


def sessions_root() -> Path:
    """Parent directory holding per-session persistence subdirs."""
    return trace_data_dir() / "synthesis_sessions"


def session_dir(session_id: str) -> Path:
    """Per-session persistence directory."""
    return sessions_root() / session_id


# --- Session --------------------------------------------------------------


@dataclass
class _AnsweredQuestion:
    question_id: str
    answer: str


class SynthesisSession:
    """A single in-progress synthesis lifecycle.

    Construction is cheap — it allocates a session id and stores references
    to the injected :class:`~synthesizer.llm_client.LLMClient` and
    :class:`~synthesizer.trajectory_reader.TrajectoryReader`. No LLM work
    happens until :meth:`start_draft` is called; the HTTP route handler is
    expected to run ``start_draft`` in a background task so the POST /start
    endpoint can return an id immediately.

    The :attr:`draft`, :attr:`questions`, :attr:`answered` and
    :attr:`costs_so_far_usd` attributes are the UI-facing snapshot of
    session progress; the API layer (S-013) renders these into the HTTP
    response and the SSE event stream.
    """

    def __init__(
        self,
        *,
        trajectory_id: str,
        reader: TrajectoryReader,
        client: LLMClient,
        skills_root: Path,
        session_id: str | None = None,
        draft_fn: DraftFn | None = None,
        revise_fn: ReviseFn | None = None,
        writer: SkillWriter | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.session_id: str = session_id or uuid.uuid4().hex
        self.trajectory_id: str = trajectory_id
        self.state: SessionState = SessionState.GENERATING_DRAFT
        self.draft: DraftResult | None = None
        self.questions: list[Question] = []
        self.answered: list[_AnsweredQuestion] = []
        self.error: str | None = None
        self.costs_so_far_usd: float = 0.0

        self._reader = reader
        self._client = client
        self._skills_root = Path(skills_root)
        self._draft_fn: DraftFn = draft_fn or generate_draft
        self._revise_fn: ReviseFn | None = revise_fn
        self._writer: SkillWriter = writer or SkillWriter()
        self._clock: Callable[[], float] = clock or time.monotonic
        self._last_updated_at: float = self._clock()

    # -- state mutation ----------------------------------------------------

    def _transition(self, target: SessionState) -> None:
        allowed = _LEGAL_TRANSITIONS[self.state]
        if target not in allowed:
            legal_names = [s.value for s in sorted(allowed, key=lambda s: s.value)]
            raise IllegalStateTransition(
                f"Cannot transition from {self.state.value} to {target.value}; "
                f"legal next states are {legal_names}."
            )
        LOGGER.info(
            "session %s: %s -> %s", self.session_id, self.state.value, target.value
        )
        self.state = target
        self._last_updated_at = self._clock()

    @property
    def last_updated_at(self) -> float:
        """Clock reading at the last state transition; used by the store for TTL."""
        return self._last_updated_at

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable view of the session for HTTP responses."""
        nxt = self.next_question()
        return {
            "session_id": self.session_id,
            "trajectory_id": self.trajectory_id,
            "state": self.state.value,
            "draft": None
            if self.draft is None
            else {
                "markdown": self.draft.markdown,
                "meta": self.draft.meta,
            },
            "questions": [q.model_dump() for q in self.questions],
            "answered": [
                {"question_id": a.question_id, "answer": a.answer}
                for a in self.answered
            ],
            "next_question": nxt.model_dump() if nxt is not None else None,
            "costs_so_far_usd": self.costs_so_far_usd,
            "error": self.error,
        }

    # -- draft phase -------------------------------------------------------

    def start_draft(self) -> None:
        """Run phase-1 draft generation.

        Blocking call; callers that need non-blocking behaviour (the HTTP
        endpoint) should wrap this in a background task. On success the
        session advances to :attr:`SessionState.AWAITING_ANSWER` (or directly
        to :attr:`SessionState.AWAITING_APPROVAL` if the LLM returned no
        follow-up questions). On :class:`DraftGenerationError` the session
        transitions to :attr:`SessionState.ERRORED` with :attr:`error` set.
        """
        if self.state != SessionState.GENERATING_DRAFT:
            raise IllegalStateTransition(
                f"start_draft requires state={SessionState.GENERATING_DRAFT.value}, "
                f"got {self.state.value}."
            )
        try:
            preprocessed = preprocess_trajectory(self._reader)
            draft = self._draft_fn(
                preprocessed, self._client, reader=self._reader
            )
        except DraftGenerationError as err:
            self._fail(f"draft generation failed: {err}")
            return
        except Exception as err:  # surface every failure to UI
            self._fail(f"unexpected error during draft generation: {err}")
            return

        self.draft = draft
        self.questions = list(draft.questions)
        self.costs_so_far_usd = draft.total_cost_usd
        if self.questions:
            self._transition(SessionState.AWAITING_ANSWER)
        else:
            self._transition(SessionState.AWAITING_APPROVAL)
        self._persist_draft()

    # -- Q&A phase ---------------------------------------------------------

    def next_question(self) -> Question | None:
        """Next unanswered question, or ``None`` if the queue is empty."""
        answered_ids = {a.question_id for a in self.answered}
        for q in self.questions:
            if q.id not in answered_ids:
                return q
        return None

    def answer_question(self, question_id: str, text: str) -> None:
        """Submit an answer to the currently-awaited question.

        Transitions through :attr:`SessionState.GENERATING_REVISION` while
        the revision LLM call is in flight, then lands at
        :attr:`SessionState.AWAITING_ANSWER` (if more questions remain) or
        :attr:`SessionState.AWAITING_APPROVAL` (if all done). Empty-string
        answers are rejected before any LLM call.
        """
        if self.state != SessionState.AWAITING_ANSWER:
            raise IllegalStateTransition(
                f"answer_question requires state={SessionState.AWAITING_ANSWER.value}, "
                f"got {self.state.value}."
            )
        if not text.strip():
            raise ValueError("answer text must be non-empty")
        if self._revise_fn is None:
            raise RuntimeError(
                "SynthesisSession constructed without a revise_fn — the "
                "caller must supply one before answer_question is called."
            )
        question = self._find_question(question_id)
        if question is None:
            raise ValueError(
                f"unknown question_id {question_id!r}; "
                f"known ids: {[q.id for q in self.questions]}"
            )
        if self.draft is None:
            raise RuntimeError("answer_question called before draft was set")

        self._transition(SessionState.GENERATING_REVISION)
        try:
            revised = self._revise_fn(
                current_draft=self.draft,
                question=question,
                answer=text,
                client=self._client,
                reader=self._reader,
            )
        except DraftGenerationError as err:
            self._fail(f"revision generation failed: {err}")
            return
        except Exception as err:
            self._fail(f"unexpected error during revision: {err}")
            return

        self.answered.append(_AnsweredQuestion(question_id=question_id, answer=text))
        self.draft = revised
        # The revision LLM may prune questions rendered moot by the answer;
        # keep the remaining set but preserve the ids of answers we've
        # already logged so next_question() skips them.
        self.questions = list(revised.questions)
        self.costs_so_far_usd = revised.total_cost_usd

        if self.next_question() is not None:
            self._transition(SessionState.AWAITING_ANSWER)
        else:
            self._transition(SessionState.AWAITING_APPROVAL)
        self._persist_draft()

    def _find_question(self, question_id: str) -> Question | None:
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    # -- approval / cancel -------------------------------------------------

    def approve(self, *, slug: str | None = None) -> WrittenSkill:
        """Finalize the draft to disk via :class:`SkillWriter`.

        Optional ``slug`` overrides the LLM-suggested slug; it is validated
        via :func:`synthesizer.slug.validate_user_slug` and a user-provided
        value that collides with an existing skill raises
        :class:`~synthesizer.slug.SlugError`. An LLM-suggested slug that
        collides gets the numeric-suffix treatment from
        :func:`~synthesizer.slug.resolve_unique_slug`.
        """
        if self.state != SessionState.AWAITING_APPROVAL:
            raise IllegalStateTransition(
                f"approve requires state={SessionState.AWAITING_APPROVAL.value}, "
                f"got {self.state.value}."
            )
        if self.draft is None:
            raise RuntimeError("approve called before draft was set")

        self._transition(SessionState.SAVING)
        try:
            final_slug = self._resolve_slug(slug)
            meta = dict(self.draft.meta)
            meta["slug"] = final_slug
            written = self._writer.write(
                self.draft.parsed,
                meta,
                self._reader,
                self._skills_root,
                cost_total_usd=self.costs_so_far_usd,
            )
        except SlugError as err:
            self._fail(f"slug rejected: {err}")
            raise
        except Exception as err:
            self._fail(f"skill write failed: {err}")
            raise

        self._transition(SessionState.COMPLETED)
        self._cleanup_persistence()
        return written

    def _resolve_slug(self, user_slug: str | None) -> str:
        base = user_slug
        if base is None:
            assert self.draft is not None  # for mypy
            base_candidate = self.draft.meta.get("slug")
            if not isinstance(base_candidate, str):
                raise SlugError(
                    "draft.meta['slug'] is missing or not a string; cannot approve."
                )
            base = base_candidate
            # LLM-suggested slugs may collide; rename silently.
            return resolve_unique_slug(base, self._skills_root)

        # User-provided override: validate strictly and do not rename.
        ok, value = validate_user_slug(base, self._skills_root)
        if not ok:
            raise SlugError(value)
        return value

    def cancel(self) -> None:
        """Abort a non-terminal session.

        Terminal sessions (completed / cancelled / errored) raise
        :class:`IllegalStateTransition` — the UI should not issue a cancel
        for a session that already completed.
        """
        if self.state in TERMINAL_STATES:
            raise IllegalStateTransition(
                f"cancel is not valid from terminal state {self.state.value}."
            )
        self._transition(SessionState.CANCELLED)
        self._cleanup_persistence()

    def _fail(self, message: str) -> None:
        self.error = message
        LOGGER.warning("session %s failed: %s", self.session_id, message)
        self._transition(SessionState.ERRORED)

    # -- persistence -------------------------------------------------------

    def _persist_draft(self) -> None:
        if self.draft is None:
            return
        path = session_dir(self.session_id)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self.session_id,
            "trajectory_id": self.trajectory_id,
            "state": self.state.value,
            "markdown": self.draft.markdown,
            "meta": self.draft.meta,
            "questions": [q.model_dump() for q in self.questions],
            "answered": [
                {"question_id": a.question_id, "answer": a.answer}
                for a in self.answered
            ],
            "costs_so_far_usd": self.costs_so_far_usd,
        }
        (path / "draft.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _cleanup_persistence(self) -> None:
        path = session_dir(self.session_id)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


# --- SessionStore ---------------------------------------------------------


class SessionStore:
    """In-memory registry of live :class:`SynthesisSession` instances.

    The store is the single owner of active sessions for the duration of the
    synthesizer process. Non-terminal sessions older than
    :attr:`ttl_seconds` are evicted by :meth:`evict_stale` (called on a
    background task when :meth:`start_cleanup` is active). The clock is
    injectable so tests can fast-forward without ``time.sleep``.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.ttl_seconds: float = ttl_seconds
        self._clock: Callable[[], float] = clock or time.monotonic
        self._sessions: dict[str, SynthesisSession] = {}
        self._cleanup_cancelled: bool = False

    @property
    def clock(self) -> Callable[[], float]:
        return self._clock

    def add(self, session: SynthesisSession) -> None:
        if session.session_id in self._sessions:
            raise ValueError(
                f"session {session.session_id!r} already registered in store"
            )
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> SynthesisSession | None:
        return self._sessions.get(session_id)

    def remove(self, session_id: str) -> SynthesisSession | None:
        return self._sessions.pop(session_id, None)

    def __contains__(self, session_id: object) -> bool:
        return session_id in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)

    def ids(self) -> list[str]:
        return list(self._sessions.keys())

    def evict_stale(self) -> list[str]:
        """Drop non-terminal sessions whose age exceeds :attr:`ttl_seconds`.

        Returns the list of evicted session ids. Terminal sessions are
        *also* pruned — once a session is completed/cancelled/errored the
        UI no longer needs it in memory and this is the cheapest place to
        reclaim the space.
        """
        now = self._clock()
        evicted: list[str] = []
        for sid, session in list(self._sessions.items()):
            age = now - session.last_updated_at
            if session.state in TERMINAL_STATES:
                del self._sessions[sid]
                evicted.append(sid)
                continue
            if age > self.ttl_seconds:
                LOGGER.warning(
                    "Evicting stale session %s (state=%s, age=%.0fs > ttl=%.0fs)",
                    sid,
                    session.state.value,
                    age,
                    self.ttl_seconds,
                )
                del self._sessions[sid]
                evicted.append(sid)
        return evicted

    # Async cleanup task — deliberately decoupled from asyncio so the
    # synchronous unit tests never need an event loop. The API layer (S-013)
    # will wrap this in an asyncio.create_task and cancel it at shutdown.

    def cancel_cleanup(self) -> None:
        """Flip the cleanup flag so a running cleanup loop exits on next tick."""
        self._cleanup_cancelled = True

    def cleanup_cancelled(self) -> bool:
        return self._cleanup_cancelled
