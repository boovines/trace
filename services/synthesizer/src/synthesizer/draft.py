"""Phase-1 draft generation: trajectory digest → SKILL.md + meta + questions.

This module owns the single multimodal Claude call that turns a
:class:`~synthesizer.preprocess.PreprocessedTrajectory` into a draft skill.
It does NOT drive the user-facing Q&A (that's S-012, :mod:`synthesizer.revise`).

The call flow is:

  1. Build a user message with (a) a structured text digest and (b) base64
     image blocks for each keyframe that carries a ``screenshot_ref``.
  2. Send to Claude with :data:`~synthesizer.draft_prompt.DRAFT_SYSTEM_PROMPT`.
  3. Parse the response as a single JSON object with keys ``{markdown, meta,
     questions}``. On failure, append a corrective user turn and retry —
     once per failure mode, capped at :data:`MAX_LLM_CALLS` total.
  4. Validate the returned markdown with
     :func:`synthesizer.skill_doc.parse_skill_md`.
  5. Validate the returned meta with :func:`synthesizer.schema.validate_meta`
     and :func:`synthesizer.schema.validate_meta_against_markdown`.

Steps 3-5 each have exactly one retry attached. The hard cap of 3 total
LLM calls per :func:`generate_draft` invocation is enforced regardless of
which validation failed — a flaky model could hit every retry path and we
must not recurse indefinitely.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from jsonschema import ValidationError
from pydantic import BaseModel, ConfigDict

from synthesizer.destructive_matcher import (
    MatcherReport,
    apply_destructive_matcher,
)
from synthesizer.draft_prompt import DRAFT_OUTPUT_KEYS, DRAFT_SYSTEM_PROMPT
from synthesizer.llm_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    LLMClient,
    LLMResponse,
)
from synthesizer.preprocess import PreprocessedTrajectory
from synthesizer.schema import validate_meta, validate_meta_against_markdown
from synthesizer.skill_doc import (
    ParsedSkill,
    SkillParseError,
    parse_skill_md,
    render_skill_md,
)
from synthesizer.trajectory_reader import TrajectoryReader

LOGGER = logging.getLogger(__name__)

__all__ = [
    "MAX_LLM_CALLS",
    "DraftGenerationError",
    "DraftResult",
    "Question",
    "build_user_content",
    "generate_draft",
]

MAX_LLM_CALLS: int = 3
"""Hard upper bound on LLM round trips per :func:`generate_draft` call.

Initial call plus up to 2 corrective retries. A model that fails every retry
path would otherwise loop forever; see PRD S-007.
"""


class DraftGenerationError(RuntimeError):
    """Raised when draft generation fails after exhausting retries.

    The ``attempts`` attribute exposes the sequence of
    :class:`~synthesizer.llm_client.LLMResponse` objects that were collected
    along the way, for diagnostics and cost accounting at the call site.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: Sequence[LLMResponse],
        last_error: str,
    ) -> None:
        super().__init__(message)
        self.attempts: list[LLMResponse] = list(attempts)
        self.last_error: str = last_error


class Question(BaseModel):
    """One follow-up question produced by the drafting model."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    category: str
    text: str


@dataclass(frozen=True)
class DraftResult:
    """Output of :func:`generate_draft`.

    ``markdown`` is the SKILL.md string (already verified to parse via
    :func:`~synthesizer.skill_doc.parse_skill_md`). ``parsed`` is the result
    of that parse, cached for downstream consumers. ``meta`` is the validated
    meta dict. ``questions`` is the (possibly-empty, up to 5) list of
    follow-ups. ``llm_calls`` counts real LLM round trips made to produce
    this result. ``total_cost_usd`` is the cumulative billed cost across
    those calls.
    """

    markdown: str
    parsed: ParsedSkill
    meta: dict[str, Any]
    questions: list[Question]
    llm_calls: int
    total_cost_usd: float
    matcher_report: MatcherReport = field(default_factory=MatcherReport)


# --- Message building -------------------------------------------------------


def _format_digest_line(index: int, entry: Any) -> str:
    return (
        f"[{index}] t+{entry.timestamp_ms}ms kind={entry.kind}: "
        f"{entry.summary_text}"
    )


def _digest_text(preprocessed: PreprocessedTrajectory) -> str:
    lines = [
        _format_digest_line(i, entry)
        for i, entry in enumerate(preprocessed.digest, start=1)
    ]
    header = (
        f"Recorded trajectory digest ({preprocessed.digest_entry_count} entries, "
        f"{preprocessed.screenshots_included} screenshots):"
    )
    body = "\n".join(lines) if lines else "(empty digest)"
    return header + "\n\n" + body


def _encode_screenshot(path: Any) -> str:
    """Return a base64-encoded PNG payload for a screenshot path."""
    with open(path, "rb") as f:
        data = f.read()
    return base64.standard_b64encode(data).decode("ascii")


def build_user_content(
    preprocessed: PreprocessedTrajectory,
    reader: TrajectoryReader,
) -> list[dict[str, Any]]:
    """Construct the user-message content blocks for the draft call.

    Returns a list of Anthropic content blocks: exactly one ``text`` block
    containing the digest, followed by up to 20 ``image`` blocks (one per
    selected keyframe with a resolvable screenshot on disk).

    Screenshots whose ``screenshot_ref`` resolves to a missing file are
    silently skipped — preprocess already capped the count at 20 and the
    trajectory reader already warned on the missing file.
    """
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": _digest_text(preprocessed)}
    ]
    for entry in preprocessed.digest:
        if not entry.screenshot_ref:
            continue
        path = reader.directory / entry.screenshot_ref
        if not path.is_file():
            continue
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _encode_screenshot(path),
                },
            }
        )
    return blocks


# --- Response validation ----------------------------------------------------


class _ResponseValidationError(Exception):
    """Raised internally when a draft response fails a validation step.

    Carries a user-facing ``feedback`` string that gets fed back to the model
    as the next user turn on retry.
    """

    def __init__(self, feedback: str) -> None:
        super().__init__(feedback)
        self.feedback = feedback


def _parse_response_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    # The model may (incorrectly) wrap the JSON in ```json fences; forgive
    # this one common mistake rather than forcing a retry over it.
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise _ResponseValidationError(
            "Your previous response was not valid JSON: "
            f"{e.msg} (line {e.lineno}, column {e.colno}). "
            "Please return a single JSON object with keys "
            "markdown, meta, questions — no code fences, no commentary."
        ) from e
    if not isinstance(obj, dict):
        raise _ResponseValidationError(
            "Your previous response parsed as JSON but was not a single "
            "object. Please return one JSON object with keys "
            "markdown, meta, questions."
        )
    missing = [k for k in DRAFT_OUTPUT_KEYS if k not in obj]
    if missing:
        raise _ResponseValidationError(
            "Your previous response JSON is missing required keys: "
            f"{missing}. The object must contain exactly "
            "markdown (string), meta (object), questions (array)."
        )
    return obj


def _validate_markdown(obj: dict[str, Any]) -> tuple[str, ParsedSkill]:
    markdown = obj.get("markdown")
    if not isinstance(markdown, str):
        raise _ResponseValidationError(
            "The 'markdown' field must be a string containing the full "
            "SKILL.md body."
        )
    try:
        parsed = parse_skill_md(markdown)
    except SkillParseError as e:
        raise _ResponseValidationError(
            "Your previous SKILL.md failed to parse: "
            f"[section={e.section} line={e.line}] {e.reason}. "
            "Please fix this and return the corrected full JSON object."
        ) from e
    return markdown, parsed


def _validate_meta_and_cross(
    obj: dict[str, Any], markdown: str
) -> dict[str, Any]:
    meta = obj.get("meta")
    if not isinstance(meta, dict):
        raise _ResponseValidationError(
            "The 'meta' field must be a JSON object following the "
            "skill.meta.json schema."
        )
    try:
        validate_meta(meta)
    except ValidationError as e:
        pointer = "/".join(str(p) for p in e.absolute_path) or "<root>"
        raise _ResponseValidationError(
            "Your previous meta object failed schema validation at "
            f"{pointer}: {e.message}. Please fix the meta and return the "
            "corrected full JSON object."
        ) from e
    try:
        validate_meta_against_markdown(meta, markdown)
    except ValidationError as e:
        raise _ResponseValidationError(
            "Your previous markdown and meta disagree: "
            f"{e.message}. Fix both so destructive_steps, step_count, and "
            "parameter references match, then return the corrected full JSON object."
        ) from e
    return meta


def _extract_questions(obj: dict[str, Any]) -> list[Question]:
    raw = obj.get("questions")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise _ResponseValidationError(
            "The 'questions' field must be a JSON array of question "
            "objects (or an empty array)."
        )
    if len(raw) > 5:
        raise _ResponseValidationError(
            f"You returned {len(raw)} questions; the limit is 5. "
            "Drop the least important ones and return the corrected full JSON object."
        )
    questions: list[Question] = []
    for i, q in enumerate(raw):
        if not isinstance(q, dict):
            raise _ResponseValidationError(
                f"Question index {i} is not a JSON object — each question "
                "must have keys id, category, text."
            )
        try:
            questions.append(Question.model_validate(q))
        except Exception as e:  # pydantic ValidationError
            raise _ResponseValidationError(
                f"Question index {i} is malformed: {e}. Each question needs "
                "string fields id, category, text."
            ) from e
    return questions


def _validate_full_response(
    response: LLMResponse,
) -> tuple[str, ParsedSkill, dict[str, Any], list[Question]]:
    """Run every validation gate. Raises :class:`_ResponseValidationError`."""
    obj = _parse_response_json(response.text)
    markdown, parsed = _validate_markdown(obj)
    meta = _validate_meta_and_cross(obj, markdown)
    questions = _extract_questions(obj)
    return markdown, parsed, meta, questions


# --- Entry point ------------------------------------------------------------


def generate_draft(
    preprocessed: PreprocessedTrajectory,
    client: LLMClient,
    *,
    reader: TrajectoryReader,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    context_label: str = "synthesizer:draft",
) -> DraftResult:
    """Generate a draft SKILL.md + meta + follow-up questions from a trajectory.

    Uses ``client`` for the underlying LLM call(s); pass a fake-mode client
    in tests. ``reader`` is required so keyframe screenshots can be
    base64-encoded into the user message.

    Retries once per validation failure class (JSON parse, markdown parse,
    meta validation). Raises :class:`DraftGenerationError` after
    :data:`MAX_LLM_CALLS` attempts.
    """
    user_content = build_user_content(preprocessed, reader)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]

    attempts: list[LLMResponse] = []
    last_error: str = ""

    for attempt in range(MAX_LLM_CALLS):
        response = client.complete(
            messages=messages,
            system=DRAFT_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            model=model,
            context_label=f"{context_label}:attempt={attempt + 1}",
        )
        attempts.append(response)

        try:
            markdown, parsed, meta, questions = _validate_full_response(response)
        except _ResponseValidationError as err:
            last_error = err.feedback
            LOGGER.info(
                "Draft attempt %d/%d failed validation: %s",
                attempt + 1,
                MAX_LLM_CALLS,
                err.feedback,
            )
            # Append the bad assistant turn + corrective user turn and loop.
            messages.append({"role": "assistant", "content": response.text})
            messages.append({"role": "user", "content": err.feedback})
            continue

        # Belt-and-suspenders: independent keyword scan over the source
        # clicks. Matches are strictly additive — the LLM's flags are never
        # unset. When the matcher adds a flag, the markdown and meta are
        # re-derived so the final DraftResult is internally consistent
        # (validate_meta_against_markdown must still pass).
        matcher = apply_destructive_matcher(parsed, reader)
        if matcher.report.added_flags:
            parsed = matcher.parsed
            markdown = render_skill_md(parsed)
            meta = dict(meta)
            combined = sorted(
                set(meta.get("destructive_steps", []))
                | set(matcher.report.added_flags)
            )
            meta["destructive_steps"] = combined
            validate_meta(meta)
            validate_meta_against_markdown(meta, markdown)

        return DraftResult(
            markdown=markdown,
            parsed=parsed,
            meta=meta,
            questions=questions,
            llm_calls=len(attempts),
            total_cost_usd=sum(r.cost_estimate_usd for r in attempts),
            matcher_report=matcher.report,
        )

    raise DraftGenerationError(
        f"Draft generation exhausted {MAX_LLM_CALLS} LLM calls without "
        f"producing a valid response. Last error: {last_error}",
        attempts=attempts,
        last_error=last_error,
    )
