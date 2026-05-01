"""Phase-2 revision: update a draft SKILL.md + meta based on a Q&A answer.

:func:`generate_revision` consumes the current :class:`~synthesizer.draft.DraftResult`,
the follow-up :class:`~synthesizer.draft.Question` the user answered, and the
answer text — and returns a new :class:`DraftResult` with the revised
markdown, meta, and (possibly pruned) remaining questions.

The retry policy mirrors :func:`~synthesizer.draft.generate_draft`: on JSON /
markdown / meta validation failure, append the bad assistant turn plus a
corrective user turn and retry. The hard cap of three total LLM calls per
revision is enforced via :data:`MAX_REVISION_LLM_CALLS`.

The destructive-keyword secondary matcher (S-008) runs AFTER the LLM returns
a structurally-valid response — the LLM may inadvertently remove a ``⚠️``
marker while restructuring text during revision, and the matcher restores
it. This is paranoia but cheap paranoia; see the PRD S-012 notes.

Cost accounting is cumulative: the returned ``DraftResult.total_cost_usd``
is the sum of the input ``current_draft.total_cost_usd`` PLUS the cost of
every LLM attempt made by this revision call, so a session accumulating
cost across draft + multiple revisions sees a monotonically-increasing
figure.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from synthesizer.destructive_matcher import apply_destructive_matcher
from synthesizer.draft import (
    DraftGenerationError,
    DraftResult,
    Question,
    _ResponseValidationError,
    _validate_full_response,
)
from synthesizer.llm_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    LLMClient,
    LLMResponse,
)
from synthesizer.revise_prompt import REVISE_SYSTEM_PROMPT
from synthesizer.schema import validate_meta, validate_meta_against_markdown
from synthesizer.skill_doc import render_skill_md
from synthesizer.trajectory_reader import TrajectoryReader

LOGGER = logging.getLogger(__name__)

__all__ = [
    "MAX_REVISION_LLM_CALLS",
    "build_revision_user_content",
    "generate_revision",
]


MAX_REVISION_LLM_CALLS: int = 3
"""Hard upper bound on LLM round trips per :func:`generate_revision` call.

Matches :data:`synthesizer.draft.MAX_LLM_CALLS`; the two constants exist
independently so a future operator can tune revision retries without
affecting the initial draft budget.
"""


# --- User message construction --------------------------------------------


def build_revision_user_content(
    current_draft: DraftResult,
    question: Question,
    answer: str,
) -> list[dict[str, Any]]:
    """Construct the user-message content for a revision call.

    The message is a single ``text`` block containing four labelled
    sections: ``CURRENT_MARKDOWN``, ``CURRENT_META``, ``ANSWERED_QUESTION``,
    and ``REMAINING_QUESTIONS``. Screenshots are intentionally omitted —
    the revision call operates on the current draft, not the raw
    trajectory, so no multimodal context is required (keeping revisions
    cheap and deterministic).
    """
    remaining = [q for q in current_draft.questions if q.id != question.id]
    remaining_json = json.dumps(
        [q.model_dump() for q in remaining], indent=2, sort_keys=True
    )
    answered_json = json.dumps(
        {
            "id": question.id,
            "category": question.category,
            "text": question.text,
            "answer": answer,
        },
        indent=2,
        sort_keys=True,
    )
    meta_json = json.dumps(current_draft.meta, indent=2, sort_keys=True)

    body = (
        "CURRENT_MARKDOWN:\n"
        "```\n"
        f"{current_draft.markdown}\n"
        "```\n"
        "\n"
        "CURRENT_META:\n"
        f"{meta_json}\n"
        "\n"
        "ANSWERED_QUESTION:\n"
        f"{answered_json}\n"
        "\n"
        "REMAINING_QUESTIONS:\n"
        f"{remaining_json}\n"
    )
    return [{"type": "text", "text": body}]


# --- Entry point ------------------------------------------------------------


def generate_revision(
    *,
    current_draft: DraftResult,
    question: Question,
    answer: str,
    client: LLMClient,
    reader: TrajectoryReader,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    context_label: str = "synthesizer:revise",
) -> DraftResult:
    """Generate a revised draft given the user's answer to ``question``.

    Raises :class:`ValueError` before any LLM call when ``answer`` is empty
    or whitespace-only. Raises :class:`~synthesizer.draft.DraftGenerationError`
    if the revision fails every retry. The returned :class:`DraftResult`
    preserves cumulative cost (``current_draft.total_cost_usd`` + revision
    attempt costs) and cumulative call count (``current_draft.llm_calls`` +
    this call's attempts).

    The destructive-keyword matcher runs after the LLM returns a valid
    response — if it adds any flags the markdown is re-rendered and the
    meta's ``destructive_steps`` set is unioned with the added flags, then
    the cross-check validator is run again to guarantee the result is
    internally consistent.
    """
    if not answer or not answer.strip():
        raise ValueError("answer text must be non-empty")

    user_content = build_revision_user_content(current_draft, question, answer)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]

    attempts: list[LLMResponse] = []
    last_error: str = ""

    for attempt in range(MAX_REVISION_LLM_CALLS):
        response = client.complete(
            messages=messages,
            system=REVISE_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            model=model,
            context_label=f"{context_label}:q={question.id}:attempt={attempt + 1}",
        )
        attempts.append(response)

        try:
            markdown, parsed, meta, questions = _validate_full_response(response)
        except _ResponseValidationError as err:
            last_error = err.feedback
            LOGGER.info(
                "Revision attempt %d/%d failed validation: %s",
                attempt + 1,
                MAX_REVISION_LLM_CALLS,
                err.feedback,
            )
            messages.append({"role": "assistant", "content": response.text})
            messages.append({"role": "user", "content": err.feedback})
            continue

        # Re-apply the destructive-keyword matcher: the revision LLM may
        # inadvertently drop a ⚠️ marker while rewriting step text, and
        # this is the single cheap insurance policy that restores it.
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
            llm_calls=current_draft.llm_calls + len(attempts),
            total_cost_usd=current_draft.total_cost_usd
            + sum(r.cost_estimate_usd for r in attempts),
            matcher_report=matcher.report,
        )

    raise DraftGenerationError(
        f"Revision generation exhausted {MAX_REVISION_LLM_CALLS} LLM calls "
        f"without producing a valid response. Last error: {last_error}",
        attempts=attempts,
        last_error=last_error,
    )
