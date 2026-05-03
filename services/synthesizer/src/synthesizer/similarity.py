"""Snapshot similarity scoring for synthesized vs. golden SKILL.md pairs.

S-014 acceptance signal for the synthesizer: compare a generated
:class:`~synthesizer.skill_doc.ParsedSkill` against a hand-crafted golden
:class:`~synthesizer.skill_doc.ParsedSkill` and return a structured
:class:`SimilarityScore`. The overall pass/fail threshold for the real-mode
smoke test (S-017) is ``overall >= 0.80``; the destructive-match dimension is
non-negotiable and must be ``1.0`` on every reference workflow.

The score has four dimensions:

* ``overall`` — LLM-judged holistic equivalence (0.0-1.0).
* ``step_coverage`` — LLM-judged "do both describe the same workflow?".
* ``parameter_match`` — **structural**, 1.0 iff the (name, type, required)
  triple set matches between generated and golden exactly; else 0.0.
* ``destructive_match`` — **structural**, 1.0 iff the set of destructive step
  numbers matches exactly; else 0.0.

Binary facts (parameter and destructive step comparison) are computed
locally rather than asked of the LLM — that's faster, free, and removes a
source of non-determinism. The LLM call (a single Haiku 4.5 round trip)
scores only the two dimensions that need judgment.

The scorer uses Claude Haiku 4.5 because this rubric is simple and a cheaper
model keeps smoke-test runs under the $2 cap.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from synthesizer.llm_client import LLMClient
from synthesizer.skill_doc import ParsedSkill, render_skill_md

__all__ = [
    "DEFAULT_SIMILARITY_MAX_TOKENS",
    "SIMILARITY_MODEL",
    "SIMILARITY_SYSTEM_PROMPT",
    "SimilarityScore",
    "SimilarityScoringError",
    "build_similarity_user_content",
    "score_skill_similarity",
]

LOGGER = logging.getLogger(__name__)

SIMILARITY_MODEL: str = "claude-haiku-4-5"
"""Haiku 4.5 is used for the rubric call — cheap, accurate enough for
binary-leaning judgments, and roughly 5x less expensive than Sonnet.
Pricing lives in :data:`synthesizer.llm_client.PRICING_USD_PER_MTOK`.
"""

DEFAULT_SIMILARITY_MAX_TOKENS: int = 1024
"""Haiku's rubric response is short (two floats plus a reasoning blurb);
1024 is comfortable headroom without inflating cost.
"""


SIMILARITY_SYSTEM_PROMPT = """You score how semantically equivalent two SKILL.md workflow descriptions are.

You will receive a GENERATED skill (produced by the synthesizer) and a GOLDEN skill (hand-crafted ground truth). Both follow the same strict SKILL.md format: H1 title, one-line description, `## Parameters`, `## Preconditions`, `## Steps`, `## Expected outcome`, optional `## Notes`. Steps are numbered 1..N. Destructive steps are prefixed `⚠️ [DESTRUCTIVE]`.

Your job: score two dimensions on a 0.0-1.0 scale.

## Dimensions

  - step_coverage: Do the two skills describe the SAME underlying workflow at the same level of detail? Score 1.0 if every meaningful action in the golden skill is represented in the generated skill (and vice versa), regardless of wording. Score lower if the generated skill is missing steps, collapses multiple actions into one, or invents actions that the golden skill does not include.
  - overall: Holistic semantic equivalence. Weight step_coverage heavily but also consider the title, description, expected outcome, and preconditions. Score 1.0 for essentially identical skills (wording differences are fine), 0.8 for minor drift, 0.5 for a workflow that overlaps but differs materially, 0.0 for completely unrelated workflows.

Parameters and destructive steps are scored structurally by the caller — do NOT score those in this call. Focus on semantic fidelity.

## Output shape

Return EXACTLY one JSON object with these three keys and no others:

  {
    "step_coverage": 0.0-1.0,
    "overall": 0.0-1.0,
    "reasoning": "<1-3 sentences explaining the scores>"
  }

Do NOT wrap the JSON in markdown code fences. Do NOT prepend commentary. Your entire response must parse as a single JSON object. Scores must be finite floats in the closed interval [0.0, 1.0].
"""


class SimilarityScore(BaseModel):
    """Rubric-based similarity score for a generated-vs-golden skill pair.

    ``overall`` and ``step_coverage`` come from a Haiku 4.5 call.
    ``parameter_match`` and ``destructive_match`` are computed structurally
    (binary 0.0 or 1.0 based on set equality) so they do not depend on model
    output; this matters because destructive-flag fidelity is a safety
    invariant and the PRD's S-017 smoke test asserts
    ``destructive_match == 1.0`` on all five reference workflows.

    ``reasoning`` is a short prose explanation from the LLM, useful for
    smoke-report readability but not machine-consumed.
    """

    model_config = ConfigDict(frozen=True)

    overall: float = Field(ge=0.0, le=1.0)
    step_coverage: float = Field(ge=0.0, le=1.0)
    parameter_match: float = Field(ge=0.0, le=1.0)
    destructive_match: float = Field(ge=0.0, le=1.0)
    reasoning: str


class SimilarityScoringError(RuntimeError):
    """Raised when the LLM response cannot be parsed into a
    :class:`SimilarityScore`.

    The PRD intentionally does NOT prescribe retries here — unlike the
    draft/revision path, similarity scoring is a leaf operation on a
    short, structured rubric and a bad response is most likely a prompt
    or model issue worth surfacing immediately rather than masking.
    """


# --- Structural scoring helpers --------------------------------------------


def _parameter_signature(skill: ParsedSkill) -> frozenset[tuple[str, str, bool]]:
    """Return a frozenset of ``(name, type, required)`` triples for a skill.

    Defaults and descriptions are intentionally ignored — a smoke test that
    flagged a description-wording drift as a "parameter mismatch" would be
    noisy. Required-ness matters because a required vs. optional parameter
    is a meaningful behavioral difference.
    """
    return frozenset((p.name, p.type, p.required) for p in skill.parameters)


def _destructive_signature(skill: ParsedSkill) -> frozenset[int]:
    """Return a frozenset of destructive step numbers."""
    return frozenset(step.number for step in skill.steps if step.destructive)


def _compute_parameter_match(
    generated: ParsedSkill, golden: ParsedSkill
) -> float:
    return 1.0 if _parameter_signature(generated) == _parameter_signature(golden) else 0.0


def _compute_destructive_match(
    generated: ParsedSkill, golden: ParsedSkill
) -> float:
    return (
        1.0 if _destructive_signature(generated) == _destructive_signature(golden) else 0.0
    )


# --- User message construction ---------------------------------------------


def build_similarity_user_content(
    generated: ParsedSkill, golden: ParsedSkill
) -> list[dict[str, Any]]:
    """Build the single user-message content block for the rubric call.

    The LLM sees both skills rendered back through
    :func:`~synthesizer.skill_doc.render_skill_md` rather than any source
    markdown — that normalizes whitespace and guarantees the model sees the
    exact canonical form both skills will have on disk.
    """
    body = (
        "GENERATED:\n"
        "```\n"
        f"{render_skill_md(generated)}\n"
        "```\n"
        "\n"
        "GOLDEN:\n"
        "```\n"
        f"{render_skill_md(golden)}\n"
        "```\n"
    )
    return [{"type": "text", "text": body}]


# --- Response parsing ------------------------------------------------------


def _parse_rubric_response(text: str) -> tuple[float, float, str]:
    """Strictly parse the Haiku rubric response into (step_coverage, overall, reasoning).

    Tolerates an accidental ```json fence (the model occasionally adds one
    despite the prompt's instruction); everything else is a hard error.
    """
    stripped = text.strip()
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
        raise SimilarityScoringError(
            f"Similarity response was not valid JSON: {e.msg} "
            f"(line {e.lineno}, column {e.colno})."
        ) from e
    if not isinstance(obj, dict):
        raise SimilarityScoringError(
            "Similarity response parsed as JSON but was not an object."
        )
    for key in ("step_coverage", "overall", "reasoning"):
        if key not in obj:
            raise SimilarityScoringError(
                f"Similarity response is missing required key {key!r}."
            )
    step_coverage = _coerce_score(obj["step_coverage"], "step_coverage")
    overall = _coerce_score(obj["overall"], "overall")
    reasoning = obj["reasoning"]
    if not isinstance(reasoning, str):
        raise SimilarityScoringError("'reasoning' must be a string.")
    return step_coverage, overall, reasoning


def _coerce_score(value: Any, key: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as e:
        raise SimilarityScoringError(
            f"'{key}' must be a number in [0.0, 1.0]; got {value!r}."
        ) from e
    if score != score or score < 0.0 or score > 1.0:  # NaN or out of range
        raise SimilarityScoringError(
            f"'{key}' must be a finite number in [0.0, 1.0]; got {score!r}."
        )
    return score


# --- Entry point -----------------------------------------------------------


def score_skill_similarity(
    generated: ParsedSkill,
    golden: ParsedSkill,
    client: LLMClient,
    *,
    model: str = SIMILARITY_MODEL,
    max_tokens: int = DEFAULT_SIMILARITY_MAX_TOKENS,
    context_label: str = "synthesizer:similarity",
) -> SimilarityScore:
    """Score ``generated`` against ``golden`` and return a :class:`SimilarityScore`.

    The ``parameter_match`` and ``destructive_match`` fields are computed
    locally (binary structural equality). The ``step_coverage`` and
    ``overall`` fields come from a single Haiku 4.5 call; the ``reasoning``
    field is the model's short prose explanation.

    Raises :class:`SimilarityScoringError` if the LLM response cannot be
    parsed or validated — no retry is attempted (see class docstring).
    """
    parameter_match = _compute_parameter_match(generated, golden)
    destructive_match = _compute_destructive_match(generated, golden)

    user_content = build_similarity_user_content(generated, golden)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    response = client.complete(
        messages=messages,
        system=SIMILARITY_SYSTEM_PROMPT,
        model=model,
        max_tokens=max_tokens,
        context_label=context_label,
    )

    step_coverage, overall, reasoning = _parse_rubric_response(response.text)

    return SimilarityScore(
        overall=overall,
        step_coverage=step_coverage,
        parameter_match=parameter_match,
        destructive_match=destructive_match,
        reasoning=reasoning,
    )
