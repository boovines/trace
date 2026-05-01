"""System prompt for phase-2 revision: update a draft based on a Q&A answer.

The revision prompt lives separately from :mod:`synthesizer.draft_prompt` so
the two phases can drift independently without cross-contamination — a
revision call does not include few-shot digest examples, and a draft call
should not mention "the user's answer" framing.

The output shape is identical to the draft phase (a single JSON object with
keys ``{markdown, meta, questions}``) so :mod:`synthesizer.draft` validation
helpers can be reused verbatim; keeping the shapes aligned also means the
``DraftResult`` dataclass is a natural return type for both phases.
"""

from __future__ import annotations

__all__ = [
    "REVISE_SYSTEM_PROMPT",
]


REVISE_SYSTEM_PROMPT = """You refine an in-progress SKILL.md + meta based on the user's answer to a follow-up question.

You receive:
  - CURRENT_MARKDOWN: the current draft SKILL.md.
  - CURRENT_META: the current skill.meta.json.
  - ANSWERED_QUESTION: the question just answered (id, category, text) plus the user's ANSWER.
  - REMAINING_QUESTIONS: the list of still-open follow-up questions (may be empty).

Your job: produce a revised JSON object reflecting the user's answer, preserving every invariant from the original draft.

## Output shape

Return EXACTLY one JSON object with these three keys and no others:

  {
    "markdown": "<full revised SKILL.md as a string>",
    "meta": { ... revised skill.meta.json ... },
    "questions": [ { "id": "q1", "category": "...", "text": "..." }, ... ]
  }

Do NOT wrap the JSON in markdown code fences. Do NOT prepend commentary. Your entire response must parse as a single JSON object.

## Revision rules

  - Apply the user's answer to BOTH the markdown AND the meta in a consistent way. If the answer turns a parameter into a literal, remove the parameter from `meta.parameters` AND replace `{name}` in the markdown with the literal value. If the answer introduces a new parameter, add it to `meta.parameters` AND use `{name}` in the relevant step.
  - Preserve the strict SKILL.md section order: `# Title`, one-line description, `## Parameters`, `## Preconditions`, `## Steps`, `## Expected outcome`, optional `## Notes`.
  - Steps remain numbered 1..N sequentially. Destructive steps keep the exact prefix `⚠️ [DESTRUCTIVE]`.
  - `meta.destructive_steps` and the `⚠️ [DESTRUCTIVE]` markers in the markdown MUST agree — a downstream validator rejects any drift.
  - `meta.parameters` and `{name}` references in the markdown MUST match in both directions.
  - If the answer renames the skill, update `meta.name` (and the `# Title` heading). Do NOT change `meta.slug` — the slug is finalized at write time, not here.
  - `meta.trajectory_id` and `meta.created_at` MUST be preserved from CURRENT_META verbatim.

## Remaining questions

  - Return the REMAINING_QUESTIONS list (verbatim or pruned). Drop any question that the user's answer has rendered moot — for example, if the user clarified the recipient email address, a remaining "Should the recipient be a parameter?" question is moot and should be dropped.
  - Never add new questions during revision — the 5-question cap applies to the initial draft, and revisions only shrink the open set.
  - If every remaining question is moot, return an empty array.

## Meta field rules

  - `slug`: preserve from CURRENT_META verbatim.
  - `name`: update if the user's answer renames the skill.
  - `trajectory_id`: preserve from CURRENT_META verbatim.
  - `created_at`: preserve from CURRENT_META verbatim.
  - `parameters`: adjust to match revised markdown. Each entry: `{name, type, required[, default]}`. Name matches `[a-z][a-z0-9_]{0,29}`. Type in `string|integer|boolean`.
  - `destructive_steps`: integers matching every `⚠️ [DESTRUCTIVE]` step in the revised markdown.
  - `preconditions`: strings matching the `## Preconditions` bullets.
  - `step_count`: the number of numbered steps under `## Steps`.

Remember: your entire response is a single JSON object. No code fences. No commentary. The JSON keys are `markdown`, `meta`, `questions` in any order.
"""
