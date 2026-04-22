"""The system prompt for draft SKILL.md generation.

The prompt has three parts, concatenated at module import time:

1. A fixed instruction block describing the task, the required output JSON shape,
   the strict SKILL.md format, and the rules for destructive flagging and
   parameter extraction.
2. Two few-shot examples loaded from ``fixtures/prompt_examples/*.txt`` — one
   canonical example pair per file, showing a trajectory digest and the
   expected JSON output.
3. A final reminder that the response must be a single JSON object with
   exactly the keys ``{markdown, meta, questions}`` — no prose, no code
   fences, no preamble.

The prompt is a *load-bearing* part of the synthesizer: prompt drift is the
most common cause of regressions in this pipeline, so the prompt text itself
lives in source control and is covered by :mod:`synthesizer.draft` retry
tests that assert the module still recovers when the LLM deviates.

The few-shot example files are plain text with the shape::

    DIGEST:
    <digest lines, one per entry>

    OUTPUT:
    <JSON object>

They are fixtures (not Python literals) so they can be inspected and edited
without touching code.
"""

from __future__ import annotations

from pathlib import Path

from synthesizer.schema import _find_repo_root

__all__ = [
    "DRAFT_OUTPUT_KEYS",
    "DRAFT_SYSTEM_PROMPT",
    "PROMPT_EXAMPLES_DIR",
    "load_prompt_examples",
]


DRAFT_OUTPUT_KEYS: tuple[str, ...] = ("markdown", "meta", "questions")


def _prompt_examples_dir() -> Path:
    repo_root = _find_repo_root(Path(__file__).resolve())
    return repo_root / "fixtures" / "prompt_examples"


PROMPT_EXAMPLES_DIR: Path = _prompt_examples_dir()


def load_prompt_examples() -> list[str]:
    """Return the raw text of every ``fixtures/prompt_examples/*.txt`` file.

    The order is the directory's sorted filename order, so the prompt is
    deterministic across machines. Returns an empty list if the directory
    is missing — callers should have at least 2 examples committed.
    """
    if not PROMPT_EXAMPLES_DIR.is_dir():
        return []
    examples: list[str] = []
    for path in sorted(PROMPT_EXAMPLES_DIR.glob("*.txt")):
        examples.append(path.read_text(encoding="utf-8").rstrip() + "\n")
    return examples


_INSTRUCTIONS = """You convert recorded desktop workflow trajectories into executable SKILL.md files.

You receive:
  - A DIGEST: an ordered, numbered list of events extracted from a macOS screen recording.
  - Up to 20 keyframe screenshots captured around meaningful moments in the trajectory.

Your job: produce three things in a single JSON object.

## Output shape

Return EXACTLY one JSON object with these three keys and no others:

  {
    "markdown": "<full SKILL.md as a string>",
    "meta": { ... skill.meta.json ... },
    "questions": [ { "id": "q1", "category": "...", "text": "..." }, ... ]
  }

Do NOT wrap the JSON in markdown code fences. Do NOT prepend commentary.
Your entire response must parse as a single JSON object.

## SKILL.md format (strict)

The markdown MUST follow this section order exactly:

  # <Title>
  <one-line description paragraph>

  ## Parameters
  - `name` (type, required|optional[, default: <value>]) — <description>
  ... OR the single line `_None._` when there are no parameters

  ## Preconditions
  - <precondition sentence>
  ... OR `_None._`

  ## Steps
  1. <first step>
  2. ⚠️ [DESTRUCTIVE] <second step text>
  ...

  ## Expected outcome
  <one-paragraph description of the final state>

  ## Notes  (optional)
  <freeform notes>

Rules:
  - Steps are numbered 1..N sequentially — never skip, never duplicate.
  - Prefix any destructive step with the EXACT string `⚠️ [DESTRUCTIVE]` (emoji, space, bracket, uppercase word, bracket) and nothing else.
  - A destructive step is one that sends/submits/deletes/publishes/posts/purchases/pays/confirms/authorizes/approves/shares or otherwise commits a change that is hard to reverse.
  - Parameter types are limited to `string`, `integer`, `boolean`.
  - Parameter names match `[a-z][a-z0-9_]{0,29}`.
  - Reference parameters inside step text with `{name}` — for example `Click {recipient_email}` — whenever a value in the trajectory plausibly varies between runs.
  - Every `{name}` reference in the markdown MUST correspond to an entry in `meta.parameters` with the same name, and vice versa.

## skill.meta.json shape

The `meta` object MUST contain these keys:
  - `slug`: matches `^[a-z][a-z0-9_]{2,39}$` — lowercase, underscores only.
  - `name`: 1-100 chars, human-readable.
  - `trajectory_id`: UUID string. Use `"00000000-0000-0000-0000-000000000000"` as a placeholder; the caller overwrites it.
  - `created_at`: ISO-8601 datetime string with timezone. Use `"2026-01-01T00:00:00+00:00"` as a placeholder.
  - `parameters`: array of `{name, type, required[, default]}` objects. Order and names must match the markdown.
  - `destructive_steps`: array of integers listing the 1-indexed step numbers flagged with `⚠️ [DESTRUCTIVE]`.
  - `preconditions`: array of strings matching the `## Preconditions` bullets.
  - `step_count`: the number of numbered steps under `## Steps`.

`meta.destructive_steps` and the `⚠️ [DESTRUCTIVE]` markers in the markdown MUST agree — a downstream validator will reject any drift.

## Follow-up questions

Include up to 5 questions (fewer is fine; zero is fine when the digest is unambiguous). Each question:
  - `id`: a short unique id like `q1`, `q2`, ...
  - `category`: one of `parameterization`, `intent`, `destructive`, `precondition`, `naming`.
  - `text`: the question itself — a single sentence, directed to the user.

Ask a question when:
  - A typed value looks like it could vary between runs (ask if it should be a parameter).
  - A click target looks borderline-destructive.
  - A precondition is plausible but unverifiable from the trajectory alone.
  - The skill name you chose is a guess.

Do NOT ask more than 5 questions. Prefer fewer questions with clearer stakes over many speculative ones.

## Operating rules

  - Only write steps that correspond to events in the digest. Do not invent steps.
  - If a step has no obvious verb from the digest, describe what the user likely intended, but flag it with a follow-up question.
  - Prefer parameters over literals when a value could reasonably change between runs.
  - Keep step text terse and imperative: "Click the Send button" not "The user clicks the Send button".

## Few-shot examples
"""

_CLOSING = """
Remember: your entire response is a single JSON object. No code fences. No commentary. The JSON keys are `markdown`, `meta`, `questions` in any order.
"""


def _build_system_prompt() -> str:
    examples = load_prompt_examples()
    example_block = "\n\n".join(
        f"### Example {i + 1}\n\n{text.rstrip()}" for i, text in enumerate(examples)
    )
    return (_INSTRUCTIONS + example_block + "\n" + _CLOSING).strip() + "\n"


DRAFT_SYSTEM_PROMPT: str = _build_system_prompt()
