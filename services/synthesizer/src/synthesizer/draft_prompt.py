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
  - **Spatially ground each step.** When the trajectory provides screenshots, write step prose that names *where* on screen the target lives — e.g. "Click the **Send** button at the bottom-left of the reply pane (visible in screenshot 3)" instead of just "Click Send". Anchor on landmarks the agent can see: app name, pane/panel/sidebar location, button label or aria-label, relative position ("top-right", "below the search bar", "in the third row"). Cite the keyframe by its 1-based index in the screenshots that arrived with this trajectory. Vague prose like "Click the button" is a regression — it gives the pixel-grounded fallback nothing to anchor to.

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
  - `steps` (OPTIONAL but strongly preferred): per-step execution metadata. See "Execution hints" below.

`meta.destructive_steps` and the `⚠️ [DESTRUCTIVE]` markers in the markdown MUST agree — a downstream validator will reject any drift.

## Execution hints (the agent's most useful output)

Trace's runner picks an execution tier per step in this priority order:

  1. **MCP** — call a published MCP server function directly (deterministic, fast, semantic).
  2. **browser_dom** — Playwright DOM action against a known selector (when the step is in a browser and no MCP fits).
  3. **computer_use** — pixel-grounded fallback using the SKILL.md prose (last resort).

For every step you can map onto an MCP function from the catalog below, emit a `meta.steps[]` entry with an ordered `execution_hints` array. The runner will try them top-down and stop at the first tier whose preconditions are satisfiable.

Each `meta.steps[]` entry shape:

```
{
  "number": <1-based step index in ## Steps>,
  "intent": "<short verb_phrase, e.g. send_email, create_event, set_status>",
  "screenshot_ref": "<NNNN.png>",              // optional: keyframe the prose grounds against
  "execution_hints": [<hint1>, <hint2>, ...]   // most-preferred first
}
```

`screenshot_ref` ties spatially-grounded step prose ("Click the Send button at the bottom-left of the reply pane") to the actual keyframe the synth saw, so the runner's computer-use fallback can show the agent the same visual. Use the filename of the most relevant keyframe from the trajectory (zero-padded `NNNN.png`, e.g. `"0003.png"`). Omit when no screenshot meaningfully grounds the step (e.g. a pure-keyboard step with no visible target).

Execution hint shapes by tier:

```
// tier=mcp — preferred when the step matches a function in the catalog below
{ "tier": "mcp",
  "mcp_server": "<catalog server name>",
  "function":   "<catalog function name>",
  "arguments":  { "<arg>": "<value or {param} substitution>" } }

// tier=browser_dom — preferred when the step targets a known browser surface
{ "tier": "browser_dom",
  "url_pattern": "https://...",      // optional; the URL the runner should be on
  "selector":    "button[aria-label='Send']",
  "action":      "click" | "type" | "navigate" | "scroll" | "submit",
  "value":       "<text or URL — required for type/navigate>" }

// tier=computer_use — always include this as the final fallback
{ "tier": "computer_use",
  "summary": "<one-sentence description for the pixel-grounded agent>" }
```

Rules:

  - Do NOT include hints whose `mcp_server` / `function` is not in the catalog. Make up a function name and the response is rejected.
  - Argument values may use `{parameter_name}` substitutions when they reference values you also declared in `meta.parameters`.
  - When in doubt, omit the MCP hint and just emit the computer_use fallback. A safe fallback beats a hallucinated MCP call.
  - Steps that are pure UI navigation (clicking into a window, scrolling, etc.) usually only deserve a `computer_use` hint — don't reach for MCP.
  - Steps you don't list in `meta.steps` are fine; the runner falls back to computer-use using the markdown prose.

{MCP_CATALOG}



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
    from synthesizer.mcp_catalog import format_for_prompt as _mcp_catalog

    examples = load_prompt_examples()
    example_block = "\n\n".join(
        f"### Example {i + 1}\n\n{text.rstrip()}" for i, text in enumerate(examples)
    )
    instructions = _INSTRUCTIONS.replace("{MCP_CATALOG}", _mcp_catalog())
    return (instructions + example_block + "\n" + _CLOSING).strip() + "\n"


DRAFT_SYSTEM_PROMPT: str = _build_system_prompt()
