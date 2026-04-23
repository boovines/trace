# Synthesizer (Module 2) — agent notes

See [/CLAUDE.md](../../CLAUDE.md) for project-wide context, safety invariants,
and reference workflows. This file is module-scoped; it exists so that future
Ralph iterations (including on other branches that need to consume the
synthesizer) can pick up the accumulated lessons without replaying 20
iterations of progress.txt.

---

## Purpose

The Synthesizer converts a recorded trajectory (produced by Module 1 / Recorder)
into a pair of on-disk artifacts that the Runner and UI consume:

- `SKILL.md` — human-readable, strict-format markdown. Sections in locked
  order: H1 title → one-line description → `## Parameters` → `## Preconditions`
  → `## Steps` → `## Expected outcome` → optional `## Notes`.
- `skill.meta.json` — machine-readable metadata validated against
  `/contracts/skill-meta.schema.json`.

Pipeline (three phases, mirrors the PRD):

1. **Draft.** A single multimodal Claude Sonnet 4.5 call consumes a
   preprocessed trajectory digest plus up to 20 keyframe screenshots and
   returns `{markdown, meta, questions}` as JSON. The secondary destructive
   keyword matcher runs over the parsed result and never removes the LLM's
   flags — only adds missing ones.
2. **Q&A revision.** Up to 5 follow-up questions are answered one at a time by
   the user via SSE. Each answer triggers a Sonnet call that returns an updated
   `{markdown, meta, questions}` shape; the destructive matcher re-runs after
   every revision.
3. **Approval + write.** `SkillWriter` atomically commits `SKILL.md` +
   `skill.meta.json` + up to 5 preview screenshots under
   `~/Library/Application Support/Trace[-dev]/skills/<slug>/`, then registers
   the row in the shared SQLite `index.db`.

The module is strictly a distiller — it drives no real UI actions. The Runner
is the only component that executes anything.

---

## Key files

Source (`services/synthesizer/src/synthesizer/`):

| File | Responsibility |
|---|---|
| `api.py` | FastAPI `APIRouter` with all 10 synthesis + skills routes (SSE stream, start/answer/approve/delete, skills list/get/preview). Module-level `_STORE`/`_DRAFT_FN`/`_REVISE_FN` are monkey-patchable for tests via `reset_for_tests()`. |
| `schema.py` | JSON-schema loader (`load_meta_schema`), `validate_meta`, `validate_meta_against_markdown` (markdown/meta cross-check). |
| `skill_doc.py` | `ParsedSkill`/`Step`/`Parameter` (frozen pydantic), `parse_skill_md`, `render_skill_md`, `extract_parameter_refs`. Round-trip-stable hand-parser. |
| `trajectory_reader.py` | `TrajectoryReader` — read-only, validates metadata + every events.jsonl line, sorts by `seq`. |
| `preprocess.py` | Collapses scroll runs / idle gaps, picks ≤20 keyframes, emits `DigestEntry` list. Never collapses clicks, text_inputs, or app_switches. |
| `llm_client.py` | `LLMClient` wrapper: fake mode, retries (5 max, exp backoff on 429/5xx), cost logging to `costs.jsonl`. Nothing else in this module imports the Anthropic SDK directly. |
| `draft_prompt.py` | `DRAFT_SYSTEM_PROMPT` with few-shot examples loaded from `fixtures/prompt_examples/`. |
| `draft.py` | `generate_draft(preprocessed, client, *, reader)`, `build_user_content(...)`. Validates + retries up to 3 total LLM calls. Runs the destructive matcher. |
| `destructive_matcher.py` | `DESTRUCTIVE_KEYWORDS` + `apply_destructive_matcher`. Additive only; matches click `target.label` (not step prose). |
| `slug.py` | `slugify`, `resolve_unique_slug`, `validate_user_slug`. Regex `^[a-z][a-z0-9_]{2,39}$`. |
| `writer.py` | `SkillWriter.write(...)` — full validation → atomic tmp/fsync/rename → preview copy → index.db commit. All-or-nothing. |
| `session.py` | `SynthesisSession` state machine + `SessionStore` in-memory registry with `evict_stale`. Injectable `draft_fn`/`revise_fn`. |
| `revise_prompt.py` | `REVISE_SYSTEM_PROMPT` (separate from draft — revision has no few-shots because the current draft IS the context). |
| `revise.py` | `generate_revision(...)`. Mirrors `draft.py` validation + retry shape. Re-applies destructive matcher post-response. |
| `similarity.py` | `score_skill_similarity(generated, golden, client)` → `SimilarityScore`. Uses Haiku 4.5. Binary facts (param/destructive match) computed locally; only `step_coverage`/`overall`/`reasoning` come from the LLM. |
| `budget.py` | `BudgetMonitor` — per-session + daily USD caps. Reads config.json, loaded on every session + request. |
| `check_fixtures.py` | Validates every `fixtures/skills/<slug>/` pair (schema, round-trip, cross-check, trajectory_id linkage). |
| `check_contracts.py` | Cross-module gate: schema validity, trajectory fixture validity, skill fixture validity, fixture linkage. |

Tests (`tests/synthesizer/`): one `test_<module>.py` per source file plus
`test_e2e_fake.py` (full pipeline, fake-mode) and `test_real_smoke.py` (gated
by `TRACE_REAL_API_TESTS`). `conftest.py` owns the hermetic env (autouse
`_force_fake_mode` + `_no_stray_anthropic_calls` + respx `anthropic_mock`).

Contracts (`/contracts/`): `trajectory.schema.json`, `skill-meta.schema.json`
— LOCKED once shipped.

Fixtures (`/fixtures/`): `trajectories/<slug>/` (from Recorder),
`skills/<slug>/` (hand-crafted golden pairs). **Never regenerated by Ralph.**

Scripts (`/scripts/`): `check_fixtures.sh`, `check_contracts.sh` — both run in
under 10 seconds, both wired into the quality gates.

---

## Prompt engineering notes

- **Output shape is locked.** Both draft and revision prompts require a single
  JSON object with keys `{markdown, meta, questions}`. The shared
  `_parse_response_json` / `_validate_full_response` helpers live in
  `draft.py` and are reused verbatim by `revise.py` so hash-stability of
  fake-mode retry tests holds across phases. If a future story needs a new
  response shape, add a new validator — do not fork these.
- **Few-shots in the draft prompt are files, not string constants.**
  `fixtures/prompt_examples/*.txt` is read at import time in sorted filename
  order. Changing an example silently invalidates every canned fake-mode hash
  — regenerate canned responses in lockstep or commit both together.
- **Destructive flagging is structural, not promptic.** The LLM is instructed
  to prefix destructive steps with `⚠️ [DESTRUCTIVE]` and to list them in
  `meta.destructive_steps`, but `apply_destructive_matcher` runs after every
  draft AND every revision and will add any flag the LLM missed. Prompt
  wording changes should never be load-bearing for destructive correctness.
- **Revision prompt omits few-shots deliberately.** The current `markdown` +
  `meta` are the "few-shots" at revision time; adding more would bloat the
  context for no accuracy gain. If drift is observed on revisions, prefer
  tightening the per-question category guidance over adding examples.
- **Labelled sections in the revision user message** (`CURRENT_MARKDOWN`,
  `CURRENT_META`, `ANSWERED_QUESTION`, `REMAINING_QUESTIONS`) are ALL-CAPS
  section headers so Claude doesn't confuse them with content. Keep the
  convention if adding sections.
- **Retry feedback is a pure function of the bad response.** On validation
  failure we append `(assistant: bad_response, user: feedback_text)` and
  re-call. The feedback strings are produced by
  `_parse_response_json`/`_validate_markdown`/`_validate_meta_and_cross` and
  must stay byte-stable across refactors — fake-mode tests register canned
  responses keyed by request hash, and changing feedback wording requires
  regenerating every affected canned response.
- **Haiku handles the similarity rubric.** 5× cheaper than Sonnet with no
  material accuracy loss for the binary-ish rubric. Keep binary facts
  (parameter signature, destructive step set) out of the prompt — compute
  them locally and feed only `step_coverage` + `overall` + `reasoning` to
  the model.
- **Long prose in prompt modules bypasses ruff E501** via
  `[tool.ruff.lint.per-file-ignores]` in `/pyproject.toml`. When adding a
  new LLM-prompt module, add it to the same per-file-ignores rather than
  hand-wrapping paragraphs — hard wraps degrade prompt readability for the
  model.
- **The LLM sometimes forgets `meta.destructive_steps`** even when it put
  `⚠️ [DESTRUCTIVE]` markers in the markdown. Both `validate_meta_against_markdown`
  and the secondary matcher catch this; the draft-call retry loop
  surfaces it back to the model. No prompt tweak has been needed yet, but if
  it grows frequent, consider adding an explicit reminder in the system
  prompt rather than relaxing the validator.

---

## Common gotchas

1. **`httpx.ASGITransport` buffers streaming responses.** It does NOT stream —
   `ac.stream("GET", url).aiter_lines()` hangs forever on an open SSE
   connection. Test `_sse_event_generator` directly with a `_FakeRequest` stub
   exposing `is_disconnected()`. Full ASGI SSE integration belongs in a
   uvicorn-backed smoke test, not the unit suite. (See `test_api.py`.)

2. **`_sse_event_generator` can emit 3 events in one poll iteration.** When a
   session is already past its first transition, a single `snapshot()` will
   yield `state_change`, `draft_ready`, and `question_ready` before the next
   sleep. Naive "wait for N events" tests hang because the Nth event needs
   another state change. Break at the expected count, call `disconnect()` on
   the fake request, and let the generator exit cleanly.

3. **`asyncio.create_task` without a strong reference is GC'd mid-flight.**
   `api.py` keeps a module-level `_BACKGROUND_TASKS: set[asyncio.Task[None]]`
   plus an `add_done_callback(_BACKGROUND_TASKS.discard)`. Downstream async
   fire-and-forget work should use the same `_schedule_background` helper.

4. **`pytest` exits 5 on "no tests collected", not 0.** Every new test module
   needs at least one assertion or `_exits 0_` will fail. (Bit the scaffold
   story.)

5. **uv workspace dev extras.** `uv sync --extra dev` at the root does NOT
   pick up per-member dev extras. Use `uv sync --all-extras --all-packages`
   or `--package <name>` to install a workspace member's dev deps.

6. **`jsonschema` has no PEP 561 stubs.** Use
   `# type: ignore[import-untyped]` on the import line. Don't pull in
   `types-jsonschema` — it lags behind.

7. **`rfc3339-validator` is not a transitive dep.** `jsonschema`'s
   `FormatChecker` validates `uuid` out-of-the-box but `date-time` requires
   this optional package. Prefer `type: "string"` + an explicit regex pattern
   for any field where date-time enforcement is load-bearing.

8. **Contracts live via upward walk.** `contracts/*.schema.json` is located at
   runtime by walking up from `Path(__file__)` until a directory containing
   `contracts/` is found. Keeps the package working from both the editable
   workspace install and a future published wheel — do not hard-code a
   relative path.

9. **Sessions never rehydrate across process restarts.** `draft.json` is
   written after every transition, but v1 intentionally does NOT reload it
   on service boot — the user re-synthesizes. Documented PRD limitation.

10. **Approve-with-user-slug is strict; approve-without is silent.** When
    `slug=None`, `resolve_unique_slug` silently appends `_2`/`_3`/… on
    collision. When the caller passes `slug="..."`, a collision raises
    `SlugError` (→ HTTP 409). The asymmetry is intentional — users who type
    a slug expect to see their input or an error, not a silent rename.

11. **Path traversal defense lives in two places.** `_resolve_skill_dir(slug)`
    rejects `/`, `\`, leading `.` in the slug; `get_skill_preview` repeats the
    same guard on the filename. Both are necessary because a legal slug with a
    traversal filename is still an attack vector.

12. **DELETE `/skills/{slug}` order matters.** Index row first, then
    directory. Reverse order would leave a ghost row if the rmtree succeeded
    but the commit failed.

13. **Golden fixtures are sacred.** `fixtures/skills/<slug>/` pairs are
    hand-crafted ground truth for S-014/S-017. Ralph iterations MUST NOT
    regenerate them from a synthesized draft — that collapses the snapshot
    comparison into a tautology. `scripts/check_fixtures.sh` + S-016
    `fixtures/skills/README.md` both document this rule.

14. **Fixture trajectory linkage is by UUID, not slug.** `meta.trajectory_id`
    → `fixtures/trajectories/<dir>/metadata.json.id`. Slug renames don't
    break the link. Any new cross-fixture validator should scan by id.

15. **The destructive matcher binds by label substring, not positional
    index.** `apply_destructive_matcher` looks at each click's
    `target.label` and finds the earliest un-bound step whose text contains
    that label substring. Positional pairing breaks on non-click steps
    (text_input, app_switch) that misalign the index with
    `iter_events_by_type("click")`. If the draft prompt ever stops emitting
    `Clicked button labeled "X"` in step text, the matcher silently stops
    flagging — keep the label in the summary.

16. **The SKILL.md parser is hand-rolled for a reason.** Commonmark/mistune
    aggressively normalize whitespace and drop structure; our round-trip
    invariant `parse(render(p)) == p` would fail every time. Don't swap in
    a generic markdown library.

---

## How to run tests

### Standard suite (what Ralph runs)

```bash
# Lint
uv run ruff check services/synthesizer tests/synthesizer scripts/

# Type check
uv run mypy --strict services/synthesizer

# Unit + integration tests (hermetic, fake-mode LLM)
uv run pytest tests/synthesizer/

# Fixture + contract gates
./scripts/check_fixtures.sh
./scripts/check_contracts.sh
```

All of the above runs without network access, without a real API key, and
without writing outside `tmp_path`. The `conftest.py` autouse fixtures force
`TRACE_LLM_FAKE_MODE=1`, pin `ANTHROPIC_API_KEY=test-dummy-key`, and install
a respx network guard that raises on any unmocked call to
`api.anthropic.com`.

### Real-API smoke test (pre-merge gate)

**Not** part of the Ralph loop. Run manually before merging
`feat/synthesizer` to `main`:

```bash
TRACE_REAL_API_TESTS=1 ANTHROPIC_API_KEY=sk-ant-... \
  uv run pytest tests/synthesizer/test_real_smoke.py -v
```

Asserts:

- `destructive_match == 1.0` on all 5 fixtures (non-negotiable).
- `overall >= 0.80` on at least 4 of 5 fixtures (tolerates LLM drift).
- Total spend < $2 per run (hard cap).

Expected cost: ~$0.30–$0.85 per run. Writes `smoke_report.json`; attach it to
the PR description.

### Useful environment variables

| Variable | Purpose |
|---|---|
| `TRACE_LLM_FAKE_MODE=1` | Force fake mode (auto-set in tests). |
| `TRACE_REAL_API_TESTS=1` | Enable the gated real-API smoke test. |
| `TRACE_PROFILE=dev` | Use `~/Library/Application Support/Trace-dev/`. |
| `TRACE_DATA_DIR=<path>` | Override the data dir (tests use `tmp_path`). |
| `TRACE_FAKE_RESPONSES_DIR=<path>` | Override the canned-response directory. |
| `ANTHROPIC_API_KEY=...` | Required only for the smoke test. |

See also [`tests/synthesizer/README.md`](../../tests/synthesizer/README.md)
for the test-suite deep-dive and `smoke_report.json` schema.
