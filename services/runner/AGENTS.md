# services/runner/AGENTS.md

Branch: `feat/runner`. Module 3 of four. Completion promise: `RUNNER_DRY_RUN_DONE`.

See [/CLAUDE.md](../../CLAUDE.md) for project-wide context, safety invariants,
and reference workflows. This file documents the runner-specific patterns,
safety model, and lessons discovered over the Ralph dry-run phase (X-001
through X-024). Read this alongside `progress.txt` whenever you touch
`services/runner/**` or `tests/runner/**`.

---

## Purpose

The runner executes a SKILL.md end-to-end via Claude Sonnet 4.5 with the
`computer_20250124` tool. Input: a synthesizer-produced `skill/<slug>/`
directory + caller-supplied parameters. Output: a `run/<run_id>/` directory
(metadata + transcript + events + screenshots), plus a live UI channel over
WebSocket for status changes, confirmation prompts, cost warnings, and a
`done` terminator.

Two execution modes share the same orchestration path:

- **`dry_run`** — the default and the only mode allowed under Ralph. Uses
  `DryRunInputAdapter` (records clicks instead of posting events) and
  `TrajectoryScreenSource` (replays keyframes from a reference trajectory
  instead of capturing the real display). `LLM` calls can be real (via
  `ClaudeRuntime` → Anthropic) or faked via `TRACE_LLM_FAKE_MODE=1`.
- **`execute`** — live. Requires `TRACE_ALLOW_LIVE=1`. Uses `LiveInputAdapter`
  (CGEventPost) and `LiveScreenSource` (CGWindowListCreateImage). NEVER
  reached by Ralph — see "Dry-run vs live mode" below.

---

## Key files

Runtime code (`services/runner/src/runner/`):

- `safety.py` — `require_live_mode()` / `is_live_mode_allowed()` /
  `LIVE_MODE_ENV_VAR`. The single source of truth for the `TRACE_ALLOW_LIVE`
  check. Every live code path calls `require_live_mode()` in `__init__`.
- `schema.py` — `RunMetadata` pydantic model + JSON-schema validator
  (`load_run_metadata_schema` / `validate_run_metadata`). Paired 1:1 with
  `contracts/run-metadata.schema.json`.
- `destructive.py` — `load_destructive_keywords()` +
  `matches_destructive_keyword(text)`. Backed by
  `contracts/destructive_keywords.json` (14 locked keywords).
- `skill_loader.py` — `load_skill` + `substitute_parameters` +
  `LoadedSkill`. Relies on `synthesizer.skill_doc` for SKILL.md parsing and
  `synthesizer.schema` for meta↔markdown symmetry.
- `run_writer.py` — `RunWriter`, thread-safe atomic metadata writer +
  JSONL event/transcript appenders + PNG screenshot writer. Writes the run
  directory at `0o700`. `update_status` re-runs pydantic validation via
  `model_dump() | updates → model_validate`.
- `run_index.py` — SQLite-backed `RunIndex` for fast listing; reconciled at
  service startup; mirrors every metadata write through the writer.
- `coords.py` — `DisplayInfo`, `ImageMapping`, `DryRunDisplayInfo`,
  `capture_and_normalize`, and the `resized_pixels_to_points` helper.
- `input_adapter.py` — `InputAdapter` Protocol + `DryRunInputAdapter`.
- `screen_source.py` — `ScreenSource` Protocol + `TrajectoryScreenSource`.
- `live_input.py` — `LiveInputAdapter` (CGEventPost). Lives in its own
  module to keep `input_adapter` free of PyObjC. One-way import only.
- `live_screen.py` — `LiveScreenSource` (CGWindowListCreateImage).
- `agent_runtime.py` / `claude_runtime.py` — LLM abstraction. `ClaudeRuntime`
  wraps `AsyncAnthropic`; fake mode matches `fixtures/llm_responses/runner_<slug>.json`
  by marker substring. `set_image_mapping(...)` is called after every
  screenshot to keep the `computer_20250124` tool's dims in sync with the
  rendered frame.
- `execution_prompt.py` — `build_execution_prompt(loaded_skill, mode)` +
  the six `Final` protocol constants (`PREAMBLE`, `DESTRUCTIVE_PROTOCOL`,
  `COMPLETION_PROTOCOL`, `ERROR_PROTOCOL`, `VERIFICATION_PROTOCOL`,
  `DRY_RUN_NOTICE`). Load-bearing for safety — do not edit the protocol
  strings without bumping the parser tests.
- `parser.py` — `parse_agent_response(AgentResponse) -> ParsedAction`.
  The first destructive gate at the text-tag level; `ConfirmationRequest`
  ALWAYS drops a concurrent `tool_use`.
- `pre_action_gate.py` — `apply_gate_to_tool_call(...)`. The third
  destructive gate at the AX-label level. Dry-run short-circuits; execute
  mode queries the `AXResolver` Protocol. Degrades open on resolver
  exceptions (`Unknown`) but closed on destructive-keyword hits
  (`RequireConfirmation`).
- `dispatcher.py` — `dispatch_tool_call` returns a `ToolResult`
  (content_blocks, new_image_mapping, is_error). Never raises on malformed
  tool_input. All 10 `computer_20250124` actions live here.
- `budget.py` + `budget_config.py` — `RunBudget` / `BudgetTracker` for
  per-run caps, `RunnerBudgetConfig` for daily + per-run cost caps, and
  the `load_runner_budget` / `sum_daily_runner_cost_usd` helpers.
- `confirmation.py` — `ConfirmationQueue`: the async mailbox that carries
  confirmation prompts UI-ward and decisions back. Note the deliberately
  shared name with `runner.parser.ConfirmationRequest` (parser = agent tag
  signal; queue = full UI prompt).
- `kill_switch.py` — `KillSwitch` (per-run `asyncio.Event` + reason +
  killed flag). `get_global_kill_switch()` is the process singleton used by
  the API.
- `executor.py` — orchestration boundary. Builds the prompt, registers the
  kill switch, walks the turn loop, enforces budgets, races kill events
  against LLM calls / confirmation awaits, writes screenshots, finalizes
  status via `_finalize_succeeded|failed|aborted|budget_exceeded|agent_stuck`.
- `run_manager.py` — FastAPI-facing orchestrator. Owns a `_BackgroundLoop`
  (separate daemon thread + event loop) and submits run coroutines via
  `asyncio.run_coroutine_threadsafe`. Wires adapters, the broadcaster, the
  confirmation queue, and the kill switch together per run.
- `observing_writer.py` — `ObservingRunWriter` subclasses `RunWriter` and
  republishes status/event/turn writes as broadcaster messages.
- `event_stream.py` — `EventBroadcaster`, the per-run publisher used by the
  WebSocket route. Cross-loop safe via `loop.call_soon_threadsafe`.
- `api.py` — FastAPI routes: `POST /run/start`, `GET /run/{run_id}`,
  `POST /run/{run_id}/confirm`, `POST /run/{run_id}/abort`, `GET /runs`,
  `GET /run/{run_id}/events`, `GET /run/{run_id}/screenshots/{filename}`,
  `WS /run/{run_id}/stream`.
- `paths.py` — profile root + runs/skills/trajectories roots.

Tests live under `tests/runner/`. Fixtures for the five reference skills live
at `fixtures/skills/<slug>/`, fake-mode LLM scripts at
`fixtures/llm_responses/runner_<slug>.json`, and reference trajectories at
`fixtures/trajectories/<uuid>/`.

---

## Safety model (three-layer destructive gate)

Destructive actions (Send, Submit, Delete, Publish, Post, Purchase, Pay, Buy,
Transfer, Confirm, Authorize, Approve, Share + the 14 keywords in
`contracts/destructive_keywords.json`) are blocked at three independent
layers. **All three must fail open simultaneously for a destructive click to
execute without a user confirmation.** This is deliberate belt-and-suspenders
engineering — each layer has independent failure modes.

### Layer 1 — Skill ⚠️ flag → prompt instruction → `<needs_confirmation>` tag

The synthesizer classifies steps as destructive at generation time and writes
`⚠️ ` on those lines of SKILL.md plus a `destructive_steps: [N, ...]` array in
`skill.meta.json`. `synthesizer.schema.validate_meta_against_markdown` enforces
symmetric agreement between the two — either-direction drift is an error.

At prompt-construction time (`execution_prompt.build_execution_prompt`),
destructive steps render with an explicit `⚠️ [DESTRUCTIVE] ` prefix inside
the fenced workflow block, and the prompt carries the verbatim
`DESTRUCTIVE_PROTOCOL` telling the agent to emit exactly
`<needs_confirmation step="N"/>` (self-closing, attribute quoted) before
executing. The literal tag string is the interface.

At response-parsing time (`parser.parse_agent_response`), a successful
`<needs_confirmation step="N"/>` match returns `ConfirmationRequest(N)` AND
drops any concurrent `tool_use` block in the same turn. A malformed
confirmation attempt (detected by the loose `<\s*needs_confirmation\b`
pattern) also drops the tool_use and returns `UnknownAction` — we do NOT
silently fall through to the tool call when the model tried and failed to
issue a confirmation.

### Layer 2 — Parser-level tag gate

The parser is a pure-text gate. It runs on `AgentResponse.content_blocks`
(plain dicts, not SDK block objects) and implements the priority order
`ConfirmationRequest > WorkflowComplete > WorkflowFailed > ToolCallAction >
UnknownAction`. The parser NEVER raises — every defect returns
`UnknownAction`. See `services/runner/src/runner/parser.py` and the
corresponding safety-ordering fixture tests in
`tests/runner/test_parser.py`.

### Layer 3 — Pre-action AX-label gate (harness-layer)

Even if layers 1 and 2 fail — e.g. the synthesizer missed a ⚠️, or the agent
issued a tool_use without a confirmation tag — the harness inspects the
target accessibility element just before every click and re-checks the
destructive-keyword list against the AX label.

`pre_action_gate.apply_gate_to_tool_call(action, image_mapping, ax_resolver,
mode)` returns `AllowAction | RequireConfirmation | Unknown`. Decision tree:

1. Mode is `dry_run` → `AllowAction` (resolver never consulted).
2. Action is not a click-class action → `AllowAction`.
3. `ax_resolver` is `None` or raises → `Unknown` (degrades open but logs).
4. Target role ∉ `ACTIONABLE_AX_ROLES` ({AXButton, AXLink, AXMenuItem,
   AXCheckBox}) → `AllowAction` (a static text that happens to say "Delete
   everything" is not actionable).
5. Destructive keyword matches the label (`matches_destructive_keyword` from
   `destructive.py`) → `RequireConfirmation`.
6. Else → `AllowAction`.

The executor consumes the gate result and, on `RequireConfirmation`, goes
through the exact same `ConfirmationQueue.push_request` /
`await_decision` path as a layer-1 confirmation — so the UI sees the same
message shape whether the model self-flagged or the harness overrode it.

### TRACE_ALLOW_LIVE — the big red switch above all three layers

Every live code path (`LiveInputAdapter.__init__`, `LiveScreenSource.__init__`,
any run started in `mode=execute`) calls `safety.require_live_mode()` which
raises `LiveModeNotAllowed` unless `TRACE_ALLOW_LIVE == "1"` (exact string
equality; `"true"`, `"yes"`, `"0"`, empty, whitespace all fail shut).

Ralph iterations, the test suite, and the integration layer's default
development environment MUST run with this flag unset. The only contexts
where it's enabled are (a) developer-driven live smoke tests (see "How
live testing works" below) and (b) the X-025 pre-merge human checklist.

---

## Dry-run vs live mode

The runner supports two modes, selected by the `mode` field in
`POST /run/start`:

| Capability | `dry_run` | `execute` |
|---|---|---|
| Requires `TRACE_ALLOW_LIVE=1` | No | Yes (400 at API if unset) |
| Input adapter | `DryRunInputAdapter` | `LiveInputAdapter` (CGEventPost) |
| Screen source | `TrajectoryScreenSource` | `LiveScreenSource` (CGWindowListCreateImage) |
| Pre-action gate | Short-circuits (`AllowAction`) | Queries `AXResolver` |
| Execution prompt | Includes `DRY_RUN_NOTICE` | No dry-run notice |
| Layer-1 and layer-2 confirmation | Active | Active |
| Allowed under Ralph | Yes (this is the only allowed mode) | No |

Fake mode (`TRACE_LLM_FAKE_MODE=1`) is orthogonal: it substitutes the LLM's
responses with canned fixtures from `fixtures/llm_responses/runner_*.json`.
Combining `mode=dry_run` with `TRACE_LLM_FAKE_MODE=1` gives a fully
hermetic e2e test that never touches Anthropic and never drives input
events. This is what the Ralph suite (`test_e2e_fake.py`,
`test_replay_correctness.py`) uses and what the X-021 integration test
ships.

### Enabling live mode for a manual smoke test

```bash
# Disposable test machine only — see services/runner/README.md for the full
# pre-merge procedure.
TRACE_ALLOW_LIVE=1 TRACE_PROFILE=dev \
    uv run python -m uvicorn gateway.main:app --host 127.0.0.1 --port 8765

# Then from another shell:
curl -X POST http://127.0.0.1:8765/run/start \
  -H 'Content-Type: application/json' \
  -d '{"skill_slug": "notes_daily", "mode": "execute", "parameters": {}}'
```

Dev profile (`TRACE_PROFILE=dev`) gives lower cost caps ($0.50 per run, $2
daily) and writes everything under `~/Library/Application
Support/Trace-dev/`. NEVER start a live run under the prod profile during
hand-testing — the daily cap will silently absorb charges until you hit
$20.

### Live-mode-adjacent tests

A handful of tests exercise the live adapters via mocked PyObjC seams —
those run on every `uv run pytest tests/runner/` invocation. A smaller set
(marked `@pytest.mark.live_input`) actually post synthetic CGEvents through
the HID tap and are skipped unless `--run-live-input` is passed:

```bash
# Normal (Ralph, CI): live tests skipped.
uv run pytest tests/runner/

# Opt-in: ONLY on a disposable machine with Accessibility permission granted.
TRACE_ALLOW_LIVE=1 uv run pytest tests/runner/test_live_input.py \
    --run-live-input -m live_input
```

See `services/runner/README.md` for the full warning and
`tests/runner/test_live_input.py` for the opt-in test bodies.

---

## Common gotchas

These are the costly-to-rediscover lessons from the Ralph dry-run phase.
Read before touching anything in this module.

1. **`CGEventPost` takes points, not pixels.** `LiveInputAdapter` snapshots
   a `DisplayInfo` at construction and every click/move routes through
   `resized_pixels_to_points(..., image_mapping)` so Claude-space
   coordinates (resized pixels) become display-point coordinates before the
   Quartz call. A 2880×1800 Retina capture downscaled to a 1568×980
   Claude-space image yields a factor ≈ 0.9184. On the dry-run's
   1440×900 blank canvas that factor is 0.5 — tests assert on either,
   depending on whether a screenshot has dispatched yet (see gotcha #3).

2. **PyObjC bridged calls return `None` on failure and DO NOT raise.**
   Every `CGEventCreate*` / `CGWindowListCreateImage` call must be followed
   by `if x is None: raise LiveInputError(...)` / `ScreenCaptureError`. The
   live modules wrap these in `_create_*` helpers. Don't bypass them.

3. **Image-mapping lifecycle is load-bearing for coordinate math.** The
   executor seeds `_default_image_mapping()` from `DryRunDisplayInfo`
   downscaled to longest edge 1568 — matching
   `ClaudeRuntime._default_display_dims()` exactly. On the first
   `screenshot` tool call the dispatcher returns a new mapping (derived
   from the actual screen source) which the executor propagates via
   `agent_runtime.set_image_mapping(...)` before the next turn. In
   practice, the first turn's click math uses the 1568-longest-edge
   default, and every subsequent turn uses the source's actual dims.
   `TrajectoryScreenSource`'s 1440×900 blank canvas is already under 1568
   so `capture_and_normalize` does NOT downscale it — the mapping factor
   flips from 0.918× to 0.5× the moment the first screenshot dispatches.
   A `new_image_mapping=None` from the dispatcher means "no screenshot was
   taken" (mouse_move, error path); the executor keeps the previous
   mapping in that case rather than clearing it.

4. **`bool` subclasses `int` in Python, so JSON numeric fields need an
   explicit bool-rejection before the numeric check.** Every integer /
   float input from dict-shaped tool_input goes through
   `isinstance(x, bool)` first and `isinstance(x, int | float)` second.
   Without this, `scroll_amount=True` coerces to 1, `coordinate=[True,
   True]` maps to point 1 pixel, and the pre-action gate silently accepts
   bool coordinates. See `dispatcher._parse_coordinate` and
   `pre_action_gate._normalize_coord` for the pattern.

5. **`asyncio.Queue`/`Event`/`Future` are NOT thread-safe.** The API
   layer's HTTP threads and the background run loop live on different
   loops. Use `loop.call_soon_threadsafe(fn)` for any cross-loop delivery
   (see `EventBroadcaster.publish`, `KillSwitch.kill`,
   `ConfirmationQueue.submit_decision`). Same-loop calls bypass the
   threadsafe hop so existing "call kill, immediately check
   `event.is_set()`" test contracts still hold.

6. **`httpx` cancellation is what delivers the 2-second abort
   guarantee.** The executor's `_run_turn_cancellable` races the LLM call
   against the kill event and `.cancel()`s the turn task on kill; the
   `asyncio.CancelledError` propagates through
   `AsyncAnthropic.messages.create` → `httpx` → the underlying socket
   read, cancelling the in-flight request. Tests exercise this by patching
   the SDK's `messages.create` with a `sleep_forever` coroutine rather
   than via `respx` delayed responses (respx async side_effects are
   historically flaky).

7. **`AuthenticationError` subclasses `APIStatusError` in the Anthropic
   SDK.** Catch it BEFORE `APIStatusError` in the retry try/except or a
   401 gets swallowed into the retry loop. `RateLimitError` (429) is
   retryable and flows through the same `APIStatusError` catch branched
   on `status_code`.

8. **`AsyncAnthropic(max_retries=0)` is REQUIRED.** The SDK's built-in
   retry defaults to 2 and would silently hide our own exponential-backoff
   retry policy (1, 2, 4, 8, 16s × 5 attempts). Without this, the
   "single 429 then success" test passes even if our retry policy is
   broken. Patch `runner.claude_runtime.asyncio.sleep` (module-local
   reference) in tests to skip the real waits.

9. **`Message.content` items are typed block objects, but the rest of the
   runner expects dicts.** `ClaudeRuntime` converts them via `[block.model_dump()
   for block in msg.content]` before handing off to `parser.parse_agent_response`.
   The parser duck-types via `isinstance(block, dict)` and `.get("type")` —
   never add support for real SDK block objects at the parser boundary.

10. **`model_copy(update=...)` in pydantic v2 does NOT re-run validators.**
    `RunWriter.update_status` rebuilds via `RunMetadata.model_validate(
    current.model_dump() | updates)` so a bad status enum fails loudly.
    `model_copy(update=...)` silently accepts anything.

11. **Atomic writes must patch the module-local `os.fsync`.** Tests that
    simulate an fsync failure must patch `runner.run_writer.os.fsync`,
    not `os.fsync` globally. The module imports `os` at top level and
    uses `os.fsync(fd)` — the global patch does not intercept. Same
    pattern applies everywhere `os.fsync` / `asyncio.sleep` / `time.sleep`
    / `CGEventPost` is mocked.

12. **`Path.mkdir(mode=0o700)` is subject to the process umask.** Always
    follow with an explicit `os.chmod(path, 0o700)` — the umask commonly
    strips group/other bits but can also leave them in non-standard
    shells.

13. **Pillow's `UnidentifiedImageError` is NOT re-exported from
    `PIL.Image` under `mypy --strict`.** Import it from the top-level
    `PIL` package: `from PIL import Image, UnidentifiedImageError`.

14. **Fake-mode fixture click coords must be in bounds.** The dispatcher's
    `_parse_coordinate` silently returns an error result on out-of-bounds
    coords; the click is never forwarded to the adapter. Tests that only
    assert on status won't catch this. Keep fixture coords `x < 1440`
    and `y < 900` for the dry-run blank-canvas 1440×900 screen source.
    See `fixtures/llm_responses/runner_slack_status.json` — the
    `[1480, 70]` bug that X-022 caught.

15. **SQLite's default NULL ordering is ascending-first.** `RunIndex.list`
    wraps `ORDER BY started_at DESC` with `COALESCE(started_at, '') DESC`
    so `pending` rows (no `started_at` yet) sort to the bottom where the
    UI wants them.

16. **`pytest_addoption` must live at the test-tree root conftest
    (`tests/conftest.py`).** Pytest discovers plugin hooks only from
    rootdir conftests at startup; a subdirectory conftest either errors
    out ("already parsed") or silently ignores the hook depending on the
    invocation path.

17. **Circular import trap between `input_adapter` and `live_input`.**
    `live_input` imports `MouseButton` / `ScrollDirection` literals from
    `input_adapter`. Do NOT re-export `LiveInputAdapter` from
    `input_adapter` — that reintroduces the cycle. Keep the direction
    strictly one-way: `live_input` depends on `input_adapter`, never the
    reverse. Callers that need the live adapter import it explicitly:
    `from runner.live_input import LiveInputAdapter`.

18. **Datetime round-trip: pydantic v2 emits `Z`, not `+00:00`.**
    Round-trip-identity tests and on-disk fixtures must use the canonical
    `Z` suffix via `model_dump(mode="json")`. Adding a `@field_serializer`
    to switch to `+00:00` would break every fixture in
    `fixtures/skills/*/skill.meta.json`.

---

## How to run tests

All commands run from repo root. Use `uv run` — `uv sync --extra dev
--extra macos` keeps pytest/mypy/ruff installed.

```bash
# Full runner test suite (fast path, ~30s end-to-end).
uv run pytest tests/runner/

# Narrower: only budget + executor + e2e.
uv run pytest tests/runner/test_budget.py tests/runner/test_executor.py \
             tests/runner/test_e2e_fake.py

# Lint + type check — both must pass before committing.
uv run ruff check services/runner tests/runner
uv run mypy --strict services/runner

# Contracts sanity check (JSON schemas + destructive keyword list).
scripts/check_contracts.sh

# Live-opt-in tests — disposable machine only (see "Dry-run vs live mode").
TRACE_ALLOW_LIVE=1 uv run pytest tests/runner/test_live_input.py \
    --run-live-input -m live_input
```

Tests that need a real event loop use the module-level `pytestmark =
pytest.mark.asyncio` convention. The `anthropic_mock` fixture (respx)
blocks all network access; never mark a test that talks to the real API.
The `live_mode_allowed` fixture opts in to `TRACE_ALLOW_LIVE=1` for the
duration of one test and reverts on teardown — every other test has the
flag force-unset by the autouse `_force_live_mode_unset` fixture in
`tests/runner/conftest.py`.

---

## How live testing works (pre-merge, not pre-Ralph-complete)

The runner branch has a two-phase completion model. Reading the PRD's
X-024 and X-025 together is the canonical summary; this is the operational
view:

### Phase A — Autonomous Ralph dry-run (X-001 … X-024)

Ralph iterates under `mode=dry_run` only. Every story's `passes: true`
flip is machine-verified via the `uv run ruff check` / `mypy --strict` /
`pytest` triad. When X-001 through X-024 all show `passes: true`, the loop
emits `RUNNER_DRY_RUN_DONE`. **X-025 does NOT block Ralph completion.**
Ralph's prompt forbids it from modifying or reading X-025 at all.

### Phase B — Human live-execution checklist (X-025)

A human operator follows `tests/runner/live_checklist.md` on a real
machine with `TRACE_ALLOW_LIVE=1`, against disposable test accounts
(test Gmail, test Slack, etc.), one reference skill at a time. For each
skill the human verifies:

- The confirmation prompt fires correctly for each destructive step.
- Clicking Confirm executes the destructive step as expected.
- Clicking Abort cancels within 2 seconds.
- The kill-switch hotkey cancels within 2 seconds.
- The final app state matches the SKILL.md's Expected-outcome section.
- Total live-run cost across all five skills is under $5.

A sign-off section at the bottom of `live_checklist.md` (tester name,
date, macOS version, hardware, test accounts used) gates the PR merge
into `main`. `scripts/check_live_signoff.sh` parses the checklist and
exits non-zero if the sign-off is missing — used by the PR template / CI
to block merge.

**`tests/runner/live_checklist.md` is a human-edited file and Ralph MUST
NOT touch it.** The top of the file carries a header enforcing this. If
you see Ralph's output diffing this file, something has gone wrong in
the prompt and the change must be reverted.

### The split: why two phases

Dry-run gives us fast, hermetic, deterministic verification of the entire
pipeline — every branch, every error path, every safety layer — in under
a minute on a developer laptop. Live execution is the one thing dry-run
cannot verify: that the AX resolver actually works against Chrome and
Electron in practice, that Retina coord math lands on the right pixels,
that the Quartz event tap doesn't get throttled under real macOS CPU
load. That verification needs a human at the screen watching real
behavior — we don't automate it because the cost of a regression slipping
through is unbounded (real messages sent, real purchases made, real
calendar entries created on the wrong account). Ralph doing live runs
unattended is not a safety model anyone is willing to sign.

---

## When the user reports something non-trivial

Append to `progress.txt` in the matching `## <date> - <story-id>` block.
Graduate lessons that apply to *any* runner change here; cross-module
gotchas go to [/CLAUDE.md](../../CLAUDE.md) instead.
