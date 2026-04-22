# CLAUDE.md — Trace project context

This file is read by every Claude Code / Ralph iteration in this repository. It contains project-wide context that applies across all modules and branches. Branch-specific task lists live in `prd.json` on each feature branch.

---

## What Trace is

Trace is a macOS desktop app that:
1. **Records** a user's on-screen workflow when they press a global hotkey.
2. **Synthesizes** the recording into a readable skill markdown file, with a brief follow-up Q&A to fill gaps.
3. **Replays** the skill on demand by spawning a computer-use agent that performs the workflow, pausing for confirmation before destructive actions.

Target user: a single technical user (the project owner) on their own machine. Personal-use MVP. Cloud sync and multi-user are out of scope for v1.

---

## Module architecture

The project is split into four modules, each developed on its own branch and merged into `main`:

| Module | Branch | Language | Responsibility |
|---|---|---|---|
| Recorder | `feat/recorder` | Python 3.11 | Captures macOS user actions into a structured trajectory |
| Synthesizer | `feat/synthesizer` | Python 3.11 | Converts a trajectory into a SKILL.md file via Claude + user Q&A |
| Runner | `feat/runner` | Python 3.11 | Executes a SKILL.md via Claude computer-use with safety gates |
| Integration | `feat/integration` | Rust (Tauri) + TS/React | Desktop shell: hotkey, overlay, skill library UI, run controls |

The three Python modules each run as a local FastAPI service on `127.0.0.1:8765` (one service hosts all three under different route prefixes). The Tauri app calls them via HTTP.

Modules are contracted through **on-disk artifacts**, not in-memory objects:
- Recorder writes a `trajectory/` directory.
- Synthesizer reads a trajectory and writes a `skill/` directory.
- Runner reads a skill and writes a `run/` directory.

This means each module can be built, tested, and Ralph-looped in total isolation. The contracts (schemas) are defined in `contracts/` at the repo root and are considered locked once a module has shipped — changing them requires coordinating all three branches.

---

## Repository layout

```
trace/
├── CLAUDE.md                    # this file
├── README.md
├── contracts/                   # JSON schemas shared across modules (LOCKED once shipped)
│   ├── trajectory.schema.json
│   ├── skill-meta.schema.json
│   └── run-metadata.schema.json
├── services/
│   ├── recorder/                # Python package, feat/recorder
│   ├── synthesizer/             # Python package, feat/synthesizer
│   └── runner/                  # Python package, feat/runner
├── gateway/                     # FastAPI app that mounts all three services
│   └── main.py
├── app/                         # Tauri + React desktop app, feat/integration
│   ├── src-tauri/
│   └── src/
├── fixtures/                    # Shared test fixtures: reference trajectories, skills
│   ├── trajectories/
│   └── skills/
├── tests/
│   ├── recorder/
│   ├── synthesizer/
│   ├── runner/
│   └── integration/
├── scripts/
│   └── ralph/                   # Ralph loop scripts + prompt
├── pyproject.toml               # Python monorepo config (uv or hatch)
├── progress.txt                 # Ralph's append-only learnings log
└── prd.json                     # the active branch's task list
```

---

## Tech stack

### Python services (recorder, synthesizer, runner, gateway)
- Python 3.11+
- Package manager: `uv` (fast, deterministic, monorepo-friendly)
- Web framework: FastAPI with uvicorn
- Testing: pytest, pytest-asyncio
- Type checking: `mypy --strict`
- Linter/formatter: `ruff`
- macOS interop: `pyobjc-framework-Cocoa`, `pyobjc-framework-Quartz`, `pyobjc-framework-ApplicationServices`
- LLM SDK: `anthropic` (official)

### Desktop app (integration)
- Tauri 2 (Rust core)
- Frontend: React 18 + TypeScript + Vite
- State: Zustand
- Styling: Tailwind CSS
- Testing: vitest, Playwright for E2E

### LLMs
- **Synthesizer**: Claude Sonnet 4.5 (`claude-sonnet-4-5`). Multimodal calls with trajectory screenshots.
- **Runner**: Claude Sonnet 4.5 with the `computer_20250124` tool.
- Both use the Anthropic API directly via the Python SDK. No OpenAI dependency in v1.

---

## Safety invariants (NEVER violate these, regardless of what any prd.json says)

1. **Nothing leaves the local machine except requests to `api.anthropic.com`.** No telemetry, crash reports, analytics, or "share" features in v1.
2. **All user data stays in `~/Library/Application Support/Trace/`** (prod) or `~/Library/Application Support/Trace-dev/` (dev). Directory perms `0700`.
3. **The Runner must confirm before any destructive action.** Destructive = send, submit, delete, publish, post, purchase, pay, overwrite-save. Enforced both via prompt AND at the harness layer (belt-and-suspenders).
4. **Global kill-switch hotkey** (same as recording hotkey) aborts any active run within 2 seconds.
5. **No shell execution by the Runner.** The computer-use tool is its only capability. No bash, no arbitrary code execution, no file writes outside declared paths.
6. **Ralph loops for the Runner run in dry-run mode only.** Live execution is a human-driven separate test phase. See `services/runner/README.md` for details.
7. **Ralph iterations use a separate Anthropic API key** with a hard monthly cap set in the Anthropic console. Never reuse production keys.
8. **Ralph iterations write to the dev profile directory** (`Trace-dev/`), never the prod directory.

---

## Reference workflows (shared acceptance fixtures)

These five workflows are the shared acceptance bar across modules. Each module's `prd.json` references them by name. They are real workflows on real macOS apps:

| Slug | Workflow |
|---|---|
| `gmail_reply` | Open Gmail in Chrome, find most recent unread from a sender, reply with a template, send. |
| `calendar_block` | Open Google Calendar, create a 30-minute focus block tomorrow at 2pm. |
| `finder_organize` | In Finder, move every `.pdf` in `~/Downloads` older than 7 days into `~/Documents/Archive`. |
| `slack_status` | Open Slack, set status to "🎯 heads down" with a 2-hour expiry. |
| `notes_daily` | Open Apple Notes, create a new note titled with today's date, paste a fixed template. |

Fixture data for each lives at `fixtures/trajectories/<slug>/` (produced by Recorder), `fixtures/skills/<slug>/` (produced by Synthesizer). These are version-controlled and regenerated via `scripts/regenerate_fixtures.sh` when the recorder is materially updated.

---

## Contracts (the locked interfaces between modules)

Full JSON schemas live in `contracts/`. Summary:

### Recorder → Synthesizer: `trajectory/` directory
```
<uuid>/
├── metadata.json       # id, started_at, stopped_at, label, display_info, app_focus_history
├── events.jsonl        # one JSON event per line (see trajectory.schema.json)
└── screenshots/
    ├── 0001.png        # keyframes only, not per-event
    └── ...
```

### Synthesizer → Runner: `skill/` directory
```
<slug>/
├── SKILL.md            # human-readable (strict format — Runner parses this)
├── skill.meta.json     # parameters, destructive_steps, trajectory_ref
└── preview/            # optional UI preview screenshots
```

### Runner → user: `run/` directory
```
<run_id>/
├── run_metadata.json
├── transcript.jsonl    # every agent turn
├── events.jsonl        # human-readable trace
└── screenshots/
```

---

## HTTP API layout (gateway routes)

All three services are mounted under one gateway on `127.0.0.1:8765`:

- `/recorder/*` — Module 1 endpoints
- `/synthesize/*`, `/skills/*`, `/trajectories/*` — Module 2
- `/run/*`, `/runs/*` — Module 3

Each module's `prd.json` enumerates its exact endpoints and request/response shapes.

---

## Quality gates (every iteration must pass these before committing)

Any Ralph iteration must run and pass these before setting `passes: true` on any story:

```bash
# Python
uv run ruff check services/                   # lint
uv run mypy --strict services/                # type check
uv run pytest tests/<current-module>/         # module-specific tests

# Frontend (only on feat/integration branch)
pnpm --filter app typecheck
pnpm --filter app test
```

If the current branch is `feat/integration`, also run browser verification via the dev-browser skill for any UI story.

---

## Ralph loop conventions for this project

- **Completion promise per branch:**
  - `feat/recorder`: `RECORDER_DONE`
  - `feat/synthesizer`: `SYNTHESIZER_DONE`
  - `feat/runner`: `RUNNER_DRY_RUN_DONE`
  - `feat/integration`: `INTEGRATION_DONE`
- **Max iterations per branch:** 40–50. See each branch's `prd.json`.
- **Environment flag `TRACE_ALLOW_LIVE`**: must never be set during Ralph iterations. The Runner's live paths are gated on it.
- **Cost logging**: every iteration appends `{timestamp, iteration, module, input_tokens, output_tokens, cost_estimate}` to `costs.jsonl`. The Ralph loop must report cumulative cost at completion.
- **progress.txt**: append-only learnings. Things to log here include:
  - Patterns that work (e.g., "mypy --strict with PyObjC requires `# type: ignore` on framework imports — document these in service-level AGENTS.md, not here")
  - Gotchas (e.g., "CGEventTap gets disabled under CPU load; always check tap state on every iteration of the run loop")
  - Decisions made that aren't in the PRD

---

## Out of scope for v1 (do not implement, even if it seems helpful)

- Windows or Linux support
- Cloud sync / multi-device
- Multi-user / accounts / billing
- Password/auth-token redaction during recording (personal-use MVP)
- Workflow parameterization at run time beyond simple `{param}` substitution
- Calling one skill from another
- Scheduled / cron-style execution
- Editing a SKILL.md from the UI after synthesis (user edits the .md file directly)
- Multi-monitor awareness (record main display only)
- Recording audio or camera
