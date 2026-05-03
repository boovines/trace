# Trace — work plan and handoff

> Living document. Last updated 2026-05-02.
>
> Captures the full state of the Trace project as of right now: what's
> shipped, what's in flight, what's planned, and exactly enough context
> for a fresh Claude Code session to pick up where the current one left
> off.

---

## 1. What Trace is

A personal-use macOS desktop app:

1. **Records** a user's on-screen workflow when they press a global hotkey.
2. **Synthesizes** that recording into a readable `SKILL.md` plus a
   structured `skill.meta.json` containing per-step **execution hints**
   (MCP > DOM > computer-use) — the synthesizer's job is to produce
   *agent context*, not just human prose.
3. **Replays** the skill on demand by routing each step through the
   highest-tier dispatcher available (real MCP API call → Playwright
   DOM action → pixel-grounded computer-use), with confirmation gates
   for destructive actions.

Target user: a single technical user (project owner) on their own Mac.
Personal-use MVP. Cloud sync, multi-user, multi-monitor are out of
scope for v1.

## 2. Architecture

Four modules, each on its own branch, talking through **on-disk
artifacts**:

| Module | Branch | Lang | Job |
|---|---|---|---|
| Recorder | `feat/recorder` | Python | Capture macOS user actions → `trajectory/` dir |
| Synthesizer | `feat/synthesizer` | Python | Trajectory → `SKILL.md` + `skill.meta.json` via Claude + Q&A |
| Runner | `feat/runner` | Python | SKILL.md → executed run via tiered dispatch |
| Integration / Dashboard | `feat/recorder` (added there) | Vite + React + Recharts | Read-only observability frontend at `dashboard/` |

All three Python services run as **one process** behind a gateway on
`127.0.0.1:8765`. Module contracts (locked once shipped) live in
`contracts/`:

- `trajectory.schema.json` — recorder output, synthesizer input
- `skill-meta.schema.json` — synthesizer output, runner input (now
  carries optional per-step `execution_hints` chain)
- `run-metadata.schema.json` — runner output

Tech stack:

- **Python 3.11**, `uv` workspace, `mypy --strict`, `ruff`, `pytest`
- **macOS interop** via PyObjC (`Cocoa`, `Quartz`, `ApplicationServices`)
- **LLM**: Claude Sonnet 4.5 via the official `anthropic` SDK directly
  (no LangChain). Synth uses multimodal (text + screenshot keyframes);
  runner uses the `computer_20250124` tool.
- **MCP**: official `mcp>=1.0` SDK. Stdio subprocess transport.
  Configured via `~/.config/trace/mcp_servers.json` (Claude-desktop-
  compatible shape).
- **Frontend**: Vite + React 18 + Recharts at `dashboard/` (no router,
  no Zustand, plain CSS in `styles.css`). Currently a single screen of
  recorder usage stats; tab framework + new screens TBD in Step 4.

## 3. Files and where to find them

```
trace/
├── CLAUDE.md                   # Project-wide context (read first)
├── contracts/                  # Locked JSON schemas (recorder/synth/runner share)
├── services/
│   ├── recorder/               # feat/recorder
│   ├── synthesizer/            # feat/synthesizer (uses src/ layout)
│   └── runner/                 # feat/runner (uses src/ layout)
├── gateway/main.py             # FastAPI app mounting all three services on :8765
├── dashboard/                  # Vite + React + Recharts (added by PR #11)
│   ├── src/App.tsx             # Single-page stats UI; tab framework lives HERE
│   ├── src/api.ts              # gateway client
│   ├── src/components/         # Header, charts, tables
│   └── src/styles.css          # plain CSS (no Tailwind)
├── fixtures/
│   ├── trajectories/           # 5 reference workflows (R-013)
│   ├── skills/                 # synth golden output (gmail_reply has hint chain)
│   └── llm_responses/          # synth fake-mode canned responses
├── scripts/
│   ├── ralph/                  # Ralph loop driver + per-branch prd.json
│   ├── demo_recording.py       # recorder live demo
│   └── demo_pipeline.py        # synth pipeline demo
├── tests/{recorder,synthesizer,runner}/
└── plan.md                     # ← this file
```

Local checkouts (each holds one branch):

```
~/Development/trace                 # main
~/Development/trace-recorder        # feat/recorder
~/Development/trace-synthesizer     # feat/synthesizer
~/Development/trace-runner          # feat/runner
```

## 4. Status snapshot

### Merged PRs (in order)

| PR | Branch | Title | What it shipped |
|---|---|---|---|
| #1 | feat/scaffold | initial scaffold | repo layout, contracts/, ralph |
| #2 | feat/recorder | recorder MVP | R-001..R-015 (capture, fixtures, smoke) |
| #4 | feat/recorder | recorder smoke fixes | session tap, chars, lsappinfo, downscale, tap-disabled echo, field_label |
| #5 | feat/runner | runner MVP | X-001..X-024 (parser, dispatcher, kill-switch, etc.) |
| #6 | feat/synthesizer | synth + tiered hints | S-001..S-020 + canonical-schema realignment + MCP catalog + execution_hints in `skill-meta.schema.json` + draft prompt |
| #7 | feat/runner | tier resolver (Step 2) | `runner.execution_hints` resolver, tier_selected events |
| #8 | feat/runner | MCP probe (Step 3a) | `runner.mcp_client` probe layer, populated CapabilityRegistry |
| #9 | feat/runner | MCP dispatch (Step 3b) | `MCPCallDispatcher`, `substitute_parameters`, primer message, non-destructive dispatch |
| #10 | feat/runner | destructive MCP (Step 3c) | confirmation-aware dispatch via existing `ConfirmationQueue` |
| #11 | feat/recorder | dashboard | Vite + React + Recharts at `dashboard/`, recorder stats |

### What this means today

The recorder → synthesizer → runner pipeline works end to end with:

- **Recorder**: real keystroke + click + scroll + app_switch capture
  (`lsappinfo`-driven), 5 reference fixtures committed, demo script at
  `scripts/demo_recording.py`.
- **Synthesizer**: trajectory in → SKILL.md + skill.meta.json out, with
  per-step `execution_hints` chain (MCP > DOM > computer-use). Live MCP
  catalog of 22 functions across 5 servers. Demo at
  `scripts/demo_pipeline.py`.
- **Runner**: reads `meta.steps[].execution_hints`, probes live MCP
  servers at startup, **actually invokes** non-destructive MCP calls,
  and gates destructive MCP calls through the existing confirmation
  queue. Pre-execution results land in a primer user-message so the
  agent skips already-done steps. Computer-use remains the universal
  fallback.
- **Dashboard**: single-screen Vite/React stats over the recorder's
  data (range selector, daily activity, top apps, event mix). No tab
  framework yet.

### What's not shipped

- **Browser DOM tier** — `tier=browser_dom` hints are recognised but
  never executed.
- **Dashboard tab framework** — currently a single screen.
- **Browser-agent observability tab** — no live-run UI yet (events are
  in `events.jsonl` only).
- **Synth visual grounding** — synth's prose doesn't reference
  screenshots spatially.

That's exactly the scope of Steps 4 and 5 below.

## 5. Step-by-step plan (history + future)

### Done

| Step | PR(s) | Outcome |
|---|---|---|
| 1 | #6 | Synth emits `meta.steps[].execution_hints` and a primer prompt section explaining MCP catalog |
| 2 | #7 | Runner's `pick_hint` resolver + `tier_selected` events (logging-only) |
| 3a | #8 | MCP capability probe — registry populated from live servers |
| 3b | #9 | Real MCP dispatch for non-destructive steps + agent primer message |
| 3c | #10 | Confirmation-aware MCP dispatch for destructive steps; entire MCP track closed |

### Step 4 — Playwright DOM tier + agent observability dashboard (CURRENT)

Two coupled tracks. Sequencing: 4.1 lands first; 4.2 in parallel; 4.3
needs both; 4.4 last.

#### 4.1 — Playwright DOM dispatcher (~1.5k LOC, runner)

Mirror the MCP dispatch pattern at the `browser_dom` tier.

- Add `playwright>=1.40` runtime dep on the runner package.
- Add `await playwright install chromium` to the `uv sync` story (or
  detect missing browser at startup and surface a clear error).
- New module `services/runner/src/runner/browser_dom_dispatcher.py`:
  - `BrowserDOMDispatcher` async-context-managed (mirrors
    `MCPCallDispatcher`).
  - `__aenter__`: launch Chromium (`playwright.async_api.async_playwright`),
    create a context, optionally attach to an existing CDP endpoint
    via `TRACE_CDP_ENDPOINT` env.
  - `dispatch(hint, parameters)` — substitute params into
    `url_pattern` / `selector` / `value`, navigate if needed, perform
    `action` (click/type/navigate/scroll/submit), return a structured
    result (mirrors `MCPCallResult`).
- Extend `runner.execution_hints.CapabilityRegistry`: replace `browser_dom: bool`
  with `browser_dom_capability: BrowserDOMCapability | None` carrying
  whether Chromium is launchable + the CDP endpoint if any.
- New `runner.browser_dom_probe.probe_browser_dom() -> BrowserDOMCapability | None`
  called by `RunManager._get_capability_registry` alongside the MCP probe.
- `Executor`:
  - Accept optional `browser_dom_dispatcher` kwarg (mirror MCP wiring).
  - Extend `_pre_execute_mcp_steps` (or split into a generic
    `_pre_execute_tiered_steps`) so `tier=browser_dom` hints flow
    through the same destructive/confirmation path.
  - `_dispatch_one_dom` mirrors `_dispatch_one_mcp`.
- Tests: stub Playwright via `playwright-fluent` or a custom
  `_FakeBrowserContext` so CI doesn't launch a real browser. Cover
  navigate/click/type/submit, `{param}` substitution, success/error
  paths, destructive-confirm flow.

#### 4.2 — Dashboard tab framework + WebSocket client (~800 LOC, dashboard)

The `dashboard/` from PR #11 is single-screen. Add a tab system without
introducing a router; everything stays one page-load, just URL hash
state for deep-linking.

- New `src/router.ts` with a tiny hash-based store (`useTabState()`
  hook backed by `window.location.hash`). No external dep.
- Refactor `App.tsx` to render a tab bar + the active tab's component.
- Tabs at launch:
  - **Stats** (existing screen, refactored into `src/tabs/StatsTab.tsx`)
  - **Runs** (new): list runs from `GET /runs`, click into one to see
    `run_metadata.json` + the events stream.
  - **Browser Agent** (new, the meat — see 4.3 below)
- New `src/ws.ts`: typed WebSocket client that subscribes to
  `/run/{run_id}/stream`, emits typed events into a callback. Reconnect
  with exponential backoff. Distinguish Trace event types (the runner
  already broadcasts `tier_selected`, `mcp_dispatched`,
  `confirmation_requested`, etc.).
- Add `recharts` is already in deps; might need `clsx` for the tab UI.
- TypeScript types for run events and tier decisions live in
  `src/types.ts` — kept in sync with the runner's
  `runner.execution_hints` and `runner.event_stream` shapes.

#### 4.3 — Browser Agent tab (~1.2k LOC, dashboard + gateway)

The new tab consumes the WS stream from 4.2 and renders, for the
active run:

- **Tier ribbon per step** — lit pill (`MCP` / `DOM` / `COMPUTER_USE`)
  showing what the runner picked. Hover tooltip exposes the
  `considered` chain + `unsupported_reasons` (data the runner already
  ships in `tier_selected` events).
- **Live Playwright video** for `tier=browser_dom` steps — Playwright's
  `context.tracing.start({screenshots: true, snapshots: true})`
  produces frames we serve as JPEG-stream over a new gateway endpoint:
  `GET /run/{run_id}/playwright_frames` (chunked
  `multipart/x-mixed-replace`). UI shows a viewport-shaped card with
  cursor overlay.
- **DOM action log** — per step: ordered list of executed
  `(selector, action, value)` tuples with success/failure markers and
  per-action duration.
- **MCP call timeline** — for `tier=mcp` steps: the
  `server.function(args)` line and the structured `content_text`
  response. JSON-formatted when it parses; raw otherwise.
- **Confirmation modal** — when the run blocks on
  `confirmation_requested`, modal pops with the prospective action and
  Approve/Decline buttons wired to `POST /run/{id}/confirm` and
  `POST /run/{id}/abort` (existing endpoints).

Backend touches:

- Runner needs to broadcast Playwright frame URLs as a new
  `playwright_frame` event type (or write them to disk and let the
  dashboard poll). Frame writer lives in
  `runner.browser_dom_dispatcher`; the run's `screenshots/` dir grows
  a `dom_frames/` subdir.
- Gateway adds `GET /run/{run_id}/dom_frames/{seq}.png` static-serve
  route.
- Existing `EventBroadcaster` already handles JSON event fan-out; just
  add the new event types.

#### 4.4 — Confirmation modal wired to existing endpoints (~300 LOC, dashboard)

Hook the modal from 4.3 to the existing confirmation API. The runner
already exposes `POST /run/{id}/confirm` (decision) and a
`confirmation_request` WebSocket event with the destructive_reason.
The dashboard now finally surfaces this to a real UI rather than
events.jsonl.

#### Step 4 sequencing summary

| PR | Scope | Risk | Depends on |
|---|---|---|---|
| 4.1 | Playwright dispatcher + tier wiring | medium (first browser dep) | — |
| 4.2 | Dashboard tab framework + WS client + Runs/Stats tabs | low (pure frontend) | — |
| 4.3 | Browser Agent tab + frame streaming | medium | 4.1 + 4.2 |
| 4.4 | Confirmation modal | low | 4.2 |

Total ~3.8k LOC across four PRs.

### Step 5 — Synthesizer visual grounding (~2 hours, synthesizer)

Smaller, surgical change to the synth's draft prompt + parser.

- Update `services/synthesizer/src/synthesizer/draft_prompt.py` to
  instruct the LLM to write **spatially-grounded** step prose:
  "Click the Send button at the bottom-left of the reply pane (visible
  in screenshot 3)" instead of "Click Send."
- Add a per-step `screenshot_ref` field to `meta.steps[]` so the runner
  can show the relevant keyframe alongside the prose during
  computer-use fallback.
- Update `fixtures/skills/<workflow>/SKILL.md` golden examples to
  match.
- Bump 1-2 prompt-shape tests in `tests/synthesizer/test_draft.py`.
- No schema-breaking changes; backward compatible.

Improves the universal fallback (computer-use) without depending on
any tier-specific work.

### Possible Steps 6+ (open)

- **Recorder browser-context enrichment**: capture URL + DOM-stable
  selectors + ARIA role/name for browser-window clicks. Makes Step 4's
  DOM tier reliable without needing the synth to invent selectors at
  draft time. ~1 day.
- **Process MCP**: register the runner itself as an MCP server so the
  synth can invoke the runner via Claude desktop. Future loop closure.
- **Tauri shell**: wrap the dashboard in a Tauri 2 app for distribution.
  The `app/` scaffold is already in the repo with empty `src` and
  `src-tauri`.

## 6. How to run things

```bash
# All three Python services share the gateway on 127.0.0.1:8765
cd ~/Development/trace
TRACE_DEV_MODE=1 uv run uvicorn gateway.main:app --host 127.0.0.1 --port 8765

# Recorder demo (live capture + summary)
uv run python scripts/demo_recording.py

# Synth pipeline demo (loads a trajectory, prints what would go to Claude)
uv run python scripts/demo_pipeline.py --no-llm --fixture gmail_reply
# With ANTHROPIC_API_KEY set, drops --no-llm to actually call Claude.

# Dashboard
cd dashboard
npm install
npm run dev          # opens at http://localhost:5173, proxies /api → :8765
npm run typecheck

# Quality gates per package
cd ~/Development/trace
uv run ruff check services/<pkg> tests/<pkg>
uv run mypy --strict services/<pkg>
uv run pytest tests/<pkg>

# MCP server config (for Steps 3+)
cat > ~/.config/trace/mcp_servers.json <<'EOF'
{
  "mcpServers": {
    "gmail":  {"command": "uvx", "args": ["mcp-gmail"]},
    "slack":  {"command": "uvx", "args": ["mcp-slack"]}
  }
}
EOF
```

## 7. Important conventions and gotchas

- **Permissions on macOS**: the runner Python process needs Accessibility,
  Screen Recording, Input Monitoring granted to **`/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11`** (the actual binary the venv resolves to). Granting "Terminal" or "Python.app" alone doesn't propagate. Use the auto-prompt path: have the recorder call `AXIsProcessTrustedWithOptions` with `prompt: True` once and click through.
- **Event-tap layer**: `kCGSessionEventTap`, NOT `kCGHIDEventTap`. HID-layer taps only see system hotkeys for unprivileged processes.
- **App focus**: `lsappinfo` shells out to LaunchServices and updates live; `NSWorkspace.frontmostApplication()` doesn't (caches in non-Cocoa processes). Don't use NSWorkspace.
- **Synth event shape**: the recorder emits canonical events
  (`{seq, timestamp_ms, type, app:{...}, target, payload}`). Synth's
  `TrajectoryReader` translates legacy-flat fixtures at load time so
  both shapes work; the contract truth is canonical.
- **MCP catalog**: lives at `services/synthesizer/src/synthesizer/mcp_catalog.py`. **The synthesizer's prompt embeds it verbatim**. When you add a new MCP server, update the catalog dict; the next synth run will know about it. The schema-validator also rejects hints whose function isn't in the catalog.
- **Destructive gate**: there are TWO matchers. Synth-side keyword matcher (belt-and-suspenders) annotates SKILL.md; runner-side
  `_pre_execute_mcp_steps` reads `meta.destructive_steps` and forces
  confirmation for any tier=mcp hit. Both must agree.
- **Pre-execution ordering**: the runner pre-executes MCP steps **at run start**, before the agent loop runs. The agent's first user message is a "Steps [N] have been pre-executed" primer. The agent is expected to skip those and continue from the first non-pre-executed step.

## 8. Test-suite quick reference

| Module | tests | passing | what's covered |
|---|---|---|---|
| recorder | `tests/recorder/` | 304 | session, event_tap, focus_tracker (incl. lsappinfo), text_aggregator, screenshot, fixtures, e2e |
| synthesizer | `tests/synthesizer/` | 474 + 1 skipped | preprocessor, draft, MCP catalog, execution_hints schema, golden fixtures, full pipeline fake-mode |
| runner | `tests/runner/` | 637 + 10 skipped | parser, dispatcher (computer-use), execution_hints resolver, mcp_client probe + dispatch + confirmation, executor end-to-end |

Total **1,415 passing** across the three modules.

---

## 9. For a new Claude Code session — copy-paste prompt

Save this file (`plan.md`) at the repo root and start the new session
with:

```text
Read /Users/justinhou/Development/trace-runner/plan.md fully before
doing anything. It captures the entire project state, what's shipped,
and the exact scope of the next two Steps (4 and 5).

The current work item is Step 4. Use the four-PR sequencing in
plan.md §5. Start with PR 4.1 (Playwright DOM dispatcher) unless I
say otherwise.

When in doubt about how to integrate something, mirror the existing
MCP dispatch pattern in services/runner/src/runner/mcp_client.py and
the executor's _pre_execute_mcp_steps / _confirm_mcp_destructive
helpers — they're the canonical templates for tiered execution. The
dashboard pattern lives at dashboard/src/App.tsx; the tab framework
in §4.2 should use a tiny hash-state hook, not a router dep.

Quality bar before any commit: ruff check + mypy --strict + pytest
green for the touched module(s). Commit messages follow the existing
pattern (see git log for examples).
```
