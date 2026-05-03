# Handoff prompt — paste this into a new Claude Code session

Copy everything between the dashes below into the first message of a
new Claude Code conversation. The session you're starting will then
have the same project context the previous one had.

---

You are picking up Trace, a personal-use macOS desktop app that
records → synthesizes → replays on-screen workflows. The full
project plan, history, and conventions are in
`/Users/justinhou/Development/trace-runner/plan.md` — **read that
file in full before responding**. Everything you need is in there:
architecture, the four module branches, file layout, what's
shipped, what's left, and the conventions/gotchas that will save
you from rediscovering the same macOS / MCP / schema traps the
last session hit.

The four working checkouts on disk:

```
~/Development/trace                 # main
~/Development/trace-recorder        # feat/recorder
~/Development/trace-synthesizer     # feat/synthesizer
~/Development/trace-runner          # feat/runner
```

The active branch right now is `feat/runner`. PRs #1, #2, #4–#11
are all merged into main. The MCP execution track (Steps 3a/3b/3c)
is complete. The next work is Step 4.

## What to do

Start **Step 4** as scoped in `plan.md` §5. It's split into four
PRs — do them in this order, one per session if needed:

1. **4.1 — Playwright DOM dispatcher.** Mirror the MCP dispatch
   pattern (`services/runner/src/runner/mcp_client.py` +
   `Executor._pre_execute_mcp_steps` /
   `_confirm_mcp_destructive`). New module
   `runner.browser_dom_dispatcher`, extended `CapabilityRegistry`,
   wired through `RunManager`.
2. **4.2 — Dashboard tab framework + WS client.** Refactor the
   single-screen dashboard at `dashboard/src/App.tsx` to use
   hash-state tabs (no router dep). Add Stats / Runs / Browser
   Agent tabs. New `src/ws.ts` typed WebSocket client subscribing
   to `/run/{run_id}/stream` (events the runner already broadcasts).
3. **4.3 — Browser Agent tab.** The meat of observability —
   per-step tier ribbons, live Playwright frames, DOM action log,
   MCP timeline, confirmation modal. Adds a chunked-JPEG endpoint
   on the gateway.
4. **4.4 — Confirmation modal wired to existing endpoints.**

After Step 4, **Step 5** is a ~2-hour synth-prompt change for
visual grounding (also scoped in `plan.md`).

## Working conventions

- One PR per logical change. Commit messages match the existing
  style (see git log for examples).
- Quality bar before every commit: `ruff check` + `mypy --strict` +
  `pytest` all green for the touched module(s). Run gates from the
  top-level checkout, not subdirs.
- Confirm with me before opening a PR. After every push, share the
  PR URL plus a one-paragraph summary of what landed.
- Follow the safety invariants in `CLAUDE.md` — especially: nothing
  leaves the local machine except `api.anthropic.com`, no shell
  execution by the runner, destructive actions always go through
  the confirmation queue, kill-switch hotkey aborts within 2s.
- Ralph-loop iterations write to `~/Library/Application Support/Trace-dev/`,
  never the prod profile. Set `TRACE_DEV_MODE=1` in any process
  you start.

## First action

Read `plan.md`. Then propose the smallest unit of work that ships
**PR 4.1** and wait for me to say go before writing any code. Don't
start adding new deps, schema fields, or modules without checking
the plan first — most of the patterns are already established.

---

## What's NOT in this prompt (intentionally)

- The full PR-by-PR history. Read `plan.md` §4 for the table.
- The exact line numbers / file paths for every existing pattern.
  Read the modules; the plan tells you which ones are canonical
  templates.
- API keys / secrets. None of those are needed for the next steps;
  fake mode covers synth tests and Playwright runs locally.
