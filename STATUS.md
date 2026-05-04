# Trace — status at a glance

> Skim of where the project is right now. For the full plan, read
> `plan.md`. For the new-Claude-Code-session handoff prompt, see
> `HANDOFF_PROMPT.md`.

## Done (all merged into main)

| PR | Branch | What it shipped |
|---|---|---|
| #1 | feat/scaffold | Repo scaffold |
| #2 | feat/recorder | Recorder MVP (R-001..R-015) |
| #4 | feat/recorder | Recorder smoke fixes (`lsappinfo`, downscale, etc.) |
| #5 | feat/runner | Runner MVP (X-001..X-024) |
| #6 | feat/synthesizer | Synth + tiered hints + MCP catalog |
| #7 | feat/runner | Step 2: tier resolver, observation-only logging |
| #8 | feat/runner | Step 3a: MCP capability probe |
| #9 | feat/runner | Step 3b: MCP dispatch (non-destructive) |
| #10 | feat/runner | Step 3c: confirmation-aware MCP dispatch (destructive) |
| #11 | feat/recorder | Vite + React dashboard, recorder usage stats |

**1,415 tests passing** (recorder 304 / synth 474 / runner 637).

## End-to-end pipeline

Today, with everything in main, you can:

1. Press a hotkey, do a workflow on your Mac → recorder writes a
   trajectory directory.
2. Run the synth pipeline against it → produces `SKILL.md` with prose
   + `skill.meta.json` with per-step `execution_hints` chain
   (MCP > DOM > computer-use).
3. Replay the skill via the runner → it probes live MCP servers,
   pre-executes non-destructive MCP steps, gates destructive ones
   through a confirmation queue, then drives any remaining steps via
   computer-use. Pre-executed step results are injected into the
   agent's first message as a primer.
4. Watch the recorder usage stats on the Vite dashboard at
   `http://localhost:5173`.

## Next up

**Step 4** — Playwright DOM tier + agent observability dashboard tab.
Four PRs:

- **4.1** Playwright dispatcher in the runner (mirrors MCP pattern).
- **4.2** Dashboard tab framework + typed WebSocket client.
- **4.3** Browser Agent tab: tier ribbons, live Playwright frames,
  DOM action log, MCP timeline, confirmation modal.
- **4.4** Confirmation modal wired to the existing
  `/run/{id}/confirm` endpoint.

**Step 5** — Synth visual grounding (~2 hours, prompt-only change).

See `plan.md` §5 for the full scope.

## How to verify the current state locally

```bash
cd ~/Development/trace
git checkout main && git pull
TRACE_DEV_MODE=1 uv run uvicorn gateway.main:app --host 127.0.0.1 --port 8765 &

# All gates green
uv run ruff check services/recorder services/synthesizer services/runner tests
uv run mypy --strict services/recorder services/synthesizer services/runner
uv run pytest tests/

# Dashboard
cd dashboard && npm install && npm run dev
```

## Where each artifact lives

- Locked schemas: `contracts/{trajectory,skill-meta,run-metadata}.schema.json`
- Synth's MCP catalog: `services/synthesizer/src/synthesizer/mcp_catalog.py`
- Runner's tier resolver: `services/runner/src/runner/execution_hints.py`
- Runner's MCP layer: `services/runner/src/runner/mcp_client.py`
- Reference fixtures: `fixtures/trajectories/<workflow>/`,
  `fixtures/skills/<workflow>/`
- Dashboard: `dashboard/src/`
- Demo scripts: `scripts/demo_recording.py`, `scripts/demo_pipeline.py`
