# Trace

A macOS desktop app that records, synthesizes, and replays on-screen workflows.

See [CLAUDE.md](./CLAUDE.md) for the full project spec, module architecture, contracts, and safety invariants.

## Layout

- `contracts/` — JSON schemas shared across modules (locked once a module ships)
- `services/recorder/` — Module 1: captures macOS user actions into a trajectory (branch: `feat/recorder`)
- `services/synthesizer/` — Module 2: converts a trajectory into a SKILL.md (branch: `feat/synthesizer`)
- `services/runner/` — Module 3: executes a SKILL.md via Claude computer-use (branch: `feat/runner`)
- `gateway/` — FastAPI app that mounts all three services on `127.0.0.1:8765`
- `app/` — Tauri 2 + React desktop shell (branch: `feat/integration`)
- `fixtures/` — Shared reference trajectories and skills
- `tests/` — Per-module test suites
- `scripts/ralph/` — Ralph loop scripts

## Quickstart

```bash
# Python side
uv sync
uv run uvicorn gateway.main:app --host 127.0.0.1 --port 8765 --reload

# Frontend (once the Tauri app is scaffolded on feat/integration)
pnpm --filter app dev
```

## Quality gates

```bash
uv run ruff check services/ gateway/
uv run mypy --strict services/ gateway/
uv run pytest tests/
```
