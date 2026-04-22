# app — Trace desktop shell

Tauri 2 + React 18 + TypeScript + Vite. Developed on branch `feat/integration`.
Completion promise: `INTEGRATION_DONE`.

The initial Tauri scaffold (`pnpm create tauri-app`) is run on that branch, not
here on `main`. This directory exists so paths referenced in CLAUDE.md resolve
and so `pnpm --filter app` can eventually target it.

Responsibilities:
- Global recording hotkey (same hotkey doubles as the run kill-switch).
- Recording-overlay window.
- Skill library UI.
- Run controls (including destructive-step confirmation prompts).

Talks to the Python gateway at `http://127.0.0.1:8765`.
