# services/runner

Module 3. Developed on branch `feat/runner`. Completion promise: `RUNNER_DRY_RUN_DONE`.

Executes a SKILL.md on behalf of the user via Claude Sonnet 4.5 with the
`computer_20250124` tool. Safety gates (see root CLAUDE.md §Safety invariants):

- Must confirm before any destructive step — enforced both in the system prompt
  AND in the harness, as a belt-and-suspenders measure.
- Global kill-switch hotkey aborts any active run within 2 seconds.
- No shell execution, no file writes outside declared paths.
- Ralph iterations run in dry-run mode only. Live execution is a human-driven
  separate test phase, gated on `TRACE_ALLOW_LIVE=1`, which must **never** be
  set during Ralph iterations.

Contracts: reads [skill-meta.schema.json](../../contracts/skill-meta.schema.json),
writes per [run-metadata.schema.json](../../contracts/run-metadata.schema.json).
