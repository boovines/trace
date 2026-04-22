# services/runner

Module 3. Developed on branch `feat/runner`. Completion promise: `RUNNER_DRY_RUN_DONE`.

Executes a SKILL.md on behalf of the user via Claude Sonnet 4.5 with the
`computer_20250124` tool.

## SAFETY

The runner drives real mouse and keyboard input on the user's machine. Three
independent gating layers protect against destructive actions; all three must
be bypassed simultaneously for a destructive action to run unattended.

1. **Skill-level markers.** The synthesizer writes ⚠️ `[DESTRUCTIVE]` on steps
   it classifies as destructive. Those markers are injected into the execution
   system prompt as per-step confirmation requirements.
2. **Harness-level keyword matching.** Before every tool call, the harness
   inspects the target accessibility-element label against
   `contracts/destructive_keywords.json` (the same list the synthesizer uses —
   single source of truth). A match forces a confirmation prompt even if the
   skill failed to flag the step.
3. **Budget and rate limiting.** Every run has a hard per-run token budget and
   a per-minute action rate limit. Runaway loops trip one of those guards and
   the run is aborted.

### The `TRACE_ALLOW_LIVE` flag

Above all three layers sits a single environment variable:

- `TRACE_ALLOW_LIVE=1` → live adapters (CGEventPost, CGWindowListCreateImage)
  may be instantiated.
- Anything else (unset, `"0"`, `"true"`, etc.) → live adapters refuse to
  instantiate and raise `LiveModeNotAllowed`.

**Ralph iterations MUST run with this flag unset.** The Ralph loop prompt
explicitly forbids setting it. The dry-run adapters (`DryRunInputAdapter`,
`TrajectoryScreenSource`) are always safe to use and do not consult the flag.

Any test or code path that constructs a `LiveInputAdapter` or
`LiveScreenSource` **also** requires `TRACE_ALLOW_LIVE=1` AND is marked with
`@pytest.mark.live_input` — both opt-ins must be explicit. Run these tests only
on a disposable test machine, never on a production workstation.

## Contracts

- Reads [skill-meta.schema.json](../../contracts/skill-meta.schema.json)
  (Module 2 output).
- Writes per [run-metadata.schema.json](../../contracts/run-metadata.schema.json).
