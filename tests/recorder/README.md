# Recorder tests

Hermetic unit tests (the default run) stub every macOS framework via a
`types.ModuleType` injected into `sys.modules`. They pass anywhere Python
3.11 runs, including Ralph's sandbox.

## macOS integration tests

Tests marked `@pytest.mark.macos` hit real macOS APIs
(`CGEventTap`, `AXUIElement`, `CGWindowListCreateImage`). They are skipped
by default by `conftest.py`. To run them:

1. Run on a real Mac (darwin).
2. Grant the terminal running `pytest` three permissions in **System
   Settings → Privacy & Security**:
   - **Accessibility** — required for `CGEventTap` and `AXUIElement`
     queries.
   - **Input Monitoring** — required for `CGEventTap` on recent macOS
     releases.
   - **Screen Recording** — required once screenshot tests land (R-009).
3. Opt in via environment variable:

   ```sh
   TRACE_RUN_MACOS_TESTS=1 uv run pytest tests/recorder/ -m macos
   ```

If permissions are missing, the tests call `pytest.skip(...)` with a
message indicating which framework refused — they never falsely pass.

## Why this is opt-in

Ralph iterations run without those permissions. If the macOS tests ran by
default, every iteration would either fail (noise) or pass trivially
(silent skip). Making them opt-in keeps the default `pytest` green on the
sandbox while still giving the human tester a single command to verify
real hardware before shipping the module.

## Reference-workflow fixtures

`fixtures/trajectories/<slug>/` holds one canonical recording per
reference workflow (`gmail_reply`, `calendar_block`, `finder_organize`,
`slack_status`, `notes_daily`). `test_fixtures.py` asserts structural
minimums on each: schema validity, monotonic 1-indexed `seq`, at least
one click with an AX target, one `text_input` with a `field_label`, one
`app_switch`, valid PNG per `screenshot_ref`, and total size under 5 MB.

The checked-in fixtures are **synthetic** — produced by
`scripts/generate_synthetic_fixtures.py` so downstream module tests
(Synthesizer, Runner) can run without a real Mac. A human tester
replaces them with real recordings during smoke testing per
`scripts/regenerate_fixtures.sh`. Rerun the synthesizer whenever the
schema changes:

```sh
uv run python scripts/generate_synthetic_fixtures.py
uv run pytest tests/recorder/test_fixtures.py
```
