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
