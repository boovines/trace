# Synthesizer tests

This suite covers the Module 2 (Synthesizer) Python package under
`services/synthesizer/`. All tests run with `TRACE_LLM_FAKE_MODE=1`
auto-applied via `conftest.py`'s `_force_fake_mode` autouse fixture, plus a
respx network guard that turns any unmocked call to `api.anthropic.com` into
an error. Nothing in the standard suite hits the live API.

## Running the standard suite

From the repo root:

```bash
uv run pytest tests/synthesizer/
```

Quality gates (the same commands the Ralph loop runs before committing):

```bash
uv run ruff check services/synthesizer tests/synthesizer
uv run mypy --strict services/synthesizer
uv run pytest tests/synthesizer/
./scripts/check_fixtures.sh
```

The standard suite is hermetic — no network access, no real API key
required, no on-disk writes outside `tmp_path`.

## Running the real-API smoke test (S-017)

`test_real_smoke.py` is the **pre-merge gate** for the synthesizer module. It
generates a draft for each of the five reference trajectories using real
Claude Sonnet 4.5, scores each draft against its hand-crafted golden via
Claude Haiku 4.5 (the `score_skill_similarity` rubric), and writes the
result to `tests/synthesizer/smoke_report.json`.

The smoke test is gated behind `TRACE_REAL_API_TESTS=1` and is **not** part
of the Ralph loop — Ralph emits `SYNTHESIZER_DONE` based on fake-mode tests
only. A human runs this before merging `feat/synthesizer` to `main` and
attaches the resulting `smoke_report.json` to the PR description.

To run it:

```bash
TRACE_REAL_API_TESTS=1 ANTHROPIC_API_KEY=sk-ant-... \
  uv run pytest tests/synthesizer/test_real_smoke.py -v
```

### What it asserts

* `destructive_match == 1.0` on **all five** fixtures — non-negotiable. The
  secondary keyword matcher (S-008) is the structural enforcement, so this
  should be deterministic across runs.
* `overall >= 0.80` on **at least four** of the five fixtures — tolerates
  LLM drift; `overall` is a Haiku-judged holistic score.
* Total spend across the entire run < **$2** — caps a runaway retry loop.

### Expected costs

A clean run (no retries, no failures) typically spends well under the $2
cap. A rough breakdown:

* Five Sonnet 4.5 draft calls @ ~$0.05–$0.15 each (depends on trajectory
  size) — $0.25–$0.75 total.
* Five Haiku 4.5 similarity calls @ ~$0.005–$0.02 each — under $0.10 total.
* **Typical total**: $0.30–$0.85.

The PRD's `apiCostBudgetForSmoke` budget is $2 per run; the assertion is the
hard cap, but if you observe sustained spend above $1 per run consider
investigating prompt drift or retry storms before merging.

### Reading the smoke report

The generated `smoke_report.json` contains:

* `timestamp`, `model_draft`, `model_similarity` — provenance for the run.
* `thresholds` — the assertion thresholds in force at run time.
* `results` — aggregate counts (`total_cost_usd`, `overall_pass_count`,
  `destructive_pass_count`, `fixture_count`).
* `fixtures[]` — per-fixture detail: `slug`, `draft_llm_calls`,
  `draft_cost_usd`, `step_count`, `destructive_step_count`, `scores`
  (`overall`, `step_coverage`, `parameter_match`, `destructive_match`), and
  the LLM-produced `reasoning` blurb explaining the rubric scores.

If a fixture fails on `overall`, read its `reasoning` first — it usually
explains whether the model produced wrong steps, missed steps, or just
worded things differently.

## Useful environment variables

| Variable | Purpose |
|---|---|
| `TRACE_LLM_FAKE_MODE=1` | Force fake mode (auto-set in tests). |
| `TRACE_REAL_API_TESTS=1` | Enable the gated real-API smoke test. |
| `TRACE_PROFILE=dev` | Use `~/Library/Application Support/Trace-dev/` for cost log etc. (auto-set in tests). |
| `TRACE_DATA_DIR=<path>` | Override the data directory entirely (used by tests with `tmp_path`). |
| `TRACE_FAKE_RESPONSES_DIR=<path>` | Override the fake-mode canned-response directory (used by tests). |
| `ANTHROPIC_API_KEY=...` | Required for the smoke test; auto-set to a dummy value in the standard suite. |
