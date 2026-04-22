# services/synthesizer

Module 2. Developed on branch `feat/synthesizer`. Completion promise: `SYNTHESIZER_DONE`.

Converts a recorded trajectory into a `skill/` directory (SKILL.md +
skill.meta.json) using Claude Sonnet 4.5 with multimodal calls over keyframe
screenshots, plus a user Q&A pass to fill gaps. See the repo-root
[CLAUDE.md](../../CLAUDE.md) for project-wide context, safety invariants, and
reference workflows; this module's contracts read
[trajectory.schema.json](../../contracts/trajectory.schema.json) and write per
[skill-meta.schema.json](../../contracts/skill-meta.schema.json).

## Layout

Source lives under `src/synthesizer/` (src layout). Tests live at the repo root
under `tests/synthesizer/`.

## Quality gates

```bash
uv run ruff check services/synthesizer
uv run mypy --strict services/synthesizer
uv run pytest tests/synthesizer/
```
