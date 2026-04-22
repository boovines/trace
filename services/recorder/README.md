# services/recorder

Module 1. Developed on branch `feat/recorder`. Completion promise: `RECORDER_DONE`.

Captures macOS user actions (mouse, keyboard, scroll, app focus changes, keyframe
screenshots) into a `trajectory/` directory per the
[trajectory.schema.json](../../contracts/trajectory.schema.json) contract.

See the root [CLAUDE.md](../../CLAUDE.md) for the overall architecture and safety
invariants.

## Layout

```
services/recorder/
├── pyproject.toml          # declares trace-recorder workspace member + runtime deps
└── src/recorder/           # source goes here (src layout)
    ├── __init__.py
    ├── __main__.py         # `uv run python -m recorder` → standalone uvicorn on 127.0.0.1:8765
    └── api.py              # FastAPI router mounted at /recorder by the gateway
```

Add new modules (schema.py, writer.py, event_tap.py, …) under `src/recorder/`.

## Quality gates

```bash
uv sync --extra dev
uv run ruff check services/recorder
uv run mypy --strict services/recorder
uv run pytest tests/recorder/
```

Mypy overrides for PyObjC live in the **root** pyproject.toml (`[[tool.mypy.overrides]]`).
When importing a new PyObjC framework here, add its top-level module name to that override
list so `--strict` mode doesn't flag the untyped call graph.
