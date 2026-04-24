"""Run the recorder service standalone for development.

Example:
    uv run python -m recorder

In production the gateway mounts the recorder router directly; this entry
point is useful for smoke-testing the module in isolation.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from recorder.api import router


def build_app() -> FastAPI:
    app = FastAPI(title="Trace recorder (standalone)", version=__import__("recorder").__version__)
    app.include_router(router, prefix="/recorder", tags=["recorder"])
    return app


def main() -> None:
    uvicorn.run(build_app(), host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
