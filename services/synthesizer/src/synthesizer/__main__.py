"""Run the synthesizer FastAPI service standalone (for dev only)."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from synthesizer.api import router


def build_app() -> FastAPI:
    app = FastAPI(title="Trace Synthesizer")
    app.include_router(router)
    return app


def main() -> None:
    uvicorn.run(build_app(), host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
