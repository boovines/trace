"""Gateway FastAPI app that mounts recorder, synthesizer, and runner routers.

Runs on 127.0.0.1:8765. All three services share one process so there is a single
port for the Tauri app to talk to.
"""

from __future__ import annotations

from fastapi import FastAPI

from runner.api import router as runner_router
from services.recorder.recorder.api import router as recorder_router
from services.synthesizer.synthesizer.api import router as synthesizer_router

app = FastAPI(title="Trace gateway", version="0.1.0")

app.include_router(recorder_router, prefix="/recorder", tags=["recorder"])
app.include_router(synthesizer_router, tags=["synthesizer"])
app.include_router(runner_router, tags=["runner"])


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
