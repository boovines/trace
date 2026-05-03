"""Gateway FastAPI app that mounts recorder, synthesizer, and runner routers.

Runs on 127.0.0.1:8765. All three services share one process so there is a single
port for the Tauri app to talk to.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from recorder.api import router as recorder_router
from recorder.api import stats_router, trajectories_router
from services.runner.runner.api import router as runner_router
from services.synthesizer.synthesizer.api import router as synthesizer_router

logger = logging.getLogger(__name__)

app = FastAPI(title="Trace gateway", version="0.1.0")

app.include_router(recorder_router, prefix="/recorder", tags=["recorder"])
app.include_router(trajectories_router, tags=["recorder"])
app.include_router(stats_router, tags=["recorder"])
app.include_router(synthesizer_router, tags=["synthesizer"])
app.include_router(runner_router, tags=["runner"])


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


# --------------------------------------------------------------- dashboard
#
# The dashboard is a Vite + React + TypeScript app under /dashboard/. We serve
# the built ``dashboard/dist`` folder as static files at ``GET /dashboard/``
# when present. In dev the user runs ``pnpm dev`` separately and Vite proxies
# ``/stats/*`` back to this gateway (see dashboard/vite.config.ts), so the
# React code uses the same relative paths in both modes.

_DASHBOARD_DIST = Path(__file__).resolve().parent.parent / "dashboard" / "dist"

if _DASHBOARD_DIST.is_dir():
    app.mount(
        "/dashboard",
        StaticFiles(directory=str(_DASHBOARD_DIST), html=True),
        name="dashboard",
    )
else:
    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/dashboard/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard_not_built() -> HTMLResponse:
        """Fallback when ``dashboard/dist`` doesn't exist yet.

        Tells the user how to either build the static bundle or run the Vite
        dev server. Avoids a confusing 404 in the browser when someone hits
        ``/dashboard`` after a fresh checkout.
        """
        return HTMLResponse(
            "<!doctype html><meta charset=utf-8>"
            "<title>Trace dashboard</title>"
            "<style>body{font-family:-apple-system,system-ui,sans-serif;"
            "background:#0e1116;color:#e6edf3;padding:48px;line-height:1.6;}"
            "code{background:#1f2630;padding:2px 6px;border-radius:4px;}"
            "</style>"
            "<h1>Trace dashboard not built</h1>"
            "<p>Run one of:</p>"
            "<pre><code>cd dashboard && pnpm install && pnpm dev"
            "  # dev server on :5173 (proxies to this gateway)</code></pre>"
            "<pre><code>cd dashboard && pnpm install && pnpm build"
            "  # builds dashboard/dist for this gateway to serve</code></pre>",
            status_code=503,
        )
