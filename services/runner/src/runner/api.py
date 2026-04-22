"""Runner HTTP routes.

Mounted by the gateway; exposes /run/* and /runs/*. All endpoints are stubs
pending implementation in later stories on feat/runner.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI

router = APIRouter()


@router.get("/run/status")
def status() -> dict[str, str]:
    return {"module": "runner", "status": "scaffold"}


app = FastAPI(title="Trace runner", version="0.1.0")
app.include_router(router)
