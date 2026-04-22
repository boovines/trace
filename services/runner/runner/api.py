"""Runner HTTP routes. Mounted at root by the gateway; paths include /run/* and /runs/*.

Endpoints are stubs pending implementation on feat/runner.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/run/status")
def status() -> dict[str, str]:
    return {"module": "runner", "status": "scaffold"}
