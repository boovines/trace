"""Recorder HTTP routes. Mounted at /recorder by the gateway.

Endpoints are stubs pending implementation on feat/recorder.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/status")
def status() -> dict[str, str]:
    return {"module": "recorder", "status": "scaffold"}
