"""Synthesizer HTTP routes. Mounted at root by the gateway; paths include
/synthesize/*, /skills/*, and /trajectories/* per CLAUDE.md.

Endpoints are stubs pending implementation on feat/synthesizer.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/synthesize/status")
def status() -> dict[str, str]:
    return {"module": "synthesizer", "status": "scaffold"}
