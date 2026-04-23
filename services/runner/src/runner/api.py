"""Runner HTTP routes.

Mounted by the gateway; exposes /run/* and /runs/*. Most endpoints are stubs
pending implementation in later stories; the abort endpoint lands here with
X-018 so the global hotkey can reach the kill switch.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI

from runner.kill_switch import KillSwitch, get_global_kill_switch

router = APIRouter()


@router.get("/run/status")
def status() -> dict[str, str]:
    return {"module": "runner", "status": "scaffold"}


@router.post("/run/{run_id}/abort")
def abort_run(run_id: str) -> dict[str, object]:
    """Trigger the X-018 kill switch for ``run_id``.

    Idempotent: a second call (or a call for a run that never started / has
    already finished) returns ``killed=False`` and still 200s. The API layer
    deliberately does not distinguish "unknown run_id" from "already finished"
    because the caller (Tauri hotkey handler) has no way to tell them apart
    and either way there is nothing further to do.
    """

    switch: KillSwitch = get_global_kill_switch()
    killed = switch.kill(run_id, reason="user_abort")
    return {"run_id": run_id, "killed": killed}


app = FastAPI(title="Trace runner", version="0.1.0")
app.include_router(router)
