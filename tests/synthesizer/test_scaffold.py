"""Smoke tests for the synthesizer scaffold (S-001).

These exist so the module-level ``pytest`` run exits 0 rather than 5
("no tests collected"), and to prove the ``anthropic_mock`` fixture actually
intercepts calls to ``api.anthropic.com``.
"""

from __future__ import annotations

import httpx
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from synthesizer import __version__
from synthesizer.api import router


def test_package_import_has_version() -> None:
    assert __version__ == "0.1.0"


def test_api_router_mounts_and_serves_status() -> None:
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        resp = client.get("/synthesize/status")
    assert resp.status_code == 200
    assert resp.json() == {"module": "synthesizer", "status": "ok"}


def test_anthropic_mock_intercepts_requests(anthropic_mock: respx.MockRouter) -> None:
    route = anthropic_mock.post("/v1/messages").mock(
        return_value=httpx.Response(200, json={"id": "msg_fake", "type": "message"})
    )
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        json={"model": "claude-sonnet-4-5", "messages": []},
    )
    assert route.called
    assert resp.status_code == 200
    assert resp.json()["id"] == "msg_fake"
