"""Smoke test: gateway imports cleanly and /healthz returns ok."""

from __future__ import annotations

from fastapi.testclient import TestClient

from gateway.main import app


def test_healthz() -> None:
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_module_scaffold_endpoints() -> None:
    client = TestClient(app)
    assert client.get("/recorder/status").json()["module"] == "recorder"
    assert client.get("/synthesize/status").json()["module"] == "synthesizer"
    assert client.get("/run/status").json()["module"] == "runner"
