"""Tests for memovex.api — multi-agent REST surface, validation, persistence.

These tests exercise the FastAPI app directly with the TestClient so they
do not require a running server, Qdrant, or Redis.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """Build a fresh app instance with persistence pointing at tmp_path.

    We reload the api module so the env-driven config picks up the test
    snapshot directory, then yield a TestClient. Triggering the lifespan
    (via TestClient as a context manager) ensures shutdown handlers run
    and snapshots are written.
    """
    monkeypatch.setenv("MEMOVEX_PERSISTENCE_ENABLED", "true")
    monkeypatch.setenv("MEMOVEX_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDINGS_ENABLED", "false")

    # Force a clean import so module-level config reads the patched env
    sys.modules.pop("memovex.api", None)
    api = importlib.import_module("memovex.api")

    with TestClient(api.app) as client:
        yield client, api, tmp_path

    # Reset for the next test
    sys.modules.pop("memovex.api", None)


# -----------------------------------------------------------------------------
# Health & basic shape
# -----------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, api_client):
        client, _api, _tmp = api_client
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["version"] == "1.1.0"
        assert set(body["allowed_agents"]) == {"claude", "hermes", "openclaw"}
        assert body["persistence_enabled"] is True

    def test_unknown_agent_returns_404(self, api_client):
        client, _api, _tmp = api_client
        r = client.get("/api/nobody/stats")
        assert r.status_code == 404

    def test_invalid_agent_id_format_returns_422(self, api_client):
        client, _api, _tmp = api_client
        # Uppercase / dot don't match the agent_id regex → 422 from FastAPI
        r = client.get("/api/Claude/stats")
        assert r.status_code == 422


# -----------------------------------------------------------------------------
# All three documented agents must work end-to-end
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("agent_id", ["claude", "hermes", "openclaw"])
class TestAllAgents:
    def test_store_and_retrieve(self, api_client, agent_id):
        client, _api, _tmp = api_client
        r = client.post(
            f"/api/{agent_id}/store",
            json={"text": f"agent {agent_id} prefers concise answers",
                  "confidence": 0.9, "salience": 0.8},
        )
        assert r.status_code == 200, r.text
        mid = r.json()["memory_id"]
        assert mid

        r = client.post(
            f"/api/{agent_id}/retrieve",
            json={"query": "concise answers", "top_k": 3},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["agent_id"] == agent_id
        assert any(mid == h["memory_id"] for h in body["results"])

    def test_stats_route(self, api_client, agent_id):
        client, _api, _tmp = api_client
        r = client.get(f"/api/{agent_id}/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["agent_id"] == agent_id
        assert "memories" in body
        assert "wisdom" in body

    def test_prefetch_and_context_alias(self, api_client, agent_id):
        client, _api, _tmp = api_client
        client.post(
            f"/api/{agent_id}/store",
            json={"text": "los gatos duermen mucho durante el día",
                  "confidence": 0.9},
        )
        r1 = client.post(f"/api/{agent_id}/prefetch",
                         json={"query": "gatos", "max_tokens": 500})
        r2 = client.post(f"/api/{agent_id}/context",
                         json={"query": "gatos", "max_tokens": 500})
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["agent_id"] == agent_id
        assert r2.json()["agent_id"] == agent_id

    def test_wisdom_route(self, api_client, agent_id):
        client, _api, _tmp = api_client
        r = client.get(f"/api/{agent_id}/wisdom")
        assert r.status_code == 200
        body = r.json()
        assert body["agent_id"] == agent_id
        assert "wisdom_memories" in body

    def test_graph_stats_route(self, api_client, agent_id):
        client, _api, _tmp = api_client
        r = client.get(f"/api/{agent_id}/graph/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["agent_id"] == agent_id
        assert "nodes" in body
        assert "edges" in body

    def test_store_chain_route(self, api_client, agent_id):
        client, _api, _tmp = api_client
        r = client.post(
            f"/api/{agent_id}/store_chain",
            json={
                "text": "Madrid es capital de España que está en Europa",
                "hops": [
                    {"source": "Madrid", "via": "capital_de", "target": "España"},
                    {"source": "España", "via": "ubicada_en", "target": "Europa"},
                ],
                "confidence": 0.9,
            },
        )
        assert r.status_code == 200
        assert r.json()["memory_id"]


# -----------------------------------------------------------------------------
# Validation errors
# -----------------------------------------------------------------------------

class TestValidation:
    def test_store_empty_text_rejected(self, api_client):
        client, _api, _tmp = api_client
        r = client.post("/api/claude/store", json={"text": ""})
        assert r.status_code == 422

    def test_store_oversize_text_rejected(self, api_client):
        client, _api, _tmp = api_client
        r = client.post("/api/claude/store", json={"text": "x" * 200_000})
        assert r.status_code == 422

    def test_store_unknown_memory_type_rejected(self, api_client):
        client, _api, _tmp = api_client
        r = client.post(
            "/api/claude/store",
            json={"text": "hi", "memory_type": "not_a_real_type"},
        )
        assert r.status_code == 422

    def test_retrieve_top_k_bounds(self, api_client):
        client, _api, _tmp = api_client
        r = client.post("/api/claude/retrieve",
                        json={"query": "x", "top_k": 999})
        assert r.status_code == 422

    def test_confidence_bounds(self, api_client):
        client, _api, _tmp = api_client
        r = client.post("/api/claude/store",
                        json={"text": "ok", "confidence": 2.0})
        assert r.status_code == 422


# -----------------------------------------------------------------------------
# Agent isolation
# -----------------------------------------------------------------------------

class TestIsolation:
    def test_memories_isolated_between_agents(self, api_client):
        client, _api, _tmp = api_client
        client.post("/api/claude/store",
                    json={"text": "secret_for_claude_only", "confidence": 0.9})
        r = client.post("/api/openclaw/retrieve",
                        json={"query": "secret_for_claude_only", "top_k": 5})
        assert r.status_code == 200
        # openclaw's bank is independent — the claude memory must not leak
        for hit in r.json()["results"]:
            assert "secret_for_claude_only" not in hit["text"]


# -----------------------------------------------------------------------------
# Persistence: snapshots survive a restart
# -----------------------------------------------------------------------------

class TestPersistence:
    def test_snapshot_survives_restart(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMOVEX_PERSISTENCE_ENABLED", "true")
        monkeypatch.setenv("MEMOVEX_SNAPSHOT_DIR", str(tmp_path))
        monkeypatch.setenv("EMBEDDINGS_ENABLED", "false")

        # First "process": store and shutdown (snapshot written)
        sys.modules.pop("memovex.api", None)
        api = importlib.import_module("memovex.api")
        with TestClient(api.app) as client:
            r = client.post(
                "/api/claude/store",
                json={"text": "remember me across restart", "confidence": 0.9},
            )
            assert r.status_code == 200
            mid = r.json()["memory_id"]
        # Snapshot file must exist
        snap = tmp_path / "claude_snapshot.json"
        assert snap.exists(), "shutdown handler must have written the snapshot"
        assert snap.stat().st_size > 0

        # Second "process": fresh app, same snapshot dir
        sys.modules.pop("memovex.api", None)
        api2 = importlib.import_module("memovex.api")
        with TestClient(api2.app) as client:
            r = client.post(
                "/api/claude/retrieve",
                json={"query": "remember restart", "top_k": 5},
            )
            assert r.status_code == 200
            ids = [h["memory_id"] for h in r.json()["results"]]
            assert mid in ids, "snapshot must restore memories on first agent use"

    def test_force_snapshot_endpoint(self, api_client):
        client, _api, tmp_path = api_client
        client.post("/api/hermes/store",
                    json={"text": "force-snapshot test memory", "confidence": 0.8})
        r = client.post("/api/hermes/snapshot")
        assert r.status_code == 200
        body = r.json()
        assert body["agent_id"] == "hermes"
        assert body["saved"] >= 1
        assert (tmp_path / "hermes_snapshot.json").exists()
