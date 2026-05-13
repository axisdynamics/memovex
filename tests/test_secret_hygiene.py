"""Regression tests for MemoVex secret hygiene and operator redaction.

These tests encode the local production failure: token-like text could be stored,
retrieved, and snapshotted with no server-side redaction or deletion path.
"""

from __future__ import annotations

import importlib
import json
import sys

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

TOKEN = "ghp_SANDBOXOnlyTokenValue1234567890abcdef"


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMOVEX_PERSISTENCE_ENABLED", "true")
    monkeypatch.setenv("MEMOVEX_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDINGS_ENABLED", "false")
    monkeypatch.setenv("QDRANT_HOST", "127.0.0.1")
    monkeypatch.setenv("REDIS_HOST", "127.0.0.1")

    sys.modules.pop("memovex.api", None)
    api = importlib.import_module("memovex.api")
    with TestClient(api.app) as client:
        yield client, tmp_path
    sys.modules.pop("memovex.api", None)


def test_store_rejects_secret_only_payload(api_client):
    client, _tmp_path = api_client

    r = client.post("/api/hermes/store", json={"text": TOKEN})

    assert r.status_code == 422
    assert "secret" in r.text.lower() or "credential" in r.text.lower()


def test_store_redacts_token_before_retrieve_and_snapshot(api_client):
    client, tmp_path = api_client

    r = client.post(
        "/api/hermes/store",
        json={"text": f"Operator pasted token {TOKEN} during a broken push"},
    )
    assert r.status_code == 200, r.text
    mid = r.json()["memory_id"]

    r = client.post("/api/hermes/retrieve", json={"query": "broken push token", "top_k": 5})
    assert r.status_code == 200
    hit_text = "\n".join(hit["text"] for hit in r.json()["results"] if hit["memory_id"] == mid)
    assert TOKEN not in hit_text
    assert "<redacted" in hit_text.lower()

    r = client.post("/api/hermes/snapshot")
    assert r.status_code == 200
    snapshot_text = (tmp_path / "hermes_snapshot.json").read_text()
    assert TOKEN not in snapshot_text
    assert "<redacted" in snapshot_text.lower()


def test_delete_memory_endpoint_removes_memory_from_retrieve_and_snapshot(api_client):
    client, tmp_path = api_client
    marker = "delete-me-unique-marker"
    r = client.post("/api/hermes/store", json={"text": f"{marker} operational note"})
    assert r.status_code == 200, r.text
    mid = r.json()["memory_id"]

    r = client.delete(f"/api/hermes/memory/{mid}")
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] is True

    r = client.post("/api/hermes/retrieve", json={"query": marker, "top_k": 5})
    assert r.status_code == 200
    assert all(hit["memory_id"] != mid for hit in r.json()["results"])

    r = client.post("/api/hermes/snapshot")
    assert r.status_code == 200
    snapshot = json.loads((tmp_path / "hermes_snapshot.json").read_text())
    assert all(mem["memory_id"] != mid for mem in snapshot["memories"])


def test_redact_endpoint_deletes_matching_memories(api_client):
    client, _tmp_path = api_client
    marker = "compromised-secret-marker-xyz"
    keep = client.post("/api/hermes/store", json={"text": "normal durable preference"})
    drop = client.post("/api/hermes/store", json={"text": f"must purge {marker} now"})
    assert keep.status_code == 200
    assert drop.status_code == 200
    keep_id = keep.json()["memory_id"]
    drop_id = drop.json()["memory_id"]

    r = client.post("/api/hermes/redact", json={"contains": marker})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matched"] == 1
    assert body["deleted"] == 1
    assert drop_id in body["memory_ids"]

    r = client.post("/api/hermes/retrieve", json={"query": marker, "top_k": 10})
    assert r.status_code == 200
    ids = {hit["memory_id"] for hit in r.json()["results"]}
    assert drop_id not in ids

    r = client.post("/api/hermes/retrieve", json={"query": "normal durable preference", "top_k": 10})
    assert r.status_code == 200
    ids = {hit["memory_id"] for hit in r.json()["results"]}
    assert keep_id in ids


def test_redact_endpoint_deletes_legacy_secret_pattern_without_echoing_secret(api_client):
    client, _tmp_path = api_client
    # Simulate an old raw secret already loaded before server-side redaction existed.
    client.get("/api/hermes/stats")
    api = sys.modules["memovex.api"]
    from memovex.core.resonance_engine import text_to_symbols, tokenize
    from memovex.core.types import Memory, MemoryType

    legacy = Memory(
        memory_id="legacy-secret-memory",
        text=f"legacy accidental token {TOKEN}",
        memory_type=MemoryType.EPISODIC,
        base64_symbols=text_to_symbols(f"legacy accidental token {TOKEN}"),
        symbolic_keys=set(tokenize(f"legacy accidental token {TOKEN}")),
        confidence=0.9,
        salience=0.9,
    )
    api._agents["hermes"]._memory_store.add(legacy)

    r = client.post("/api/hermes/redact", json={"secret_patterns": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matched"] == 1
    assert body["deleted"] == 1
    assert "legacy-secret-memory" in body["memory_ids"]

    r = client.post("/api/hermes/retrieve", json={"query": "legacy accidental token", "top_k": 10})
    assert r.status_code == 200
    assert all(hit["memory_id"] != "legacy-secret-memory" for hit in r.json()["results"])
