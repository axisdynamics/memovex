"""
memovex — OpenClaw Agent Plugin.

Pre-configured MemoVexOrchestrator for the OpenClaw agent.
Namespace: agent_id="openclaw"
  - Qdrant collection: memovex_openclaw
  - Redis prefix:      memovex:openclaw:*
  - Snapshot:          ~/.memovex/openclaw_snapshot.json
                       (legacy fallback: ~/.memorybank/openclaw_snapshot.json)

Usage:
    from memovex.plugins.openclaw_plugin import create_openclaw_memory

    bank = create_openclaw_memory()
    mid  = bank.store("user prefers dark mode", tags={"preferences"})
    ctx  = bank.prefetch("what are user preferences?")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_AGENT_ID = "openclaw"
_DEFAULT_SNAPSHOT = Path.home() / ".memovex" / "openclaw_snapshot.json"
_LEGACY_SNAPSHOT = Path.home() / ".memorybank" / "openclaw_snapshot.json"


def create_openclaw_memory(
    qdrant_host: Optional[str] = None,
    qdrant_port: Optional[int] = None,
    redis_host: Optional[str] = None,
    redis_port: Optional[int] = None,
    embeddings_enabled: bool = True,
    embedding_device: str = "cpu",
    snapshot_path: Optional[str] = None,
    auto_load_snapshot: bool = True,
):
    """
    Factory for an OpenClaw-scoped MemoVexOrchestrator.

    Auto-loads the local snapshot so each invocation starts with the
    full memory state persisted from previous sessions.
    """
    from ..core.memory_bank import MemoVexOrchestrator

    bank = MemoVexOrchestrator(
        agent_id=_AGENT_ID,
        embeddings_enabled=embeddings_enabled,
        embedding_device=embedding_device,
    )
    bank.initialize()

    qhost = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
    qport = qdrant_port or int(os.getenv("QDRANT_PORT", "6333"))
    rhost = redis_host or os.getenv("REDIS_HOST", "localhost")
    rport = redis_port or int(os.getenv("REDIS_PORT", "6379"))

    if bank.connect_qdrant(host=qhost, port=qport):
        logger.info("[openclaw] Qdrant connected → collection memovex_openclaw")
    else:
        logger.debug("[openclaw] Qdrant unavailable — using snapshot only")

    if bank.connect_redis(host=rhost, port=rport):
        logger.info("[openclaw] Redis connected → namespace memovex:openclaw:*")

    if auto_load_snapshot:
        snap = snapshot_path or str(_DEFAULT_SNAPSHOT)
        n = bank.load_snapshot(snap)
        if n == 0 and snapshot_path is None and _LEGACY_SNAPSHOT.exists():
            n = bank.load_snapshot(str(_LEGACY_SNAPSHOT))
            if n:
                logger.info("[openclaw] Migrated %d memories from legacy snapshot at %s",
                            n, _LEGACY_SNAPSHOT)
        if n:
            logger.debug("[openclaw] Loaded %d memories from snapshot", n)

    bank._snapshot_path = snapshot_path or str(_DEFAULT_SNAPSHOT)

    return bank


def save_openclaw_memory(bank, snapshot_path: Optional[str] = None) -> int:
    """Save the bank state to the OpenClaw snapshot file."""
    path = snapshot_path or getattr(bank, "_snapshot_path", str(_DEFAULT_SNAPSHOT))
    return bank.save_snapshot(path)
