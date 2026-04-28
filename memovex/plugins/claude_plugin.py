"""
memovex — Claude Agent Plugin.

Pre-configured MemoVexOrchestrator for the Claude agent.
Namespace: agent_id="claude"
  - Qdrant collection: memovex_claude
  - Redis prefix:      memovex:claude:*
  - Snapshot:          ~/.claude/memovex/claude_snapshot.json
                       (legacy fallback: ~/.claude/memorybank/claude_snapshot.json)

The snapshot provides cross-process persistence for Claude Code hooks:
the Stop hook saves state, the UserPromptSubmit hook loads it — so
memories survive across hook invocations without Qdrant being mandatory.

Usage:
    from memovex.plugins.claude_plugin import create_claude_memory

    bank = create_claude_memory()
    results = bank.retrieve("user preferences")
    context = bank.prefetch("what does the user prefer?")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_AGENT_ID = "claude"
_DEFAULT_SNAPSHOT = Path.home() / ".claude" / "memovex" / "claude_snapshot.json"
_LEGACY_SNAPSHOT = Path.home() / ".claude" / "memorybank" / "claude_snapshot.json"


def create_claude_memory(
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
    Factory for a Claude-scoped MemoVexOrchestrator.

    Automatically loads the local snapshot so hook invocations start
    with the full memory state from previous turns.
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
        logger.info("[claude] Qdrant connected → collection memovex_claude")
    else:
        logger.debug("[claude] Qdrant unavailable — using snapshot only")

    if bank.connect_redis(host=rhost, port=rport):
        logger.info("[claude] Redis connected → namespace memovex:claude:*")

    # Load snapshot for cross-process persistence
    if auto_load_snapshot:
        snap = snapshot_path or str(_DEFAULT_SNAPSHOT)
        n = bank.load_snapshot(snap)
        # Backward-compat: if the new path is empty but the legacy one exists,
        # load it once. Subsequent saves go to the new path.
        if n == 0 and snapshot_path is None and _LEGACY_SNAPSHOT.exists():
            n = bank.load_snapshot(str(_LEGACY_SNAPSHOT))
            if n:
                logger.info("[claude] Migrated %d memories from legacy snapshot at %s",
                            n, _LEGACY_SNAPSHOT)
        if n:
            logger.debug("[claude] Loaded %d memories from snapshot", n)

    # Attach snapshot path so callers can save later
    bank._snapshot_path = snapshot_path or str(_DEFAULT_SNAPSHOT)

    return bank


def save_claude_memory(bank, snapshot_path: Optional[str] = None) -> int:
    """Save the bank state to the Claude snapshot file."""
    path = snapshot_path or getattr(bank, "_snapshot_path", str(_DEFAULT_SNAPSHOT))
    return bank.save_snapshot(path)
