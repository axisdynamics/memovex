"""
memovex — Hermes Agent Plugin.

Pre-configured MemoVexOrchestrator for the Hermes VEX agent.
Namespace: agent_id="hermes"
  - Qdrant collection: memovex_hermes
  - Redis prefix:      memovex:hermes:*

Includes a VEX compatibility layer so Hermes can call its existing
plugin interface (prefetch / sync_turn / store_memory) without changes.

Usage (drop-in replacement for Resonant Memory VEX v2.3):
    from memovex.plugins.hermes_plugin import create_hermes_memory

    plugin = create_hermes_memory()
    plugin.initialize()

    # Existing VEX API still works:
    context = plugin.prefetch("¿qué sé sobre el usuario?")
    plugin.sync_turn(user_msg, assistant_msg, session_id="s1")
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_AGENT_ID = "hermes"


class HermesMemoryPlugin:
    """
    memovex orchestrator wrapped in the Resonant VEX v2.3 plugin API.

    Hermes interacts with this via the same methods it used before,
    so no changes are needed to the Hermes agent code.
    """

    def __init__(self, bank):
        self._bank = bank

    # ------------------------------------------------------------------
    # VEX-compatible API (matches Resonant Memory VEX v2.3 interface)
    # ------------------------------------------------------------------

    def prefetch(self, query: str, max_tokens: int = 1000) -> str:
        """Return LLM-ready context for a query. Drop-in for VEX prefetch."""
        return self._bank.prefetch(query, max_tokens=max_tokens)

    def sync_turn(self, user_message: str, assistant_message: str,
                  session_id: Optional[str] = None,
                  confidence: float = 0.7) -> None:
        """
        Store a conversation turn as two episodic memories.
        Mirrors the VEX sync_turn interface.
        """
        from ..core.types import MemoryType
        self._bank.store(
            text=f"User: {user_message}",
            memory_type=MemoryType.EPISODIC,
            session_id=session_id,
            confidence=confidence,
            salience=0.5,
        )
        self._bank.store(
            text=f"Hermes: {assistant_message}",
            memory_type=MemoryType.EPISODIC,
            session_id=session_id,
            confidence=confidence,
            salience=0.6,
        )

    def store_memory(self, text: str, memory_type: str = "episodic",
                     entities=None, tags=None,
                     confidence: float = 0.7,
                     salience: float = 0.5) -> str:
        """Generic store — mirrors VEX store_memory."""
        from ..core.types import MemoryType
        try:
            mt = MemoryType(memory_type)
        except ValueError:
            mt = MemoryType.EPISODIC
        return self._bank.store(
            text=text,
            memory_type=mt,
            entities=set(entities or []),
            tags=set(tags or []),
            confidence=confidence,
            salience=salience,
        )

    def corroborate(self, memory_id: str) -> None:
        self._bank.corroborate(memory_id)

    def promote_wisdom(self, memory_id: str, notes: str = "") -> None:
        self._bank.promote_to_wisdom(memory_id, notes=notes)

    def shutdown(self) -> None:
        self._bank.shutdown()

    # ------------------------------------------------------------------
    # Direct access to the underlying orchestrator
    # ------------------------------------------------------------------

    @property
    def bank(self):
        return self._bank

    def stats(self) -> dict:
        return self._bank.stats()


def create_hermes_memory(
    qdrant_host: Optional[str] = None,
    qdrant_port: Optional[int] = None,
    redis_host: Optional[str] = None,
    redis_port: Optional[int] = None,
    embeddings_enabled: bool = True,
    embedding_device: str = "cpu",
) -> HermesMemoryPlugin:
    """
    Factory for a Hermes-scoped MemoVexOrchestrator wrapped in
    the VEX-compatible plugin interface.
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
        logger.info("[hermes] Qdrant connected → collection memovex_hermes")
    else:
        logger.warning("[hermes] Qdrant unavailable — semantic search uses BoW")

    if bank.connect_redis(host=rhost, port=rport):
        logger.info("[hermes] Redis connected → namespace memovex:hermes:*")
    else:
        logger.warning("[hermes] Redis unavailable — caching disabled")

    return HermesMemoryPlugin(bank)
