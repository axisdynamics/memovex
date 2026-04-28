"""
memovex — Memobase / Letta Adapter.

Integrates Letta (formerly MemGPT / Memobase) as an external provider.
Bridges Letta's hierarchical memory (core ↔ archival ↔ recall) into
memovex, with symbol injection for cross-provider indexing.

Requires: pip install letta
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemobaseAdapter:
    """
    Adapter for Letta / Memobase.

    Pulls memories from Letta's archival store and registers them as
    SEMANTIC memories in memovex.  Symbol injection lets the
    resonance engine index Letta memories via base64 fingerprints.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {
            "base_url": "http://localhost:8080",
        }
        self._client = None

    def initialize(self) -> None:
        try:
            from letta import create_client
            self._client = create_client(base_url=self._config.get("base_url"))
            logger.info("Memobase/Letta adapter initialized at %s",
                        self._config.get("base_url"))
        except ImportError:
            logger.warning("letta not installed — Memobase adapter disabled")
        except Exception as e:
            logger.error("Memobase init failed: %s", e)

    @property
    def available(self) -> bool:
        return self._client is not None

    def store_memory(self, memory) -> str:
        if not self.available:
            return ""
        try:
            from ..core.resonance_engine import text_to_symbols
            symbols = list(memory.base64_symbols or text_to_symbols(memory.text))
            result = self._client.insert_archival_memory(
                agent_id=self._config.get("agent_id", "default"),
                memory=memory.text,
            )
            logger.debug("Memobase stored: %s (symbols injected: %d)",
                         memory.memory_id, len(symbols))
            return memory.memory_id
        except Exception as e:
            logger.debug("Memobase store failed: %s", e)
            return ""

    def retrieve(self, query: str, top_k: int = 5) -> List:
        if not self.available:
            return []

        from ..core.types import Memory, MemoryType, RetrievalResult
        from ..core.resonance_engine import text_to_symbols, compute_symbolic_resonance

        results = []
        try:
            archival = self._client.get_archival_memory(
                agent_id=self._config.get("agent_id", "default"),
                query=query,
                limit=top_k * 2,
            )
            query_symbols = text_to_symbols(query)

            for item in (archival or []):
                text = item.text if hasattr(item, "text") else str(item)
                mem_symbols = text_to_symbols(text)
                resonance = compute_symbolic_resonance(query_symbols, mem_symbols)

                mem = Memory(
                    memory_id=f"memobase-{hash(text) & 0xFFFFFFFF}",
                    text=text,
                    memory_type=MemoryType.SEMANTIC,
                    base64_symbols=mem_symbols,
                    provider="memobase",
                )
                results.append(RetrievalResult(
                    memory=mem,
                    total_score=resonance,
                    channel_scores={"symbolic": resonance},
                    provider="memobase",
                ))

        except Exception as e:
            logger.debug("Memobase retrieve failed: %s", e)

        results.sort(key=lambda r: r.total_score, reverse=True)
        return results[:top_k]

    def sync_to_memorybank(self, orchestrator, limit: int = 100) -> int:
        """
        Pull archival memories from Letta and store them in memovex.
        Returns number of memories synced.
        """
        if not self.available:
            return 0
        synced = 0
        try:
            from ..core.types import MemoryType
            archival = self._client.get_archival_memory(
                agent_id=self._config.get("agent_id", "default"),
                limit=limit,
            )
            for item in (archival or []):
                text = item.text if hasattr(item, "text") else str(item)
                orchestrator.store(
                    text=text,
                    memory_type=MemoryType.SEMANTIC,
                    provider="memobase",
                )
                synced += 1
        except Exception as e:
            logger.warning("Memobase sync failed: %s", e)
        logger.info("Synced %d memories from Memobase", synced)
        return synced

    def shutdown(self) -> None:
        self._client = None
        logger.info("Memobase adapter shut down")
