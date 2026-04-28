"""
memovex — Mem0 Adapter.

Integrates Mem0 (open source memory system) as an external provider
within memovex, with symbolic resonance injection.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class Mem0Adapter:
    """
    Adapter for Mem0 (https://github.com/mem0ai/mem0).

    Bridges Mem0's SQL + Vector memory into memovex.
    Injects base64 symbols into Mem0's metadata for cross-provider resonance.

    Requires: pip install mem0ai
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {
            "db_path": "memovex_mem0.db",
            "embedding_model": "all-MiniLM-L6-v2",
        }
        self._client = None

    def initialize(self) -> None:
        """Initialize Mem0 client."""
        try:
            from mem0 import MemoryClient
            self._client = MemoryClient(
                db_path=self._config.get("db_path"),
                embedding_model=self._config.get("embedding_model"),
            )
            logger.info("Mem0 adapter initialized")
        except ImportError:
            logger.warning("mem0ai not installed. Mem0 adapter disabled.")
        except Exception as e:
            logger.error(f"Mem0 init failed: {e}")

    @property
    def available(self) -> bool:
        return self._client is not None

    def store_memory(self, memory) -> str:
        """Store memory via Mem0 with symbol injection."""
        if not self.available:
            return ""

        try:
            # Inyectar símbolos base64 como metadata para resonancia cruzada
            metadata = {
                "memory_id": memory.memory_id,
                "memory_type": memory.memory_type.value,
                "provider": memory.provider,
                "base64_symbols": json.dumps(list(memory.base64_symbols)),
                "entities": json.dumps(list(memory.entities)),
                "tags": json.dumps(list(memory.tags)),
                "confidence": memory.confidence,
            }

            result = self._client.add(
                text=memory.text,
                user_id=memory.session_id or "default",
                metadata=metadata,
            )
            return result.get("id", "")

        except Exception as e:
            logger.error(f"Mem0 store failed: {e}")
            return ""

    def retrieve(self, query: str, top_k: int = 5) -> List:
        """Retrieve from Mem0 with symbolic resonance boost."""
        if not self.available:
            return []

        from ..core.types import Memory, MemoryType, RetrievalResult
        from ..core.resonance_engine import text_to_symbols, compute_symbolic_resonance

        results = []

        try:
            mem0_results = self._client.search(
                query=query,
                limit=top_k * 2,  # Fetch more for resonance re-ranking
            )

            query_symbols = text_to_symbols(query)

            for item in mem0_results:
                text = item.get("text", "")
                score = item.get("score", 0.0)
                metadata = item.get("metadata", {})

                # Parse injected symbols
                mem_symbols = set()
                try:
                    mem_symbols = set(json.loads(metadata.get("base64_symbols", "[]")))
                except (json.JSONDecodeError, TypeError):
                    pass

                # Compute symbolic resonance boost
                resonance = compute_symbolic_resonance(query_symbols, mem_symbols)

                # Boost final score with resonance
                final_score = max(score, resonance * 0.35 + score * 0.65)

                mem = Memory(
                    memory_id=metadata.get("memory_id", f"mem0-{hash(text)}"),
                    text=text,
                    memory_type=MemoryType.EPISODIC,
                    base64_symbols=mem_symbols,
                    provider="mem0",
                )

                results.append(RetrievalResult(
                    memory=mem,
                    total_score=final_score,
                    channel_scores={"semantic": score, "symbolic": resonance},
                    provider="mem0",
                ))

        except Exception as e:
            logger.error(f"Mem0 retrieve failed: {e}")

        results.sort(key=lambda r: r.total_score, reverse=True)
        return results[:top_k]

    def shutdown(self) -> None:
        """Cleanup Mem0 connection."""
        self._client = None
        logger.info("Mem0 adapter shut down")
