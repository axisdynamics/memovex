"""
OpenClaw plugin for MemoVex.

Provides a clean memory API for the OpenClaw agent with:
- Persistent snapshot storage (~/.memorybank/openclaw_snapshot.json)
- Multi-channel retrieval (10 weighted channels)
- Optional Qdrant vector search and Redis caching
- WisdomStore curation pipeline

Quick start:
    from plugins.openclaw.plugin import OpenClawMemory

    mem = OpenClawMemory()
    mem.remember("user prefers concise answers", tags=["preferences"])
    context = mem.recall("how should I format my replies?")
    mem.save()
    mem.shutdown()
"""

from __future__ import annotations

import sys
import os
from typing import Optional, Set, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class OpenClawMemory:
    """
    High-level memory interface for the OpenClaw agent.

    Wraps MemoVexOrchestrator with a simpler, opinionated API:
    - remember() instead of store()
    - recall()   instead of prefetch()
    - save()     to persist the snapshot
    - teach()    to store procedural / how-to knowledge
    - reflect()  to store reasoning chains

    The underlying orchestrator is always accessible via `.bank`.
    """

    def __init__(
        self,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        embeddings_enabled: bool = True,
        embedding_device: str = "cpu",
        snapshot_path: Optional[str] = None,
    ):
        from memovex.plugins.openclaw_plugin import create_openclaw_memory
        self.bank = create_openclaw_memory(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            redis_host=redis_host,
            redis_port=redis_port,
            embeddings_enabled=embeddings_enabled,
            embedding_device=embedding_device,
            snapshot_path=snapshot_path,
            auto_load_snapshot=True,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def remember(
        self,
        text: str,
        memory_type: str = "episodic",
        tags: Optional[Set[str]] = None,
        confidence: float = 0.7,
        salience: float = 0.5,
        session_id: Optional[str] = None,
    ) -> str:
        """Store a memory. Returns the memory_id."""
        from memovex.core.types import MemoryType
        try:
            mt = MemoryType(memory_type)
        except ValueError:
            mt = MemoryType.EPISODIC
        return self.bank.store(
            text=text,
            memory_type=mt,
            tags=set(tags or []),
            confidence=confidence,
            salience=salience,
            session_id=session_id,
        )

    def recall(self, query: str, max_tokens: int = 800) -> str:
        """Return LLM-ready context string for a query."""
        return self.bank.prefetch(query, max_tokens=max_tokens)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Return ranked memory dicts for a query."""
        results = self.bank.retrieve(query, top_k=top_k)
        return [
            {
                "text": r.memory.text,
                "score": round(r.total_score, 4),
                "type": r.memory.memory_type.value,
                "tags": list(r.memory.tags),
                "memory_id": r.memory.memory_id,
            }
            for r in results
        ]

    def teach(self, text: str, tags: Optional[Set[str]] = None,
              confidence: float = 0.8) -> str:
        """Store procedural / how-to knowledge."""
        return self.remember(
            text, memory_type="procedural",
            tags=(tags or set()) | {"procedural"},
            confidence=confidence, salience=0.7,
        )

    def reflect(self, text: str, hops: List[dict],
                confidence: float = 0.75) -> str:
        """Store a reasoning chain with graph edges."""
        return self.bank.store_reasoning_chain(
            text=text, hops=hops, confidence=confidence,
        )

    def reinforce(self, memory_id: str) -> None:
        """Add evidence to a memory (may trigger WisdomStore promotion)."""
        self.bank.corroborate(memory_id)

    def promote(self, memory_id: str, notes: str = "") -> None:
        """Manually promote a memory to WISDOM level."""
        self.bank.promote_to_wisdom(memory_id, notes=notes)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> int:
        """Persist all memories to the snapshot file. Returns count saved."""
        from memovex.plugins.openclaw_plugin import save_openclaw_memory
        return save_openclaw_memory(self.bank)

    def shutdown(self) -> None:
        """Save snapshot and release resources."""
        self.save()
        self.bank.shutdown()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return self.bank.stats()

    def wisdom_summary(self) -> dict:
        return self.bank.wisdom_summary()


# ---------------------------------------------------------------------------
# Low-level factory (for callers that want the raw orchestrator)
# ---------------------------------------------------------------------------

def create_openclaw_memory(**kwargs):
    """Return a raw MemoVexOrchestrator scoped to OpenClaw."""
    from memovex.plugins.openclaw_plugin import create_openclaw_memory as _factory
    return _factory(**kwargs)


def save_openclaw_memory(bank, snapshot_path: Optional[str] = None) -> int:
    from memovex.plugins.openclaw_plugin import save_openclaw_memory as _save
    return _save(bank, snapshot_path=snapshot_path)


__all__ = ["OpenClawMemory", "create_openclaw_memory", "save_openclaw_memory"]
