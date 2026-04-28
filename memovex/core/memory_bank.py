"""
memovex — Memory Orchestrator.

Central orchestrator that coordinates multiple memory providers,
manages the resonance engine, handles homeostasis, and provides
a unified API for memory operations.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .resonance_engine import (
    DEFAULT_CHANNEL_WEIGHTS,
    MemoryStore,
    ResonanceEngine,
    text_to_symbols,
    compute_symbolic_resonance,
    extract_entities,
    tokenize,
)
from .types import Memory, MemoryType, RetrievalResult
from .homeostasis import HomeostasisManager
from .wisdom_store import WisdomStore, WisdomLevel

logger = logging.getLogger(__name__)


class MemoVexOrchestrator:
    """
    Central orchestrator for memovex.

    Each instance is scoped to one agent via `agent_id`.  All Qdrant
    collections and Redis keys are namespaced, so multiple orchestrators
    sharing the same infrastructure remain fully isolated.
    """

    def __init__(self, agent_id: str = "default",
                 weights: Optional[Dict[str, float]] = None,
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 embedding_device: str = "cpu",
                 embeddings_enabled: bool = True):
        self.agent_id = agent_id
        self._initialized = False
        self._lock = threading.Lock()

        # Core stores
        self._memory_store = MemoryStore()
        self._resonance_engine = ResonanceEngine(
            self._memory_store,
            weights=weights or dict(DEFAULT_CHANNEL_WEIGHTS),
        )

        # External providers
        self._providers: Dict[str, Any] = {}

        # Integrations (optional, set via connect_* methods)
        self._qdrant = None
        self._chroma = None
        self._redis = None
        self._graph_store = None

        # Homeostasis
        self._homeostasis = HomeostasisManager(self._memory_store)

        # Wisdom curation pipeline
        self._wisdom_store = WisdomStore()
        self._resonance_engine._wisdom_store = self._wisdom_store

        # Embedding model (lazy-loaded)
        self._embeddings_enabled = embeddings_enabled
        self._embedding_model_name = embedding_model_name
        self._embedding_device = embedding_device

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def initialize(self) -> None:
        if self._initialized:
            return

        logger.info("memovex initializing (agent=%s)...", self.agent_id)

        # Wire embedding model
        if self._embeddings_enabled:
            from .tokenizer import get_default_model
            em = get_default_model(
                model_name=self._embedding_model_name,
                device=self._embedding_device,
            )
            self._resonance_engine._embedding_model = em

        # Wire graph store (always available; NetworkX is optional)
        from ..integrations.graph_store import GraphStore
        self._graph_store = GraphStore()
        self._resonance_engine._graph_store = self._graph_store

        # Start homeostasis
        self._homeostasis.start()
        self._initialized = True
        logger.info("memovex initialized (agent=%s)", self.agent_id)

    def connect_qdrant(self, host: str = "localhost", port: int = 6333,
                       collection: Optional[str] = None) -> bool:
        from ..integrations.qdrant_store import QdrantStore
        col = collection or f"memovex_{self.agent_id}"
        self._qdrant = QdrantStore(host=host, port=port, collection=col)
        ok = self._qdrant.connect()
        logger.info("agent=%s qdrant_connect host=%s port=%d collection=%s ok=%s",
                    self.agent_id, host, port, col, ok)
        return ok

    def connect_chroma(self, persist_directory: Optional[str] = "./data/chroma",
                       collection: Optional[str] = None) -> bool:
        """Wire a ChromaDB store as an alternative/companion to Qdrant.

        When both Qdrant and Chroma are connected, Qdrant takes precedence
        for semantic search; Chroma still receives upserts so the on-disk
        index stays in sync. Use this when you want a local, fileless
        persistence layer instead of a Qdrant server.
        """
        from ..integrations.chroma_store import ChromaStore
        col = collection or f"memovex_{self.agent_id}"
        self._chroma = ChromaStore(collection=col, persist_directory=persist_directory)
        ok = self._chroma.connect()
        logger.info("agent=%s chroma_connect persist=%s collection=%s ok=%s",
                    self.agent_id, persist_directory, col, ok)
        return ok

    def connect_redis(self, host: str = "localhost", port: int = 6379) -> bool:
        from ..integrations.redis_cache import RedisCache
        self._redis = RedisCache(host=host, port=port, namespace=self.agent_id)
        ok = self._redis.connect()
        logger.info("agent=%s redis_connect host=%s port=%d ok=%s",
                    self.agent_id, host, port, ok)
        return ok

    def shutdown(self) -> None:
        self._initialized = False
        self._homeostasis.stop()
        if self._qdrant:
            self._qdrant.disconnect()
        if self._chroma:
            self._chroma.disconnect()
        if self._redis:
            self._redis.disconnect()
        for name, provider in self._providers.items():
            try:
                provider.shutdown()
            except Exception as e:
                logger.warning("Error shutting down provider %s: %s", name, e)
        logger.info("memovex shut down (agent=%s)", self.agent_id)

    # -----------------------------------------------------------------------
    # Provider registration
    # -----------------------------------------------------------------------

    def register_provider(self, name: str, provider: Any) -> None:
        self._providers[name] = provider
        logger.info("Registered provider: %s", name)

    def get_provider(self, name: str) -> Optional[Any]:
        return self._providers.get(name)

    # -----------------------------------------------------------------------
    # Memory operations
    # -----------------------------------------------------------------------

    def store(self, text: str, memory_type: MemoryType = MemoryType.EPISODIC,
              entities: Optional[Set[str]] = None,
              tags: Optional[Set[str]] = None,
              session_id: Optional[str] = None,
              confidence: float = 0.7, salience: float = 0.5,
              provider: str = "native") -> str:
        """Store a memory across all active providers."""

        symbols = text_to_symbols(text, max_keywords=10)
        if entities is None:
            entities = extract_entities(text)

        # Compute embedding if model is available
        embedding = None
        em = self._resonance_engine._embedding_model
        if em is not None:
            embedding = em.encode(text)

        memory = Memory(
            memory_id=str(uuid.uuid4()),
            text=text,
            memory_type=memory_type,
            entities=entities or set(),
            base64_symbols=symbols,
            symbolic_keys=set(tokenize(text)),
            tags=tags or set(),
            session_id=session_id,
            confidence=confidence,
            salience=salience,
            provider=provider,
            embedding=embedding,
        )

        # Store in local MemoryStore
        self._memory_store.add(memory)

        # Persist embedding to Qdrant if available
        if self._qdrant and embedding:
            self._qdrant.upsert(
                memory_id=memory.memory_id,
                vector=embedding,
                payload={"memory_id": memory.memory_id,
                         "memory_type": memory.memory_type.value},
            )

        # Mirror to Chroma when connected (alternative on-disk vector store)
        if self._chroma and embedding:
            self._chroma.upsert(
                memory_id=memory.memory_id,
                vector=embedding,
                payload={"memory_id": memory.memory_id,
                         "memory_type": memory.memory_type.value},
            )

        # Register in wisdom pipeline
        self._wisdom_store.register(
            memory.memory_id,
            confidence=memory.confidence,
            salience=memory.salience,
        )

        # Propagate to external providers
        for pname, prov in self._providers.items():
            try:
                prov.store_memory(memory)
            except Exception as e:
                logger.debug("Provider %s store failed: %s", pname, e)

        # Track recency in Redis
        if self._redis:
            self._redis.record_access(memory.memory_id)

        return memory.memory_id

    def store_reasoning_chain(self, text: str, hops: List[Dict],
                               entities: Optional[Set[str]] = None,
                               confidence: float = 0.7) -> str:
        """Store a reasoning chain and build its graph edges."""
        graph_nodes: Set[str] = set()
        graph_edges = []
        for hop in hops:
            source = hop.get("source", "")
            target = hop.get("target", "")
            via = hop.get("via", "")
            if source:
                graph_nodes.add(source)
            if target:
                graph_nodes.add(target)
            if source and target:
                graph_edges.append((source, via, target))

        memory = Memory(
            memory_id=str(uuid.uuid4()),
            text=text,
            memory_type=MemoryType.REASONING,
            entities=entities or extract_entities(text),
            base64_symbols=text_to_symbols(text, max_keywords=10),
            symbolic_keys=set(tokenize(text)),
            graph_nodes=list(graph_nodes),
            graph_edges=graph_edges,
            reasoning_hops=hops,
            confidence=confidence,
            salience=min(0.5 + confidence * 0.3, 1.0),
            provider="native",
        )
        self._memory_store.add(memory)
        self._wisdom_store.register(memory.memory_id, confidence=confidence,
                                    salience=memory.salience)

        # Build graph edges in GraphStore
        if self._graph_store:
            self._graph_store.add_from_hops(hops, memory.memory_id, confidence)

        return memory.memory_id

    def retrieve(self, query: str, top_k: int = 5,
                 channels: Optional[List[str]] = None,
                 renormalize: bool = True) -> List[RetrievalResult]:
        """Retrieve memories using multi-channel resonance.

        Args:
            query: query string.
            top_k: maximum results.
            channels: optional subset of channel names. When given,
                only those channels contribute to the total score and,
                by default, the score is renormalized so it stays in
                roughly the [0, 1] range. Pass ``renormalize=False`` to
                keep absolute weights instead (the score will be capped
                by the sum of the active channels' weights).
            renormalize: see ``channels``. Has no effect when ``channels``
                is None.
        """

        # Optional: boost from Qdrant semantic search (fall back to Chroma
        # when Qdrant is absent but Chroma is wired)
        vec_store = self._qdrant or self._chroma
        if vec_store:
            em = self._resonance_engine._embedding_model
            if em is not None:
                qvec = em.encode(query)
                if qvec:
                    qdrant_hits = vec_store.search(qvec, top_k=top_k * 2)
                    for mid, score in qdrant_hits:
                        mem = self._memory_store.get(mid)
                        if mem:
                            mem.qdrant_score = score

        results = self._resonance_engine.search(
            query, top_k=top_k, channels=channels, renormalize=renormalize)

        # External providers
        for pname, prov in self._providers.items():
            try:
                for pr in prov.retrieve(query, top_k=top_k):
                    self._merge_result(results, pr, provider=pname)
            except Exception as e:
                logger.debug("Provider %s retrieve failed: %s", pname, e)

        results.sort(key=lambda r: r.total_score, reverse=True)

        # Record recency in Redis
        if self._redis:
            for r in results[:top_k]:
                self._redis.record_access(r.memory.memory_id)

        return results[:top_k]

    def prefetch(self, query: str, max_tokens: int = 1000) -> str:
        """Generate LLM-ready context string respecting a token budget."""
        # Check Redis cache first
        cache_key = query[:128]
        if self._redis:
            cached = self._redis.get_prefetch(cache_key)
            if cached:
                return cached

        results = self.retrieve(query, top_k=5)
        if not results:
            return ""

        formatted = []
        for r in results:
            top_channel = max(r.channel_scores, key=lambda k: r.channel_scores[k],
                              default="unknown")
            ch_score = r.channel_scores.get(top_channel, 0.0)
            formatted.append(
                f"[{r.memory.memory_type.value}] "
                f"(score:{r.total_score:.3f}, {top_channel}:{ch_score:.3f}) "
                f"{r.memory.text[:250]} [{r.provider}]"
            )

        budgeted = []
        for line in formatted:
            tokens = max(1, len(line) // 4)
            if tokens >= max_tokens:
                budgeted.append(line[:max_tokens * 4])
                break
            budgeted.append(line)
            max_tokens -= tokens

        result = "\n".join(budgeted)

        # Cache in Redis
        if self._redis:
            self._redis.cache_prefetch(cache_key, result, ttl=60)

        return result

    # -----------------------------------------------------------------------
    # Wisdom operations
    # -----------------------------------------------------------------------

    def corroborate(self, memory_id: str, delta_confidence: float = 0.05) -> None:
        """Add evidence to a memory, potentially promoting it toward WISDOM."""
        self._wisdom_store.corroborate(memory_id, delta_confidence)

    def promote_to_wisdom(self, memory_id: str, notes: str = "") -> None:
        self._wisdom_store.promote(memory_id, WisdomLevel.WISDOM, notes=notes)

    def wisdom_summary(self) -> Dict:
        return self._wisdom_store.count()

    # -----------------------------------------------------------------------
    # Graph operations
    # -----------------------------------------------------------------------

    def get_reasoning_chains(self, entity: str) -> List[Memory]:
        """All reasoning chains that mention an entity."""
        entity_lower = entity.lower()
        return [
            mem for mem in self._memory_store.all()
            if mem.memory_type == MemoryType.REASONING
            and any(n.lower() == entity_lower for n in mem.graph_nodes)
        ]

    def traverse_graph(self, start_entity: str, max_depth: int = 2) -> List[Memory]:
        """BFS through reasoning chains starting from an entity."""
        if self._graph_store and self._graph_store.available:
            mids = self._graph_store.memory_ids_for_entities({start_entity})
            neighbors = self._graph_store.neighbors(start_entity, depth=max_depth)
            mids |= self._graph_store.memory_ids_for_entities(set(neighbors))
            return [m for mid in mids if (m := self._memory_store.get(mid))]

        # Fallback: manual BFS
        visited: Set[str] = set()
        results: List[Memory] = []
        queue = [(start_entity, 0)]
        while queue:
            entity, depth = queue.pop(0)
            if entity.lower() in visited or depth > max_depth:
                continue
            visited.add(entity.lower())
            for chain in self.get_reasoning_chains(entity):
                if chain not in results:
                    results.append(chain)
                    for node in chain.graph_nodes:
                        if node.lower() not in visited:
                            queue.append((node, depth + 1))
        return results

    def graph_stats(self) -> Dict:
        if self._graph_store:
            return self._graph_store.stats()
        return {"nodes": 0, "edges": 0, "networkx_available": False}

    # -----------------------------------------------------------------------
    # Symbolic operations
    # -----------------------------------------------------------------------

    def compute_symbols(self, text: str) -> Set[str]:
        return text_to_symbols(text, max_keywords=12)

    def find_by_symbol(self, symbol: str) -> List[Memory]:
        ids = self._memory_store.by_base64_symbol.get(symbol, set())
        return [m for mid in ids if (m := self._memory_store.get(mid))]

    def compute_resonance(self, query: str, memory_text: str) -> float:
        return compute_symbolic_resonance(
            text_to_symbols(query), text_to_symbols(memory_text)
        )

    # -----------------------------------------------------------------------
    # Snapshot persistence (for hook-based cross-process use)
    # -----------------------------------------------------------------------

    def save_snapshot(self, path: str) -> int:
        """
        Dump all in-memory memories to a JSON file.
        Returns number of memories saved.
        Used by Stop hooks to persist state across invocations.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        memories = [m.to_dict() for m in self._memory_store.all()]
        wisdom = {
            mid: {
                "level": e.level.value,
                "confidence": e.confidence,
                "salience": e.salience,
                "evidence_count": e.evidence_count,
                "notes": e.notes,
            }
            for mid, e in self._wisdom_store._entries.items()
        }
        snapshot = {"version": "1.1", "memories": memories, "wisdom": wisdom}
        with open(p, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        logger.debug("Snapshot saved: %d memories → %s", len(memories), path)
        return len(memories)

    def load_snapshot(self, path: str) -> int:
        """
        Load memories from a JSON snapshot file into the in-memory store.
        Skips memories that are already present (by memory_id).
        Returns number of memories loaded.
        """
        p = Path(path)
        if not p.exists():
            return 0
        try:
            with open(p, encoding="utf-8") as f:
                snapshot = json.load(f)
        except Exception as e:
            logger.warning("Could not load snapshot from %s: %s", path, e)
            return 0

        loaded = 0
        for d in snapshot.get("memories", []):
            mid = d.get("memory_id", "")
            if not mid or self._memory_store.get(mid) is not None:
                continue
            try:
                mem = Memory.from_dict(d)
                self._memory_store.add(mem)
                loaded += 1
            except Exception:
                continue

        # Restore wisdom levels
        from .wisdom_store import WisdomLevel
        for mid, wd in snapshot.get("wisdom", {}).items():
            if self._wisdom_store._entries.get(mid) is None:
                e = self._wisdom_store.register(
                    mid,
                    confidence=wd.get("confidence", 0.0),
                    salience=wd.get("salience", 0.0),
                )
            else:
                e = self._wisdom_store._entries[mid]
            try:
                e.level = WisdomLevel(wd.get("level", "raw"))
                e.evidence_count = wd.get("evidence_count", 0)
                e.notes = wd.get("notes", "")
            except ValueError:
                pass

        logger.debug("Snapshot loaded: %d memories from %s", loaded, path)
        return loaded

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    def stats(self) -> Dict:
        return {
            "memories": self._memory_store.count(),
            "wisdom": self.wisdom_summary(),
            "graph": self.graph_stats(),
            "providers": list(self._providers.keys()),
            "embeddings_enabled": self._embeddings_enabled,
            "qdrant_connected": self._qdrant is not None and self._qdrant.available,
            "chroma_connected": self._chroma is not None and self._chroma.available,
            "redis_connected": self._redis is not None and self._redis.available,
        }

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _merge_result(self, results: List[RetrievalResult],
                      new_result: RetrievalResult,
                      provider: str = "external") -> None:
        for existing in results:
            if existing.memory.text == new_result.memory.text:
                existing.total_score = max(existing.total_score, new_result.total_score)
                existing.provider = f"{existing.provider}+{provider}"
                return
        new_result.provider = provider
        results.append(new_result)
