"""
memovex — Qdrant Vector Store integration.

Provides persistent vector storage and semantic search via Qdrant.
Gracefully degrades if qdrant-client is not installed.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed — Qdrant store disabled")


class QdrantStore:
    """
    Vector store backed by Qdrant.

    All operations are no-ops when Qdrant is unavailable so the rest
    of the system continues to function with BoW-only semantic scoring.
    """

    def __init__(self, host: str = "localhost", port: int = 6333,
                 collection: str = "memovex_vex",
                 vector_size: int = 384):
        self._host = host
        self._port = port
        self._collection = collection
        self._vector_size = vector_size
        self._client: Optional[Any] = None

    def connect(self) -> bool:
        if not _QDRANT_AVAILABLE:
            return False
        try:
            self._client = QdrantClient(host=self._host, port=self._port, timeout=5)
            self._ensure_collection()
            logger.info("Qdrant connected at %s:%s (collection=%s)",
                        self._host, self._port, self._collection)
            return True
        except Exception as e:
            logger.warning("Qdrant connection failed: %s — semantic search degraded to BoW", e)
            self._client = None
            return False

    @property
    def available(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert(self, memory_id: str, vector: List[float],
               payload: Optional[Dict] = None) -> bool:
        if not self.available:
            return False
        try:
            self._client.upsert(
                collection_name=self._collection,
                points=[qmodels.PointStruct(
                    id=self._to_qdrant_id(memory_id),
                    vector=vector,
                    payload=payload or {},
                )],
            )
            return True
        except Exception as e:
            logger.debug("Qdrant upsert failed for %s: %s", memory_id, e)
            return False

    def search(self, vector: List[float], top_k: int = 10,
               score_threshold: float = 0.0) -> List[Tuple[str, float]]:
        """Return list of (memory_id, score) sorted by score desc."""
        if not self.available:
            return []
        try:
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=top_k,
                score_threshold=score_threshold,
            )
            return [(h.payload.get("memory_id", str(h.id)), h.score) for h in hits]
        except Exception as e:
            logger.debug("Qdrant search failed: %s", e)
            return []

    def delete(self, memory_id: str) -> bool:
        if not self.available:
            return False
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=qmodels.Filter(
                    must=[qmodels.FieldCondition(
                        key="memory_id",
                        match=qmodels.MatchValue(value=memory_id),
                    )]
                ),
            )
            return True
        except Exception as e:
            logger.debug("Qdrant delete failed for %s: %s", memory_id, e)
            return False

    def disconnect(self) -> None:
        self._client = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=self._vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s'", self._collection)

    @staticmethod
    def _to_qdrant_id(memory_id: str) -> str:
        try:
            uuid.UUID(memory_id)
            return memory_id
        except ValueError:
            import hashlib
            h = hashlib.md5(memory_id.encode()).hexdigest()
            return str(uuid.UUID(h))
