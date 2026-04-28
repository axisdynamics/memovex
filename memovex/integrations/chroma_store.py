"""
memovex — ChromaDB Vector Store (alternative to Qdrant).

Drop-in replacement for QdrantStore with the same interface.
Useful for local/edge deployments without a Qdrant server.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed — ChromaDB store disabled")


class ChromaStore:
    """
    Vector store backed by ChromaDB.

    ChromaDB persists to disk by default (persist_directory).
    Set persist_directory=None for an ephemeral in-memory store
    (useful for tests).
    """

    def __init__(self, collection: str = "memovex_vex",
                 persist_directory: Optional[str] = "./data/chroma"):
        self._collection_name = collection
        self._persist_directory = persist_directory
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None

    def connect(self) -> bool:
        if not _CHROMA_AVAILABLE:
            return False
        try:
            if self._persist_directory:
                self._client = chromadb.PersistentClient(
                    path=self._persist_directory,
                )
            else:
                self._client = chromadb.EphemeralClient()
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB connected (collection=%s, persist=%s)",
                        self._collection_name, self._persist_directory)
            return True
        except Exception as e:
            logger.warning("ChromaDB connection failed: %s", e)
            self._client = None
            return False

    @property
    def available(self) -> bool:
        return self._collection is not None

    def upsert(self, memory_id: str, vector: List[float],
               payload: Optional[Dict] = None) -> bool:
        if not self.available:
            return False
        try:
            self._collection.upsert(
                ids=[memory_id],
                embeddings=[vector],
                metadatas=[payload or {}],
            )
            return True
        except Exception as e:
            logger.debug("ChromaDB upsert failed for %s: %s", memory_id, e)
            return False

    def search(self, vector: List[float], top_k: int = 10,
               score_threshold: float = 0.0) -> List[Tuple[str, float]]:
        if not self.available:
            return []
        try:
            res = self._collection.query(
                query_embeddings=[vector],
                n_results=min(top_k, self._collection.count()),
            )
            ids = res["ids"][0]
            # Chroma returns distances; convert cosine distance → similarity
            distances = res["distances"][0]
            results = [
                (mid, max(0.0, 1.0 - dist))
                for mid, dist in zip(ids, distances)
                if (1.0 - dist) >= score_threshold
            ]
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        except Exception as e:
            logger.debug("ChromaDB search failed: %s", e)
            return []

    def delete(self, memory_id: str) -> bool:
        if not self.available:
            return False
        try:
            self._collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            logger.debug("ChromaDB delete failed for %s: %s", memory_id, e)
            return False

    def count(self) -> int:
        if not self.available:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def disconnect(self) -> None:
        self._client = None
        self._collection = None
