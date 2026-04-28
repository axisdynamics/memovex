"""
memovex — Tokenizer & Text Utilities.

Central module for tokenization, entity extraction, and embedding
generation.  Embedding support requires sentence-transformers; the
module falls back to BoW when it is not available.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: sentence-transformers
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.info(
        "sentence-transformers not installed — semantic channel uses BoW. "
        "Install with: pip install sentence-transformers"
    )


class EmbeddingModel:
    """
    Lazy-loaded sentence embedding model.

    Encoding is cached per-text to avoid re-computing identical strings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._model: Optional["SentenceTransformer"] = None
        self._cache: Dict[str, List[float]] = {}

    @property
    def available(self) -> bool:
        return _ST_AVAILABLE

    @property
    def vector_size(self) -> int:
        return 384 if "MiniLM" in self._model_name else 768

    def _load(self) -> None:
        if self._model is not None or not _ST_AVAILABLE:
            return
        try:
            self._model = SentenceTransformer(self._model_name, device=self._device)
            logger.info("Loaded embedding model '%s' on %s", self._model_name, self._device)
        except Exception as e:
            logger.warning("Could not load embedding model '%s': %s", self._model_name, e)

    def encode(self, text: str) -> Optional[List[float]]:
        """Return embedding vector or None if unavailable."""
        if not _ST_AVAILABLE:
            return None
        if text in self._cache:
            return self._cache[text]
        self._load()
        if self._model is None:
            return None
        try:
            vec = self._model.encode(text, convert_to_numpy=True).tolist()
            if len(self._cache) > 2000:
                # Evict oldest 500 entries
                for k in list(self._cache.keys())[:500]:
                    del self._cache[k]
            self._cache[text] = vec
            return vec
        except Exception as e:
            logger.debug("Embedding encode failed: %s", e)
            return None

    def encode_batch(self, texts: List[str],
                     batch_size: int = 32) -> List[Optional[List[float]]]:
        if not _ST_AVAILABLE:
            return [None] * len(texts)
        self._load()
        if self._model is None:
            return [None] * len(texts)
        try:
            vecs = self._model.encode(texts, batch_size=batch_size,
                                       convert_to_numpy=True).tolist()
            for text, vec in zip(texts, vecs):
                self._cache[text] = vec
            return vecs
        except Exception as e:
            logger.warning("Batch encode failed: %s", e)
            return [None] * len(texts)


# ---------------------------------------------------------------------------
# Tokenization helpers (re-exported from resonance_engine for convenience)
# ---------------------------------------------------------------------------

STOPWORDS: Set[str] = {
    "el", "la", "los", "las", "de", "del", "y", "e", "o", "u", "a", "en",
    "un", "una", "que", "es", "se", "con", "por", "para", "al", "lo", "su",
    "sus", "como", "mas", "pero", "sin", "si", "no", "todo", "este",
    "esta", "esto", "eso", "esa", "ese", "the", "an", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "about", "up", "down",
}

ENTITY_RE = re.compile(
    r"\b[A-Z\xc1\xc9\xcd\xd3\xda\xd1][a-z\xe1\xe9\xed\xf3\xfa\xf1]+"
    r"(?:\s+[A-Z\xc1\xc9\xcd\xd3\xda\xd1][a-z\xe1\xe9\xed\xf3\xfa\xf1]+)*\b"
)


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"\b\w+\b", text.lower())
            if t not in STOPWORDS and len(t) > 2]


def extract_entities(text: str) -> Set[str]:
    return {m.group(0) for m in ENTITY_RE.finditer(text)}


def cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two dense vectors."""
    import math
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# Module-level default model instance (lazy-loaded on first use)
_default_model: Optional[EmbeddingModel] = None


def get_default_model(model_name: str = "all-MiniLM-L6-v2",
                      device: str = "cpu") -> EmbeddingModel:
    global _default_model
    if _default_model is None:
        _default_model = EmbeddingModel(model_name=model_name, device=device)
    return _default_model
