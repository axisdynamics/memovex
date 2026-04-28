"""
memovex — Resonance Engine.

Multi-channel weighted retrieval with base64 symbolic resonance,
inspired by Resonant Memory VEX v2.3 and expanded with graph
traversal, reasoning chains, and procedural channels.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from .types import (
    ChannelType,
    ChannelResult,
    Memory,
    MemoryType,
    QueryFeatures,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default channel weights (v1.1)
# ---------------------------------------------------------------------------
# Rationale for ordering:
#   semantic (0.32) — real embedding similarity via Qdrant; strongest signal
#   entity   (0.13) — exact named-entity overlap; orthogonal to semantic
#   graph_traversal (0.12) — multi-hop entity paths; needed for complex queries
#   wisdom   (0.10) — high-confidence curated knowledge; acts as a boost
#   temporal (0.08) — year/date matching; essential for temporal queries
#   recency  (0.06) — freshness decay; prevents stale retrieval
#   reasoning_chain (0.05) — inference traces; complements graph
#   symbolic (0.05) — token overlap (subsumes the old base64_resonance)
#   procedural (0.05) — how-to/skill memories matched to "how"-style queries
#   usage    (0.02) — access frequency; minor quality signal
#   tag      (0.02) — category labels; coarse-grained filter
#
# base64_resonance is intentionally removed as a scored channel: it was
# functionally identical to 'symbolic' but with an SHA-256 hash layer that
# introduced collision risk and no semantic gain. Base64 symbols are still
# generated and stored for cross-provider indexing (see MemoryStore indices).

DEFAULT_CHANNEL_WEIGHTS: Dict[str, float] = {
    "semantic":          0.32,
    "entity":            0.13,
    "graph_traversal":   0.12,
    "wisdom":            0.10,
    "temporal":          0.08,
    "recency":           0.06,
    "reasoning_chain":   0.05,
    "symbolic":          0.05,
    "procedural":        0.05,
    "usage":             0.02,
    "tag":               0.02,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STOPWORDS: Set[str] = {
    "el", "la", "los", "las", "de", "del", "y", "e", "o", "u", "a", "en",
    "un", "una", "que", "es", "se", "con", "por", "para", "al", "lo", "su",
    "sus", "como", "mas", "pero", "sin", "que", "si", "no", "todo", "este",
    "esta", "esto", "eso", "esa", "ese", "the", "a", "an", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can", "need",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "about", "up", "down",
}

ENTITY_PATTERN = re.compile(r"\b[A-Z\xc1\xc9\xcd\xd3\xda\xd1][a-z\xe1\xe9\xed\xf3\xfa\xf1]+(?:\s+[A-Z\xc1\xc9\xcd\xd3\xda\xd1][a-z\xe1\xe9\xed\xf3\xfa\xf1]+)*\b")
YEAR_PATTERN = re.compile(r"\b(20\d{2}|19\d{2})\b")
DATE_PATTERN = re.compile(
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|"
    r"July|August|September|October|November|December)\s+(\d{4})\b",
    re.IGNORECASE,
)


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"\b\w+\b", text.lower())
            if t not in STOPWORDS and len(t) > 2]


def extract_entities(text: str) -> Set[str]:
    return {m.group(0) for m in ENTITY_PATTERN.finditer(text)}


def extract_years(text: str) -> Set[str]:
    return {m.group(1) for m in YEAR_PATTERN.finditer(text)}


def extract_dates(text: str) -> List[str]:
    return [m.group(0) for m in DATE_PATTERN.finditer(text)]


def bow_vector(text: str) -> Dict[str, float]:
    tokens = tokenize(text)
    if not tokens:
        return {}
    freq = Counter(tokens)
    total = max(len(tokens), 1)
    return {t: c / total for t, c in freq.items()}


def cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in set(a) | set(b))
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# In-Memory Store with Multi-Index
# ---------------------------------------------------------------------------

class MemoryStore:
    """In-memory memory store with secondary indices for fast lookup."""

    def __init__(self) -> None:
        self.memories: Dict[str, Memory] = {}
        self.by_base64_symbol: Dict[str, Set[str]] = defaultdict(set)
        self.by_entity: Dict[str, Set[str]] = defaultdict(set)
        self.by_symbolic: Dict[str, Set[str]] = defaultdict(set)
        self.by_session: Dict[str, Set[str]] = defaultdict(set)
        self.by_type: Dict[str, Set[str]] = defaultdict(set)
        self.by_tag: Dict[str, Set[str]] = defaultdict(set)

    def add(self, memory: Memory) -> None:
        self.memories[memory.memory_id] = memory
        for sym in memory.base64_symbols:
            self.by_base64_symbol[sym].add(memory.memory_id)
        for entity in memory.entities:
            self.by_entity[entity.lower()].add(memory.memory_id)
        for key in memory.symbolic_keys:
            self.by_symbolic[key].add(memory.memory_id)
        for tag in memory.tags:
            self.by_tag[tag.lower()].add(memory.memory_id)
        if memory.session_id:
            self.by_session[memory.session_id].add(memory.memory_id)
        self.by_type[memory.memory_type.value].add(memory.memory_id)

    def get(self, memory_id: str) -> Optional[Memory]:
        return self.memories.get(memory_id)

    def all(self) -> List[Memory]:
        return list(self.memories.values())

    def count(self) -> int:
        return len(self.memories)

    def candidates_by_base64(self, symbols: Set[str]) -> Set[str]:
        out: Set[str] = set()
        for sym in symbols:
            out |= self.by_base64_symbol.get(sym, set())
        return out

    def candidates_by_entity(self, entities: Set[str]) -> Set[str]:
        out: Set[str] = set()
        for e in entities:
            out |= self.by_entity.get(e.lower(), set())
        return out

    def candidates_by_symbolic(self, keys: Set[str]) -> Set[str]:
        out: Set[str] = set()
        for key in keys:
            out |= self.by_symbolic.get(key, set())
        return out

    def candidates_by_tag(self, tags: Set[str]) -> Set[str]:
        out: Set[str] = set()
        for tag in tags:
            out |= self.by_tag.get(tag.lower(), set())
        return out

    def candidates_by_type(self, mem_type: str) -> Set[str]:
        return self.by_type.get(mem_type, set()).copy()

    def register_access(self, memory_id: str) -> None:
        mem = self.memories.get(memory_id)
        if mem:
            mem.access_count += 1
            mem.last_accessed = time.time()
            mem.usage_score = min(mem.usage_score + 0.15, 1.0)

    def decay_all(self, amount: float = 0.03) -> None:
        for mem in self.memories.values():
            mem.usage_score = max(0.0, mem.usage_score - amount)

    def prune(self, max_memories: int = 5000) -> List[str]:
        if len(self.memories) <= max_memories:
            return []
        scored = [(0.40 * m.salience + 0.25 * m.usage_score + 0.15 * m.confidence
                    + 0.10 * min(m.access_count / 100.0, 1.0), m.memory_id)
                   for m in self.memories.values()]
        scored.sort(reverse=True)
        survivors = {mid for _, mid in scored[:max_memories]}
        removed = [mid for mid in list(self.memories) if mid not in survivors]
        for mid in removed:
            self._remove_from_indices(mid)
        return removed

    def _remove_from_indices(self, memory_id: str) -> None:
        """Remove a memory from all secondary indices before deleting it."""
        mem = self.memories.pop(memory_id, None)
        if mem is None:
            return
        for sym in mem.base64_symbols:
            self.by_base64_symbol[sym].discard(memory_id)
        for entity in mem.entities:
            self.by_entity[entity.lower()].discard(memory_id)
        for key in mem.symbolic_keys:
            self.by_symbolic[key].discard(memory_id)
        for tag in mem.tags:
            self.by_tag[tag.lower()].discard(memory_id)
        if mem.session_id:
            self.by_session[mem.session_id].discard(memory_id)
        self.by_type[mem.memory_type.value].discard(memory_id)


# ---------------------------------------------------------------------------
# Symbolic Resonance Engine (Base64)
# ---------------------------------------------------------------------------

def keyword_to_symbol(keyword: str, length: int = 6) -> str:
    """Convert a keyword to a base64 symbol using SHA-256."""
    import base64
    h = hashlib.sha256(keyword.encode("utf-8")).digest()
    b64 = base64.b64encode(h).decode("ascii")
    return b64.replace("=", "")[:length]


def text_to_symbols(text: str, max_keywords: int = 12,
                    symbol_length: int = 6) -> Set[str]:
    """Generate a set of base64 symbols from text."""
    raw = re.findall(r"\b[\wáéíóúñÁÉÍÓÚÑ]+(?:'[\w]+)?\b", text.lower())
    freq: Dict[str, int] = {}
    for token in raw:
        token = token.strip("'\".,!?;:")
        if token and token not in STOPWORDS and len(token) > 2:
            freq[token] = freq.get(token, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: -x[1])[:max_keywords]
    return {keyword_to_symbol(t, symbol_length) for t, _ in sorted_tokens}


def compute_symbolic_resonance(
    query_symbols: Set[str],
    memory_symbols: Set[str],
    method: str = "jaccard",
) -> float:
    if not query_symbols or not memory_symbols:
        return 0.0
    intersection = query_symbols & memory_symbols
    if method == "jaccard":
        union = query_symbols | memory_symbols
        return len(intersection) / max(len(union), 1)
    else:
        return len(intersection) / max(len(query_symbols), 1)


# ---------------------------------------------------------------------------
# Resonance Engine
# ---------------------------------------------------------------------------

class ResonanceEngine:
    """Multi-channel resonance engine with 10 scored channels."""

    def __init__(self, store: MemoryStore,
                 weights: Optional[Dict[str, float]] = None):
        self.store = store
        self.weights = weights or dict(DEFAULT_CHANNEL_WEIGHTS)
        self._query_symbols: Set[str] = set()
        self._wisdom_store = None        # set by MemoVexOrchestrator
        self._embedding_model = None     # set by MemoVexOrchestrator
        self._graph_store = None         # set by MemoVexOrchestrator
        self._query_embedding: Optional[List[float]] = None

    def search(self, query: str, top_k: int = 10,
               channels: Optional[List[str]] = None,
               renormalize: bool = True) -> List[RetrievalResult]:
        """Multi-channel search with optional channel filter.

        Args:
            query: query string.
            top_k: maximum results.
            channels: subset of channels to score on (default: all weighted
                channels). Unknown channel names are silently ignored.
            renormalize: when ``channels`` is given and this is True
                (default), divide the total by the sum of the active
                channels' weights, so scores stay in roughly the same
                [0, 1] range regardless of which channels are active.
                Set False if you want absolute weights (the score will
                be capped by the sum of the active channels' weights).
        """
        qf = self._extract_features(query)
        self._query_symbols = text_to_symbols(query, max_keywords=12)

        # Pre-compute query embedding once for all candidate comparisons
        self._query_embedding = None
        if self._embedding_model is not None:
            self._query_embedding = self._embedding_model.encode(query)

        # Collect candidates from sparse indices
        candidate_ids: Set[str] = self._collect_candidates(qf)

        # Fallback to all memories if no candidates
        if not candidate_ids and self.store.count() > 0:
            candidate_ids = {m.memory_id for m in self.store.all()}

        # Resolve & validate active channels — silently drop unknown ones
        if channels is None:
            active_channels = list(self.weights.keys())
        else:
            active_channels = [c for c in channels if c in self.weights]

        results: List[RetrievalResult] = []

        for mid in candidate_ids:
            mem = self.store.get(mid)
            if mem is None:
                continue

            score, channel_scores = self._score_channels(
                mem, qf, active_channels, renormalize=renormalize)

            if score > 0.001:
                results.append(RetrievalResult(
                    memory=mem,
                    total_score=score,
                    channel_scores=channel_scores,
                    channels_used=len(active_channels),
                ))

        results.sort(key=lambda r: r.total_score, reverse=True)

        for r in results[:top_k]:
            self.store.register_access(r.memory.memory_id)

        return results[:top_k]

    def _extract_features(self, query: str) -> QueryFeatures:
        qf = QueryFeatures(raw_text=query)
        qf.tokens = set(tokenize(query))
        qf.entities = extract_entities(query)
        qf.base64_symbols = text_to_symbols(query, max_keywords=12)

        # Infer time hint
        years = extract_years(query)
        if years:
            qf.time_hint = sorted(years)[0]

        # Infer tags
        ql = query.lower()
        if any(w in ql for w in ("live", "lives", "where", "location", "city", "town", "country", "move", "moved")):
            qf.tags.add("location")
        if any(w in ql for w in ("like", "likes", "prefer", "favorite", "love", "enjoy")):
            qf.tags.add("preference")
        if any(w in ql for w in ("work", "works", "job", "company", "career", "employ")):
            qf.tags.add("work")
        if any(w in ql for w in ("wife", "husband", "married", "partner", "girlfriend", "boyfriend", "relationship")):
            qf.tags.add("relationship")
        if any(w in ql for w in ("eat", "food", "breakfast", "lunch", "dinner", "restaurant")):
            qf.tags.add("food")
        if any(w in ql for w in ("school", "university", "college", "study", "student", "class")):
            qf.tags.add("education")
        if any(w in ql for w in ("how", "steps", "process", "procedure", "way to", "method")):
            qf.query_type = "procedural"
        if any(w in ql for w in ("why", "because", "reason", "cause", "explain")):
            qf.query_type = "reasoning"

        return qf

    def _collect_candidates(self, qf: QueryFeatures) -> Set[str]:
        candidate_ids: Set[str] = set()
        if qf.base64_symbols:
            candidate_ids |= self.store.candidates_by_base64(qf.base64_symbols)
        if qf.entities:
            candidate_ids |= self.store.candidates_by_entity(qf.entities)
        if qf.tokens:
            candidate_ids |= self.store.candidates_by_symbolic(qf.tokens)
        if qf.tags:
            candidate_ids |= self.store.candidates_by_tag(qf.tags)
        return candidate_ids

    def _score_channels(self, mem: Memory, qf: QueryFeatures,
                        active_channels: List[str],
                        renormalize: bool = True) -> Tuple[float, Dict[str, float]]:
        scores: Dict[str, float] = {}
        mem_tokens = set(tokenize(mem.text))

        for channel in active_channels:
            score = self._score_single_channel(mem, qf, mem_tokens, channel)
            scores[channel] = score

        # Sum only across active channels — channels not in active_channels
        # contribute 0 by construction.
        active_weight_sum = sum(self.weights.get(ch, 0.0) for ch in active_channels)
        weighted = sum(self.weights.get(ch, 0.0) * scores.get(ch, 0.0)
                        for ch in active_channels)

        # When the caller restricted the channel set, optionally rescale so
        # the score still lives in roughly [0, 1] instead of being capped by
        # the sum of the active weights.
        full_weight_sum = sum(self.weights.values())
        is_subset = abs(active_weight_sum - full_weight_sum) > 1e-9
        if renormalize and is_subset and active_weight_sum > 0:
            total = weighted / active_weight_sum
        else:
            total = weighted

        if mem.qdrant_score > 0:
            sem_weight = self.weights.get("semantic", 0.18)
            total = sem_weight * mem.qdrant_score + (1.0 - sem_weight) * total

        return total, scores

    def _score_single_channel(self, mem: Memory, qf: QueryFeatures,
                               mem_tokens: Set[str], channel: str) -> float:
        if channel == "semantic":
            # Use dense embeddings when available; fall back to BoW cosine
            if (self._query_embedding is not None
                    and mem.embedding is not None):
                from .tokenizer import cosine
                return max(0.0, cosine(self._query_embedding, mem.embedding))
            mem_bow = bow_vector(mem.text)
            query_bow = bow_vector(qf.raw_text)
            return cosine_sparse(query_bow, mem_bow)

        elif channel == "symbolic":
            overlap = len(qf.tokens & mem_tokens)
            return min(overlap / max(len(qf.tokens), 1), 1.0)

        elif channel == "entity":
            mem_entities_lower = {e.lower() for e in mem.entities}
            qf_entities_lower = {e.lower() for e in qf.entities}
            entity_overlap = len(qf_entities_lower & mem_entities_lower)
            return min(entity_overlap / max(len(qf_entities_lower), 1), 1.0)

        elif channel == "tag":
            tag_overlap = len(qf.tags & mem.tags)
            return min(tag_overlap / max(len(qf.tags), 1), 1.0)

        elif channel == "temporal":
            return self._temporal_score(mem, qf.time_hint)

        elif channel == "recency":
            return self._recency_score(mem)

        elif channel == "usage":
            return min(mem.usage_score, 1.0)

        elif channel == "wisdom":
            return self._wisdom_score(mem)

        elif channel == "graph_traversal":
            return self._graph_traversal_score(mem, qf)

        elif channel == "reasoning_chain":
            return self._reasoning_chain_score(mem, qf)

        elif channel == "procedural":
            return self._procedural_score(mem, qf)

        return 0.0

    def _temporal_score(self, mem: Memory, hint: Optional[str]) -> float:
        if hint is None:
            return 0.5
        if mem.event_time and hint:
            if hint in mem.event_time:
                return 1.0
            hint_year = re.search(r"\b(20\d{2})\b", hint)
            mem_year = re.search(r"\b(20\d{2})\b", mem.event_time)
            if hint_year and mem_year and hint_year.group(1) == mem_year.group(1):
                return 0.8
        return 0.3

    def _recency_score(self, mem: Memory) -> float:
        age_days = (time.time() - mem.ingested_at) / 86400.0
        if age_days <= 1:
            return 1.0
        if age_days <= 7:
            return 0.7
        if age_days <= 30:
            return 0.4
        if age_days <= 90:
            return 0.2
        return 0.1

    def _wisdom_score(self, mem: Memory) -> float:
        """Score from WisdomStore curation level, with confidence/salience fallback."""
        if self._wisdom_store is not None:
            return self._wisdom_store.wisdom_score(mem.memory_id)
        # Fallback when WisdomStore is not wired (should not happen post-init)
        if mem.confidence > 0.8 and mem.salience > 0.7:
            return 0.8 + (mem.confidence * mem.salience * 0.2)
        if mem.confidence > 0.6 and mem.salience > 0.5:
            return 0.5
        return 0.0

    def _graph_traversal_score(self, mem: Memory, qf: QueryFeatures) -> float:
        """
        Score based on graph reachability from query entities.

        When GraphStore is available, checks if query entities can reach
        the memory's entities within 2 hops.  Falls back to direct node
        overlap when GraphStore is absent.
        """
        if not qf.entities:
            return 0.0

        if self._graph_store is not None and self._graph_store.available:
            reachable = set()
            for entity in qf.entities:
                reachable.update(
                    self._graph_store.neighbors(entity, depth=2)
                )
            mem_entities_lower = {e.lower() for e in mem.entities}
            mem_nodes_lower = {n.lower() for n in mem.graph_nodes}
            target = mem_entities_lower | mem_nodes_lower
            overlap = len(reachable & target)
            return min(overlap / max(len(qf.entities), 1), 1.0)

        # Fallback: direct node overlap
        if not mem.graph_nodes:
            return 0.0
        qf_entities_lower = {e.lower() for e in qf.entities}
        graph_nodes_lower = {n.lower() for n in mem.graph_nodes}
        overlap = len(qf_entities_lower & graph_nodes_lower)
        return min(overlap / max(len(qf_entities_lower), 1), 1.0)

    def _reasoning_chain_score(self, mem: Memory, qf: QueryFeatures) -> float:
        """
        Score based on reasoning hop overlap.
        Matches if query shares entities with reasoning hops.
        """
        if not mem.reasoning_hops or not qf.entities:
            return 0.0
        hop_entities = set()
        for hop in mem.reasoning_hops:
            for key in ("source", "target", "via"):
                val = hop.get(key, "")
                if val:
                    hop_entities.add(val.lower())
        qf_entities_lower = {e.lower() for e in qf.entities}
        overlap = len(qf_entities_lower & hop_entities)
        return min(overlap / max(len(qf_entities_lower), 1), 1.0)

    def _procedural_score(self, mem: Memory, qf: QueryFeatures) -> float:
        """
        Score based on procedural pattern matching.
        Checks if memory has procedural tags and query is procedural.
        """
        if qf.query_type != "procedural":
            return 0.0
        if "procedure" in mem.tags or "skill" in mem.tags or "workflow" in mem.tags:
            return 0.6
        return 0.0
