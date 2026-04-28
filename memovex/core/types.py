"""
memovex — Core types and data models.

Defines the shared data structures used across all memory channels,
providers, and adapters.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Memory Types
# ---------------------------------------------------------------------------

class MemoryType(str, Enum):
    """Types of memory stored in the system."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    SYMBOLIC = "symbolic"
    RESONANT = "resonant"
    REASONING = "reasoning"
    PROCEDURAL = "procedural"
    WISDOM = "wisdom"
    EXPLICIT = "explicit_memory"


class ChannelType(str, Enum):
    """Retrieval channels for multi-channel resonance.

    base64_resonance is intentionally absent: it was consolidated into
    'symbolic' (token overlap). Base64 symbols are still generated and
    stored for cross-provider indexing, but are not a scored channel.
    """
    SEMANTIC = "semantic"
    ENTITY = "entity"
    GRAPH_TRAVERSAL = "graph_traversal"
    WISDOM = "wisdom"
    TEMPORAL = "temporal"
    REASONING_CHAIN = "reasoning_chain"
    RECENCY = "recency"
    SYMBOLIC = "symbolic"
    USAGE = "usage"
    TAG = "tag"
    PROCEDURAL = "procedural"


@dataclass
class Memory:
    """Unified memory record across all channels and providers."""

    memory_id: str
    text: str
    memory_type: MemoryType = MemoryType.EPISODIC

    # Metadata
    speaker: Optional[str] = None
    session_id: Optional[str] = None
    turn_index: Optional[int] = None
    event_time: Optional[str] = None

    # Indices
    entities: Set[str] = field(default_factory=set)
    base64_symbols: Set[str] = field(default_factory=set)
    symbolic_keys: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)

    # Graph (for reasoning chains)
    graph_nodes: List[str] = field(default_factory=list)
    graph_edges: List[Tuple[str, str, str]] = field(default_factory=list)
    reasoning_hops: List[Dict[str, Any]] = field(default_factory=list)

    # Scores
    confidence: float = 0.7
    salience: float = 0.5
    usage_score: float = 0.0
    access_count: int = 0

    # Timestamps
    ingested_at: float = field(default_factory=time.time)
    last_accessed: Optional[float] = None

    # Dense embedding (sentence-transformers or compatible)
    embedding: Optional[List[float]] = None

    # External provider tracking
    provider: str = "native"
    external_id: Optional[str] = None
    qdrant_score: float = 0.0
    qdrant_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "text": self.text,
            "memory_type": self.memory_type.value,
            "speaker": self.speaker,
            "session_id": self.session_id,
            "event_time": self.event_time,
            "entities": list(self.entities),
            "base64_symbols": list(self.base64_symbols),
            "symbolic_keys": list(self.symbolic_keys),
            "tags": list(self.tags),
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "reasoning_hops": self.reasoning_hops,
            "confidence": self.confidence,
            "salience": self.salience,
            "usage_score": self.usage_score,
            "access_count": self.access_count,
            "ingested_at": self.ingested_at,
            "last_accessed": self.last_accessed,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Memory":
        try:
            mt = MemoryType(d.get("memory_type", "episodic"))
        except ValueError:
            mt = MemoryType.EPISODIC
        return cls(
            memory_id=d["memory_id"],
            text=d.get("text", ""),
            memory_type=mt,
            speaker=d.get("speaker"),
            session_id=d.get("session_id"),
            event_time=d.get("event_time"),
            entities=set(d.get("entities", [])),
            base64_symbols=set(d.get("base64_symbols", [])),
            symbolic_keys=set(d.get("symbolic_keys", [])),
            tags=set(d.get("tags", [])),
            graph_nodes=d.get("graph_nodes", []),
            graph_edges=d.get("graph_edges", []),
            reasoning_hops=d.get("reasoning_hops", []),
            confidence=float(d.get("confidence", 0.7)),
            salience=float(d.get("salience", 0.5)),
            usage_score=float(d.get("usage_score", 0.0)),
            access_count=int(d.get("access_count", 0)),
            ingested_at=float(d.get("ingested_at", 0.0)),
            last_accessed=d.get("last_accessed"),
            provider=d.get("provider", "native"),
        )


@dataclass
class QueryFeatures:
    """Features extracted from a query for multi-channel retrieval."""

    raw_text: str
    tokens: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    time_hint: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    base64_symbols: Set[str] = field(default_factory=set)
    query_type: Optional[str] = None  # factual, procedural, reasoning, etc.


@dataclass
class ChannelResult:
    """Result from a single channel retrieval."""

    memory: Memory
    score: float
    channel: ChannelType
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Aggregated result from multi-channel retrieval."""

    memory: Memory
    total_score: float
    channel_scores: Dict[str, float]
    provider: str = "native"
    channels_used: int = 0
