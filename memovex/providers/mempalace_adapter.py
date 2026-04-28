"""
memovex — MemPalace Adapter.

Integrates MemPalace (property graph memory) as an external provider.
Pulls MemPalace graph walks and exposes them to memovex channels,
particularly graph_traversal and entity.

Requires: pip install mempalace  (or a local checkout)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class MemPalaceAdapter:
    """
    Adapter for MemPalace property-graph memory.

    Falls back to a file-based palace store when the mempalace package
    is not installed, enabling local-only usage without dependencies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {
            "data_dir": "./data/mempalace",
            "index_file": "palace_index.json",
        }
        self._palace = None
        self._local_index: Dict[str, Dict] = {}

    def initialize(self) -> None:
        try:
            import mempalace
            data_dir = self._config.get("data_dir", "./data/mempalace")
            self._palace = mempalace.Palace(data_dir=data_dir)
            logger.info("MemPalace adapter initialized at %s", data_dir)
        except ImportError:
            logger.warning("mempalace not installed — using file-based fallback")
            self._load_local_index()
        except Exception as e:
            logger.error("MemPalace init failed: %s", e)

    def _load_local_index(self) -> None:
        index_path = Path(self._config.get("data_dir", "./data/mempalace")) / \
                     self._config.get("index_file", "palace_index.json")
        if index_path.exists():
            try:
                with open(index_path) as f:
                    self._local_index = json.load(f)
                logger.info("MemPalace local index loaded (%d entries)",
                            len(self._local_index))
            except Exception as e:
                logger.warning("Could not load MemPalace index: %s", e)

    def _save_local_index(self) -> None:
        data_dir = Path(self._config.get("data_dir", "./data/mempalace"))
        data_dir.mkdir(parents=True, exist_ok=True)
        index_path = data_dir / self._config.get("index_file", "palace_index.json")
        try:
            with open(index_path, "w") as f:
                json.dump(self._local_index, f, indent=2)
        except Exception as e:
            logger.warning("Could not save MemPalace index: %s", e)

    @property
    def available(self) -> bool:
        return self._palace is not None or bool(self._local_index)

    def store_memory(self, memory) -> str:
        if self._palace is not None:
            try:
                self._palace.add(
                    text=memory.text,
                    entities=list(memory.entities),
                    tags=list(memory.tags),
                    metadata={"memory_id": memory.memory_id},
                )
                return memory.memory_id
            except Exception as e:
                logger.debug("MemPalace store failed: %s", e)
                return ""

        # File-based fallback
        from ..core.resonance_engine import text_to_symbols
        self._local_index[memory.memory_id] = {
            "text": memory.text,
            "entities": list(memory.entities),
            "tags": list(memory.tags),
            "symbols": list(memory.base64_symbols or text_to_symbols(memory.text)),
        }
        self._save_local_index()
        return memory.memory_id

    def retrieve(self, query: str, top_k: int = 5) -> List:
        from ..core.types import Memory, MemoryType, RetrievalResult
        from ..core.resonance_engine import (
            text_to_symbols, compute_symbolic_resonance,
            extract_entities, tokenize,
        )

        query_symbols = text_to_symbols(query)
        query_entities = {e.lower() for e in extract_entities(query)}
        query_tokens = set(tokenize(query))
        results = []

        if self._palace is not None:
            try:
                palace_results = self._palace.search(query=query, limit=top_k * 2)
                for item in (palace_results or []):
                    text = item.get("text", "")
                    entities: Set[str] = set(item.get("entities", []))
                    mem_symbols = text_to_symbols(text)
                    score = self._score_item(
                        text, entities, mem_symbols,
                        query_tokens, query_entities, query_symbols,
                    )
                    mem = Memory(
                        memory_id=f"mempalace-{hash(text) & 0xFFFFFFFF}",
                        text=text,
                        memory_type=MemoryType.SEMANTIC,
                        entities=entities,
                        base64_symbols=mem_symbols,
                        provider="mempalace",
                    )
                    results.append(RetrievalResult(
                        memory=mem, total_score=score,
                        channel_scores={"symbolic": score}, provider="mempalace",
                    ))
            except Exception as e:
                logger.debug("MemPalace palace search failed: %s", e)

        else:
            for mid, entry in self._local_index.items():
                text = entry.get("text", "")
                entities = set(entry.get("entities", []))
                mem_symbols = set(entry.get("symbols", []))
                score = self._score_item(
                    text, entities, mem_symbols,
                    query_tokens, query_entities, query_symbols,
                )
                if score > 0.0:
                    mem = Memory(
                        memory_id=mid,
                        text=text,
                        memory_type=MemoryType.SEMANTIC,
                        entities=entities,
                        base64_symbols=mem_symbols,
                        provider="mempalace",
                    )
                    results.append(RetrievalResult(
                        memory=mem, total_score=score,
                        channel_scores={"symbolic": score}, provider="mempalace",
                    ))

        results.sort(key=lambda r: r.total_score, reverse=True)
        return results[:top_k]

    def _score_item(self, text: str, entities: Set[str], mem_symbols: Set[str],
                    query_tokens: Set[str], query_entities: Set[str],
                    query_symbols: Set[str]) -> float:
        from ..core.resonance_engine import tokenize, compute_symbolic_resonance
        token_overlap = len(query_tokens & set(tokenize(text))) / max(len(query_tokens), 1)
        entity_overlap = len(query_entities & {e.lower() for e in entities}) / \
                         max(len(query_entities), 1) if query_entities else 0.0
        sym_score = compute_symbolic_resonance(query_symbols, mem_symbols)
        return 0.4 * token_overlap + 0.4 * entity_overlap + 0.2 * sym_score

    def shutdown(self) -> None:
        self._palace = None
        logger.info("MemPalace adapter shut down")
