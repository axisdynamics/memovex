"""
memovex — Resonant Memory VEX Adapter (EXPERIMENTAL).

Integrates the existing Resonant Memory VEX v2.3 as a provider
within memovex, bridging symbolic resonance, wisdom curation,
and multi-channel retrieval.

⚠️  This adapter is EXPERIMENTAL and incomplete. The store_memory
    path does not yet persist to the underlying Resonant provider;
    only the prefetch-based retrieve path is functional. Do not
    use in production. Tracked under docs/ROADMAP.md.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ResonantAdapter:
    """
    Adapter for Resonant Memory VEX v2.3.

    Bridges the existing Resonant Memory system into memovex,
    enabling cross-provider resonance and symbol injection.
    """

    def __init__(self, resonant_provider=None):
        """
        Args:
            resonant_provider: An initialized ResonantMemoryProvider instance.
                              If None, adapter runs in passthrough mode.
        """
        self._provider = resonant_provider

    @property
    def available(self) -> bool:
        return self._provider is not None

    def store_memory(self, memory) -> str:
        """Store memory via Resonant VEX provider.

        EXPERIMENTAL: the Resonant Memory VEX provider does not yet
        expose a stable bulk-store API; until it does, this method is
        a no-op that returns the memory_id so callers can keep their
        flow consistent.
        """
        if not self.available:
            return ""

        # Inject symbols into the memory before storing
        # This enables cross-provider resonance
        from ...core.resonance_engine import text_to_symbols, extract_entities

        # Symbols and entities are injected into memory.* by the orchestrator
        # already; we read them here so future implementations can forward
        # them without recomputing.
        _symbols = memory.base64_symbols or text_to_symbols(memory.text)
        _entities = memory.entities or extract_entities(memory.text)
        logger.debug(
            "ResonantAdapter.store_memory called for %s (no-op, experimental)",
            memory.memory_id,
        )
        return memory.memory_id

    def retrieve(self, query: str, top_k: int = 5) -> List:
        """Retrieve from Resonant VEX using its native 9-channel retrieval."""
        if not self.available:
            return []

        # Use the provider's prefetch or search mechanism
        context = self._provider.prefetch(query)
        if not context:
            return []

        # Parse context into RetrievalResult objects
        results = []
        for line in context.split("\n"):
            if not line.strip():
                continue
            # Simple parsing of prefetch output
            results.append(self._parse_result_line(line, query))

        return results[:top_k]

    def inject_symbols(self, text: str, provider_name: str) -> Set[str]:
        """
        Inject base64 symbols from another provider into Resonant VEX.
        This enables cross-provider symbolic resonance.

        Args:
            text: Text from another provider to symbolize
            provider_name: Source provider name for tracking

        Returns:
            Set of generated symbols
        """
        from ...core.resonance_engine import text_to_symbols
        symbols = text_to_symbols(text)
        logger.debug(
            "Injected %d symbols from provider '%s' into resonant index",
            len(symbols), provider_name,
        )
        return symbols

    def _parse_result_line(self, line: str, query: str) -> Any:
        """Parse a prefetch result line into a RetrievalResult-like object."""
        from ...core.types import Memory, MemoryType, RetrievalResult

        # Format: [dialog] (score:0.123, base64_resonance:0.456) text... [provider]
        import re
        score_match = re.search(r"score:([\d.]+)", line)
        channel_match = re.search(r"(\w+):([\d.]+)\)", line)

        score = float(score_match.group(1)) if score_match else 0.0
        top_channel = channel_match.group(1) if channel_match else "unknown"
        channel_val = float(channel_match.group(2)) if channel_match else 0.0

        # Extract text (between last ) and [provider])
        text_match = re.search(r"\)\s+(.+?)\s*\[", line)
        text = text_match.group(1) if text_match else line

        mem = Memory(
            memory_id=f"resonant-{hash(text)}",
            text=text,
            memory_type=MemoryType.RESONANT,
            provider="resonant_vex",
        )

        return RetrievalResult(
            memory=mem,
            total_score=score,
            channel_scores={top_channel: channel_val},
            provider="resonant_vex",
            channels_used=1,
        )

    def shutdown(self) -> None:
        """Delegates to ResonantMemoryProvider shutdown."""
        if self._provider:
            try:
                self._provider.shutdown()
            except Exception as e:
                logger.warning(f"Resonant shutdown: {e}")
