"""
memovex — WisdomStore.

Four-level curation pipeline: RAW → PROCESSED → CURATED → WISDOM.
Memories promoted to WISDOM are protected from pruning and receive a
score boost via the 'wisdom' retrieval channel.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class WisdomLevel(str, Enum):
    RAW = "raw"
    PROCESSED = "processed"
    CURATED = "curated"
    WISDOM = "wisdom"


# Thresholds for automatic promotion
_PROMOTE_TO_PROCESSED = 0.40   # confidence
_PROMOTE_TO_CURATED = 0.60    # confidence + corroboration
_PROMOTE_TO_WISDOM = 0.80     # confidence + high salience + evidence count


@dataclass
class WisdomEntry:
    memory_id: str
    level: WisdomLevel = WisdomLevel.RAW
    confidence: float = 0.0
    salience: float = 0.0
    evidence_count: int = 0       # number of times corroborated
    promoted_at: float = field(default_factory=time.time)
    notes: str = ""


class WisdomStore:
    """Tracks curation level for all memories and exposes wisdom-boosted retrieval."""

    def __init__(self) -> None:
        self._entries: Dict[str, WisdomEntry] = {}

    # ------------------------------------------------------------------
    # Registration & promotion
    # ------------------------------------------------------------------

    def register(self, memory_id: str, confidence: float = 0.0,
                 salience: float = 0.0) -> WisdomEntry:
        entry = WisdomEntry(
            memory_id=memory_id,
            confidence=confidence,
            salience=salience,
        )
        self._entries[memory_id] = entry
        self._auto_promote(entry)
        return entry

    def corroborate(self, memory_id: str, delta_confidence: float = 0.05) -> None:
        """Record an additional piece of evidence for a memory."""
        entry = self._entries.get(memory_id)
        if entry is None:
            return
        entry.evidence_count += 1
        entry.confidence = min(1.0, entry.confidence + delta_confidence)
        self._auto_promote(entry)

    def promote(self, memory_id: str, level: WisdomLevel,
                notes: str = "") -> None:
        """Manually promote a memory to a specific level."""
        entry = self._entries.get(memory_id)
        if entry is None:
            logger.warning("promote: unknown memory_id %s", memory_id)
            return
        entry.level = level
        entry.promoted_at = time.time()
        entry.notes = notes
        logger.info("Memory %s promoted to %s", memory_id, level.value)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_level(self, memory_id: str) -> Optional[WisdomLevel]:
        entry = self._entries.get(memory_id)
        return entry.level if entry else None

    def is_wisdom(self, memory_id: str) -> bool:
        return self.get_level(memory_id) == WisdomLevel.WISDOM

    def wisdom_score(self, memory_id: str) -> float:
        """Return a score [0, 1] reflecting curation level for channel weighting."""
        level = self.get_level(memory_id)
        if level is None:
            return 0.0
        return {
            WisdomLevel.RAW:       0.0,
            WisdomLevel.PROCESSED: 0.3,
            WisdomLevel.CURATED:   0.6,
            WisdomLevel.WISDOM:    1.0,
        }[level]

    def list_wisdom(self) -> List[WisdomEntry]:
        return [e for e in self._entries.values() if e.level == WisdomLevel.WISDOM]

    def count(self) -> Dict[str, int]:
        counts: Dict[str, int] = {lvl.value: 0 for lvl in WisdomLevel}
        for e in self._entries.values():
            counts[e.level.value] += 1
        return counts

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _auto_promote(self, entry: WisdomEntry) -> None:
        old_level = entry.level
        if (entry.confidence >= _PROMOTE_TO_WISDOM
                and entry.salience >= 0.70
                and entry.evidence_count >= 2):
            entry.level = WisdomLevel.WISDOM
        elif (entry.confidence >= _PROMOTE_TO_CURATED
              and entry.evidence_count >= 1):
            entry.level = WisdomLevel.CURATED
        elif entry.confidence >= _PROMOTE_TO_PROCESSED:
            entry.level = WisdomLevel.PROCESSED

        if entry.level != old_level:
            entry.promoted_at = time.time()
            logger.debug(
                "Auto-promoted %s: %s → %s",
                entry.memory_id, old_level.value, entry.level.value,
            )
