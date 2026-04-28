"""
memovex — Homeostasis Manager.

Automatic decay, pruning, and memory health management.
Inspired by Resonant Memory VEX v2.3 homeostasis.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .resonance_engine import MemoryStore

logger = logging.getLogger(__name__)

# Config
HOMEOSTASIS_INTERVAL = 300          # 5 minutes between cycles
HOMEOSTASIS_DECAY_AMOUNT = 0.03     # Per-cycle usage decay
MEMORYSTORE_MAX_MEMORIES = 5000     # Max before aggressive pruning
WISDOM_PROTECTION = True            # Don't prune WISDOM-level memories
LOW_CONFIDENCE_PRUNE = 0.2         # Memories below this confidence are pruned first


class HomeostasisManager:
    """Background manager for memory decay and pruning."""

    def __init__(self, memory_store: MemoryStore):
        self._store = memory_store
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()
        self._decay_amount = HOMEOSTASIS_DECAY_AMOUNT
        self._max_memories = MEMORYSTORE_MAX_MEMORIES

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._homeostasis_loop,
            daemon=True,
            name="memovex-homeostasis",
        )
        self._thread.start()
        logger.info(
            "Homeostasis started (interval=%ss, decay=%.2f, max=%d)",
            HOMEOSTASIS_INTERVAL, self._decay_amount, self._max_memories,
        )

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def run_cycle_now(self) -> None:
        """Run a single homeostasis cycle (for testing or manual trigger)."""
        self._decay()
        self._prune()

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _homeostasis_loop(self) -> None:
        while self._running:
            # Sleep is interruptible: stop() sets the event and we exit immediately
            if self._stop_event.wait(timeout=HOMEOSTASIS_INTERVAL):
                break
            try:
                before = self._store.count()
                self._decay()
                removed = self._prune()
                if removed:
                    logger.info(
                        "Homeostasis cycle: pruned %d memories (%d -> %d)",
                        len(removed), before, self._store.count(),
                    )
            except Exception as e:
                logger.warning("Homeostasis cycle error: %s", e)

    def _decay(self) -> None:
        self._store.decay_all(amount=self._decay_amount)

    def _prune(self) -> list:
        if self._store.count() <= self._max_memories:
            return []

        # First pass: prune low-confidence memories
        removed_low = []
        if self._store.count() > self._max_memories * 0.8:
            for mem in list(self._store.all()):
                if mem.confidence < LOW_CONFIDENCE_PRUNE:
                    self._store._remove_from_indices(mem.memory_id)
                    removed_low.append(mem.memory_id)

        # Second pass: score-based pruning
        removed = self._store.prune(max_memories=self._max_memories)
        return removed_low + removed
