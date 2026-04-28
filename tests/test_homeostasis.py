"""Tests for HomeostasisManager — decay and pruning."""

import time
import pytest
from memovex.core.resonance_engine import MemoryStore, text_to_symbols
from memovex.core.homeostasis import HomeostasisManager
from memovex.core.types import Memory, MemoryType


def make_memory(mid: str, confidence: float = 0.7,
                salience: float = 0.5, usage: float = 0.5) -> Memory:
    text = f"memoria de test identificador {mid}"
    m = Memory(
        memory_id=mid,
        text=text,
        memory_type=MemoryType.EPISODIC,
        base64_symbols=text_to_symbols(text),
        confidence=confidence,
        salience=salience,
        usage_score=usage,
    )
    return m


class TestHomeostasisDecay:
    def test_decay_reduces_usage(self):
        store = MemoryStore()
        m = make_memory("m1", usage=0.5)
        store.add(m)
        mgr = HomeostasisManager(store)
        mgr._decay()
        assert store.get("m1").usage_score < 0.5

    def test_decay_does_not_go_negative(self):
        store = MemoryStore()
        m = make_memory("m1", usage=0.0)
        store.add(m)
        mgr = HomeostasisManager(store)
        mgr._decay()
        assert store.get("m1").usage_score == 0.0


class TestHomeostasisPrune:
    def test_prune_removes_excess(self):
        store = MemoryStore()
        for i in range(10):
            store.add(make_memory(f"m{i}", confidence=0.7, salience=0.5))
        mgr = HomeostasisManager(store)
        mgr._max_memories = 5
        mgr._prune()
        assert store.count() <= 5

    def test_prune_clears_low_confidence_first(self):
        store = MemoryStore()
        store.add(make_memory("low", confidence=0.1, salience=0.1))
        for i in range(5):
            store.add(make_memory(f"high-{i}", confidence=0.9, salience=0.9))
        mgr = HomeostasisManager(store)
        mgr._max_memories = 5
        mgr._prune()
        assert store.get("low") is None

    def test_prune_keeps_high_salience(self):
        store = MemoryStore()
        store.add(make_memory("keeper", confidence=0.9, salience=0.99))
        for i in range(5):
            store.add(make_memory(f"low-{i}", confidence=0.3, salience=0.1))
        mgr = HomeostasisManager(store)
        mgr._max_memories = 1
        mgr._prune()
        assert store.get("keeper") is not None

    def test_indices_clean_after_prune(self):
        store = MemoryStore()
        for i in range(6):
            store.add(make_memory(f"m{i}"))
        mgr = HomeostasisManager(store)
        mgr._max_memories = 3
        removed = mgr._prune()
        for mid in removed:
            for ids in store.by_entity.values():
                assert mid not in ids


class TestHomeostasisCycle:
    def test_run_cycle_now_does_not_raise(self):
        store = MemoryStore()
        for i in range(3):
            store.add(make_memory(f"m{i}"))
        mgr = HomeostasisManager(store)
        mgr.run_cycle_now()

    def test_start_stop(self):
        store = MemoryStore()
        mgr = HomeostasisManager(store)
        mgr.start()
        assert mgr._running is True
        mgr.stop()
        assert mgr._running is False
