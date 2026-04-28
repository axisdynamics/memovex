"""Tests for MemoryStore — storage and index correctness."""

import pytest
from memovex.core.resonance_engine import MemoryStore, text_to_symbols
from memovex.core.types import Memory, MemoryType


def make_memory(i: int, confidence: float = 0.7, salience: float = 0.5) -> Memory:
    text = f"memoria numero {i} sobre el tema principal del test"
    return Memory(
        memory_id=f"test-{i}",
        text=text,
        memory_type=MemoryType.EPISODIC,
        base64_symbols=text_to_symbols(text),
        entities={f"Entidad{i}"},
        symbolic_keys={"memoria", "tema", "test"},
        tags={"test"},
        confidence=confidence,
        salience=salience,
    )


class TestMemoryStoreAdd:
    def test_add_and_get(self):
        store = MemoryStore()
        m = make_memory(1)
        store.add(m)
        assert store.get("test-1") is m

    def test_count(self):
        store = MemoryStore()
        for i in range(5):
            store.add(make_memory(i))
        assert store.count() == 5

    def test_by_entity_index(self):
        store = MemoryStore()
        m = make_memory(1)
        store.add(m)
        assert "test-1" in store.by_entity["entidad1"]

    def test_by_tag_index(self):
        store = MemoryStore()
        store.add(make_memory(1))
        assert "test-1" in store.by_tag["test"]

    def test_by_type_index(self):
        store = MemoryStore()
        store.add(make_memory(1))
        assert "test-1" in store.by_type["episodic"]


class TestMemoryStorePrune:
    def test_prune_reduces_count(self):
        store = MemoryStore()
        for i in range(10):
            store.add(make_memory(i))
        removed = store.prune(max_memories=5)
        assert store.count() == 5
        assert len(removed) == 5

    def test_prune_cleans_entity_index(self):
        store = MemoryStore()
        for i in range(6):
            store.add(make_memory(i))
        removed = store.prune(max_memories=3)
        for mid in removed:
            for ids in store.by_entity.values():
                assert mid not in ids

    def test_prune_cleans_tag_index(self):
        store = MemoryStore()
        for i in range(6):
            store.add(make_memory(i))
        removed = store.prune(max_memories=3)
        for mid in removed:
            for ids in store.by_tag.values():
                assert mid not in ids

    def test_prune_cleans_base64_index(self):
        store = MemoryStore()
        for i in range(6):
            store.add(make_memory(i))
        removed = store.prune(max_memories=3)
        for mid in removed:
            for ids in store.by_base64_symbol.values():
                assert mid not in ids

    def test_prune_cleans_type_index(self):
        store = MemoryStore()
        for i in range(6):
            store.add(make_memory(i))
        removed = store.prune(max_memories=3)
        for mid in removed:
            for ids in store.by_type.values():
                assert mid not in ids

    def test_prune_no_op_when_below_limit(self):
        store = MemoryStore()
        for i in range(3):
            store.add(make_memory(i))
        removed = store.prune(max_memories=5)
        assert removed == []
        assert store.count() == 3

    def test_prune_prefers_high_salience(self):
        store = MemoryStore()
        low = make_memory(0, salience=0.1)
        high = make_memory(1, salience=0.9)
        store.add(low)
        store.add(high)
        removed = store.prune(max_memories=1)
        assert "test-0" in removed
        assert store.get("test-1") is high


class TestMemoryStoreDecay:
    def test_decay_reduces_usage_score(self):
        store = MemoryStore()
        m = make_memory(1)
        m.usage_score = 0.5
        store.add(m)
        store.decay_all(amount=0.1)
        assert store.get("test-1").usage_score == pytest.approx(0.4)

    def test_decay_does_not_go_below_zero(self):
        store = MemoryStore()
        m = make_memory(1)
        m.usage_score = 0.0
        store.add(m)
        store.decay_all(amount=0.5)
        assert store.get("test-1").usage_score == 0.0


class TestMemoryStoreAccess:
    def test_register_access_increments_count(self):
        store = MemoryStore()
        store.add(make_memory(1))
        store.register_access("test-1")
        assert store.get("test-1").access_count == 1

    def test_register_access_updates_usage_score(self):
        store = MemoryStore()
        store.add(make_memory(1))
        store.register_access("test-1")
        assert store.get("test-1").usage_score > 0
