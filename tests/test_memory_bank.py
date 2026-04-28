"""Tests for MemoVexOrchestrator — integration tests."""

import pytest
from memovex.core.memory_bank import MemoVexOrchestrator
from memovex.core.types import MemoryType


@pytest.fixture
def bank():
    b = MemoVexOrchestrator(embeddings_enabled=False)
    b.initialize()
    yield b
    b.shutdown()


class TestOrchestratorStore:
    def test_store_returns_id(self, bank):
        mid = bank.store("Santiago es la capital de Chile")
        assert isinstance(mid, str)
        assert len(mid) > 0

    def test_store_increases_count(self, bank):
        before = bank._memory_store.count()
        bank.store("nueva memoria de test")
        assert bank._memory_store.count() == before + 1

    def test_store_registers_in_wisdom(self, bank):
        mid = bank.store("memoria con alta confianza", confidence=0.9, salience=0.8)
        assert bank._wisdom_store.get_level(mid) is not None

    def test_store_reasoning_chain(self, bank):
        mid = bank.store_reasoning_chain(
            text="Paris es capital de Francia que está en Europa",
            hops=[
                {"source": "Paris", "via": "capital_de", "target": "Francia"},
                {"source": "Francia", "via": "ubicada_en", "target": "Europa"},
            ],
            confidence=0.9,
        )
        assert mid
        assert bank._memory_store.get(mid).memory_type == MemoryType.REASONING


class TestOrchestratorRetrieve:
    def test_retrieve_returns_list(self, bank):
        bank.store("el perro ladra en el parque")
        results = bank.retrieve("perro parque")
        assert isinstance(results, list)

    def test_retrieve_finds_relevant_memory(self, bank):
        bank.store("Santiago es la capital de Chile", confidence=0.9, salience=0.8)
        bank.store("Tokio es la capital de Japón", confidence=0.9, salience=0.8)
        results = bank.retrieve("capital de Chile", top_k=3)
        texts = [r.memory.text for r in results]
        assert any("Chile" in t for t in texts)

    def test_retrieve_top_k_respected(self, bank):
        for i in range(10):
            bank.store(f"memoria de tema general numero {i}")
        results = bank.retrieve("tema general", top_k=3)
        assert len(results) <= 3

    def test_prefetch_returns_string(self, bank):
        bank.store("el gato duerme todo el día")
        result = bank.prefetch("gato dormir")
        assert isinstance(result, str)

    def test_prefetch_empty_on_no_match(self, bank):
        result = bank.prefetch("xyzzy no existe en ninguna memoria")
        assert isinstance(result, str)


class TestOrchestratorWisdom:
    def test_corroborate_promotes_memory(self, bank):
        mid = bank.store("dato importante", confidence=0.85, salience=0.75)
        bank.corroborate(mid)
        bank.corroborate(mid)
        assert bank._wisdom_store.is_wisdom(mid)

    def test_promote_to_wisdom_manual(self, bank):
        mid = bank.store("dato curado manualmente", confidence=0.1)
        bank.promote_to_wisdom(mid, notes="revisado por experto")
        assert bank._wisdom_store.is_wisdom(mid)

    def test_wisdom_summary_returns_dict(self, bank):
        summary = bank.wisdom_summary()
        assert "wisdom" in summary
        assert "raw" in summary


class TestOrchestratorGraph:
    def test_reasoning_chain_builds_graph(self, bank):
        bank.store_reasoning_chain(
            text="Madrid es capital de España",
            hops=[{"source": "Madrid", "via": "capital_de", "target": "España"}],
            confidence=0.9,
        )
        chains = bank.get_reasoning_chains("Madrid")
        assert len(chains) >= 1

    def test_traverse_graph_follows_hops(self, bank):
        bank.store_reasoning_chain(
            text="Paris en Francia, Francia en Europa",
            hops=[
                {"source": "Paris", "via": "en", "target": "Francia"},
                {"source": "Francia", "via": "en", "target": "Europa"},
            ],
            confidence=0.9,
        )
        results = bank.traverse_graph("Paris", max_depth=2)
        entities_found = {n for m in results for n in m.graph_nodes}
        assert "Francia" in entities_found or "Paris" in entities_found

    def test_graph_stats_returns_dict(self, bank):
        stats = bank.graph_stats()
        assert "nodes" in stats
        assert "edges" in stats


class TestOrchestratorStats:
    def test_stats_returns_expected_keys(self, bank):
        s = bank.stats()
        assert "memories" in s
        assert "wisdom" in s
        assert "graph" in s
        assert "providers" in s

    def test_provider_registration(self, bank):
        class FakeProvider:
            def store_memory(self, m): return ""
            def retrieve(self, q, top_k=5): return []
            def shutdown(self): pass

        bank.register_provider("fake", FakeProvider())
        assert "fake" in bank.stats()["providers"]
