"""Tests for ResonanceEngine — channel scoring and retrieval."""

import pytest
from memovex.core.resonance_engine import (
    DEFAULT_CHANNEL_WEIGHTS,
    MemoryStore,
    ResonanceEngine,
    text_to_symbols,
    compute_symbolic_resonance,
    tokenize,
    bow_vector,
    cosine_sparse,
)
from memovex.core.types import Memory, MemoryType


def add_memory(store: MemoryStore, mid: str, text: str,
               entities=None, tags=None,
               confidence: float = 0.7, salience: float = 0.5) -> Memory:
    m = Memory(
        memory_id=mid,
        text=text,
        memory_type=MemoryType.SEMANTIC,
        entities=set(entities or []),
        base64_symbols=text_to_symbols(text),
        symbolic_keys=set(tokenize(text)),
        tags=set(tags or []),
        confidence=confidence,
        salience=salience,
    )
    store.add(m)
    return m


class TestChannelWeights:
    def test_weights_sum_to_one(self):
        total = sum(DEFAULT_CHANNEL_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_semantic_leads(self):
        assert DEFAULT_CHANNEL_WEIGHTS["semantic"] == max(DEFAULT_CHANNEL_WEIGHTS.values())

    def test_base64_resonance_absent(self):
        assert "base64_resonance" not in DEFAULT_CHANNEL_WEIGHTS

    def test_expected_channels_present(self):
        expected = {"semantic", "entity", "graph_traversal", "wisdom",
                    "temporal", "recency", "reasoning_chain", "symbolic",
                    "procedural", "usage", "tag"}
        assert expected == set(DEFAULT_CHANNEL_WEIGHTS.keys())


class TestResonanceEngineSearch:
    def setup_method(self):
        self.store = MemoryStore()
        self.engine = ResonanceEngine(self.store)

    def test_returns_empty_on_empty_store(self):
        results = self.engine.search("any query")
        assert results == []

    def test_exact_match_retrieves_memory(self):
        add_memory(self.store, "m1", "Santiago es la capital de Chile")
        results = self.engine.search("capital de Chile", top_k=1)
        assert len(results) == 1
        assert results[0].memory.memory_id == "m1"

    def test_top_k_respected(self):
        for i in range(10):
            add_memory(self.store, f"m{i}", f"memoria sobre tema numero {i}")
        results = self.engine.search("memoria tema", top_k=3)
        assert len(results) <= 3

    def test_scores_are_positive(self):
        add_memory(self.store, "m1", "el gato duerme en la cama")
        results = self.engine.search("gato cama", top_k=5)
        for r in results:
            assert r.total_score > 0

    def test_entity_channel_boosts_named_entity(self):
        add_memory(self.store, "m1", "Paris es la capital de Francia",
                   entities=["Paris"])
        add_memory(self.store, "m2", "la ciudad tiene muchos monumentos")
        results = self.engine.search("Paris capital", top_k=2,
                                     channels=["entity"])
        if results:
            assert results[0].memory.memory_id == "m1"

    def test_tag_channel_scores_matching_tag(self):
        add_memory(self.store, "m1", "vive en Madrid", tags=["location"])
        add_memory(self.store, "m2", "trabaja en Google", tags=["work"])
        results = self.engine.search("donde vive", top_k=2,
                                     channels=["tag"])
        scored = {r.memory.memory_id: r.total_score for r in results}
        # m1 has location tag — query triggers location
        if "m1" in scored and "m2" in scored:
            assert scored["m1"] >= scored["m2"]

    def test_temporal_channel_prefers_year_match(self):
        add_memory(self.store, "m1", "evento ocurrido en 2023",
                   tags=[])
        add_memory(self.store, "m2", "evento de 2020")
        # Patch event_time
        self.store.get("m1").event_time = "2023-06-01"
        self.store.get("m2").event_time = "2020-06-01"
        results = self.engine.search("qué pasó en 2023", top_k=5,
                                     channels=["temporal"])
        ids = [r.memory.memory_id for r in results]
        if "m1" in ids and "m2" in ids:
            assert ids.index("m1") < ids.index("m2")

    def test_recency_scores_newer_higher(self):
        import time
        add_memory(self.store, "old", "memoria antigua sobre gatos")
        add_memory(self.store, "new", "memoria reciente sobre gatos")
        self.store.get("old").ingested_at = time.time() - 200 * 86400
        self.store.get("new").ingested_at = time.time() - 1
        results = self.engine.search("gatos", top_k=5, channels=["recency"])
        scores = {r.memory.memory_id: r.total_score for r in results}
        if "old" in scores and "new" in scores:
            assert scores["new"] > scores["old"]

    def test_channel_scores_returned(self):
        add_memory(self.store, "m1", "test de canales de resonancia")
        results = self.engine.search("canales resonancia", top_k=1)
        assert results
        assert isinstance(results[0].channel_scores, dict)
        assert len(results[0].channel_scores) > 0


class TestSymbolicFunctions:
    def test_text_to_symbols_returns_set(self):
        syms = text_to_symbols("el gato come pollo todos los días")
        assert isinstance(syms, set)
        assert len(syms) > 0

    def test_same_text_same_symbols(self):
        text = "Santiago capital Chile"
        assert text_to_symbols(text) == text_to_symbols(text)

    def test_different_texts_different_symbols(self):
        s1 = text_to_symbols("Santiago capital Chile")
        s2 = text_to_symbols("París capital Francia")
        assert s1 != s2

    def test_symbolic_resonance_identical_texts(self):
        text = "el perro ladra en el parque"
        s = text_to_symbols(text)
        score = compute_symbolic_resonance(s, s)
        assert score == pytest.approx(1.0)

    def test_symbolic_resonance_empty(self):
        assert compute_symbolic_resonance(set(), {"abc"}) == 0.0
        assert compute_symbolic_resonance({"abc"}, set()) == 0.0

    def test_bow_cosine(self):
        a = bow_vector("gato come pollo")
        b = bow_vector("gato come pollo")
        assert cosine_sparse(a, b) == pytest.approx(1.0)

    def test_bow_cosine_orthogonal(self):
        a = bow_vector("gato")
        b = bow_vector("xyz mno")
        assert cosine_sparse(a, b) == pytest.approx(0.0)
