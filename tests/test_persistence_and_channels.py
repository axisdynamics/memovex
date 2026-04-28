"""Tests for orchestrator-level snapshot/restore and channel renormalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memovex.core.memory_bank import MemoVexOrchestrator
from memovex.core.types import MemoryType
from memovex.core.resonance_engine import (
    DEFAULT_CHANNEL_WEIGHTS,
    MemoryStore,
    ResonanceEngine,
    text_to_symbols,
    tokenize,
)
from memovex.core.types import Memory


@pytest.fixture
def bank():
    b = MemoVexOrchestrator(agent_id="snaptest", embeddings_enabled=False)
    b.initialize()
    yield b
    b.shutdown()


# -----------------------------------------------------------------------------
# Snapshot / Restore
# -----------------------------------------------------------------------------

class TestSnapshotRoundTrip:
    def test_save_then_load_recovers_memories(self, tmp_path, bank):
        bank.store("memoria semantica importante",
                   memory_type=MemoryType.SEMANTIC,
                   confidence=0.9, salience=0.8)
        bank.store("episodio relevante de la sesion",
                   memory_type=MemoryType.EPISODIC,
                   confidence=0.7, salience=0.6)
        path = tmp_path / "snap.json"
        n = bank.save_snapshot(str(path))
        assert n == 2
        assert path.exists()

        # Fresh orchestrator, same snapshot
        b2 = MemoVexOrchestrator(agent_id="snaptest2", embeddings_enabled=False)
        b2.initialize()
        loaded = b2.load_snapshot(str(path))
        b2.shutdown()
        assert loaded == 2

    def test_load_idempotent(self, tmp_path, bank):
        bank.store("memoria duplicada test", confidence=0.7)
        path = tmp_path / "snap.json"
        bank.save_snapshot(str(path))

        b2 = MemoVexOrchestrator(agent_id="snaptest3", embeddings_enabled=False)
        b2.initialize()
        first = b2.load_snapshot(str(path))
        second = b2.load_snapshot(str(path))   # already present, must skip
        b2.shutdown()
        assert first == 1
        assert second == 0

    def test_load_missing_path_returns_zero(self, tmp_path, bank):
        n = bank.load_snapshot(str(tmp_path / "does-not-exist.json"))
        assert n == 0

    def test_load_corrupted_snapshot_returns_zero(self, tmp_path, bank):
        bad = tmp_path / "bad.json"
        bad.write_text("{ this is not valid json")
        n = bank.load_snapshot(str(bad))
        assert n == 0

    def test_snapshot_preserves_wisdom_level(self, tmp_path, bank):
        mid = bank.store("hecho de alta confianza permanente",
                         confidence=0.95, salience=0.9)
        bank.corroborate(mid)
        bank.corroborate(mid)
        # Should be at WISDOM now
        assert bank._wisdom_store.is_wisdom(mid)

        path = tmp_path / "snap.json"
        bank.save_snapshot(str(path))

        b2 = MemoVexOrchestrator(agent_id="snaptest4", embeddings_enabled=False)
        b2.initialize()
        b2.load_snapshot(str(path))
        try:
            assert b2._wisdom_store.is_wisdom(mid)
        finally:
            b2.shutdown()


# -----------------------------------------------------------------------------
# channels=  semantics  —  renormalize True/False both behave correctly
# -----------------------------------------------------------------------------

class TestChannelRenormalization:
    def _populate(self):
        store = MemoryStore()
        text = "Santiago es la capital de Chile en Sudamerica"
        m = Memory(
            memory_id="m1",
            text=text,
            memory_type=MemoryType.SEMANTIC,
            entities={"Santiago", "Chile"},
            base64_symbols=text_to_symbols(text),
            symbolic_keys=set(tokenize(text)),
            confidence=0.9,
            salience=0.8,
        )
        store.add(m)
        return store, m

    def test_default_no_subset_no_renorm(self):
        """Without restricting channels, total ≤ sum(weights) = 1.0."""
        store, _ = self._populate()
        engine = ResonanceEngine(store)
        results = engine.search("capital de Chile", top_k=1)
        assert results
        assert 0.0 < results[0].total_score <= 1.0

    def test_subset_renormalize_expands_score(self):
        """With renormalize=True, single-channel score reaches a similar
        magnitude to the full search instead of being capped by the
        small per-channel weight."""
        store, _ = self._populate()
        engine = ResonanceEngine(store)

        full = engine.search("capital de Chile", top_k=1)
        subset = engine.search("capital de Chile", top_k=1,
                               channels=["entity"], renormalize=True)
        assert full and subset
        # Renormalized subset must not silently shrink to entity_weight (0.13)
        assert subset[0].total_score > DEFAULT_CHANNEL_WEIGHTS["entity"] + 1e-6

    def test_subset_no_renormalize_capped_by_weight(self):
        """With renormalize=False, a single-channel score is capped by
        that channel's weight."""
        store, _ = self._populate()
        engine = ResonanceEngine(store)
        subset = engine.search(
            "capital de Chile", top_k=1,
            channels=["entity"], renormalize=False,
        )
        assert subset
        assert subset[0].total_score <= DEFAULT_CHANNEL_WEIGHTS["entity"] + 1e-9

    def test_unknown_channel_silently_ignored(self):
        """Misspelled or removed channels must not crash the search."""
        store, _ = self._populate()
        engine = ResonanceEngine(store)
        # base64_resonance was removed in v1.1 — must not raise
        results = engine.search("Chile", top_k=1, channels=["base64_resonance"])
        # Score may be 0 (no active channel matched) but no exception
        assert isinstance(results, list)

    def test_orchestrator_passes_renormalize_through(self, tmp_path):
        bank = MemoVexOrchestrator(agent_id="renorm", embeddings_enabled=False)
        bank.initialize()
        try:
            bank.store("Madrid es la capital de España",
                       confidence=0.9, salience=0.8)
            r1 = bank.retrieve("capital de España", top_k=1,
                                channels=["entity"], renormalize=True)
            r2 = bank.retrieve("capital de España", top_k=1,
                                channels=["entity"], renormalize=False)
            if r1 and r2:
                assert r1[0].total_score >= r2[0].total_score
        finally:
            bank.shutdown()


# -----------------------------------------------------------------------------
# Procedural channel  —  now part of DEFAULT_CHANNEL_WEIGHTS
# -----------------------------------------------------------------------------

class TestProceduralChannel:
    def test_procedural_in_weights(self):
        assert "procedural" in DEFAULT_CHANNEL_WEIGHTS
        assert DEFAULT_CHANNEL_WEIGHTS["procedural"] > 0

    def test_procedural_scores_how_query_against_procedural_tag(self, tmp_path):
        bank = MemoVexOrchestrator(agent_id="proc", embeddings_enabled=False)
        bank.initialize()
        try:
            bank.store(
                "para reiniciar el servicio ejecute systemctl restart memovex",
                memory_type=MemoryType.PROCEDURAL,
                tags={"procedure"},
                confidence=0.9,
            )
            results = bank.retrieve(
                "how do I restart the service?", top_k=2,
                channels=["procedural"], renormalize=True,
            )
            assert results
            assert results[0].total_score > 0
        finally:
            bank.shutdown()


# -----------------------------------------------------------------------------
# Public package surface
# -----------------------------------------------------------------------------

class TestPackageImports:
    def test_top_level_imports(self):
        from memovex import (
            MemoVexOrchestrator, MemoryType, ChannelType, Memory,
            RetrievalResult, DEFAULT_CHANNEL_WEIGHTS, MemoryStore,
            ResonanceEngine, WisdomLevel, WisdomStore, __version__,
        )
        assert __version__ == "1.1.0"
        assert callable(MemoVexOrchestrator)
        assert isinstance(DEFAULT_CHANNEL_WEIGHTS, dict)

    def test_api_app_importable(self):
        from memovex.api import app
        # FastAPI sets this attr
        assert app.title == "memovex API"

    def test_lazy_plugin_factories(self):
        import memovex
        for name in ("create_claude_memory", "create_hermes_memory",
                     "create_openclaw_memory", "HermesMemoryPlugin"):
            obj = getattr(memovex, name)
            assert callable(obj) or hasattr(obj, "__init__")


# -----------------------------------------------------------------------------
# Provider modules — import smoke & json availability
# -----------------------------------------------------------------------------

class TestProviderImports:
    def test_mem0_adapter_imports(self):
        from memovex.providers.mem0_adapter import Mem0Adapter
        adapter = Mem0Adapter()
        # Must be safe to call retrieve before initialize — no NameError on json
        out = adapter.retrieve("anything")
        assert out == []

    def test_memobase_adapter_imports(self):
        from memovex.providers.memobase_adapter import MemobaseAdapter
        MemobaseAdapter()

    def test_mempalace_adapter_imports(self):
        from memovex.providers.mempalace_adapter import MemPalaceAdapter
        MemPalaceAdapter()

    def test_reasoning_bank_imports(self):
        from memovex.providers.reasoning_bank import ReasoningBankAdapter
        ReasoningBankAdapter()

    def test_experimental_resonant_adapter_imports(self):
        from memovex.providers.experimental.resonant_adapter import ResonantAdapter
        ResonantAdapter()
