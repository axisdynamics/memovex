"""Tests for clean MemoryRouter implementation."""

import time

from memovex.core.memory_router import LocalMemoryEntry, LocalMemoryIndex, MemoryRouter


class SlowSemanticClient:
    def __init__(self, delay_s=0.02):
        self.calls = 0
        self.delay_s = delay_s

    def retrieve(self, query, top_k=5, channels=None):
        self.calls += 1
        time.sleep(self.delay_s)
        return [{"text": f"semantic result for {query}", "route": "semantic"}]


def make_index():
    return LocalMemoryIndex.from_texts(
        [
            "Sustrato provider failover config yaml",
            "VEX Constellation Shared Chronicle network map",
            "User has institutional email for AxisDynamics promotion",
        ],
        source="fixture",
    )


def test_local_index_returns_confident_match():
    index = make_index()
    results = index.search("Sustrato provider failover", top_k=1)
    assert results
    assert results[0]["source"] == "fixture"
    assert results[0]["score"] >= 0.55


def test_fast_mode_never_calls_semantic():
    semantic = SlowSemanticClient()
    router = MemoryRouter(make_index(), semantic, threshold=0.55)
    out = router.retrieve("unknown query with no local match", mode="fast")
    assert out["route"] == "local"
    assert semantic.calls == 0
    router.shutdown()


def test_balanced_mode_uses_semantic_when_local_is_weak():
    semantic = SlowSemanticClient(delay_s=0)
    router = MemoryRouter(make_index(), semantic, threshold=0.55)
    out = router.retrieve("unknown query with no local match", mode="balanced")
    assert out["route"] == "semantic"
    assert semantic.calls == 1
    router.shutdown()


def test_local_then_async_semantic_returns_before_slow_semantic_finishes():
    semantic = SlowSemanticClient(delay_s=0.05)
    router = MemoryRouter(make_index(), semantic, threshold=0.55, background_workers=1)
    out = router.retrieve("unknown query with no local match", mode="local_then_async_semantic")
    assert out["route"] == "local_background_scheduled"
    assert out["latency_ms"] < 25
    assert out["background_pending"] is True
    assert router.wait_background(timeout=1)
    stats = router.background_stats()
    assert stats["submitted"] == 1
    assert stats["completed"] == 1
    assert stats["errors"] == 0
    router.shutdown()


def test_local_then_async_semantic_uses_cache_after_background_fill():
    semantic = SlowSemanticClient(delay_s=0.01)
    router = MemoryRouter(make_index(), semantic, threshold=0.55, cache_ttl_s=60)
    first = router.retrieve("unknown query with no local match", mode="local_then_async_semantic")
    assert first["route"] == "local_background_scheduled"
    assert router.wait_background(timeout=1)
    second = router.retrieve("unknown query with no local match", mode="local_then_async_semantic")
    assert second["route"] == "semantic_cache"
    assert semantic.calls == 1
    router.shutdown()


def test_local_then_async_semantic_deduplicates_pending_query():
    semantic = SlowSemanticClient(delay_s=0.05)
    router = MemoryRouter(make_index(), semantic, threshold=0.55, cache_ttl_s=60)
    first = router.retrieve("unknown query with no local match", mode="local_then_async_semantic")
    second = router.retrieve("unknown query with no local match", mode="local_then_async_semantic")
    assert first["route"] == "local_background_scheduled"
    assert second["route"] in {"local_background_pending", "semantic_cache"}
    assert router.wait_background(timeout=1)
    assert semantic.calls == 1
    router.shutdown()


def test_local_memory_entry_preserves_stable_hash():
    a = LocalMemoryEntry.from_text("same text", source="fixture")
    b = LocalMemoryEntry.from_text("same text", source="fixture")
    assert a.hash == b.hash


def test_router_public_api_exports_are_available():
    import memovex

    assert memovex.LocalMemoryIndex is LocalMemoryIndex
    assert memovex.MemoryRouter is MemoryRouter
