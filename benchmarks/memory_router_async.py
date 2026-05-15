#!/usr/bin/env python3
"""Benchmark MemoryRouter async hot path with sandbox-only fixtures.

No live MemoVex agents or external services are used. The semantic client is a
small in-process fake that simulates L2 latency and verifies cache warming.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from memovex.core.memory_router import LocalMemoryIndex, MemoryRouter  # noqa: E402


class SimulatedSemanticClient:
    def __init__(self, delay_s: float = 0.02):
        self.calls = 0
        self.delay_s = delay_s

    def retrieve(self, query: str, top_k: int = 5, channels=None):
        self.calls += 1
        time.sleep(self.delay_s)
        return [{"text": f"semantic sandbox result for {query}", "route": "semantic"}]


def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    frac = k - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def stats(values):
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def make_entries(n=500):
    topics = [
        "memovex router local fast semantic cache latency coherence",
        "vex constellation chronicle network peer autoresponder bridge",
        "sustrato provider failover model config fallback chain",
        "hermes agent memory provider plugin config sandbox",
        "axis dynamics launch campaign institutional email promotion",
        "qdrant vector semantic retrieval embeddings namespace",
        "redis cache prefetch recency ttl warm route",
        "memobase local text file memory user profile delimiter",
        "benchmark speed p95 median rss coherence synthetic fixture",
        "bio digital witness silence cellular consciousness",
    ]
    return [
        f"SANDBOX_ONLY router-bench-{i:03d}: {topics[i % len(topics)]}; token_unico_{i:03d}; no usar en agentes reales."
        for i in range(n)
    ]


def main():
    random.seed(123)
    entries = make_entries(500)
    local = LocalMemoryIndex.from_texts(entries, source="sandbox")
    semantic = SimulatedSemanticClient(delay_s=0.02)
    router = MemoryRouter(
        local,
        semantic,
        threshold=0.55,
        cache_ttl_s=120,
        background_workers=1,
        background_start_delay_s=0.25,
    )

    known_queries = [f"token_unico_{i:03d}" for i in range(300)]
    unknown_queries = [f"unknown_async_query_{i} no_local_match_{i}" for i in range(100)]
    first_pass = known_queries + unknown_queries
    random.shuffle(first_pass)

    first_latencies = []
    first_routes = {}
    for query in first_pass:
        out = router.retrieve(query, mode="local_then_async_semantic")
        first_latencies.append(out["latency_ms"])
        first_routes[out["route"]] = first_routes.get(out["route"], 0) + 1

    drained = router.wait_background(timeout=40)
    background = router.background_stats()

    second_latencies = []
    second_routes = {}
    for query in unknown_queries:
        out = router.retrieve(query, mode="local_then_async_semantic")
        second_latencies.append(out["latency_ms"])
        second_routes[out["route"]] = second_routes.get(out["route"], 0) + 1

    report = {
        "mode": "local_then_async_semantic",
        "fixture_entries": len(entries),
        "first_pass_queries": len(first_pass),
        "first_pass_latency_ms": stats(first_latencies),
        "first_pass_routes": first_routes,
        "background_drained": drained,
        "background_stats": background,
        "second_pass_unknown_queries": len(unknown_queries),
        "second_pass_latency_ms": stats(second_latencies),
        "second_pass_routes": second_routes,
        "semantic_calls": semantic.calls,
        "targets": {
            "first_pass_p95_lt_2ms": percentile(first_latencies, 95) < 2,
            "background_drained": drained,
            "second_pass_unknown_median_lt_5ms": statistics.median(second_latencies) < 5,
            "unknown_cache_fill_gte_95pct": second_routes.get("semantic_cache", 0) / len(unknown_queries) >= 0.95,
        },
    }
    path = RESULTS / "memory_router_async.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report={path}")
    print(
        "first_pass median/p95 "
        f"{report['first_pass_latency_ms']['median']:.4f}/"
        f"{report['first_pass_latency_ms']['p95']:.4f} ms routes={first_routes}"
    )
    print(
        "second_pass median/p95 "
        f"{report['second_pass_latency_ms']['median']:.4f}/"
        f"{report['second_pass_latency_ms']['p95']:.4f} ms routes={second_routes}"
    )
    print(f"background={background} targets={report['targets']}")
    router.shutdown()


if __name__ == "__main__":
    main()
