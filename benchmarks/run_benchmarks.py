"""
memovex — Benchmark Runner (v1.1).

Evaluates MemoryBank performance on HotpotQA and MuSiQue samples
and compares against the Resonant VEX v2.3 baseline.

Usage:
    python benchmarks/run_benchmarks.py [--no-embeddings]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from memovex.core.memory_bank import MemoVexOrchestrator
from memovex.core.resonance_engine import DEFAULT_CHANNEL_WEIGHTS
from memovex.core.types import MemoryType

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark datasets
# ---------------------------------------------------------------------------

HOTPOTQA_SAMPLE: List[Dict[str, Any]] = [
    {"question": "What is the capital of France?",
     "answer": "Paris", "entities": ["France"]},
    {"question": "Which country is Tokyo the capital of?",
     "answer": "Japan", "entities": ["Tokyo"]},
    {"question": "What language do people speak in Brazil?",
     "answer": "Portuguese", "entities": ["Brazil"]},
    {"question": "What is the largest planet in the solar system?",
     "answer": "Jupiter", "entities": ["Jupiter"]},
]

MUSIQUE_SAMPLE: List[Dict[str, Any]] = [
    {
        "question": "In which continent is the country whose capital is Paris?",
        "answer": "Europe",
        "hops": [
            {"source": "Paris", "via": "capital_of", "target": "France"},
            {"source": "France", "via": "located_in", "target": "Europe"},
        ],
        "entities": ["Paris"],
    },
    {
        "question": "What is the official language of the country where Tokyo is located?",
        "answer": "Japanese",
        "hops": [
            {"source": "Tokyo", "via": "capital_of", "target": "Japan"},
            {"source": "Japan", "via": "official_language", "target": "Japanese"},
        ],
        "entities": ["Tokyo"],
    },
    {
        "question": "Which ocean borders the country where Buenos Aires is the capital?",
        "answer": "Atlantic",
        "hops": [
            {"source": "Buenos Aires", "via": "capital_of", "target": "Argentina"},
            {"source": "Argentina", "via": "borders", "target": "Atlantic Ocean"},
        ],
        "entities": ["Buenos Aires"],
    },
]

# Baseline metrics from Resonant VEX v2.3
BASELINE = {"r1": 0.384, "r10": 0.485}


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------

def seed_data(bank: MemoVexOrchestrator) -> None:
    for item in HOTPOTQA_SAMPLE:
        bank.store(
            text=f"{item['question']} → {item['answer']}",
            memory_type=MemoryType.SEMANTIC,
            entities=set(item.get("entities", [])),
            confidence=0.9, salience=0.8,
        )

    for item in MUSIQUE_SAMPLE:
        bank.store(
            text=f"{item['question']} → {item['answer']}",
            memory_type=MemoryType.SEMANTIC,
            entities=set(item.get("entities", [])),
            confidence=0.9, salience=0.8,
        )
        bank.store_reasoning_chain(
            text=f"Reasoning: {item['question']} → {item['answer']}",
            hops=item["hops"],
            entities=set(item.get("entities", [])),
            confidence=0.85,
        )
        # Corroborate to test wisdom promotion
        mid_list = [
            mid for mid, m in bank._memory_store.memories.items()
            if item["answer"] in m.text
        ]
        for mid in mid_list:
            bank.corroborate(mid)
            bank.corroborate(mid)


def evaluate_r1(bank: MemoVexOrchestrator,
                items: List[Dict]) -> float:
    correct = 0
    for item in items:
        results = bank.retrieve(item["question"], top_k=1)
        if results and item["answer"].lower() in results[0].memory.text.lower():
            correct += 1
            logger.info("  ✓ %s", item["question"])
        else:
            got = results[0].memory.text[:60] if results else "(no result)"
            logger.info("  ✗ %s → got: %s", item["question"], got)
    return correct / max(len(items), 1)


def evaluate_r10(bank: MemoVexOrchestrator,
                 items: List[Dict]) -> float:
    correct = 0
    for item in items:
        results = bank.retrieve(item["question"], top_k=10)
        texts = [r.memory.text.lower() for r in results]
        if any(item["answer"].lower() in t for t in texts):
            correct += 1
            logger.info("  ✓ %s", item["question"])
        else:
            logger.info("  ✗ %s", item["question"])
    return correct / max(len(items), 1)


def latency_test(bank: MemoVexOrchestrator,
                 items: List[Dict]) -> float:
    latencies = []
    for item in items:
        t0 = time.time()
        bank.retrieve(item["question"], top_k=5)
        latencies.append(time.time() - t0)
    return sum(latencies) / max(len(latencies), 1)


def run_benchmark(embeddings_enabled: bool = False) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("memovex Benchmark Suite (v1.1)")
    logger.info("Channels: %d  Weights: %s",
                len(DEFAULT_CHANNEL_WEIGHTS),
                {k: round(v, 2) for k, v in DEFAULT_CHANNEL_WEIGHTS.items()})
    logger.info("=" * 60)

    bank = MemoVexOrchestrator(embeddings_enabled=embeddings_enabled)
    bank.initialize()

    logger.info("\n--- Seeding data ---")
    seed_data(bank)
    logger.info("Memories: %d | Wisdom: %s",
                bank._memory_store.count(), bank.wisdom_summary())
    logger.info("Graph: %s", bank.graph_stats())

    logger.info("\n--- R@1: Single-hop (HotpotQA) ---")
    r1 = evaluate_r1(bank, HOTPOTQA_SAMPLE)
    logger.info("  R@1 = %.3f", r1)

    logger.info("\n--- R@10: Multi-hop (MuSiQue) ---")
    r10 = evaluate_r10(bank, MUSIQUE_SAMPLE)
    logger.info("  R@10 = %.3f", r10)

    logger.info("\n--- Latency ---")
    avg_ms = latency_test(bank, HOTPOTQA_SAMPLE + MUSIQUE_SAMPLE) * 1000
    logger.info("  Average latency = %.1f ms", avg_ms)

    logger.info("\n--- Graph Traversal ---")
    for item in MUSIQUE_SAMPLE:
        for hop in item.get("hops", []):
            entity = hop.get("source", "")
            if entity:
                chains = bank.get_reasoning_chains(entity)
                logger.info("  Entity '%s': %d chains", entity, len(chains))
                traversed = bank.traverse_graph(entity, max_depth=2)
                logger.info("    Traversal depth-2: %d memories", len(traversed))

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    summary = {
        "r1_accuracy": round(r1, 3),
        "r10_accuracy": round(r10, 3),
        "avg_latency_ms": round(avg_ms, 1),
        "memories_total": bank._memory_store.count(),
        "wisdom_summary": bank.wisdom_summary(),
        "graph_stats": bank.graph_stats(),
        "channels_active": len(DEFAULT_CHANNEL_WEIGHTS),
        "embeddings": embeddings_enabled,
    }
    logger.info(json.dumps(summary, indent=2))

    logger.info("\n--- vs Baseline (Resonant VEX v2.3) ---")
    logger.info("  Metric     | Baseline | MemoryBank | Delta")
    logger.info("  ---------- | -------- | ---------- | -----")
    logger.info("  R@1        | %.3f    | %-8.3f   | %+.3f",
                BASELINE["r1"], r1, r1 - BASELINE["r1"])
    logger.info("  R@10       | %.3f    | %-8.3f   | %+.3f",
                BASELINE["r10"], r10, r10 - BASELINE["r10"])

    bank.shutdown()
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", action="store_true",
                   help="Enable sentence-transformers embeddings")
    args = p.parse_args()
    run_benchmark(embeddings_enabled=args.embeddings)


if __name__ == "__main__":
    main()
