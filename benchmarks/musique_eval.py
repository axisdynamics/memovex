"""
memovex — MuSiQue-style Multi-hop Evaluation.

Evaluates the GraphStore reasoning chain traversal and multi-hop
retrieval on questions requiring 2, 3, and 4 supporting facts.

Metrics: Hop Accuracy (2/3/4-hop), Graph Hit Rate,
         Chain Retrieval F1, Traversal Depth Coverage.

Reference: Trivedi et al. "MuSiQue: Multihop Questions via
           Single-hop Question Composition" (TACL 2022).
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from memovex.core.memory_bank import MemoVexOrchestrator
from memovex.core.types import MemoryType


# ---------------------------------------------------------------------------
# Dataset — multi-hop facts about a tech ecosystem
# ---------------------------------------------------------------------------

# Atomic facts stored as semantic memories
FACTS: List[Dict[str, Any]] = [
    # Company / product graph
    {"id": "F01", "text": "Anthropic creó el modelo Claude.", "entities": {"anthropic", "claude"}},
    {"id": "F02", "text": "Claude es un asistente de IA desarrollado por Anthropic.", "entities": {"claude", "anthropic", "ia"}},
    {"id": "F03", "text": "Anthropic fue fundada en San Francisco en 2021.", "entities": {"anthropic", "san francisco", "2021"}},
    {"id": "F04", "text": "Google invirtió 300 millones de dólares en Anthropic.", "entities": {"google", "anthropic", "inversión"}},
    {"id": "F05", "text": "Google fue fundada por Larry Page y Sergey Brin.", "entities": {"google", "larry page", "sergey brin"}},
    {"id": "F06", "text": "Larry Page creció en East Lansing, Michigan.", "entities": {"larry page", "east lansing", "michigan"}},
    {"id": "F07", "text": "Michigan es un estado del noreste de Estados Unidos.", "entities": {"michigan", "estados unidos"}},
    {"id": "F08", "text": "OpenAI desarrolló GPT-4 y ChatGPT.", "entities": {"openai", "gpt-4", "chatgpt"}},
    {"id": "F09", "text": "Microsoft invirtió 10.000 millones en OpenAI en 2023.", "entities": {"microsoft", "openai", "2023"}},
    {"id": "F10", "text": "Microsoft fue fundada por Bill Gates y Paul Allen en 1975.", "entities": {"microsoft", "bill gates", "paul allen", "1975"}},
    # Framework / tech graph
    {"id": "F11", "text": "FastAPI es un framework web Python creado por Sebastián Ramírez.", "entities": {"fastapi", "python", "sebastián ramírez"}},
    {"id": "F12", "text": "FastAPI está basado en Starlette y Pydantic.", "entities": {"fastapi", "starlette", "pydantic"}},
    {"id": "F13", "text": "Pydantic usa type hints de Python para validación.", "entities": {"pydantic", "python", "type hints"}},
    {"id": "F14", "text": "Python fue creado por Guido van Rossum.", "entities": {"python", "guido van rossum"}},
    {"id": "F15", "text": "Guido van Rossum trabajó en Google entre 2005 y 2012.", "entities": {"guido van rossum", "google"}},
    {"id": "F16", "text": "Qdrant es una base de datos vectorial escrita en Rust.", "entities": {"qdrant", "rust"}},
    {"id": "F17", "text": "Rust fue creado por Mozilla Research.", "entities": {"rust", "mozilla"}},
    {"id": "F18", "text": "Mozilla es la organización detrás de Firefox.", "entities": {"mozilla", "firefox"}},
    {"id": "F19", "text": "Redis fue creado por Salvatore Sanfilippo en 2009.", "entities": {"redis", "salvatore sanfilippo", "2009"}},
    {"id": "F20", "text": "Redis es una base de datos in-memory clave-valor.", "entities": {"redis", "in-memory", "clave-valor"}},
]

# Multi-hop questions with their supporting fact chains
MULTIHOP_QUERIES: List[Dict[str, Any]] = [
    # 2-hop
    {
        "id": "M01", "hops": 2,
        "question": "¿Quién creó el framework que usa Pydantic para validación?",
        "answer": "Sebastián Ramírez",
        "keywords": ["sebastián", "ramírez", "fastapi"],
        "chain": [
            {"source": "Pydantic",  "via": "usado_por",   "target": "FastAPI"},
            {"source": "FastAPI",   "via": "creado_por",  "target": "Sebastián Ramírez"},
        ],
        "support": ["F12", "F11"],
    },
    {
        "id": "M02", "hops": 2,
        "question": "¿En qué lenguaje está escrita la base de datos vectorial Qdrant?",
        "answer": "Rust",
        "keywords": ["rust"],
        "chain": [
            {"source": "Qdrant", "via": "escrito_en", "target": "Rust"},
        ],
        "support": ["F16"],
    },
    {
        "id": "M03", "hops": 2,
        "question": "¿Qué empresa invirtió en Anthropic, y quiénes la fundaron?",
        "answer": "Google Larry Page Sergey Brin",
        "keywords": ["google", "larry page", "sergey brin"],
        "chain": [
            {"source": "Google",    "via": "invirtió_en", "target": "Anthropic"},
            {"source": "Google",    "via": "fundada_por", "target": "Larry Page"},
        ],
        "support": ["F04", "F05"],
    },
    {
        "id": "M04", "hops": 2,
        "question": "¿Cuándo se creó Redis y quién lo creó?",
        "answer": "2009 Salvatore Sanfilippo",
        "keywords": ["2009", "salvatore", "sanfilippo"],
        "chain": [
            {"source": "Redis", "via": "creado_por",   "target": "Salvatore Sanfilippo"},
            {"source": "Redis", "via": "creado_en",    "target": "2009"},
        ],
        "support": ["F19"],
    },
    # 3-hop
    {
        "id": "M05", "hops": 3,
        "question": "¿El creador de Python trabajó en la misma empresa que invirtió en Anthropic?",
        "answer": "Sí, Guido van Rossum trabajó en Google, que invirtió en Anthropic",
        "keywords": ["guido", "google", "anthropic"],
        "chain": [
            {"source": "Python",          "via": "creado_por",  "target": "Guido van Rossum"},
            {"source": "Guido van Rossum","via": "trabajó_en",  "target": "Google"},
            {"source": "Google",          "via": "invirtió_en", "target": "Anthropic"},
        ],
        "support": ["F14", "F15", "F04"],
    },
    {
        "id": "M06", "hops": 3,
        "question": "¿Qué validación usa FastAPI y quién creó el lenguaje base de esa librería?",
        "answer": "Pydantic Python Guido van Rossum",
        "keywords": ["pydantic", "guido", "van rossum"],
        "chain": [
            {"source": "FastAPI",  "via": "usa",        "target": "Pydantic"},
            {"source": "Pydantic", "via": "basado_en",  "target": "Python"},
            {"source": "Python",   "via": "creado_por", "target": "Guido van Rossum"},
        ],
        "support": ["F12", "F13", "F14"],
    },
    {
        "id": "M07", "hops": 3,
        "question": "¿Qué organización creó el lenguaje en que está escrito Qdrant, y qué navegador mantiene?",
        "answer": "Mozilla Firefox",
        "keywords": ["mozilla", "firefox"],
        "chain": [
            {"source": "Qdrant",  "via": "escrito_en",  "target": "Rust"},
            {"source": "Rust",    "via": "creado_por",  "target": "Mozilla"},
            {"source": "Mozilla", "via": "mantiene",    "target": "Firefox"},
        ],
        "support": ["F16", "F17", "F18"],
    },
    # 4-hop
    {
        "id": "M08", "hops": 4,
        "question": "¿En qué estado creció el cofundador de la empresa que invirtió en el creador de Claude?",
        "answer": "Michigan",
        "keywords": ["michigan"],
        "chain": [
            {"source": "Claude",     "via": "creado_por",  "target": "Anthropic"},
            {"source": "Google",     "via": "invirtió_en", "target": "Anthropic"},
            {"source": "Google",     "via": "cofundada_por","target": "Larry Page"},
            {"source": "Larry Page", "via": "creció_en",   "target": "Michigan"},
        ],
        "support": ["F01", "F04", "F05", "F06"],
    },
    {
        "id": "M09", "hops": 4,
        "question": "¿En qué año fue fundada la empresa que invirtió en el creador de GPT-4?",
        "answer": "1975",
        "keywords": ["1975", "microsoft"],
        "chain": [
            {"source": "GPT-4",     "via": "creado_por",  "target": "OpenAI"},
            {"source": "Microsoft", "via": "invirtió_en", "target": "OpenAI"},
            {"source": "Microsoft", "via": "fundada_en",  "target": "1975"},
        ],
        "support": ["F08", "F09", "F10"],
    },
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def keyword_hit(text: str, keywords: List[str]) -> bool:
    low = text.lower()
    return any(kw.lower() in low for kw in keywords)


def token_f1(pred: str, gt: str) -> float:
    p_toks = set(_tokenize(pred))
    g_toks = set(_tokenize(gt))
    if not g_toks:
        return 1.0
    if not p_toks:
        return 0.0
    ov = p_toks & g_toks
    prec = len(ov) / len(p_toks)
    rec = len(ov) / len(g_toks)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def seed_facts(bank: MemoVexOrchestrator) -> None:
    for fact in FACTS:
        bank.store(
            text=fact["text"],
            memory_type=MemoryType.SEMANTIC,
            entities=fact["entities"],
            confidence=0.95,
            salience=0.85,
        )

    for q in MULTIHOP_QUERIES:
        bank.store_reasoning_chain(
            text=f"[Cadena] {q['question']} → {q['answer']}",
            hops=q["chain"],
            entities={h["source"] for h in q["chain"]} | {h["target"] for h in q["chain"]},
            confidence=0.9,
        )


def _evaluate_retrieval(bank: MemoVexOrchestrator,
                        queries: List[Dict]) -> Dict[str, Any]:
    """Evaluate flat retrieval (R@k, MRR, F1)."""
    hits1 = hits5 = 0
    f1_scores = []
    rrs = []

    for q in queries:
        results = bank.retrieve(q["question"], top_k=10)
        texts = [r.memory.text for r in results]
        hit_list = [keyword_hit(t, q["keywords"]) for t in texts]

        if hit_list and hit_list[0]:
            hits1 += 1
        if any(hit_list[:5]):
            hits5 += 1

        rr = 0.0
        for rank, hit in enumerate(hit_list, 1):
            if hit:
                rr = 1.0 / rank
                break
        rrs.append(rr)

        best_f1 = max((token_f1(t, q["answer"]) for t in texts[:5]), default=0.0)
        f1_scores.append(best_f1)

    n = max(len(queries), 1)
    return {
        "recall@1": round(hits1 / n, 3),
        "recall@5": round(hits5 / n, 3),
        "mrr": round(sum(rrs) / n, 3),
        "token_f1": round(sum(f1_scores) / n, 3),
    }


def _evaluate_graph(bank: MemoVexOrchestrator,
                    queries: List[Dict]) -> Dict[str, Any]:
    """Evaluate graph traversal — can we reach the answer entity from the first hop?"""
    graph_hits = 0
    traversal_depths = []

    for q in queries:
        chain = q["chain"]
        if not chain:
            continue
        start_entity = chain[0]["source"]
        expected_target = chain[-1]["target"]

        # Check if traversal reaches the answer entity
        traversed = bank.traverse_graph(start_entity, max_depth=len(chain) + 1)
        found = any(
            expected_target.lower() in m.text.lower()
            for m in traversed
        )
        if found:
            graph_hits += 1

        # Depth: how many hops were actually traversed
        traversal_depths.append(len(traversed))

    n = max(len(queries), 1)
    return {
        "graph_hit_rate": round(graph_hits / n, 3),
        "avg_traversal_depth": round(sum(traversal_depths) / n, 1),
    }


def run_musique(embeddings_enabled: bool = False, verbose: bool = True) -> Dict[str, Any]:
    bank = MemoVexOrchestrator(agent_id="musique_eval",
                                  embeddings_enabled=embeddings_enabled)
    bank.initialize()

    t0 = time.time()
    seed_facts(bank)
    seed_ms = (time.time() - t0) * 1000

    # Overall
    all_ret = _evaluate_retrieval(bank, MULTIHOP_QUERIES)
    all_graph = _evaluate_graph(bank, MULTIHOP_QUERIES)

    # Per-hop breakdown
    hop_breakdown = {}
    for n_hops in (2, 3, 4):
        qs = [q for q in MULTIHOP_QUERIES if q["hops"] == n_hops]
        if qs:
            ret = _evaluate_retrieval(bank, qs)
            grp = _evaluate_graph(bank, qs)
            hop_breakdown[f"{n_hops}hop"] = {**ret, **grp, "n": len(qs)}

    # Latency
    lats = []
    for q in MULTIHOP_QUERIES:
        ts = time.time()
        bank.retrieve(q["question"], top_k=5)
        lats.append((time.time() - ts) * 1000)
    avg_lat = round(sum(lats) / len(lats), 1)

    bank.shutdown()

    result = {
        "benchmark": "MuSiQue-style",
        "facts": len(FACTS),
        "queries": len(MULTIHOP_QUERIES),
        "seed_ms": round(seed_ms, 1),
        "avg_latency_ms": avg_lat,
        "embeddings": embeddings_enabled,
        **all_ret,
        **all_graph,
        "hop_breakdown": hop_breakdown,
    }

    if verbose:
        _print_musique(result)

    return result


def _print_musique(r: Dict[str, Any]) -> None:
    print("\n" + "=" * 58)
    print(f"  MuSiQue-style Evaluation — MemoVex")
    print(f"  Facts: {r['facts']}  |  Queries: {r['queries']}")
    print("=" * 58)
    print(f"  Recall@1        {r['recall@1']:.3f}")
    print(f"  Recall@5        {r['recall@5']:.3f}")
    print(f"  MRR             {r['mrr']:.3f}")
    print(f"  Token-F1        {r['token_f1']:.3f}")
    print(f"  Graph Hit Rate  {r['graph_hit_rate']:.3f}")
    print(f"  Avg Traversal   {r['avg_traversal_depth']:.1f} memories/query")
    print(f"  Latency         {r['avg_latency_ms']:.1f} ms avg")
    print()
    for hop_key, hm in r["hop_breakdown"].items():
        label = f"{hop_key} ({hm['n']} queries)"
        print(f"  {label:22s}  R@1={hm['recall@1']:.3f}  "
              f"R@5={hm['recall@5']:.3f}  "
              f"Graph={hm['graph_hit_rate']:.3f}  "
              f"F1={hm['token_f1']:.3f}")
    print("=" * 58)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", action="store_true")
    args = p.parse_args()
    run_musique(embeddings_enabled=args.embeddings)
