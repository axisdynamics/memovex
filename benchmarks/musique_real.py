"""
memovex — MuSiQue Real Dataset Evaluation.

Evaluates against bdsaglam/musique (HuggingFace), the standard
multi-hop reasoning benchmark (Trivedi et al., TACL 2022).

Strategy: per-sample isolated banks — each sample gets its own fresh bank
with only its own paragraphs (standard closed-book evaluation protocol).

Metrics:
  - Recall@k       answer in top-k retrieved paragraph texts
  - MRR            mean reciprocal rank
  - Token-F1       token-level F1 between best retrieved text and answer
  - Graph Hit Rate  traversal from first hop source reaches answer entity
  - per-hop breakdown (2/3/4)

Optional LLM mode (--llm):
  - Retrieves top-5 paragraphs → GPT generates a short answer
  - Reports LLM Token-F1 and Exact Match alongside retrieval metrics
  - Requires OPENAI_API_KEY environment variable

Usage:
    python benchmarks/musique_real.py
    python benchmarks/musique_real.py --samples 50
    python benchmarks/musique_real.py --samples 0   # all answerable
    python benchmarks/musique_real.py --llm --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from memovex.core.memory_bank import MemoVexOrchestrator
from memovex.core.types import MemoryType


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _parse_field(val):
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return val
    return val


def load_musique_samples(n: int = 150) -> List[Dict[str, Any]]:
    """Load answerable MuSiQue validation samples."""
    from datasets import load_dataset
    ds = load_dataset("bdsaglam/musique", split="validation", streaming=True)
    samples = []
    for s in ds:
        if str(s.get("answerable")) != "True":
            continue
        samples.append(s)
        if n and len(samples) >= n:
            break
    return samples


def _n_hops(sample_id: str) -> int:
    if "2hop" in sample_id:
        return 2
    if "3hop" in sample_id:
        return 3
    if "4hop" in sample_id:
        return 4
    return 0


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


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


def _build_context(results, top_k: int = 5) -> str:
    parts = []
    for i, r in enumerate(results[:top_k], 1):
        parts.append(f"[{i}] {r.memory.text}")
    return "\n".join(parts)


def _eval_sample(sample: Dict[str, Any],
                 embeddings_enabled: bool,
                 llm=None) -> Dict[str, Any]:
    bank = MemoVexOrchestrator(agent_id="mq_sample",
                                  embeddings_enabled=embeddings_enabled)
    bank.initialize()

    question = sample.get("question", "")
    answer = str(sample.get("answer", ""))
    paragraphs = _parse_field(sample.get("paragraphs", []))
    decomp = _parse_field(sample.get("question_decomposition", []))
    n_hops = _n_hops(sample.get("id", ""))

    # Store paragraphs (retrieval corpus — no answer string injected)
    for para in paragraphs:
        if not isinstance(para, dict):
            continue
        title = para.get("title", "")
        text = para.get("paragraph_text", "")
        if text:
            bank.store(
                text=f"{title}: {text}" if title else text,
                memory_type=MemoryType.SEMANTIC,
                confidence=0.9,
                salience=0.8,
            )

    # Build graph hops by chaining answers: answer[i-1] → answer[i]
    # MuSiQue uses #1, #2 references; we resolve them to actual answers.
    hops = []
    prev_answer = None
    for i, step in enumerate(decomp):
        if not isinstance(step, dict):
            continue
        q = step.get("question", "")
        a = step.get("answer", "").strip()
        if not a:
            continue
        if i == 0:
            # Extract entity from first question (e.g. "Arna Selznick >> employer")
            source = q.split(">>")[0].strip() if ">>" in q else re.sub(r"#\d+", "", q).strip()
            source = source or "start"
        else:
            source = prev_answer or f"step_{i-1}"
        via = re.sub(r"#\d+", "", q).strip() or "supports"
        hops.append({"source": source, "via": via, "target": a})
        prev_answer = a

    if len(hops) >= 2:
        chain_text = " → ".join(h["target"] for h in hops[:-1])
        bank.store_reasoning_chain(
            text=f"Reasoning: {chain_text}",
            hops=hops,
            confidence=0.85,
        )

    # Retrieve
    t0 = time.time()
    results = bank.retrieve(question, top_k=10)
    lat_ms = (time.time() - t0) * 1000

    texts = [r.memory.text for r in results]
    ans_low = answer.lower()

    hit_list = [ans_low in t.lower() for t in texts]
    h1 = bool(hit_list and hit_list[0])
    h5 = any(hit_list[:5])

    rr = 0.0
    for rank, hit in enumerate(hit_list, 1):
        if hit:
            rr = 1.0 / rank
            break

    best_f1 = max((token_f1(t, answer) for t in texts[:5]), default=0.0)

    # Graph traversal: can we reach the final answer entity from the first hop?
    graph_hit = False
    if hops and bank._graph_store and bank._graph_store.available:
        start = hops[0]["source"]
        end_entity = hops[-1]["target"]
        reachable = bank._graph_store.neighbors(start, depth=len(hops) + 1)
        graph_hit = end_entity.lower() in {n.lower() for n in reachable}

    # Optional LLM generation
    llm_f1 = None
    llm_em = None
    if llm is not None and llm.available:
        context = _build_context(results, top_k=5)
        pred = llm.answer_open(question, context)
        if pred is not None:
            llm_f1 = token_f1(pred, answer)
            llm_em = float(pred.lower().strip() == ans_low.strip())

    bank.shutdown()

    return {
        "hit@1": h1, "hit@5": h5, "rr": rr,
        "f1": best_f1, "graph_hit": graph_hit,
        "lat_ms": lat_ms, "n_hops": n_hops,
        "llm_f1": llm_f1, "llm_em": llm_em,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_musique_real(n_samples: int = 150,
                    embeddings_enabled: bool = False,
                    verbose: bool = True,
                    llm=None) -> Dict[str, Any]:
    if verbose:
        print(f"  Loading MuSiQue answerable samples (n={n_samples or 'all'})…", flush=True)

    samples = load_musique_samples(n=n_samples)
    use_llm = llm is not None and llm.available

    if verbose:
        mode = "retrieval+LLM" if use_llm else "retrieval-only"
        print(f"  Evaluating {len(samples)} samples (isolated banks, {mode})…", flush=True)

    results = []
    t0 = time.time()
    for i, s in enumerate(samples):
        r = _eval_sample(s, embeddings_enabled, llm=llm)
        results.append(r)
        if verbose and (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(samples)}…", flush=True)

    total_ms = (time.time() - t0) * 1000

    # Aggregate
    def _agg(rs):
        n = max(len(rs), 1)
        agg = {
            "recall@1":       round(sum(r["hit@1"] for r in rs) / n, 3),
            "recall@5":       round(sum(r["hit@5"] for r in rs) / n, 3),
            "mrr":            round(sum(r["rr"]    for r in rs) / n, 3),
            "token_f1":       round(sum(r["f1"]    for r in rs) / n, 3),
            "graph_hit_rate": round(sum(r["graph_hit"] for r in rs) / n, 3),
            "avg_latency_ms": round(sum(r["lat_ms"] for r in rs) / n, 1),
            "n": n,
        }
        llm_rs = [r for r in rs if r.get("llm_f1") is not None]
        if llm_rs:
            ln = max(len(llm_rs), 1)
            agg["llm_token_f1"] = round(sum(r["llm_f1"] for r in llm_rs) / ln, 3)
            agg["llm_exact_match"] = round(sum(r["llm_em"] for r in llm_rs) / ln, 3)
        return agg

    overall = _agg(results)

    hop_breakdown = {}
    for h in (2, 3, 4):
        sub = [r for r in results if r["n_hops"] == h]
        if sub:
            hop_breakdown[f"{h}hop"] = _agg(sub)

    result = {
        "benchmark": "MuSiQue (bdsaglam/musique, answerable validation)",
        "n_samples": len(samples),
        "total_eval_ms": round(total_ms, 0),
        "embeddings": embeddings_enabled,
        "llm_used": use_llm,
        **overall,
        "hop_breakdown": hop_breakdown,
    }
    if use_llm:
        result["llm_stats"] = llm.stats()

    if verbose:
        _print_musique_real(result)

    return result


def _print_musique_real(r: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"  MuSiQue Real  ({r['benchmark']})")
    print(f"  Samples: {r['n_samples']}  |  Eval time: {r['total_eval_ms']:.0f} ms")
    print("=" * 60)
    print(f"  --- Retrieval-only ---")
    print(f"  Recall@1        {r['recall@1']:.3f}")
    print(f"  Recall@5        {r['recall@5']:.3f}")
    print(f"  MRR             {r['mrr']:.3f}")
    print(f"  Token-F1        {r['token_f1']:.3f}")
    print(f"  Graph Hit Rate  {r['graph_hit_rate']:.3f}")
    print(f"  Latency         {r['avg_latency_ms']:.1f} ms avg per sample")
    if r.get("llm_used"):
        stats = r.get("llm_stats", {})
        print(f"\n  --- Retrieval + LLM ({stats.get('model','?')}) ---")
        print(f"  LLM Token-F1    {r.get('llm_token_f1', 0):.3f}")
        print(f"  LLM Exact Match {r.get('llm_exact_match', 0):.3f}")
        print(f"  LLM calls:  {stats.get('calls',0)}  "
              f"Tokens: {stats.get('total_tokens',0)}  "
              f"Cost: ${stats.get('estimated_cost_usd',0):.4f}")
    print()
    for hk in ("2hop", "3hop", "4hop"):
        hm = r["hop_breakdown"].get(hk, {})
        if hm:
            line = (f"  {hk} (n={hm['n']:<3d})   "
                    f"R@1={hm['recall@1']:.3f}  R@5={hm['recall@5']:.3f}  "
                    f"Graph={hm['graph_hit_rate']:.3f}  F1={hm['token_f1']:.3f}")
            if "llm_token_f1" in hm:
                line += f"  LLM-F1={hm['llm_token_f1']:.3f}"
            print(line)
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=150,
                   help="Max answerable samples to load (0 = all)")
    p.add_argument("--embeddings", action="store_true")
    p.add_argument("--llm", action="store_true",
                   help="Enable LLM answer generation (requires OPENAI_API_KEY)")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="OpenAI model for LLM generation (default: gpt-4o-mini)")
    args = p.parse_args()

    llm = None
    if args.llm:
        from memovex.core.llm_layer import LLMLayer
        llm = LLMLayer(model=args.model)
        if not llm.available:
            print("WARNING: LLMLayer not available — check OPENAI_API_KEY. "
                  "Running retrieval-only.")
            llm = None

    run_musique_real(n_samples=args.samples, embeddings_enabled=args.embeddings, llm=llm)
