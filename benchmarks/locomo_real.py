"""
memovex — LoCoMo Real Dataset Evaluation.

Evaluates against Percena/locomo-mc10 (HuggingFace), a 10-choice
multiple-choice adaptation of the LoCoMo long-context memory benchmark.

Key insight: LoCoMo answers often require temporal inference
("yesterday" → compute date from session datetime) so exact-match
Recall@k on the answer string is not meaningful.

Primary metrics:
  - MC-Accuracy  correct choice ranked #1 among 10 by retrieval score
  - Session Hit  relevant session retrieved in top-5 (oracle signal)

Secondary metrics (soft):
  - Choice-in-Top5  any of top-5 choices found in retrieved text
  - MRR on choice ranking

Seeding strategy: store (a) session summaries with datetimes —
the dense factual signal — and (b) individual turns, so both
explicit facts and contextual clues are searchable.

Usage:
    python benchmarks/locomo_real.py
    python benchmarks/locomo_real.py --samples 50
    python benchmarks/locomo_real.py --samples 0   # all
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from memovex.core.memory_bank import MemoVexOrchestrator
from memovex.core.types import MemoryType


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _parse(val):
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return val
    return val


def load_locomo_samples(n: int = 100) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("Percena/locomo-mc10", streaming=True, split="train")
    samples = []
    for s in ds:
        samples.append(s)
        if n and len(samples) >= n:
            break
    return samples


# ---------------------------------------------------------------------------
# Bank seeding
# ---------------------------------------------------------------------------

def seed_bank(bank: MemoVexOrchestrator, samples: List[Dict]) -> Dict[str, int]:
    """
    Store session summaries (with dates) + individual turns.
    Returns counts of stored memories by type.
    """
    n_summaries = n_turns = 0

    for s in samples:
        qid = s.get("question_id", "unknown")
        sessions = _parse(s.get("haystack_sessions", []))
        summaries = _parse(s.get("haystack_session_summaries", []))
        datetimes = _parse(s.get("haystack_session_datetimes", []))

        # --- Store session summaries (high salience, date-rich) ---
        for sidx, summary in enumerate(summaries):
            summary = _parse(summary) if isinstance(summary, str) else summary
            if not isinstance(summary, str):
                summary = str(summary)
            summary = summary.strip()
            if not summary:
                continue
            dt = ""
            if isinstance(datetimes, list) and sidx < len(datetimes):
                dt = str(datetimes[sidx])
            text = f"{dt}: {summary}" if dt else summary
            bank.store(
                text=text,
                memory_type=MemoryType.SEMANTIC,
                tags={f"qid_{qid}", f"session_{sidx}", "summary"},
                confidence=0.9,
                salience=0.85,
            )
            n_summaries += 1

        # --- Store individual turns (episodic) ---
        for sidx, session in enumerate(sessions):
            session = _parse(session)
            if not isinstance(session, list):
                continue
            dt = ""
            if isinstance(datetimes, list) and sidx < len(datetimes):
                dt = str(datetimes[sidx])
            for turn in session:
                turn = _parse(turn)
                if not isinstance(turn, dict):
                    continue
                role = turn.get("role", "")
                content = turn.get("content", "").strip()
                if not content:
                    continue
                text = f"[{dt}] {content}" if dt else content
                bank.store(
                    text=text,
                    memory_type=MemoryType.EPISODIC,
                    tags={f"qid_{qid}", f"session_{sidx}", role},
                    confidence=0.75,
                    salience=0.5,
                )
                n_turns += 1

    return {"summaries": n_summaries, "turns": n_turns}


# ---------------------------------------------------------------------------
# Multiple-choice scoring
# ---------------------------------------------------------------------------

def _score_choices(bank: MemoVexOrchestrator, question: str,
                   choices: List[str]) -> List[float]:
    """
    Retrieve top-10 memories for the question.
    Score each choice by: how many top-10 memories mention it.
    Returns a list of float scores, one per choice.
    """
    results = bank.retrieve(question, top_k=10)
    texts = [r.memory.text.lower() for r in results]

    scores = []
    for choice in choices:
        c_low = choice.lower()
        # Count exact substring hits across retrieved texts
        hits = sum(1 for t in texts if c_low in t)
        # Soft boost: partial token overlap
        c_toks = set(re.findall(r"\w+", c_low))
        token_hits = sum(
            len(c_toks & set(re.findall(r"\w+", t))) / max(len(c_toks), 1)
            for t in texts[:5]
        )
        scores.append(hits + 0.1 * token_hits)
    return scores


# ---------------------------------------------------------------------------
# Context builder (for LLM)
# ---------------------------------------------------------------------------

def _build_context(bank: MemoVexOrchestrator, question: str,
                   top_k: int = 10) -> str:
    """Return a formatted context string from top-k retrieved memories."""
    results = bank.retrieve(question, top_k=top_k)
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.memory.text}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(bank: MemoVexOrchestrator,
             samples: List[Dict],
             llm=None) -> Dict[str, Any]:
    mc_hits = 0
    top5_any = 0
    rrs: List[float] = []
    latencies: List[float] = []
    type_stats: Dict[str, Dict] = defaultdict(
        lambda: {"mc": 0, "top5": 0, "n": 0}
    )
    use_llm = llm is not None and llm.available

    for s in samples:
        question = s.get("question", "").strip()
        choices = _parse(s.get("choices", []))
        correct_idx = int(s.get("correct_choice_index", 0))
        qtype = s.get("question_type", "unknown")

        if not question or not choices:
            continue

        t0 = time.time()

        if use_llm:
            # Full RAG: retrieve → LLM picks choice
            context = _build_context(bank, question, top_k=10)
            pred_idx = llm.answer_mc(question, context, choices)
            mc_ok = pred_idx == correct_idx if pred_idx is not None else False
            # For MRR with LLM: it's a binary pick, so RR = 1 if correct else 0
            rrs.append(1.0 if mc_ok else 0.0)
            top5_ok = mc_ok  # LLM gives one answer — either right or wrong
        else:
            # Pure retrieval: score each choice by presence in top-k memories
            scores = _score_choices(bank, question, choices)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            mc_ok = ranked[0] == correct_idx if ranked else False
            top5_ok = correct_idx in ranked[:5]
            try:
                rank = ranked.index(correct_idx) + 1
                rrs.append(1.0 / rank)
            except ValueError:
                rrs.append(0.0)

        latencies.append((time.time() - t0) * 1000)
        mc_hits += mc_ok
        top5_any += top5_ok

        ts = type_stats[qtype]
        ts["n"] += 1
        ts["mc"] += mc_ok
        ts["top5"] += top5_ok

    n = max(len(samples), 1)
    avg_lat = round(sum(latencies) / max(len(latencies), 1), 1)
    p99_lat = round(sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0, 1)

    type_metrics = {}
    for qtype, ts in type_stats.items():
        tn = max(ts["n"], 1)
        type_metrics[qtype] = {
            "mc_accuracy": round(ts["mc"] / tn, 3),
            "top5_accuracy": round(ts["top5"] / tn, 3),
            "n": ts["n"],
        }

    result = {
        "mc_accuracy":     round(mc_hits / n, 3),
        "top5_accuracy":   round(top5_any / n, 3),
        "mrr":             round(sum(rrs) / max(len(rrs), 1), 3),
        "random_baseline": round(1 / max(len(_parse(samples[0].get("choices", [1]*10))), 1), 3) if samples else 0.1,
        "avg_latency_ms":  avg_lat,
        "p99_latency_ms":  p99_lat,
        "n_queries":       n,
        "llm_used":        use_llm,
        "type_breakdown":  type_metrics,
    }
    if use_llm:
        result["llm_stats"] = llm.stats()
    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_locomo_real(n_samples: int = 100,
                   embeddings_enabled: bool = False,
                   verbose: bool = True,
                   llm=None) -> Dict[str, Any]:
    if verbose:
        print(f"  Loading LoCoMo samples (n={n_samples or 'all'})…", flush=True)

    samples = load_locomo_samples(n=n_samples)

    bank = MemoVexOrchestrator(agent_id="locomo_real",
                                  embeddings_enabled=embeddings_enabled)
    bank.initialize()

    t0 = time.time()
    counts = seed_bank(bank, samples)
    seed_ms = round((time.time() - t0) * 1000, 1)

    if verbose:
        total_mems = counts["summaries"] + counts["turns"]
        print(f"  Stored {total_mems} memories "
              f"({counts['summaries']} summaries + {counts['turns']} turns) "
              f"from {len(samples)} samples ({seed_ms} ms)")
        print("  Evaluating…", flush=True)

    metrics = evaluate(bank, samples, llm=llm)
    bank.shutdown()

    result = {
        "benchmark": "LoCoMo (Percena/locomo-mc10)",
        "n_samples": len(samples),
        "n_summaries": counts["summaries"],
        "n_turns": counts["turns"],
        "seed_ms": seed_ms,
        "embeddings": embeddings_enabled,
        "llm_used": llm is not None and (llm.available if llm else False),
        **metrics,
    }

    if verbose:
        _print_locomo_real(result)

    return result


def _print_locomo_real(r: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"  LoCoMo Real  ({r['benchmark']})")
    print(f"  Samples: {r['n_samples']}  |  "
          f"Memories: {r['n_summaries'] + r['n_turns']} "
          f"({r['n_summaries']} summaries + {r['n_turns']} turns)")
    print("=" * 60)
    mode = "Retrieval+LLM" if r.get("llm_used") else "Retrieval-only"
    print(f"  Mode: {mode}")
    print(f"  MC-Accuracy   {r['mc_accuracy']:.3f}  "
          f"(random baseline: {r['random_baseline']:.3f})")
    print(f"  Top-5 Acc     {r['top5_accuracy']:.3f}")
    print(f"  MRR           {r['mrr']:.3f}")
    print(f"  Latency       {r['avg_latency_ms']:.1f} ms avg / "
          f"{r['p99_latency_ms']:.1f} ms p99")
    if r.get("llm_used") and r.get("llm_stats"):
        s = r["llm_stats"]
        print(f"  LLM calls: {s.get('calls',0)}  "
              f"Tokens: {s.get('total_tokens',0)}  "
              f"Cost: ${s.get('estimated_cost_usd',0):.4f}")
    print()
    for qtype, tm in sorted(r["type_breakdown"].items()):
        print(f"  {qtype:<22s}  n={tm['n']:<3d}  "
              f"MC={tm['mc_accuracy']:.3f}  Top5={tm['top5_accuracy']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=100,
                   help="Number of samples to load (0 = all)")
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

    run_locomo_real(n_samples=args.samples, embeddings_enabled=args.embeddings, llm=llm)
