#!/usr/bin/env python3
"""
Single-configuration benchmark worker for MemoVex.

Executed as a subprocess by compare_engines.py so each configuration
gets an isolated sys.modules environment.  If --pkg-path is provided it
is prepended to sys.path before importing memovex; otherwise memovex must
already be importable in the active Python environment.

Outputs a JSON object to --output (or stdout if omitted).

Usage:
    # with explicit package path
    python runner.py \\
        --pkg-path /path/to/memovex \\
        --label "bow" \\
        --locomo-samples 50 \\
        --musique-samples 50 \\
        --output /tmp/bow.json

    # using installed package
    python runner.py \\
        --label "bow" \\
        --locomo-samples 50 \\
        --musique-samples 50 \\
        --output /tmp/bow.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Inject pkg-path BEFORE any memovex import ───────────────────────────────
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--pkg-path", default=None)
_known, _ = _pre.parse_known_args()
if _known.pkg_path:
    sys.path.insert(0, _known.pkg_path)

from memovex.core.memory_bank import MemoVexOrchestrator  # noqa: E402
from memovex.core.types import MemoryType                  # noqa: E402


# ============================================================================
# Dataset helpers
# ============================================================================

def _parse(val):
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return val
    return val


def load_locomo_samples(n: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("Percena/locomo-mc10", streaming=True, split="train")
    samples: List[Dict] = []
    for s in ds:
        samples.append(s)
        if n and len(samples) >= n:
            break
    return samples


def load_musique_samples(n: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("bdsaglam/musique", split="validation", streaming=True)
    samples: List[Dict] = []
    for s in ds:
        if str(s.get("answerable")) != "True":
            continue
        samples.append(s)
        if n and len(samples) >= n:
            break
    return samples


# ============================================================================
# LoCoMo benchmark
# ============================================================================

def _seed_locomo(bank: MemoVexOrchestrator, samples: List[Dict]) -> Dict[str, int]:
    n_summaries = n_turns = 0
    for s in samples:
        qid = s.get("question_id", "unknown")
        sessions   = _parse(s.get("haystack_sessions", []))
        summaries  = _parse(s.get("haystack_session_summaries", []))
        datetimes  = _parse(s.get("haystack_session_datetimes", []))

        for sidx, summary in enumerate(summaries):
            summary = _parse(summary) if isinstance(summary, str) else summary
            if not isinstance(summary, str):
                summary = str(summary)
            summary = summary.strip()
            if not summary:
                continue
            dt = str(datetimes[sidx]) if isinstance(datetimes, list) and sidx < len(datetimes) else ""
            text = f"{dt}: {summary}" if dt else summary
            bank.store(
                text=text,
                memory_type=MemoryType.SEMANTIC,
                tags={f"qid_{qid}", f"session_{sidx}", "summary"},
                confidence=0.9,
                salience=0.85,
            )
            n_summaries += 1

        for sidx, session in enumerate(sessions):
            session = _parse(session)
            if not isinstance(session, list):
                continue
            dt = str(datetimes[sidx]) if isinstance(datetimes, list) and sidx < len(datetimes) else ""
            for turn in session:
                turn = _parse(turn)
                if not isinstance(turn, dict):
                    continue
                role    = turn.get("role", "")
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


def _score_choices(bank: MemoVexOrchestrator, question: str, choices: List[str]) -> List[float]:
    results = bank.retrieve(question, top_k=10)
    texts = [r.memory.text.lower() for r in results]
    scores = []
    for choice in choices:
        c_low = choice.lower()
        hits = sum(1 for t in texts if c_low in t)
        c_toks = set(re.findall(r"\w+", c_low))
        token_hits = sum(
            len(c_toks & set(re.findall(r"\w+", t))) / max(len(c_toks), 1)
            for t in texts[:5]
        )
        scores.append(hits + 0.1 * token_hits)
    return scores


def _build_context(bank: MemoVexOrchestrator, question: str, top_k: int = 10) -> str:
    results = bank.retrieve(question, top_k=top_k)
    return "\n".join(f"[{i}] {r.memory.text}" for i, r in enumerate(results, 1))


def _evaluate_locomo(bank: MemoVexOrchestrator, samples: List[Dict], llm=None) -> Dict[str, Any]:
    mc_hits = top5_any = 0
    rrs: List[float] = []
    latencies: List[float] = []
    type_stats: Dict[str, Dict] = defaultdict(lambda: {"mc": 0, "top5": 0, "n": 0})
    use_llm = llm is not None and llm.available

    llm_calls = llm_tokens = 0

    for s in samples:
        question    = s.get("question", "").strip()
        choices     = _parse(s.get("choices", []))
        correct_idx = int(s.get("correct_choice_index", 0))
        qtype       = s.get("question_type", "unknown")

        if not question or not choices:
            continue

        t0 = time.time()
        if use_llm:
            context  = _build_context(bank, question, top_k=10)
            pred_idx = llm.answer_mc(question, context, choices)
            mc_ok    = pred_idx == correct_idx if pred_idx is not None else False
            rrs.append(1.0 if mc_ok else 0.0)
            top5_ok  = mc_ok
        else:
            scores = _score_choices(bank, question, choices)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            mc_ok   = ranked[0] == correct_idx if ranked else False
            top5_ok = correct_idx in ranked[:5]
            try:
                rrs.append(1.0 / (ranked.index(correct_idx) + 1))
            except ValueError:
                rrs.append(0.0)

        latencies.append((time.time() - t0) * 1000)
        mc_hits  += mc_ok
        top5_any += top5_ok
        ts = type_stats[qtype]
        ts["n"] += 1
        ts["mc"]   += mc_ok
        ts["top5"] += top5_ok

    n = max(len(samples), 1)
    avg_lat = round(sum(latencies) / max(len(latencies), 1), 1)
    p99_lat = round(sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0, 1)

    type_metrics = {
        qtype: {
            "mc_accuracy":   round(ts["mc"]   / max(ts["n"], 1), 3),
            "top5_accuracy": round(ts["top5"] / max(ts["n"], 1), 3),
            "n": ts["n"],
        }
        for qtype, ts in type_stats.items()
    }

    result: Dict[str, Any] = {
        "mc_accuracy":     round(mc_hits / n, 3),
        "top5_accuracy":   round(top5_any / n, 3),
        "mrr":             round(sum(rrs) / max(len(rrs), 1), 3),
        "random_baseline": round(1 / max(len(choices), 1), 3) if samples else 0.1,
        "avg_latency_ms":  avg_lat,
        "p99_latency_ms":  p99_lat,
        "n_queries":       n,
        "llm_used":        use_llm,
        "type_breakdown":  type_metrics,
    }
    if use_llm and hasattr(llm, "stats"):
        result["llm_stats"] = llm.stats()
    return result


def run_locomo(samples: List[Dict], label: str, embeddings_enabled: bool,
               llm=None) -> Dict[str, Any]:
    bank = MemoVexOrchestrator(agent_id=f"cmp_{label}_locomo",
                                embeddings_enabled=embeddings_enabled)
    bank.initialize()
    t0 = time.time()
    counts = _seed_locomo(bank, samples)
    seed_ms = round((time.time() - t0) * 1000, 1)
    metrics = _evaluate_locomo(bank, samples, llm=llm)
    bank.shutdown()
    return {
        "benchmark": "LoCoMo (Percena/locomo-mc10)",
        "n_samples":   len(samples),
        "n_summaries": counts["summaries"],
        "n_turns":     counts["turns"],
        "seed_ms":     seed_ms,
        **metrics,
    }


# ============================================================================
# MuSiQue benchmark
# ============================================================================

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _token_f1(pred: str, gt: str) -> float:
    p_toks = set(_tokenize(pred))
    g_toks = set(_tokenize(gt))
    if not g_toks:
        return 1.0
    if not p_toks:
        return 0.0
    ov = p_toks & g_toks
    prec = len(ov) / len(p_toks)
    rec  = len(ov) / len(g_toks)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def _n_hops(sid: str) -> int:
    for h in (2, 3, 4):
        if f"{h}hop" in sid:
            return h
    return 0


def _eval_musique_sample(sample: Dict, label: str,
                         embeddings_enabled: bool, llm=None) -> Dict[str, Any]:
    bank = MemoVexOrchestrator(agent_id=f"cmp_{label}_mq",
                                embeddings_enabled=embeddings_enabled)
    bank.initialize()

    question   = sample.get("question", "")
    answer     = str(sample.get("answer", ""))
    paragraphs = _parse(sample.get("paragraphs", []))
    decomp     = _parse(sample.get("question_decomposition", []))
    n_hops_val = _n_hops(sample.get("id", ""))

    for para in paragraphs:
        if not isinstance(para, dict):
            continue
        title = para.get("title", "")
        text  = para.get("paragraph_text", "")
        if text:
            bank.store(
                text=f"{title}: {text}" if title else text,
                memory_type=MemoryType.SEMANTIC,
                confidence=0.9,
                salience=0.8,
            )

    hops: List[Dict] = []
    prev_answer = None
    for i, step in enumerate(decomp):
        if not isinstance(step, dict):
            continue
        q = step.get("question", "")
        a = step.get("answer", "").strip()
        if not a:
            continue
        if i == 0:
            source = q.split(">>")[0].strip() if ">>" in q else re.sub(r"#\d+", "", q).strip()
            source = source or "start"
        else:
            source = prev_answer or f"step_{i-1}"
        via = re.sub(r"#\d+", "", q).strip() or "supports"
        hops.append({"source": source, "via": via, "target": a})
        prev_answer = a

    if len(hops) >= 2:
        bank.store_reasoning_chain(
            text=" → ".join(h["target"] for h in hops[:-1]),
            hops=hops,
            confidence=0.85,
        )

    t0 = time.time()
    results = bank.retrieve(question, top_k=10)
    lat_ms  = (time.time() - t0) * 1000

    texts    = [r.memory.text for r in results]
    ans_low  = answer.lower()
    hit_list = [ans_low in t.lower() for t in texts]
    h1  = bool(hit_list and hit_list[0])
    h5  = any(hit_list[:5])
    rr  = next((1.0 / (i + 1) for i, h in enumerate(hit_list) if h), 0.0)
    f1  = max((_token_f1(t, answer) for t in texts[:5]), default=0.0)

    graph_hit = False
    if hops and bank._graph_store and bank._graph_store.available:
        start      = hops[0]["source"]
        end_entity = hops[-1]["target"]
        reachable  = bank._graph_store.neighbors(start, depth=len(hops) + 1)
        graph_hit  = end_entity.lower() in {n.lower() for n in reachable}

    llm_f1 = llm_em = None
    if llm is not None and llm.available:
        ctx  = "\n".join(f"[{i}] {r.memory.text}" for i, r in enumerate(results[:5], 1))
        pred = llm.answer_open(question, ctx)
        if pred is not None:
            llm_f1 = _token_f1(pred, answer)
            llm_em = float(pred.lower().strip() == ans_low.strip())

    bank.shutdown()
    return {
        "hit@1": h1, "hit@5": h5, "rr": rr,
        "f1": f1, "graph_hit": graph_hit,
        "lat_ms": lat_ms, "n_hops": n_hops_val,
        "llm_f1": llm_f1, "llm_em": llm_em,
    }


def run_musique(samples: List[Dict], label: str, embeddings_enabled: bool,
                llm=None, verbose: bool = True) -> Dict[str, Any]:
    results = []
    t0 = time.time()
    for i, s in enumerate(samples):
        r = _eval_musique_sample(s, label, embeddings_enabled, llm=llm)
        results.append(r)
        if verbose and (i + 1) % 25 == 0:
            print(f"    musique {i+1}/{len(samples)}…", file=sys.stderr, flush=True)
    total_ms = (time.time() - t0) * 1000

    def _agg(rs: List[Dict]) -> Dict:
        n = max(len(rs), 1)
        out: Dict[str, Any] = {
            "recall@1":       round(sum(r["hit@1"]     for r in rs) / n, 3),
            "recall@5":       round(sum(r["hit@5"]     for r in rs) / n, 3),
            "mrr":            round(sum(r["rr"]        for r in rs) / n, 3),
            "token_f1":       round(sum(r["f1"]        for r in rs) / n, 3),
            "graph_hit_rate": round(sum(r["graph_hit"] for r in rs) / n, 3),
            "avg_latency_ms": round(sum(r["lat_ms"]    for r in rs) / n, 1),
            "n": n,
        }
        llm_rs = [r for r in rs if r.get("llm_f1") is not None]
        if llm_rs:
            ln = max(len(llm_rs), 1)
            out["llm_token_f1"]  = round(sum(r["llm_f1"] for r in llm_rs) / ln, 3)
            out["llm_exact_match"] = round(sum(r["llm_em"] for r in llm_rs) / ln, 3)
        return out

    overall = _agg(results)
    hop_breakdown = {
        f"{h}hop": _agg([r for r in results if r["n_hops"] == h])
        for h in (2, 3, 4)
        if any(r["n_hops"] == h for r in results)
    }

    out: Dict[str, Any] = {
        "benchmark": "MuSiQue (bdsaglam/musique, answerable validation)",
        "n_samples":    len(samples),
        "total_eval_ms": round(total_ms, 0),
        **overall,
        "hop_breakdown": hop_breakdown,
    }
    if llm is not None and llm.available and hasattr(llm, "stats"):
        out["llm_stats"] = llm.stats()
    return out


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pkg-path",        default=None,
                   help="Path to memovex package root (optional if memovex is installed)")
    p.add_argument("--label",           required=True)
    p.add_argument("--locomo-samples",  type=int, default=50)
    p.add_argument("--musique-samples", type=int, default=50)
    p.add_argument("--embeddings",      action="store_true")
    p.add_argument("--llm",             action="store_true")
    p.add_argument("--model",           default="gpt-4o-mini")
    p.add_argument("--output",          default=None,
                   help="Path to write JSON result (default: stdout)")
    args = p.parse_args()

    llm = None
    if args.llm:
        try:
            from memovex.core.llm_layer import LLMLayer
            llm = LLMLayer(model=args.model)
            if not llm.available:
                print("WARNING: LLMLayer unavailable — running retrieval-only",
                      file=sys.stderr)
                llm = None
        except ImportError:
            print("WARNING: LLMLayer not found — running retrieval-only",
                  file=sys.stderr)

    mode = ("retrieval+LLM" if llm else
            "retrieval+emb" if args.embeddings else
            "retrieval-only")
    print(f"[{args.label}] Loading LoCoMo ({args.locomo_samples} samples)…",
          file=sys.stderr, flush=True)
    locomo_samples  = load_locomo_samples(args.locomo_samples)

    print(f"[{args.label}] Loading MuSiQue ({args.musique_samples} samples)…",
          file=sys.stderr, flush=True)
    musique_samples = load_musique_samples(args.musique_samples)

    print(f"[{args.label}] Running LoCoMo ({mode})…", file=sys.stderr, flush=True)
    t0 = time.time()
    locomo_result  = run_locomo(locomo_samples,  args.label, args.embeddings, llm=llm)
    print(f"[{args.label}] LoCoMo done in {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)

    print(f"[{args.label}] Running MuSiQue ({mode})…", file=sys.stderr, flush=True)
    t0 = time.time()
    musique_result = run_musique(musique_samples, args.label, args.embeddings, llm=llm)
    print(f"[{args.label}] MuSiQue done in {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)

    import datetime
    output = {
        "label":       args.label,
        "pkg_path":    args.pkg_path,
        "run_date":    datetime.date.today().isoformat(),
        "locomo_samples":  args.locomo_samples,
        "musique_samples": args.musique_samples,
        "embeddings":  args.embeddings,
        "llm":         llm is not None,
        "mode":        mode,
        "locomo":      locomo_result,
        "musique":     musique_result,
    }

    json_out = json.dumps(output, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(json_out)
        print(f"[{args.label}] Saved → {args.output}", file=sys.stderr)
    else:
        print(json_out)


if __name__ == "__main__":
    main()
