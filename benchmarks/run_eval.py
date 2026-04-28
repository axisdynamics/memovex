"""
memovex — Unified Benchmark Runner.

Runs LoCoMo + MuSiQue evaluations and prints a combined results table
suitable for README insertion.

Usage:
    python benchmarks/run_eval.py               # synthetic datasets
    python benchmarks/run_eval.py --real        # real HuggingFace datasets
    python benchmarks/run_eval.py --real --samples 50
    python benchmarks/run_eval.py --embeddings  # with sentence-transformers
    python benchmarks/run_eval.py --json        # machine-readable output
    python benchmarks/run_eval.py --save results/latest.json
    python benchmarks/run_eval.py --real --llm                    # retrieval+GPT-4o-mini
    python benchmarks/run_eval.py --real --llm --model gpt-4o     # retrieval+GPT-4o
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.locomo_eval import run_locomo
from benchmarks.musique_eval import run_musique


BASELINE = {
    "locomo": {
        "recall@1": 0.40, "recall@5": 0.55, "mrr": 0.45, "token_f1": 0.38,
        "label": "SimpleMemory (token overlap only)",
    },
    "musique": {
        "recall@1": 0.22, "recall@5": 0.44, "mrr": 0.31, "token_f1": 0.29,
        "graph_hit_rate": 0.11,
        "label": "SimpleMemory (token overlap only)",
    },
}


def _delta(val: float, base: float) -> str:
    d = val - base
    return f"+{d:.3f}" if d >= 0 else f"{d:.3f}"


def _pct(val: float, base: float) -> str:
    if base == 0:
        return "n/a"
    return f"+{(val-base)/base*100:.1f}%" if val >= base else f"{(val-base)/base*100:.1f}%"


def print_summary_table(locomo: dict, musique: dict) -> None:
    bl_l = BASELINE["locomo"]
    bl_m = BASELINE["musique"]

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          MemoVex — Benchmark Results             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  LoCoMo-style  (60-turn conversation, 15 queries)           ║")
    print("╠═══════════════╦═══════════╦═══════════╦════════════════════╣")
    print("║  Metric       ║  Baseline ║  Vex 1.0  ║  Delta             ║")
    print("╠═══════════════╬═══════════╬═══════════╬════════════════════╣")

    for metric, label in [("recall@1","Recall@1"), ("recall@3","Recall@3"),
                           ("recall@5","Recall@5"), ("mrr","MRR"), ("token_f1","Token-F1")]:
        base = bl_l.get(metric, 0.0)
        val  = locomo.get(metric, 0.0)
        print(f"║  {label:<13s}║  {base:.3f}    ║  {val:.3f}    ║  "
              f"{_delta(val, base):<8s} {_pct(val, base):<9s}║")

    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Tier breakdown                                             ║")
    for tk, tlabel in [("tier1","Tier 1 recent"), ("tier2","Tier 2 distant"), ("tier3","Tier 3 cross-turn")]:
        tm = locomo.get("tier_breakdown", {}).get(tk, {})
        r1 = tm.get("recall@1", 0.0)
        r5 = tm.get("recall@5", 0.0)
        f1 = tm.get("token_f1", 0.0)
        print(f"║  {tlabel:<17s} R@1={r1:.3f}  R@5={r5:.3f}  F1={f1:.3f}        ║")

    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  MuSiQue-style  (20 facts, 9 multi-hop queries)             ║")
    print("╠═══════════════╦═══════════╦═══════════╦════════════════════╣")
    print("║  Metric       ║  Baseline ║  Vex 1.0  ║  Delta             ║")
    print("╠═══════════════╬═══════════╬═══════════╬════════════════════╣")

    for metric, label in [("recall@1","Recall@1"), ("recall@5","Recall@5"),
                           ("mrr","MRR"), ("token_f1","Token-F1"),
                           ("graph_hit_rate","Graph Hit")]:
        base = bl_m.get(metric, 0.0)
        val  = musique.get(metric, 0.0)
        print(f"║  {label:<13s}║  {base:.3f}    ║  {val:.3f}    ║  "
              f"{_delta(val, base):<8s} {_pct(val, base):<9s}║")

    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Hop breakdown                                              ║")
    for hk in ("2hop", "3hop", "4hop"):
        hm = musique.get("hop_breakdown", {}).get(hk, {})
        r1 = hm.get("recall@1", 0.0)
        gh = hm.get("graph_hit_rate", 0.0)
        f1 = hm.get("token_f1", 0.0)
        n  = hm.get("n", 0)
        print(f"║  {hk:<17s} R@1={r1:.3f}  Graph={gh:.3f}  F1={f1:.3f}  n={n}   ║")

    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Latency  LoCoMo {locomo.get('avg_latency_ms', 0):.1f} ms avg"
          f"  /  MuSiQue {musique.get('avg_latency_ms', 0):.1f} ms avg"
          + " " * max(0, 13 - len(f"{locomo.get('avg_latency_ms',0):.1f}") - len(f"{musique.get('avg_latency_ms',0):.1f}")) + "║")
    print("╚══════════════════════════════════════════════════════════════╝")


def print_readme_table(locomo: dict, musique: dict) -> None:
    """Markdown table for direct README insertion."""
    print()
    print("```")
    print("## Benchmark Results")
    print()
    print("### LoCoMo-style  (60-turn conversation, 15 queries)")
    print()
    print("| Metric     | Baseline | MemoVex | Delta   |")
    print("|------------|----------|--------------------|---------|")
    bl_l = BASELINE["locomo"]
    for metric, label in [("recall@1","Recall@1"), ("recall@3","Recall@3"),
                           ("recall@5","Recall@5"), ("mrr","MRR"), ("token_f1","Token-F1")]:
        base = bl_l.get(metric, 0.0)
        val  = locomo.get(metric, 0.0)
        print(f"| {label:<10s} | {base:.3f}    | {val:.3f}              | {_delta(val,base)} |")

    print()
    print("| Tier                  | Recall@1 | Recall@5 | Token-F1 |")
    print("|-----------------------|----------|----------|----------|")
    tier_labels = {"tier1": "Tier 1 (recent, turns 40-60)",
                   "tier2": "Tier 2 (distant, turns 1-20)",
                   "tier3": "Tier 3 (cross-turn inference)"}
    for tk, tlabel in tier_labels.items():
        tm = locomo.get("tier_breakdown", {}).get(tk, {})
        r1 = tm.get("recall@1", 0.0)
        r5 = tm.get("recall@5", 0.0)
        f1 = tm.get("token_f1", 0.0)
        print(f"| {tlabel:<21s} | {r1:.3f}    | {r5:.3f}    | {f1:.3f}    |")

    print()
    print("### MuSiQue-style  (20 facts, 9 multi-hop queries)")
    print()
    print("| Metric         | Baseline | MemoVex | Delta   |")
    print("|----------------|----------|--------------------|---------|")
    bl_m = BASELINE["musique"]
    for metric, label in [("recall@1","Recall@1"), ("recall@5","Recall@5"),
                           ("mrr","MRR"), ("token_f1","Token-F1"),
                           ("graph_hit_rate","Graph Hit Rate")]:
        base = bl_m.get(metric, 0.0)
        val  = musique.get(metric, 0.0)
        print(f"| {label:<14s} | {base:.3f}    | {val:.3f}              | {_delta(val,base)} |")

    print()
    print("| Hops | Recall@1 | Recall@5 | Graph Hit Rate | Token-F1 |")
    print("|------|----------|----------|----------------|----------|")
    for hk in ("2hop", "3hop", "4hop"):
        hm = musique.get("hop_breakdown", {}).get(hk, {})
        r1 = hm.get("recall@1", 0.0)
        r5 = hm.get("recall@5", 0.0)
        gh = hm.get("graph_hit_rate", 0.0)
        f1 = hm.get("token_f1", 0.0)
        n  = hm.get("n", 0)
        label = f"{hk[0]}-hop (n={n})"
        print(f"| {label:<4s} | {r1:.3f}    | {r5:.3f}    | {gh:.3f}          | {f1:.3f}    |")

    avg_l = locomo.get("avg_latency_ms", 0)
    avg_m = musique.get("avg_latency_ms", 0)
    print()
    print(f"> Latency: {avg_l:.1f} ms avg (LoCoMo) / {avg_m:.1f} ms avg (MuSiQue) — CPU, no embeddings")
    print("> Baseline: SimpleMemory (token overlap only, no channels/graph)")
    print("```")


def main() -> None:
    p = argparse.ArgumentParser(description="MemoVex benchmark runner")
    p.add_argument("--real", action="store_true",
                   help="Use real HuggingFace datasets (LoCoMo + MuSiQue)")
    p.add_argument("--samples", type=int, default=100,
                   help="Samples per benchmark when --real (0 = all)")
    p.add_argument("--embeddings", action="store_true",
                   help="Enable sentence-transformers (requires installation)")
    p.add_argument("--json", action="store_true", help="Output raw JSON")
    p.add_argument("--save", metavar="PATH", help="Save results to JSON file")
    p.add_argument("--readme", action="store_true",
                   help="Print Markdown table for README insertion")
    p.add_argument("--llm", action="store_true",
                   help="Enable LLM generation layer (requires OPENAI_API_KEY)")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="OpenAI model to use with --llm (default: gpt-4o-mini)")
    args = p.parse_args()

    # Build LLM layer if requested
    llm = None
    if args.llm:
        from memovex.core.llm_layer import LLMLayer
        llm = LLMLayer(model=args.model)
        if not llm.available:
            print("WARNING: LLMLayer unavailable — check OPENAI_API_KEY. "
                  "Running retrieval-only.")
            llm = None
        else:
            print(f"LLM generation enabled: {args.model}")

    t_total = time.time()

    if args.real:
        from benchmarks.locomo_real import run_locomo_real
        from benchmarks.musique_real import run_musique_real
        print(f"Running LoCoMo REAL evaluation (n={args.samples or 'all'})…")
        locomo = run_locomo_real(n_samples=args.samples,
                                 embeddings_enabled=args.embeddings,
                                 verbose=not args.json,
                                 llm=llm)
        print(f"\nRunning MuSiQue REAL evaluation (n={args.samples or 'all'})…")
        musique = run_musique_real(n_samples=args.samples,
                                   embeddings_enabled=args.embeddings,
                                   verbose=not args.json,
                                   llm=llm)
    else:
        print("Running LoCoMo evaluation…")
        locomo = run_locomo(embeddings_enabled=args.embeddings, verbose=not args.json)

        print("\nRunning MuSiQue evaluation…")
        musique = run_musique(embeddings_enabled=args.embeddings, verbose=not args.json)

    total_s = round(time.time() - t_total, 1)

    if args.json:
        print(json.dumps({"locomo": locomo, "musique": musique,
                          "total_seconds": total_s}, indent=2))
        return

    print_summary_table(locomo, musique)

    if args.readme:
        print_readme_table(locomo, musique)

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"locomo": locomo, "musique": musique,
                       "total_seconds": total_s}, f, indent=2)
        print(f"\nResults saved to {path}")

    print(f"\nTotal time: {total_s}s")


if __name__ == "__main__":
    main()
