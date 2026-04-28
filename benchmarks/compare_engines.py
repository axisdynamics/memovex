#!/usr/bin/env python3
"""
MemoVex benchmark: BoW retrieval vs Embeddings retrieval.

Runs runner.py as isolated subprocesses so each configuration gets a clean
Python environment (avoids sys.modules cache issues when running multiple
configurations in sequence).

Engines
-------
  bow    Bag-of-Words retrieval  — no embedding model required, fast
  emb    Semantic retrieval      — requires sentence-transformers, slower but
                                   more accurate on paraphrase queries

Usage examples
--------------
  python compare_engines.py                    # BoW only, 50 samples
  python compare_engines.py --emb              # BoW + Emb side-by-side
  python compare_engines.py --samples 100      # larger sample set
  python compare_engines.py --llm              # add LLM re-ranking (needs OPENAI_API_KEY)
  python compare_engines.py --pkg-path /path/to/memovex  # explicit package root

If --pkg-path is omitted the script auto-detects the installed memovex package.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────
BENCHMARKS_DIR = Path(__file__).resolve().parent
RUNNER         = BENCHMARKS_DIR / "runner.py"
RESULTS_DIR    = BENCHMARKS_DIR / "results"


# ── Terminal colours ──────────────────────────────────────────────────────────
RST  = "\033[0m"
BOLD = "\033[1m"
GRN  = "\033[92m"
RED  = "\033[91m"
GRY  = "\033[90m"
CYN  = "\033[96m"
YLW  = "\033[93m"


# ── Package auto-detection ─────────────────────────────────────────────────────

def _detect_pkg_path() -> Optional[Path]:
    """Return the parent directory of the installed memovex package, or None."""
    spec = importlib.util.find_spec("memovex")
    if spec and spec.submodule_search_locations:
        locations = list(spec.submodule_search_locations)
        if locations:
            return Path(locations[0]).parent
    return None


# ============================================================================
# Subprocess runner
# ============================================================================

def run_engine(pkg_path: Optional[Path], label: str, locomo_samples: int,
               musique_samples: int, embeddings: bool,
               llm: bool, model: str) -> Optional[Dict[str, Any]]:
    out_path = RESULTS_DIR / f"{label.replace(' ', '_')}_tmp.json"
    RESULTS_DIR.mkdir(exist_ok=True)

    cmd = [
        sys.executable, str(RUNNER),
        "--label",           label,
        "--locomo-samples",  str(locomo_samples),
        "--musique-samples", str(musique_samples),
        "--output",          str(out_path),
    ]
    if pkg_path:
        cmd += ["--pkg-path", str(pkg_path)]
    if embeddings:
        cmd.append("--embeddings")
    if llm:
        cmd += ["--llm", "--model", model]

    pkg_display = pkg_path.name if pkg_path else "(installed)"
    print(f"\n{BOLD}{'─'*60}{RST}")
    print(f"{CYN}▶ [{label}]{RST}  pkg: {pkg_display}")
    print(f"   cmd: {' '.join(cmd[2:])}")

    proc = subprocess.run(cmd, text=True, capture_output=False)
    if proc.returncode != 0:
        print(f"{RED}  ERROR: runner.py exited with code {proc.returncode}{RST}")
        return None

    if not out_path.exists():
        print(f"{RED}  ERROR: output file not written{RST}")
        return None

    result = json.loads(out_path.read_text())
    out_path.unlink(missing_ok=True)
    return result


# ============================================================================
# Comparison display
# ============================================================================

def _delta_str(base: float, val: float) -> str:
    d = val - base
    if abs(d) < 0.001:
        return f"{GRY}  —  {RST}"
    color = GRN if d > 0 else RED
    arrow = "▲" if d > 0 else "▼"
    return f"{color}{arrow} {d:+.3f}{RST}"


def _col(val: float) -> str:
    return f"{val:.3f}"


def _header(labels: List[str]) -> str:
    cols = "  ".join(f"{l:>10s}" for l in labels)
    return f"  {'Metric':<22s}  {cols}"


def _row(metric: str, values: List[float], base_idx: int = 0) -> str:
    base = values[base_idx]
    parts = []
    for i, v in enumerate(values):
        if i == base_idx:
            parts.append(f"{_col(v):>10s}")
        else:
            parts.append(f"{_col(v):>6s} {_delta_str(base, v)}")
    return f"  {metric:<22s}  {'  '.join(parts)}"


def _section(title: str, width: int) -> str:
    return f"\n  {GRY}── {title} {'─' * (width - len(title) - 5)}{RST}"


def print_comparison(results: List[Dict[str, Any]]) -> None:
    labels = [r["label"] for r in results]
    n = len(labels)
    W = 26 + n * 14
    W = max(W, 60)

    print(f"\n{'='*W}")
    print(f"  {BOLD}MemoVex Benchmark Results{RST}")
    if n > 1:
        print(f"  {YLW}Base (Δ reference): {labels[0]}{RST}")
    print(_header(labels))
    print(f"{'─'*W}")

    # ── LoCoMo ───────────────────────────────────────────────────────────────
    print(_section("LoCoMo  (Percena/locomo-mc10)", W))
    locomo_metrics = [
        ("mc_accuracy",    "MC Accuracy"),
        ("top5_accuracy",  "Top-5 Accuracy"),
        ("mrr",            "MRR"),
        ("avg_latency_ms", "Latency avg (ms)"),
        ("p99_latency_ms", "Latency p99 (ms)"),
    ]
    for key, label in locomo_metrics:
        vals = [r["locomo"].get(key, 0) for r in results]
        print(_row(label, vals))

    # ── LoCoMo by question type ───────────────────────────────────────────────
    all_types: set = set()
    for r in results:
        all_types |= set(r["locomo"].get("type_breakdown", {}).keys())
    if all_types:
        print(f"\n  {GRY}  by question type:{RST}")
        for qt in sorted(all_types):
            vals = [
                r["locomo"].get("type_breakdown", {}).get(qt, {}).get("mc_accuracy", 0)
                for r in results
            ]
            print(_row(f"    {qt}", vals))

    # ── LoCoMo LLM metrics ────────────────────────────────────────────────────
    if any(r["locomo"].get("llm_used") for r in results):
        print(f"\n  {GRY}  LLM re-ranking:{RST}")
        for key, lbl in [("mc_accuracy", "LLM MC Accuracy"), ("mrr", "LLM MRR")]:
            vals = [
                r["locomo"].get(key, 0) if r["locomo"].get("llm_used") else float("nan")
                for r in results
            ]
            finite = [v for v in vals if v == v]
            if finite:
                print(_row(lbl, vals))

    # ── MuSiQue ──────────────────────────────────────────────────────────────
    print(_section("MuSiQue (bdsaglam/musique, answerable)", W))
    musique_metrics = [
        ("recall@1",       "Recall@1"),
        ("recall@5",       "Recall@5"),
        ("mrr",            "MRR"),
        ("token_f1",       "Token F1"),
        ("graph_hit_rate", "Graph Hit Rate"),
        ("avg_latency_ms", "Latency avg (ms)"),
    ]
    for key, lbl in musique_metrics:
        vals = [r["musique"].get(key, 0) for r in results]
        print(_row(lbl, vals))

    # ── MuSiQue LLM metrics ───────────────────────────────────────────────────
    if any(r["musique"].get("llm_token_f1") is not None for r in results):
        print(f"\n  {GRY}  LLM re-ranking:{RST}")
        for key, lbl in [("llm_token_f1", "LLM Token F1"),
                         ("llm_exact_match", "LLM Exact Match")]:
            vals = [r["musique"].get(key, float("nan")) for r in results]
            finite = [v for v in vals if v is not None and v == v]
            if finite:
                print(_row(lbl, [v if v is not None else float("nan") for v in vals]))

    # ── MuSiQue by hop count ─────────────────────────────────────────────────
    all_hops: set = set()
    for r in results:
        all_hops |= set(r["musique"].get("hop_breakdown", {}).keys())
    if all_hops:
        print(f"\n  {GRY}  by hop count (Recall@1):{RST}")
        for hk in sorted(all_hops):
            vals = [
                r["musique"].get("hop_breakdown", {}).get(hk, {}).get("recall@1", 0)
                for r in results
            ]
            print(_row(f"    {hk}", vals))

    print(f"\n{'='*W}")

    # ── Run metadata ──────────────────────────────────────────────────────────
    print(f"\n  {BOLD}Run info{RST}")
    for r in results:
        mode  = r.get("mode", "?")
        ls    = r.get("locomo_samples", "?")
        ms    = r.get("musique_samples", "?")
        pkg   = Path(r.get("pkg_path", "installed")).name or "installed"
        lmems = r["locomo"].get("n_summaries", 0) + r["locomo"].get("n_turns", 0)
        print(f"  {r['label']:<14s}  pkg={pkg}  mode={mode}  "
              f"locomo_n={ls}({lmems} mems)  musique_n={ms}")
    print()


# ============================================================================
# Persistence
# ============================================================================

def save_results(results: List[Dict[str, Any]]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"benchmark_{ts}.json"
    payload = {
        "run_date": datetime.date.today().isoformat(),
        "engines":  results,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    latest = RESULTS_DIR / "benchmark_latest.json"
    latest.write_text(json.dumps(payload, indent=2, default=str))
    return path


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Benchmark memovex: BoW retrieval vs Embeddings retrieval.
            Datasets: LoCoMo (MC@10 long-term memory) + MuSiQue (multi-hop reasoning).
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--samples",         type=int, default=50,
                   help="LoCoMo + MuSiQue sample count (default: 50)")
    p.add_argument("--locomo-samples",  type=int, default=None,
                   help="Override LoCoMo sample count independently")
    p.add_argument("--musique-samples", type=int, default=None,
                   help="Override MuSiQue sample count independently")
    p.add_argument("--emb",            action="store_true",
                   help="Add embeddings variant alongside BoW (requires sentence-transformers)")
    p.add_argument("--llm",            action="store_true",
                   help="Enable LLM re-ranking layer (requires OPENAI_API_KEY)")
    p.add_argument("--model",          default="gpt-4o-mini",
                   help="OpenAI model for LLM layer (default: gpt-4o-mini)")
    p.add_argument("--pkg-path",       default=None,
                   help="Path to the memovex package root (auto-detected if omitted)")
    p.add_argument("--skip-bow",       action="store_true",
                   help="Skip BoW engine (run Emb only — requires --emb)")
    args = p.parse_args()

    locomo_n  = args.locomo_samples  or args.samples
    musique_n = args.musique_samples or args.samples

    # Resolve package path
    pkg_path: Optional[Path] = None
    if args.pkg_path:
        pkg_path = Path(args.pkg_path).resolve()
        if not pkg_path.exists():
            print(f"{RED}ERROR: --pkg-path not found: {pkg_path}{RST}")
            sys.exit(1)
    else:
        pkg_path = _detect_pkg_path()
        if pkg_path:
            print(f"{GRY}Auto-detected memovex at: {pkg_path}{RST}")
        else:
            print(f"{GRY}memovex not detected via importlib — "
                  f"assuming it is importable in the active environment{RST}")

    if args.skip_bow and not args.emb:
        print(f"{RED}ERROR: --skip-bow requires --emb (nothing to run){RST}")
        sys.exit(1)

    # Build engine list: (label, embeddings_enabled)
    engines: List[Tuple[str, bool]] = []
    if not args.skip_bow:
        engines.append(("bow", False))
    if args.emb:
        engines.append(("emb", True))

    print(f"\n{BOLD}MemoVex Benchmark Suite{RST}")
    print(f"  Engines        : {', '.join(label for label, _ in engines)}")
    print(f"  LoCoMo samples : {locomo_n}")
    print(f"  MuSiQue samples: {musique_n}")
    print(f"  LLM            : {'yes (' + args.model + ')' if args.llm else 'no'}")

    results = []
    for label, emb in engines:
        r = run_engine(
            pkg_path=pkg_path,
            label=label,
            locomo_samples=locomo_n,
            musique_samples=musique_n,
            embeddings=emb,
            llm=args.llm,
            model=args.model,
        )
        if r is not None:
            results.append(r)

    if not results:
        print(f"{RED}No results collected — all engines failed.{RST}")
        sys.exit(1)

    print_comparison(results)

    out_path = save_results(results)
    print(f"  Results saved → {out_path}")
    print(f"             and → {RESULTS_DIR / 'benchmark_latest.json'}\n")


if __name__ == "__main__":
    main()
