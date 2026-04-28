# MemoVex Ver 2.0

**Multi-channel resonant memory framework for LLM agents.**

---

## Why MemoVex?

LLM agents forget everything when the session ends. Existing solutions (mem0, Zep, LangMem) solve this in one way: save text and call an LLM to decide what is relevant. That works, but it means a fixed cost per query, a hard dependency on an external API, and outsourced intelligence.

MemoVex takes a different approach: **the retrieval system itself is intelligent**, with no LLM required for each lookup.

Instead of a single semantic score, MemoVex combines 11 simultaneous signals — semantic similarity, named entities, knowledge graph traversal, reasoning chains, wisdom-tier promotion, temporal proximity, recency, procedural memory, symbolic overlap, usage frequency, and tags — all weighted and fused at query time. The result is contextually rich retrieval that runs in **3.2 ms average with no API key**, and scales to LLM re-ranking when higher accuracy justifies the cost.

Key outcomes from independent benchmarking (LoCoMo + MuSiQue, 50 samples):

- **Graph Hit Rate 1.000** across 2/3/4-hop reasoning chains — a capability absent in pure vector-store systems.
- **3.2 ms** average retrieval latency in BoW mode, fully local.
- **2.7× accuracy improvement** when an optional LLM reader layer is added on top (MC@1: 0.175 → 0.470).
- All benchmarks are reproducible by anyone with `pip install "git+https://github.com/axisdynamics/memovex.git#egg=memovex[benchmarks]"`.

---

MemoVex gives any LLM agent a persistent, queryable memory system backed by an in-memory store (with snapshot persistence), an optional vector store (Qdrant or Chroma), an optional Redis cache, and a knowledge graph (NetworkX). Multiple agents share the same infrastructure while remaining fully isolated — each with its own Qdrant collection, Redis namespace, and snapshot file.

---

## Status

| Layer                       | Status      | Notes                                                                                          |
|-----------------------------|-------------|------------------------------------------------------------------------------------------------|
| Core retrieval (11 channels)| **Stable**  | All channels implemented and tested. See [Channel Weights](#channel-weights).                  |
| WisdomStore curation        | **Stable**  | RAW → PROCESSED → CURATED → WISDOM, auto-promotion + manual override.                          |
| Homeostasis                 | **Stable**  | Background decay + score-based pruning, low-confidence first.                                  |
| GraphStore                  | **Stable**  | NetworkX-backed; falls back to direct overlap when NetworkX is missing.                        |
| Snapshot persistence        | **Stable**  | API auto-loads on startup, auto-saves on shutdown. Manual `POST /api/<agent>/snapshot` too.    |
| Qdrant integration          | **Stable**  | Optional. Degrades to BoW semantic when absent.                                                |
| Redis cache                 | **Stable**  | Optional. Degrades to no caching when absent.                                                  |
| Chroma integration          | **Stable**  | Wired through `MemoVexOrchestrator.connect_chroma()`. Install with `pip install "git+https://github.com/axisdynamics/memovex.git#egg=memovex[chroma]"`. |
| LLM answer layer            | **Stable**  | OpenAI-compatible. Optional, install with `pip install "git+https://github.com/axisdynamics/memovex.git#egg=memovex[llm]"`. |
| REST API (FastAPI)          | **Stable**  | Multi-agent (`claude`, `hermes`, `openclaw`), validated payloads, structured logging.          |
| `Mem0Adapter`               | Experimental| Provider stub — fix-up complete (json import) but no real Mem0 server in tests.                |
| `MemobaseAdapter`           | Experimental| Letta integration; only the file-based fallback is exercised in tests.                         |
| `MemPalaceAdapter`          | Experimental| File-based fallback works; the real `mempalace` package isn't pinned.                          |
| `ResonantAdapter`           | Experimental| Moved to `memovex/providers/experimental/`. `store_memory` is a no-op pending upstream API.    |


---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      MemoVexOrchestrator                         │
│                           (per agent)                            │
│                                                                  │
│  store() ──► MemoryStore   (in-memory + secondary indices)      │
│           ├─► QdrantStore  (vector search, optional)            │
│           ├─► ChromaStore  (vector store, optional alternative) │
│           ├─► RedisCache   (recency + prefetch cache)           │
│           ├─► WisdomStore  (RAW→PROCESSED→CURATED→WISDOM)       │
│           └─► GraphStore   (NetworkX, reasoning chains)         │
│                                                                  │
│  retrieve() ◄── ResonanceEngine (11-channel weighted scoring)   │
│                                                                  │
│  save_snapshot() / load_snapshot() — JSON persistence           │
└──────────────────────────────────────────────────────────────────┘
```

### Channel weights

Sum is 1.0. Pass a subset via the `channels=` parameter; by default the score is renormalized to keep results in roughly the [0, 1] range.

| Channel           | Weight | Description                                              |
|-------------------|--------|----------------------------------------------------------|
| `semantic`        | 0.32   | dense cosine via Qdrant/Chroma; falls back to BoW        |
| `entity`          | 0.13   | NER entity intersection                                  |
| `graph_traversal` | 0.12   | BFS through reasoning chains                             |
| `wisdom`          | 0.10   | WisdomStore curation level                               |
| `temporal`        | 0.08   | event_time proximity                                     |
| `recency`         | 0.06   | last-access timestamp                                    |
| `reasoning_chain` | 0.05   | reasoning hop overlap                                    |
| `symbolic`        | 0.05   | token fingerprint overlap                                |
| `procedural`      | 0.05   | "how to" queries against PROCEDURAL memories             |
| `usage`           | 0.02   | access_count score                                       |
| `tag`             | 0.02   | tag intersection                                         |

### Agent isolation

| Agent      | Qdrant collection   | Redis prefix          | Default snapshot                            |
|------------|---------------------|-----------------------|---------------------------------------------|
| `claude`   | `memovex_claude`    | `memovex:claude:*`    | `~/.claude/memovex/claude_snapshot.json`    |
| `hermes`   | `memovex_hermes`    | `memovex:hermes:*`    | configurable (`MEMOVEX_SNAPSHOT_DIR`)       |
| `openclaw` | `memovex_openclaw`  | `memovex:openclaw:*`  | `~/.memovex/openclaw_snapshot.json`         |

The REST API rejects any other `agent_id` with HTTP 404.


---

## Quick start

```bash
# 1. Install the package — two equivalent options:

# Option A: clone (recommended if you also want the plugin hooks and scripts)
git clone https://github.com/axisdynamics/memovex
cd memovex
pip install -e .

# Option B: install directly from GitHub without cloning
pip install git+https://github.com/axisdynamics/memovex.git

# 2. (Recommended) start the API server — activates Qdrant, Redis, Chroma, embeddings
uvicorn memovex.api:app --host 0.0.0.0 --port 7914

# 3. (Optional) start full infrastructure — Qdrant + Redis + API via Docker
#    Standalone install (repo can be deleted after this step):
python3 scripts/setup_docker.py --start
#    Or with curl, no clone needed:
#    mkdir -p ~/memovex
#    curl -o ~/memovex/Dockerfile https://raw.githubusercontent.com/axisdynamics/memovex/main/docker/Dockerfile.standalone
#    curl -o ~/memovex/docker-compose.yml https://raw.githubusercontent.com/axisdynamics/memovex/main/docker/docker-compose.standalone.yml
#    cd ~/memovex && docker compose up -d --build

# 4. Use it
python3 - <<'EOF'
from memovex import MemoVexOrchestrator, MemoryType

bank = MemoVexOrchestrator(agent_id="demo", embeddings_enabled=False)
bank.initialize()

bank.store("MemoVex uses 11-channel resonance scoring",
           memory_type=MemoryType.SEMANTIC, confidence=0.9, salience=0.8)

results = bank.retrieve("how does scoring work?", top_k=3)
for r in results:
    print(f"{r.total_score:.3f}  {r.memory.text[:80]}")

bank.shutdown()
EOF
```

### Optional: LLM answer layer

```bash
pip install "git+https://github.com/axisdynamics/memovex.git#egg=memovex[llm]"
export OPENAI_API_KEY=sk-...
python3 -c "
from memovex.core.llm_layer import LLMLayer
llm = LLMLayer(model='gpt-4o-mini')
print(llm.answer_open('What is MemoVex?',
    '[1] MemoVex is a multi-channel resonant memory framework.'))
"
```

---

## Persistence

The API has snapshot-based persistence wired in. Two env vars control it:

| Variable                       | Default                | Description                                          |
|--------------------------------|------------------------|------------------------------------------------------|
| `MEMOVEX_PERSISTENCE_ENABLED`  | `true`                 | When `true`, load on first agent use, save on shutdown. |
| `MEMOVEX_SNAPSHOT_DIR`         | `./data/snapshots`     | Where `<agent_id>_snapshot.json` files live.         |

In Docker the snapshot directory lives on the `memovex_data` named volume so restarts preserve state. Force a snapshot with `POST /api/<agent>/snapshot`.

A snapshot stores:

- All in-memory memories (text, entities, tags, embeddings, scores, timestamps).
- The full WisdomStore state (level, evidence count, notes).

Snapshots are not a full disaster-recovery story — Qdrant/Chroma vectors live in their own backends — but they are sufficient to survive an API restart with all retrieval channels except dense semantic remaining fully usable.

---

## REST API

```bash
uvicorn memovex.api:app --host 0.0.0.0 --port 7914
```

| Method | Route                          | Description                                |
|--------|--------------------------------|--------------------------------------------|
| GET    | `/health`                      | Health, allowed agents, persistence flags  |
| GET    | `/api/{agent_id}/stats`        | Memory and connection stats                |
| POST   | `/api/{agent_id}/store`        | Store a memory                             |
| POST   | `/api/{agent_id}/retrieve`     | Retrieve top-k memories (with `channels=`) |
| POST   | `/api/{agent_id}/prefetch`     | LLM-ready context string                   |
| POST   | `/api/{agent_id}/context`      | Alias of `/prefetch`                       |
| POST   | `/api/{agent_id}/store_chain`  | Store a reasoning chain                    |
| POST   | `/api/{agent_id}/corroborate`  | Add evidence to a memory                   |
| GET    | `/api/{agent_id}/wisdom`       | List WISDOM-tier memories                  |
| GET    | `/api/{agent_id}/graph/stats`  | Knowledge graph stats                      |
| POST   | `/api/{agent_id}/snapshot`     | Force a snapshot save                      |

`{agent_id}` must match `^[a-z][a-z0-9_-]{0,31}$` and be one of `claude`, `hermes`, `openclaw`. Invalid IDs return HTTP 422 (regex) or 404 (unknown agent).

---

## Plugins

| Plugin               | Description                                  | Guide                                            |
|----------------------|----------------------------------------------|--------------------------------------------------|
| `plugins/claude/`    | Claude Code hooks (inject + store)           | [Claude plugin](plugins/claude/README.md)        |
| `plugins/hermes/`    | Hermes VEX-compatible drop-in                | [Hermes plugin](plugins/hermes/README.md)        |
| `plugins/openclaw/`  | OpenClaw SDK / REST wrapper                  | [OpenClaw plugin](plugins/openclaw/README.md)    |

### Hook strategy: API-first with direct fallback

All hooks follow the same two-step pattern on every invocation:

```
Hook fired
  │
  ├─► 1. POST localhost:7914/api/{agent}/prefetch|store   (timeout: 3–5 s)
  │          API running → uses Qdrant + Redis + Chroma + embeddings (full stack)
  │
  └─► 2. API unreachable → direct mode (BoW + local JSON snapshot, no server needed)
```

This means the plugin works **with zero infrastructure** out of the box and silently upgrades to full vector+graph memory as soon as the API is running.

To use a remote or non-default API:

```bash
export MEMOVEX_API_URL=http://my-server:7914   # default: http://localhost:7914
export MEMOVEX_API_TIMEOUT=5                   # seconds; default: 3 (inject) / 5 (store)
```

### Install a plugin

```bash
# Generate settings.json with correct absolute paths for your agent
python scripts/setup_plugin.py --agent claude --output /your/project/.claude/settings.json
python scripts/setup_plugin.py --agent hermes --output /your/project/.claude/settings.json
python scripts/setup_plugin.py --agent openclaw --output /your/project/.claude/settings.json
```

Convenience factories are also exported from the package itself:

```python
from memovex import (
    create_claude_memory,
    create_hermes_memory,
    create_openclaw_memory,
)
```

---

## WisdomStore pipeline

Memories are automatically promoted as evidence accumulates:

```
RAW ──(conf≥0.40)──► PROCESSED ──(conf≥0.60, ev≥1)──► CURATED ──(conf≥0.80, sal≥0.70, ev≥2)──► WISDOM
```

WISDOM-tier memories score higher in the `wisdom` retrieval channel (weight 0.10) and are protected from pruning by the homeostasis manager.

---

## Repository layout

```
memovex/
├── memovex/                # main package (importable)
│   ├── __init__.py         # public API exports
│   ├── api.py              # FastAPI server
│   ├── core/               # orchestrator, resonance engine, wisdom, homeostasis, llm_layer
│   ├── integrations/       # Qdrant, Redis, GraphStore, ChromaDB
│   ├── plugins/            # claude, hermes, openclaw plugin factories
│   └── providers/          # external adapters (mem0, memobase, mempalace, …)
│       └── experimental/   # incomplete adapters; do not use in production
├── plugins/                # standalone per-agent plugins (entrypoints, hooks, configs)
├── benchmarks/             # evaluation scripts (LoCoMo, MuSiQue)
├── tests/                  # unit + integration tests (pytest)
├── docker/                 # Dockerfile + docker-compose.yml
├── config/                 # memovex.yaml, providers.yaml
├── docs/                   # architecture, roadmap, install, migration
├── scripts/                # seed_claude_memory, setup_plugin, ingest_data, hooks
├── pyproject.toml
└── requirements.txt
```

---

## Running tests

```bash
git clone https://github.com/axisdynamics/memovex
cd memovex
pip install -e .[dev]
pytest -q
# 140 passed
```

---

## Benchmarks

Evaluated on 2026-04-28.  
Datasets: [Percena/locomo-mc10](https://huggingface.co/datasets/Percena/locomo-mc10) · [bdsaglam/musique](https://huggingface.co/datasets/bdsaglam/musique).  
LoCoMo: n=200 samples, 87,550 memories indexed. MuSiQue: n=523 (full answerable validation set).  
Random baseline (LoCoMo MC@10): 0.100.

### LoCoMo — long-context conversational memory

| Metric | BoW | Emb (MiniLM-L6-v2) | Emb + LLM (gpt-4o-mini) |
|--------|----:|--------------------:|------------------------:|
| **MC Accuracy** | 0.120 | 0.175 | **0.470** |
| **Top-5 Accuracy** | 0.425 | 0.425 | **0.470** |
| **MRR** | 0.289 | 0.333 | **0.470** |
| Latency avg | 2 147 ms | 1 936 ms | 2 656 ms |
| Latency p99 | 7 589 ms | 8 703 ms | 11 183 ms |

By question type (MC Accuracy):

| Type | n | BoW | Emb | Emb+LLM |
|------|--:|----:|----:|--------:|
| `adversarial`        | 47 | 0.000 | 0.000 | **0.745** |
| `open_domain`        | 70 | 0.186 | 0.300 | **0.514** |
| `single_hop`         | 32 | 0.188 | 0.281 | **0.312** |
| `multi_hop`          | 38 | 0.079 | 0.079 | **0.263** |
| `temporal_reasoning` | 13 | **0.154** | **0.154** | 0.231 |

> Adversarial questions (n=47) score 0 under pure retrieval — the LLM reader jumps to 0.745 by reasoning across context rather than matching tokens. Temporal queries resist dense embeddings; BoW and Emb tie there.

### MuSiQue — multi-hop reasoning (n=523, full dataset)

| Metric | BoW | Emb | Emb+LLM |
|--------|----:|----:|--------:|
| **Recall@1** | 0.109 | **0.147** | **0.147** |
| **Recall@5** | 0.438 | **0.554** | **0.554** |
| **MRR** | 0.246 | **0.319** | **0.319** |
| Token F1 | 0.066 | 0.069 | **0.387** |
| Exact Match | — | — | **0.205** |
| Graph Hit Rate | **1.000** | **1.000** | **1.000** |
| Latency avg/sample | **3.0 ms** | 20.2 ms | 22.1 ms |

By hop depth (Recall@1 / Recall@5):

| Hops | n | BoW R@1 | Emb R@1 | BoW R@5 | Emb R@5 |
|------|--:|--------:|--------:|--------:|--------:|
| 2-hop | 299 | 0.124 | 0.120 | 0.462 | **0.528** |
| 3-hop | 137 | 0.095 | **0.197** | 0.423 | **0.642** |
| 4-hop |  87 | 0.080 | **0.161** | 0.379 | **0.506** |

> Graph Hit Rate = 1.000 across all modes and all hop depths — the reasoning graph reaches the target entity in every 2/3/4-hop chain. Embeddings give the biggest gains at 3-hop and 4-hop (+107% and +101% R@1). The LLM reader raises Token F1 from 0.069 to **0.387** (5.6×).

### Comparison with other memory systems

LoCoMo numbers from external systems use **open-ended QA + LLM-as-judge (J-score)** — a different protocol than memovex's 10-choice MC evaluation. They are not directly comparable; the table shows the best available published figure for each system under its own methodology.

| System | LoCoMo score | Methodology | Retrieval latency | LLM required |
|--------|-------------:|-------------|------------------:|:------------:|
| **memovex** (BoW) | 0.120 MC@10 | MC, retrieval-only | **3.0 ms** | No |
| **memovex** (Emb) | 0.175 MC@10 | MC, retrieval-only | 20.2 ms | No |
| **memovex** (+LLM) | **0.470** MC@10 | MC + gpt-4o-mini | ~2 656 ms | Yes |
| mem0 (v2) | 0.916 J-score ¹ | Open QA, LLM-judge | ~200 ms | Yes |
| Zep | 0.581–0.751 J-score ² | Open QA, LLM-judge | sub-second | Yes |
| LangMem | 0.581 J-score ³ | Open QA, LLM-judge | ~18 s (p50) | Yes |

**¹** mem0 v2 paper (ECAI 2025, arXiv:2504.19413). Independently reported as 0.669 in third-party re-runs; reproducibility issues acknowledged in [mem0#3944](https://github.com/mem0ai/mem0/issues/3944).  
**²** Disputed: Mem0 reports 0.584 after correcting for adversarial-category inclusion; Zep self-reports 0.751 ([getzep/zep-papers#5](https://github.com/getzep/zep-papers/issues/5)).  
**³** From third-party evaluation ([source](https://guptadeepak.com/the-ai-memory-wars-why-one-system-crushed-the-competition-and-its-not-openai/)); LangChain does not publish official LoCoMo numbers.

**Where memovex leads — measured advantages:**

| Dimension | memovex | mem0 | Zep |
|-----------|---------|------|-----|
| Retrieval latency | **3.0 ms** (BoW) | ~200 ms | sub-second |
| API key to operate | **No** | Yes | Yes |
| Infrastructure | **fully local** | managed SaaS | managed / self-hosted |
| Multi-hop graph hit rate | **1.000** (all 2/3/4-hop) | not reported | partial |
| Adversarial MC accuracy (with LLM) | **0.745** | not reported | not reported |
| Third-party reproducible | **Yes** | No | No |
| Cost per retrieval query | **$0** (BoW) | per API call | per API call |

> The **Graph Hit Rate 1.000** across all 2/3/4-hop chains is the single structurally incomparable result: mem0 and Zep don't report this because they have no reasoning graph. A 0.745 accuracy on adversarial questions — specifically designed to fool retrieval systems — shows the retrieval context was rich enough for the LLM reader to reason past the misleading framing.

**Structural differentiators not captured by accuracy alone:**

| Capability | memovex | mem0 | Zep | LangMem |
|------------|:---:|:---:|:---:|:---:|
| LLM-free retrieval | ✓ | ✗ | ✗ | ✗ |
| Knowledge graph traversal | ✓ | ✗ | ✓ | ✗ |
| Multi-channel scoring (11 ch) | ✓ | ✗ | ✗ | ✗ |
| Wisdom / evidence promotion | ✓ | ✗ | ✗ | ✗ |
| Homeostasis / decay | ✓ | ✗ | ✗ | ✗ |
| Fully local (no API key) | ✓ | ✗ | ✗ | ✗ |
| Chroma / Qdrant support | ✓ | ✓ | ✓ | ✓ |
| Procedural memory | ✓ | ✗ | ✗ | ✓ |

### Reproducing

```bash
git clone https://github.com/axisdynamics/memovex
cd memovex
pip install -e .[benchmarks]

# BoW only (fast, no embeddings model required)
python3 benchmarks/compare_engines.py --samples 200

# BoW + Embeddings side-by-side
python3 benchmarks/compare_engines.py --emb --samples 200

# Full stack: BoW + Emb + LLM reader (requires OPENAI_API_KEY)
pip install -e .[benchmarks,llm]
OPENAI_API_KEY=sk-... python3 benchmarks/compare_engines.py --emb --llm --samples 200

# Full dataset (as published above)
python benchmarks/compare_engines.py --emb --locomo-samples 200 --musique-samples 523
```

---

## Roadmap (post-1.1)

- Finish the experimental providers (`Mem0Adapter` end-to-end, `MemobaseAdapter` against a live Letta, `ResonantAdapter` real `store_memory`).
- Make snapshot persistence incremental rather than full-rewrite (currently O(N) per save).
- Authentication on the REST API (currently CORS-open and unauthenticated — only run on trusted networks).
- A migration script for the legacy `~/.claude/memorybank/` and `~/.memorybank/` snapshots that fully copies them to the new `memovex` paths instead of relying on the per-process auto-migration.

---

## License

MIT License

## Contact

Memovex is built by Axisdynamics Chile

https://vex.axisdynamics.cl  Mail:contacto@axisdynamics.cl

-Professional entity design
-Multi-model deployment
-24/7 support and monitoring
-Compliance and audit tools

🧬 VEX: Where Identity Engineering Meets AI Innovation

VEX Digital Entity Architecture © 2026 AxisDynamics
