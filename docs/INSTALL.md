# Installation Guide

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python      | ≥ 3.10  | 3.11 recommended  |
| Docker      | ≥ 24    | for Qdrant + Redis (optional) |
| pip         | ≥ 23    | |

Optional but recommended:

- `sentence-transformers` — enables dense vector embeddings (falls back to BoW without it).
- `networkx` — enables graph traversal channel (graceful degradation without it; in `requirements.txt` by default).

---

## 1. Clone the repository

```bash
git clone https://github.com/axisdynamics/memovex
cd memovex
```

---

## 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

To enable extras (from the cloned repo):

```bash
pip install -e .[embeddings]           # dense vector search
pip install -e .[benchmarks]           # LoCoMo + MuSiQue datasets
pip install -e .[benchmarks,llm]       # full benchmark stack with LLM reader
```

Available extras: `api`, `vector`, `cache`, `chroma`, `embeddings`, `llm`, `benchmarks`, `dev`, `all`.

### Alternative: install without cloning

If you don't need the plugin hooks or scripts, install directly from GitHub:

```bash
pip install git+https://github.com/axisdynamics/memovex.git

# With extras:
pip install "git+https://github.com/axisdynamics/memovex.git#egg=memovex[embeddings]"
pip install "git+https://github.com/axisdynamics/memovex.git#egg=memovex[api,vector,cache]"
```

---

## 3. Infrastructure (Qdrant + Redis)

### Option A — standalone install (recommended)

Copies the Docker stack to a permanent directory so you can delete the repo afterwards:

```bash
python3 scripts/setup_docker.py              # installs to ~/memovex
python3 scripts/setup_docker.py --start      # install + docker compose up -d immediately
python3 scripts/setup_docker.py --dir /opt/memovex   # custom path
```

Or with curl, no clone needed at all:

```bash
mkdir -p ~/memovex
curl -o ~/memovex/Dockerfile \
  https://raw.githubusercontent.com/axisdynamics/memovex/main/docker/Dockerfile.standalone
curl -o ~/memovex/docker-compose.yml \
  https://raw.githubusercontent.com/axisdynamics/memovex/main/docker/docker-compose.standalone.yml
cd ~/memovex && docker compose up -d --build
```

Both approaches produce an identical `~/memovex/` directory:

```
~/memovex/
├── Dockerfile          ← installs memovex from GitHub (no repo needed)
├── docker-compose.yml  ← Qdrant + Redis + memovex-api
└── .env                ← optional feature flags (embeddings, chroma)
```

This starts three containers on an isolated `memovex` bridge network: `memovex-qdrant` (port 6333), `memovex-redis` (port 6379), and `memovex-api` (port 7914).

### Option B — from the cloned repo

```bash
docker compose -f docker/docker-compose.yml up -d
```

Requires staying in the repo directory since the build context points to it.

### Option C — your own Qdrant / Redis

```bash
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

### Option D — run with no infrastructure at all

The orchestrator gracefully degrades when Qdrant or Redis are absent (BoW semantic, no caching). Snapshot persistence still works against a local JSON file.

---

## 4. Verify the installation

```bash
# Run all unit + integration tests (140 of them)
pytest -q

# Smoke test
python3 -c "
from memovex import MemoVexOrchestrator
b = MemoVexOrchestrator(agent_id='test', embeddings_enabled=False)
b.initialize()
mid = b.store('Installation verified')
r = b.retrieve('installation', top_k=1)
print('OK:', r[0].memory.text if r else 'no results')
b.shutdown()
"
```

---

## 5. Agent plugin setup

### Hook strategy: API-first with direct fallback

All hooks attempt to reach the local API server first (`localhost:7914`). If the API is running, the hook gets full capabilities — Qdrant, Redis, Chroma, embeddings, and the knowledge graph — all already warm. If the API is not running, the hook falls back automatically to direct mode (BoW + local JSON snapshot), so the plugin always works regardless of infrastructure.

```
Hook fired
  │
  ├─► POST localhost:7914/api/{agent}/... (API running → full stack)
  └─► API unreachable → direct fallback (BoW + snapshot, no server needed)
```

**Recommended:** start the API before using Claude, Hermes, or OpenClaw:

```bash
uvicorn memovex.api:app --host 0.0.0.0 --port 7914
```

### Claude Code

```bash
# Generate settings.json with correct absolute paths
python3 scripts/setup_plugin.py --agent claude \
  --output /path/to/your/project/.claude/settings.json

# Verify the inject hook responds (API running or fallback)
echo '{"prompt": "test query"}' | python3 plugins/claude/hooks/memory_inject.py

# Verify the store hook responds
echo '{}' | python3 plugins/claude/hooks/memory_store.py
```

See [plugins/claude/README.md](../plugins/claude/README.md) for full setup.

### Hermes

```bash
python3 scripts/setup_plugin.py --agent hermes \
  --output /path/to/your/project/.claude/settings.json
```

```python
from memovex import create_hermes_memory

plugin = create_hermes_memory()
context = plugin.prefetch("what does the user prefer?")
plugin.sync_turn(user_msg, assistant_msg, session_id="s1")
plugin.shutdown()
```

See [plugins/hermes/README.md](../plugins/hermes/README.md) for full setup.

### OpenClaw

```bash
python3 scripts/setup_plugin.py --agent openclaw \
  --output /path/to/your/project/.claude/settings.json
```

```python
from memovex import create_openclaw_memory

bank = create_openclaw_memory()
mid = bank.store("User prefers concise answers")
context = bank.prefetch("user preferences")
bank.shutdown()
```

See [plugins/openclaw/README.md](../plugins/openclaw/README.md) for full setup.

---

## 6. REST API

The REST API serves all three agents from a single process:

```bash
# Direct
uvicorn memovex.api:app --host 0.0.0.0 --port 7914

# Via Docker Compose
docker compose -f docker/docker-compose.yml up memovex-api
```

Test:

```bash
curl http://localhost:7914/health
curl -X POST http://localhost:7914/api/claude/store \
  -H "Content-Type: application/json" \
  -d '{"text": "MemoVex installed correctly"}'
curl -X POST http://localhost:7914/api/claude/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "MemoVex", "top_k": 3}'
```

---

## Environment variables

| Variable                       | Default                    | Description                                        |
|--------------------------------|----------------------------|----------------------------------------------------|
| `QDRANT_HOST`                  | `localhost`                | Qdrant server hostname                             |
| `QDRANT_PORT`                  | `6333`                     | Qdrant REST port                                   |
| `REDIS_HOST`                   | `localhost`                | Redis hostname                                     |
| `REDIS_PORT`                   | `6379`                     | Redis port                                         |
| `EMBEDDINGS_ENABLED`           | `false`                    | Toggle sentence-transformers in the API            |
| `CHROMA_ENABLED`               | `false`                    | Enable embedded Chroma per-agent (no extra container) |
| `CHROMA_PERSIST_DIR`           | `./data/chroma`            | Base directory for per-agent Chroma collections    |
| `MEMOVEX_PERSISTENCE_ENABLED`  | `true`                     | API auto-loads/saves snapshots                     |
| `MEMOVEX_SNAPSHOT_DIR`         | `./data/snapshots`         | Where snapshots are written                        |
| `LOG_LEVEL`                    | `INFO`                     | API log level                                      |
| `MEMOVEX_API_URL`              | `http://localhost:7914`    | Hook target — change for remote API               |
| `MEMOVEX_API_TIMEOUT`          | `3` (inject) / `5` (store) | Seconds before hook falls back to direct mode      |
