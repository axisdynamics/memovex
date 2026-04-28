# OpenClaw Plugin

MemoVex integration for the OpenClaw agent.

Provides a high-level `OpenClawMemory` class with an opinionated API (`remember`, `recall`, `teach`, `reflect`) on top of the full `MemoVexOrchestrator`. Snapshot persistence is automatic — memories survive process restarts without Qdrant being required.

---

## Setup

### 1. Install MemoVex

**Option A — clone (recommended for development):**

```bash
git clone https://github.com/axisdynamics/memovex
cd memovex
pip install -e .
```

**Option B — install directly from GitHub:**

```bash
pip install git+https://github.com/axisdynamics/memovex.git
```

### 2. Option A — Python SDK (recommended)

```python
from plugins.openclaw.plugin import OpenClawMemory

mem = OpenClawMemory()

# Store a user preference
mem.remember(
    "User prefers responses in English, concise and without preamble",
    memory_type="semantic",
    tags={"preferences", "language"},
    confidence=0.9,
    salience=0.8,
)

# Retrieve context before answering
context = mem.recall("how should I format my response?")

# Store a how-to memory (matches the procedural channel)
mem.teach(
    "To restart the service: sudo systemctl restart myapp",
    tags={"ops", "restart"},
)

# Store a reasoning chain
mem.reflect(
    "Concluded that the latency is caused by N+1 queries in the user endpoint",
    hops=[
        {"source": "high_latency", "via": "profiled", "target": "db_queries"},
        {"source": "db_queries",   "via": "analyzed", "target": "n_plus_one"},
    ],
    confidence=0.85,
)

# Reinforce important memories (may promote to WISDOM)
mid = mem.remember("API rate limit is 1000 req/min", confidence=0.95)
mem.reinforce(mid)
mem.reinforce(mid)

# Persist and release
mem.shutdown()
```

### 2. Option B — REST API

Start the MemoVex API:

```bash
uvicorn memovex.api:app --host 0.0.0.0 --port 7914
```

Then call from OpenClaw:

```python
import requests

BASE = "http://localhost:7914/api/openclaw"

# Prefetch context  (also available as /context)
ctx = requests.post(f"{BASE}/prefetch",
    json={"query": "user preferences", "max_tokens": 800}).json()["context"]

# Store a memory
requests.post(f"{BASE}/store", json={
    "text": "User prefers dark mode",
    "memory_type": "semantic",
    "tags": ["preferences", "ui"],
    "confidence": 0.9,
})

# Retrieve top-k memories
results = requests.post(f"{BASE}/retrieve",
    json={"query": "UI preferences", "top_k": 5}).json()["results"]
```

---

## API reference

### `OpenClawMemory`

| Method           | Signature                                                                  | Description                                  |
|------------------|----------------------------------------------------------------------------|----------------------------------------------|
| `remember`       | `(text, memory_type, tags, confidence, salience, session_id) → memory_id` | Store a memory                               |
| `recall`         | `(query, max_tokens) → str`                                                | LLM-ready context string                     |
| `search`         | `(query, top_k) → List[dict]`                                              | Ranked memory dicts                          |
| `teach`          | `(text, tags, confidence) → memory_id`                                     | Store procedural / how-to knowledge          |
| `reflect`        | `(text, hops, confidence) → memory_id`                                     | Store reasoning chain                        |
| `reinforce`      | `(memory_id)`                                                              | Add evidence (may promote to WISDOM)         |
| `promote`        | `(memory_id, notes)`                                                       | Force promote to WISDOM level                |
| `save`           | `() → int`                                                                 | Persist snapshot, return count               |
| `shutdown`       | `()`                                                                       | Save + release resources                     |
| `stats`          | `() → dict`                                                                | System stats                                 |
| `wisdom_summary` | `() → dict`                                                                | Wisdom pipeline counts                       |

---

## Snapshot location

Memories are persisted to `~/.memovex/openclaw_snapshot.json` by default.
If a legacy snapshot at `~/.memorybank/openclaw_snapshot.json` is found and the new path is empty, it is loaded once and migrated forward — subsequent saves go to the new location.

Override with the `snapshot_path` parameter:

```python
mem = OpenClawMemory(snapshot_path="/custom/path/openclaw.json")
```

---

## Files

| File        | Description                                                  |
|-------------|--------------------------------------------------------------|
| `plugin.py` | `OpenClawMemory` class + `create_openclaw_memory()` factory  |

---

## Namespace

| Resource          | Value                  |
|-------------------|------------------------|
| Qdrant collection | `memovex_openclaw`     |
| Redis prefix      | `memovex:openclaw:*`   |
| Agent ID          | `openclaw`             |
