# Hermes Plugin

Drop-in replacement for Resonant Memory VEX 2.3 in the Hermes agent.

The plugin wraps `MemoVexOrchestrator` in the same interface Hermes already calls (`prefetch`, `sync_turn`, `store_memory`), so no changes are needed to the Hermes agent code.

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

### 2. Option A — Python SDK (same process)

```python
from memovex import create_hermes_memory

plugin = create_hermes_memory()

# --- existing Hermes call sites (unchanged) ---

# Retrieve context before building a reply
context = plugin.prefetch("what does the user prefer?", max_tokens=800)

# Store a conversation turn
plugin.sync_turn(user_message, assistant_message, session_id="session-42")

# Store an explicit memory
plugin.store_memory(
    "User prefers responses in Spanish",
    memory_type="semantic",
    tags=["preferences", "language"],
    confidence=0.9,
)

# Shutdown when the agent exits
plugin.shutdown()
```

### 2. Option B — REST API (separate process)

Start the MemoVex API:

```bash
uvicorn memovex.api:app --host 0.0.0.0 --port 7914
```

Then call it from Hermes:

```python
import requests

BASE = "http://localhost:7914/api/hermes"

# Prefetch context  (also available as /context)
ctx = requests.post(f"{BASE}/prefetch",
    json={"query": "user preferences", "max_tokens": 800}).json()["context"]

# Store a turn
requests.post(f"{BASE}/store", json={
    "text": "User: ...",
    "memory_type": "episodic",
    "session_id": "session-42",
})
```

---

## Migrating from Resonant Memory VEX 2.3

```bash
# Dry run — see what would be migrated
python3 scripts/migrate_hermes.py --dry-run

# Real migration
python3 scripts/migrate_hermes.py
```

See the comments in `scripts/migrate_hermes.py` for the full step-by-step guide.

---

## Files

| File        | Description                                                     |
|-------------|-----------------------------------------------------------------|
| `plugin.py` | `HermesMemoryPlugin` wrapper + `create_hermes_memory()` factory |

---

## Namespace

| Resource          | Value                |
|-------------------|----------------------|
| Qdrant collection | `memovex_hermes`     |
| Redis prefix      | `memovex:hermes:*`   |
| Agent ID          | `hermes`             |
