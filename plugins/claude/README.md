# Claude Code Plugin

Integrates MemoVex into Claude Code via two hooks:

- **UserPromptSubmit** — loads the snapshot and injects relevant memories as context before the LLM sees your prompt.
- **Stop** — stores the last conversation turn and saves the updated snapshot after each reply.

Because each hook is a separate Python process, persistence is handled by a JSON snapshot at `~/.claude/memovex/claude_snapshot.json` (legacy snapshots at `~/.claude/memorybank/claude_snapshot.json` are auto-migrated on first load).

---

## Setup

### 1. Install MemoVex

**Option A — clone (gives you the hook scripts):**

```bash
git clone https://github.com/axisdynamics/memovex
cd memovex
pip install -e .
```

**Option B — install from GitHub without cloning:**

```bash
pip install git+https://github.com/axisdynamics/memovex.git
# Then download the hooks separately:
mkdir -p ~/.local/lib/memovex/plugins/claude/hooks
curl -o ~/.local/lib/memovex/plugins/claude/hooks/memory_inject.py \
  https://raw.githubusercontent.com/axisdynamics/memovex/main/plugins/claude/hooks/memory_inject.py
curl -o ~/.local/lib/memovex/plugins/claude/hooks/memory_store.py \
  https://raw.githubusercontent.com/axisdynamics/memovex/main/plugins/claude/hooks/memory_store.py
```

### 2. Copy the settings template to your Claude project

```bash
cp plugins/claude/settings.json /path/to/your/project/.claude/settings.json
```

Edit the two `command` paths to point at this checkout's location.

### 3. Seed initial memories (optional but recommended)

```bash
python3 scripts/seed_claude_memory.py
```

### 4. Verify

```bash
# Inject hook — should return additionalContext with relevant memories
echo '{"prompt": "how does the memory system work?"}' \
  | python3 plugins/claude/hooks/memory_inject.py

# Store hook — should return {"decision": "allow"} and grow the snapshot
TRANSCRIPT="~/.claude/projects/<id>/<session>.jsonl"
echo "{\"stop_reason\": \"end_turn\", \"transcript_path\": \"$TRANSCRIPT\"}" \
  | python3 plugins/claude/hooks/memory_store.py
```

---

## How it works

```
User types prompt
      │
      ▼
UserPromptSubmit hook (memory_inject.py)
  1. Load snapshot from ~/.claude/memovex/claude_snapshot.json
  2. Call bank.prefetch(prompt, max_tokens=600)
  3. Return { additionalContext: "=== memovex [claude] ..." }
      │
      ▼
Claude sees: [system context] + [memory context] + [user prompt]
      │
      ▼
Stop hook (memory_store.py)
  1. Load snapshot
  2. Parse last user + assistant turn from transcript
  3. bank.store([User] ...) + bank.store([Claude] ...)
  4. save_claude_memory(bank) → snapshot updated
```

---

## Files

| File                      | Description                              |
|---------------------------|------------------------------------------|
| `plugin.py`               | Python factory (`create_claude_memory`)  |
| `hooks/memory_inject.py`  | UserPromptSubmit hook                    |
| `hooks/memory_store.py`   | Stop hook                                |
| `settings.json`           | Claude Code settings template            |

---

## Seeding project-specific memories

Use `scripts/seed_claude_memory.py` as a template to add domain-specific context:

```python
SEEDS = [
    {
        "text": "[Claude] Our API uses JWT tokens with 24h expiry.",
        "memory_type": MemoryType.SEMANTIC,
        "tags": {"auth", "api"},
        "confidence": 0.9,
        "salience": 0.8,
    },
    ...
]
```

Re-run the script any time you want to add more seeds.
