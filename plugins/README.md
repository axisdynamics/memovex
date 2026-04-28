# Agent Plugins

Each subdirectory contains a self-contained plugin for one agent. All plugins share the same underlying `MemoVexOrchestrator` but operate in completely isolated namespaces.

## Available plugins

| Directory   | Agent        | Integration style                                |
|-------------|--------------|--------------------------------------------------|
| `claude/`   | Claude Code  | Claude Code hooks (UserPromptSubmit + Stop)      |
| `hermes/`   | Hermes VEX   | Drop-in replacement for Resonant Memory VEX 2.3  |
| `openclaw/` | OpenClaw     | Python SDK + REST                                |

## Namespace isolation

```
Agent      Qdrant collection      Redis prefix
─────────  ─────────────────────  ─────────────────────
claude     memovex_claude         memovex:claude:*
hermes     memovex_hermes         memovex:hermes:*
openclaw   memovex_openclaw       memovex:openclaw:*
```

Agents never share memory — they can run against the same Qdrant and Redis instances without interference. The same applies to snapshot files (`<MEMOVEX_SNAPSHOT_DIR>/<agent_id>_snapshot.json`) and to the local plugin defaults (`~/.claude/memovex/`, `~/.memovex/`, ...).

## REST endpoints

The REST API supports all three agents at `/api/{agent_id}/...`. See the top-level `README.md` for the full route list. Any other agent id returns `404`.

## Adding a new agent

1. Copy `openclaw/` as a template.
2. Change `_AGENT_ID = "your_agent"` in the new `plugin.py`.
3. Add the new id to `_ALLOWED_AGENTS` in `memovex/api.py` if you want it exposed via REST.
4. Add a `README.md` describing the integration steps.
