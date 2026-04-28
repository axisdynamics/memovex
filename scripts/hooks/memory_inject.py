#!/usr/bin/env python3
"""
memovex — Claude Code UserPromptSubmit Hook.

Lee el prompt del usuario, recupera memorias relevantes del namespace
"claude" y las inyecta como additionalContext antes de que Claude procese
la respuesta.

Instalado via .claude/settings.json → hooks.UserPromptSubmit.
"""

import json
import sys
import os

# Ubicar el proyecto
_HOOK_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HOOK_DIR, "../.."))
sys.path.insert(0, _PROJECT_ROOT)


def _allow(context: str = "") -> None:
    out: dict = {"decision": "allow"}
    if context:
        out["hookSpecificOutput"] = {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    print(json.dumps(out))


def _silent_allow() -> None:
    """No memory context — Claude proceeds as normal."""
    print(json.dumps({"decision": "allow"}))


def main() -> None:
    try:
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
    except Exception:
        _silent_allow()
        return

    prompt = event.get("prompt", "").strip()
    if not prompt:
        _silent_allow()
        return

    try:
        from memovex.plugins.claude_plugin import create_claude_memory
        bank = create_claude_memory(embeddings_enabled=False)
    except Exception:
        _silent_allow()
        return

    try:
        context = bank.prefetch(prompt, max_tokens=600)
        if not context:
            _silent_allow()
            return

        header = "=== memovex [claude] — contexto relevante ==="
        footer = "=== fin del contexto de memoria ==="
        _allow(f"{header}\n{context}\n{footer}")
    except Exception:
        _silent_allow()
    finally:
        try:
            bank.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
