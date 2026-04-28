#!/usr/bin/env python3
"""
memovex — Claude Code Stop Hook.

Al terminar cada turno, lee el transcript, extrae el último intercambio
(user + assistant) y lo persiste en el namespace "claude" de memovex.

Instalado via .claude/settings.json → hooks.Stop.
"""

import json
import sys
import os
import re
from typing import Optional, Tuple

_HOOK_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HOOK_DIR, "../.."))
sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    """Extract plain text from a Claude message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    pass  # skip tool output
        return " ".join(p for p in parts if p).strip()
    return ""


def _read_last_exchange(transcript_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse the JSONL transcript and return (last_user_msg, last_assistant_msg).
    Returns (None, None) if the file can't be read or has no content.
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return None, None

    last_user: Optional[str] = None
    last_assistant: Optional[str] = None

    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Transcript entries have different shapes across Claude Code versions
                # Handle both wrapped {type, message} and bare {role, content} formats
                msg = entry.get("message", entry)
                role = msg.get("role", entry.get("type", ""))
                content = msg.get("content", "")

                text = _extract_text(content)
                if not text:
                    continue

                if role == "user":
                    last_user = text
                elif role == "assistant":
                    last_assistant = text

    except Exception:
        return None, None

    return last_user, last_assistant


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _allow() -> None:
    print(json.dumps({"decision": "allow"}))


def main() -> None:
    try:
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
    except Exception:
        _allow()
        return

    # Only store on normal turn completion
    stop_reason = event.get("stop_reason", "")
    if stop_reason not in ("end_turn", ""):
        _allow()
        return

    transcript_path = event.get("transcript_path", "")
    session_id = event.get("session_id", "")

    user_msg, assistant_msg = _read_last_exchange(transcript_path)
    if not user_msg or not assistant_msg:
        _allow()
        return

    try:
        from memovex.plugins.claude_plugin import create_claude_memory
        from memovex.core.types import MemoryType
        bank = create_claude_memory(embeddings_enabled=False)
    except Exception:
        _allow()
        return

    try:
        # Trim messages to avoid storing huge tool outputs
        user_short = user_msg[:500]
        asst_short = assistant_msg[:800]

        # Extract session context tags from the conversation
        tags = _infer_tags(user_msg + " " + assistant_msg)

        # Store user turn
        bank.store(
            text=f"[User] {user_short}",
            memory_type=MemoryType.EPISODIC,
            session_id=session_id or None,
            tags=tags | {"user_turn"},
            confidence=0.7,
            salience=0.5,
        )

        # Store assistant turn with slightly higher salience
        mid = bank.store(
            text=f"[Claude] {asst_short}",
            memory_type=MemoryType.EPISODIC,
            session_id=session_id or None,
            tags=tags | {"assistant_turn"},
            confidence=0.75,
            salience=0.6,
        )

        # If the assistant gave a concrete answer/decision, corroborate it
        if any(w in assistant_msg.lower() for w in
               ("implementé", "implementa", "creé", "crea", "fix", "fixed",
                "resolvé", "resolved", "completé", "completed")):
            bank.corroborate(mid)

    except Exception:
        pass

    # Always persist snapshot so next hook invocation can load it
    try:
        from memovex.plugins.claude_plugin import save_claude_memory
        save_claude_memory(bank)
    except Exception:
        pass
    finally:
        try:
            bank.shutdown()
        except Exception:
            pass

    _allow()


def _infer_tags(text: str) -> set:
    """Infer context tags from the conversation text."""
    tags = set()
    t = text.lower()
    if any(w in t for w in ("python", "def ", "class ", "import ", "función", "función")):
        tags.add("code")
    if any(w in t for w in ("docker", "compose", "container", "imagen")):
        tags.add("docker")
    if any(w in t for w in ("memoria", "memory", "qdrant", "redis", "embedding")):
        tags.add("memovex")
    if any(w in t for w in ("error", "bug", "falla", "fix", "arregl")):
        tags.add("debugging")
    if any(w in t for w in ("test", "pytest", "assert", "prueba")):
        tags.add("testing")
    return tags


if __name__ == "__main__":
    main()
