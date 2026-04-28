#!/usr/bin/env python3
"""
memovex — Claude Code Stop Hook.

Persists the last user+assistant exchange after each turn.

Strategy (API-first with direct fallback):
  1. Try POST localhost:7914/api/claude/store  (pool warm, all stores active)
  2. If API unavailable → fall back to direct MemoVexOrchestrator + snapshot
"""

import json
import sys
import os
import urllib.request
import urllib.error
from typing import Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
_AGENT = "claude"
_API_BASE = os.getenv("MEMOVEX_API_URL", "http://localhost:7914") + f"/api/{_AGENT}"
_API_TIMEOUT = int(os.getenv("MEMOVEX_API_TIMEOUT", "5"))


def _allow() -> None:
    # Stop hooks don't use "decision" — just exit 0 silently.
    pass


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return ""


def _read_last_exchange(path: str) -> Tuple[Optional[str], Optional[str]]:
    if not path or not os.path.exists(path):
        return None, None
    last_user = last_assistant = None
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = entry.get("message", entry)
                role = msg.get("role", entry.get("type", ""))
                text = _extract_text(msg.get("content", ""))
                if not text:
                    continue
                if role == "user":
                    last_user = text
                elif role == "assistant":
                    last_assistant = text
    except Exception:
        pass
    return last_user, last_assistant


def _infer_tags(text: str) -> list:
    tags = []
    t = text.lower()
    if any(w in t for w in ("python", "def ", "class ", "import ")):
        tags.append("code")
    if any(w in t for w in ("docker", "compose", "container")):
        tags.append("docker")
    if any(w in t for w in ("memoria", "memory", "qdrant", "redis", "embedding")):
        tags.append("memovex")
    if any(w in t for w in ("error", "bug", "fix", "arregl")):
        tags.append("debugging")
    if any(w in t for w in ("test", "pytest", "assert")):
        tags.append("testing")
    return tags


# ---------------------------------------------------------------------------
# API path
# ---------------------------------------------------------------------------

def _store_via_api(text: str, memory_type: str, tags: list,
                   session_id: Optional[str], confidence: float,
                   salience: float) -> Optional[str]:
    """Store a memory via API. Returns memory_id on success, None on failure."""
    payload = json.dumps({
        "text": text,
        "memory_type": memory_type,
        "tags": tags,
        "session_id": session_id,
        "confidence": confidence,
        "salience": salience,
    }).encode()
    req = urllib.request.Request(
        f"{_API_BASE}/store",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_API_TIMEOUT) as resp:
            return json.loads(resp.read()).get("memory_id")
    except Exception:
        return None


def _corroborate_via_api(memory_id: str) -> None:
    payload = json.dumps({"memory_id": memory_id}).encode()
    req = urllib.request.Request(
        f"{_API_BASE}/corroborate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=_API_TIMEOUT)
    except Exception:
        pass


def _store_exchange_via_api(user_msg: str, asst_msg: str,
                             session_id: Optional[str]) -> bool:
    tags = _infer_tags(user_msg + " " + asst_msg)
    _store_via_api(
        text=f"[User] {user_msg[:500]}",
        memory_type="episodic",
        tags=tags + ["user_turn"],
        session_id=session_id,
        confidence=0.7,
        salience=0.5,
    )
    mid = _store_via_api(
        text=f"[Claude] {asst_msg[:800]}",
        memory_type="episodic",
        tags=tags + ["assistant_turn"],
        session_id=session_id,
        confidence=0.75,
        salience=0.6,
    )
    if mid is None:
        return False  # signal fallback
    if any(w in asst_msg.lower() for w in
           ("implementé", "implementa", "creé", "crea", "fix", "fixed",
            "resolvé", "resolved", "completé", "completed")):
        _corroborate_via_api(mid)
    return True


# ---------------------------------------------------------------------------
# Direct fallback path
# ---------------------------------------------------------------------------

def _store_exchange_direct(user_msg: str, asst_msg: str,
                            session_id: Optional[str]) -> None:
    sys.path.insert(0, _PROJECT_ROOT)
    from memovex.plugins.claude_plugin import create_claude_memory, save_claude_memory
    from memovex.core.types import MemoryType

    bank = create_claude_memory(embeddings_enabled=False)
    try:
        tags = set(_infer_tags(user_msg + " " + asst_msg))
        bank.store(
            text=f"[User] {user_msg[:500]}",
            memory_type=MemoryType.EPISODIC,
            session_id=session_id or None,
            tags=tags | {"user_turn"},
            confidence=0.7,
            salience=0.5,
        )
        mid = bank.store(
            text=f"[Claude] {asst_msg[:800]}",
            memory_type=MemoryType.EPISODIC,
            session_id=session_id or None,
            tags=tags | {"assistant_turn"},
            confidence=0.75,
            salience=0.6,
        )
        if any(w in asst_msg.lower() for w in
               ("implementé", "implementa", "creé", "crea", "fix", "fixed",
                "resolvé", "resolved", "completé", "completed")):
            bank.corroborate(mid)
        save_claude_memory(bank)
    finally:
        try:
            bank.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        event = json.loads(sys.stdin.read() or "{}")
    except Exception:
        _allow()
        return

    if event.get("stop_reason", "") not in ("end_turn", ""):
        _allow()
        return

    user_msg, asst_msg = _read_last_exchange(event.get("transcript_path", ""))
    if not user_msg or not asst_msg:
        _allow()
        return

    session_id = event.get("session_id") or None

    # 1. Try API
    ok = _store_exchange_via_api(user_msg, asst_msg, session_id)

    # 2. Fallback to direct if API unreachable
    if not ok:
        try:
            _store_exchange_direct(user_msg, asst_msg, session_id)
        except Exception:
            pass

    _allow()


if __name__ == "__main__":
    main()
