#!/usr/bin/env python3
"""
memovex — OpenClaw Stop Hook.

Strategy (API-first with direct fallback):
  1. Try POST localhost:7914/api/openclaw/store (x2: user + assistant)
  2. If API unavailable → fall back to direct MemoVexOrchestrator + snapshot
"""

import json
import sys
import os
import urllib.request
from typing import Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
_AGENT = "openclaw"
_API_BASE = os.getenv("MEMOVEX_API_URL", "http://localhost:7914") + f"/api/{_AGENT}"
_API_TIMEOUT = int(os.getenv("MEMOVEX_API_TIMEOUT", "5"))


def _allow() -> None:
    # Stop hooks don't use "decision" — just exit 0 silently.
    pass


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
                try:
                    entry = json.loads(line.strip())
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


def _store_via_api(text: str, tags: list, session_id: Optional[str],
                   confidence: float, salience: float) -> bool:
    payload = json.dumps({
        "text": text,
        "memory_type": "episodic",
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
            return resp.status == 200
    except Exception:
        return False


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
    ok = _store_via_api(f"[User] {user_msg[:500]}", ["user_turn"],
                        session_id, 0.7, 0.5)
    ok = _store_via_api(f"[OpenClaw] {asst_msg[:800]}", ["assistant_turn"],
                        session_id, 0.75, 0.6) and ok

    # 2. Fallback to direct if API unreachable
    if not ok:
        try:
            sys.path.insert(0, _PROJECT_ROOT)
            from memovex.plugins.openclaw_plugin import (
                create_openclaw_memory, save_openclaw_memory,
            )
            from memovex.core.types import MemoryType
            bank = create_openclaw_memory(embeddings_enabled=False)
            try:
                bank.store(f"[User] {user_msg[:500]}", memory_type=MemoryType.EPISODIC,
                           session_id=session_id, confidence=0.7, salience=0.5)
                bank.store(f"[OpenClaw] {asst_msg[:800]}", memory_type=MemoryType.EPISODIC,
                           session_id=session_id, confidence=0.75, salience=0.6)
                save_openclaw_memory(bank)
            finally:
                try:
                    bank.shutdown()
                except Exception:
                    pass
        except Exception:
            pass

    _allow()


if __name__ == "__main__":
    main()
