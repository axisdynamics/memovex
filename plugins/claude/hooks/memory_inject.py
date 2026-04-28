#!/usr/bin/env python3
"""
memovex — Claude Code UserPromptSubmit Hook.

Retrieves relevant memories and injects them as additionalContext.

Strategy (API-first with direct fallback):
  1. Try POST localhost:7914/api/claude/prefetch  (fast, pool warm, embeddings active)
  2. If API unavailable → fall back to direct MemoVexOrchestrator + snapshot
"""

import json
import sys
import os
import urllib.request
import urllib.error

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
_AGENT = "claude"
_API_BASE = os.getenv("MEMOVEX_API_URL", "http://localhost:7914") + f"/api/{_AGENT}"
_API_TIMEOUT = int(os.getenv("MEMOVEX_API_TIMEOUT", "3"))


def _allow(context: str = "") -> None:
    out: dict = {"decision": "allow"}
    if context:
        out["hookSpecificOutput"] = {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    print(json.dumps(out))


# ---------------------------------------------------------------------------
# API path
# ---------------------------------------------------------------------------

def _prefetch_via_api(prompt: str) -> str | None:
    """Return context string from the API, or None if unavailable."""
    payload = json.dumps({"query": prompt, "max_tokens": 600}).encode()
    req = urllib.request.Request(
        f"{_API_BASE}/prefetch",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_API_TIMEOUT) as resp:
            return json.loads(resp.read()).get("context", "")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Direct fallback path
# ---------------------------------------------------------------------------

def _prefetch_direct(prompt: str) -> str:
    sys.path.insert(0, _PROJECT_ROOT)
    from memovex.plugins.claude_plugin import create_claude_memory
    bank = create_claude_memory(embeddings_enabled=False)
    try:
        return bank.prefetch(prompt, max_tokens=600)
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

    prompt = event.get("prompt", "").strip()
    if not prompt:
        _allow()
        return

    # 1. Try API
    context = _prefetch_via_api(prompt)

    # 2. Fallback to direct if API unreachable
    if context is None:
        try:
            context = _prefetch_direct(prompt)
        except Exception:
            _allow()
            return

    if context:
        _allow(f"=== memovex [claude] ===\n{context}\n=== fin ===")
    else:
        _allow()


if __name__ == "__main__":
    main()
