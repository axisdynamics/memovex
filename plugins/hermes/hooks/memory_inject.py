#!/usr/bin/env python3
"""
memovex — Hermes UserPromptSubmit Hook.

Strategy (API-first with direct fallback):
  1. Try POST localhost:7914/api/hermes/prefetch
  2. If API unavailable → fall back to direct HermesMemoryPlugin + snapshot
"""

import json
import sys
import os
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
_AGENT = "hermes"
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


def _prefetch_via_api(prompt: str):
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


def _prefetch_direct(prompt: str) -> str:
    sys.path.insert(0, _PROJECT_ROOT)
    from memovex.plugins.hermes_plugin import create_hermes_memory
    plugin = create_hermes_memory(embeddings_enabled=False)
    try:
        return plugin.prefetch(prompt, max_tokens=600)
    finally:
        try:
            plugin.shutdown()
        except Exception:
            pass


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

    context = _prefetch_via_api(prompt)
    if context is None:
        try:
            context = _prefetch_direct(prompt)
        except Exception:
            _allow()
            return

    if context:
        _allow(f"=== memovex [hermes] ===\n{context}\n=== fin ===")
    else:
        _allow()


if __name__ == "__main__":
    main()
