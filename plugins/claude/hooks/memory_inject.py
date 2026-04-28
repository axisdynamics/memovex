#!/usr/bin/env python3
"""
memovex — Claude Code UserPromptSubmit Hook.

Retrieves relevant memories and injects them as clean narrative context.
Auto-adapts formatting and thresholds based on whether embeddings (Qdrant/
semantic) or keyword fallback (native) retrieval is active.
"""

import json
import re
import sys
import os
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
_AGENT = "claude"
_API_BASE = os.getenv("MEMOVEX_API_URL", "http://localhost:7914") + f"/api/{_AGENT}"
_API_TIMEOUT = int(os.getenv("MEMOVEX_API_TIMEOUT", "3"))

# Overridable via env; defaults differ by engine (see _format_context)
_MIN_SCORE_OVERRIDE = os.getenv("MEMOVEX_MIN_SCORE")


def _allow(context: str = "") -> None:
    # UserPromptSubmit hooks don't use "decision" — that field is PreToolUse only.
    # With no context: exit silently (allow with no injection).
    # With context: output hookSpecificOutput only.
    if context:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": context,
            }
        }))


# ---------------------------------------------------------------------------
# Context formatter — strips infrastructure metadata, returns clean narrative
# ---------------------------------------------------------------------------

_ENTRY_RE = re.compile(
    r"\[(?P<mtype>\w+)\]\s*\(score:(?P<score>[0-9.]+)[^)]*\)\s*(?P<text>.+?)\s*\[(?P<engine>native|qdrant|chroma)\]",
    re.DOTALL,
)


def _detect_engine(raw: str) -> str:
    """
    Returns 'semantic' if any entry came from qdrant/chroma (embeddings active),
    'keyword' if all entries are from the native bag-of-words fallback.
    """
    engines = set(m.group("engine") for m in _ENTRY_RE.finditer(raw))
    if engines & {"qdrant", "chroma"}:
        return "semantic"
    return "keyword"


def _trim_to_sentence(text: str) -> str:
    for end in reversed(range(len(text))):
        if text[end] in ".!?":
            return text[: end + 1]
    return text


def _first_sentence(text: str) -> str:
    m = re.match(r"^(.+?[.!?])\s", text)
    return (m.group(1) if m else text[:50]).lower().strip()


def _deduplicate(entries: list) -> list:
    seen: set[str] = set()
    result = []
    for score, text in entries:
        key = _first_sentence(text)
        if key not in seen:
            seen.add(key)
            result.append((score, text))
    return result


def _format_context(raw: str) -> str:
    """
    Transforms raw memovex context into transparent narrative.
    Automatically adjusts score threshold and result count based on engine:

      semantic (qdrant/chroma) — high-fidelity vectors, stricter threshold,
                                  top 3 results.
      keyword  (native)        — bag-of-words fallback, looser threshold,
                                  top 4 results, no false confidence implied.
    """
    engine = _detect_engine(raw)

    if _MIN_SCORE_OVERRIDE:
        min_score = float(_MIN_SCORE_OVERRIDE)
        top_k = 3
    elif engine == "semantic":
        min_score = 0.20
        top_k = 3
    else:  # keyword
        min_score = 0.10
        top_k = 4

    entries = []
    for m in _ENTRY_RE.finditer(raw):
        score = float(m.group("score"))
        if score < min_score:
            continue
        text = m.group("text").strip()
        text = re.sub(r"\s*\[.*?\]\s*$", "", text).strip()
        text = _trim_to_sentence(text)
        # Skip low-score conversational fragments stored by memory_store
        if re.match(r"^\[(User|Claude)\]", text) and score < 0.40:
            continue
        if len(text) > 20:
            entries.append((score, text))

    if not entries:
        return ""

    entries.sort(key=lambda x: x[0], reverse=True)
    entries = _deduplicate(entries)[:top_k]
    bullets = "\n".join(f"• {t}" for _, t in entries)
    return f"Recuerdos relevantes:\n{bullets}"


# ---------------------------------------------------------------------------
# API path
# ---------------------------------------------------------------------------

def _prefetch_via_api(prompt: str) -> str | None:
    payload = json.dumps({"query": prompt, "max_tokens": 1200}).encode()
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
        return bank.prefetch(prompt, max_tokens=1200)
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

    context_raw = _prefetch_via_api(prompt)
    if context_raw is None:
        try:
            context_raw = _prefetch_direct(prompt)
        except Exception:
            _allow()
            return

    context = _format_context(context_raw or "")
    _allow(context)


if __name__ == "__main__":
    main()
