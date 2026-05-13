"""Security hygiene helpers for MemoVex text ingestion and recall.

This module is intentionally stdlib-only and conservative.  It redacts common
credential shapes before text enters durable memory, while preserving enough
surrounding context for useful recall.
"""

from __future__ import annotations

import re

_REDACTED = "<redacted-secret>"

_SECRET_PATTERNS = [
    # GitHub classic PATs and fine-grained PATs.
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    # OpenAI-like keys. Require a long tail to avoid redacting ordinary words.
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    # Generic assignment forms: token=..., api_key: ..., password = ...
    re.compile(
        r"(?i)\b(token|api[_-]?key|secret|password|passwd)\b\s*[:=]\s*"
        r"(['\"]?)[A-Za-z0-9_./+=@:-]{12,}\2"
    ),
]


def redact_secrets(text: str) -> str:
    """Return *text* with known credential-like values replaced.

    The function is idempotent and never raises for non-string/empty input.
    """
    if not isinstance(text, str) or not text:
        return text
    out = text
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.startswith("(?i)\\b(token"):
            out = pattern.sub(lambda m: f"{m.group(1)}=<redacted-secret>", out)
        else:
            out = pattern.sub(_REDACTED, out)
    return out


def contains_secret(text: str) -> bool:
    """True when *text* contains a known credential-like value."""
    if not isinstance(text, str) or not text:
        return False
    return redact_secrets(text) != text


def is_secret_only(text: str) -> bool:
    """True when text is basically just a credential blob.

    This lets API routes reject accidental bare-token storage while still
    accepting contextual notes after redacting the token inside them.
    """
    if not contains_secret(text):
        return False
    stripped = redact_secrets(text).replace(_REDACTED, "")
    stripped = re.sub(r"[\s:='\"`.,;()\[\]{}<>-]+", "", stripped)
    return not stripped
