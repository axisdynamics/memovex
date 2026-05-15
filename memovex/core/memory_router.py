"""Fast local + async semantic memory router.

This module is intentionally stdlib-only and side-effect free.  It provides a
small L1 local index, a TTL result cache, and a router that can either block on
semantic recall or schedule semantic recall in the background.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

MEMOBASE_DELIMITER = "\n§\n"
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_./:@-]{3,}")


def tokenize_text(text: str) -> list[str]:
    """Return normalized lexical tokens for lightweight local scoring."""
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


@dataclass(frozen=True)
class LocalMemoryEntry:
    """One local-memory record indexed by the fast L1 router."""

    source: str
    text: str
    tokens: frozenset[str]
    hash: str

    @classmethod
    def from_text(cls, text: str, source: str = "local") -> "LocalMemoryEntry":
        normalized = text.strip()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return cls(
            source=source,
            text=normalized,
            tokens=frozenset(tokenize_text(normalized)),
            hash=digest,
        )


class LocalMemoryIndex:
    """Tiny in-memory lexical index for hot-path memory lookup."""

    def __init__(self, entries: Iterable[LocalMemoryEntry] = ()):  # noqa: B008
        self.entries = list(entries)

    @classmethod
    def from_texts(cls, texts: Iterable[str], source: str = "local") -> "LocalMemoryIndex":
        return cls(LocalMemoryEntry.from_text(text, source=source) for text in texts if text and text.strip())

    @classmethod
    def from_memobase_dir(cls, memory_dir: str | Path) -> "LocalMemoryIndex":
        """Build an index from MemoBase-style MEMORY.md and USER.md files."""
        memory_dir = Path(memory_dir)
        entries: list[LocalMemoryEntry] = []
        for source, filename in (("memory", "MEMORY.md"), ("user", "USER.md")):
            path = memory_dir / filename
            if not path.exists():
                continue
            raw = path.read_text(encoding="utf-8")
            for item in (part.strip() for part in raw.split(MEMOBASE_DELIMITER)):
                if item:
                    entries.append(LocalMemoryEntry.from_text(item, source=source))
        return cls(entries)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        q_tokens = set(tokenize_text(query))
        q_lower = (query or "").lower().strip()
        if not q_tokens and not q_lower:
            return []

        scored: list[dict[str, Any]] = []
        for entry in self.entries:
            overlap = len(q_tokens & set(entry.tokens))
            denom = max(1, min(len(q_tokens), 12))
            score = overlap / denom
            text_lower = entry.text.lower()
            if q_lower and (q_lower in text_lower or text_lower in q_lower):
                score += 0.6
            if entry.source == "user" and any(
                token in q_tokens for token in {"user", "usuario", "preference", "preferencia", "email"}
            ):
                score += 0.15
            if score > 0:
                scored.append(
                    {
                        "source": entry.source,
                        "text": entry.text,
                        "score": round(min(score, 1.0), 4),
                        "hash": entry.hash,
                        "route": "local",
                    }
                )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]


class SemanticClient(Protocol):
    """Minimal protocol expected by MemoryRouter for L2 recall."""

    def retrieve(self, query: str, top_k: int = 5, channels: Optional[list[str]] = None) -> list[dict[str, Any]]:
        ...


class TTLCache:
    """Thread-safe TTL cache for semantic recall results."""

    def __init__(self, ttl_s: float = 60):
        self.ttl_s = ttl_s
        self._items: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires, value = item
            if time.monotonic() > expires:
                self._items.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._items[key] = (time.monotonic() + self.ttl_s, value)

    def __len__(self) -> int:
        now = time.monotonic()
        with self._lock:
            expired = [key for key, (expires, _) in self._items.items() if now > expires]
            for key in expired:
                self._items.pop(key, None)
            return len(self._items)


class MemoryRouter:
    """Route memory lookups through L1 local index, cache, and L2 semantic recall."""

    def __init__(
        self,
        local_index: LocalMemoryIndex,
        semantic_client: SemanticClient,
        threshold: float = 0.55,
        cache_ttl_s: float = 60,
        background_workers: int = 1,
        background_start_delay_s: float = 0.0,
    ):
        self.local_index = local_index
        self.semantic_client = semantic_client
        self.threshold = threshold
        self.cache = TTLCache(cache_ttl_s)
        self.background_start_delay_s = max(0.0, background_start_delay_s)
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, background_workers),
            thread_name_prefix="memovex-router-bg",
        )
        self._pending: dict[str, Future] = {}
        self._pending_lock = threading.Lock()
        self.background_submitted = 0
        self.background_completed = 0
        self.background_errors = 0

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        channels: Optional[list[str]] = None,
        mode: str = "balanced",
    ) -> dict[str, Any]:
        """Retrieve memories via one of four routing modes.

        Modes:
        - ``fast``: local only, never calls semantic.
        - ``balanced``/``auto``: local if confident, otherwise blocking semantic.
        - ``deep``: blocking semantic after cache check, regardless of local score.
        - ``local_then_async_semantic``: return local/cache now; fill semantic cache in background.
        """
        t0 = time.perf_counter_ns()
        local = self.local_index.search(query, top_k=min(top_k, 5))
        best = local[0]["score"] if local else 0.0
        cache_key = self._cache_key(query, top_k, channels)

        if mode == "fast":
            return self._pack("local", local[:top_k], t0, local_score=best)

        if mode == "local_then_async_semantic":
            if best >= self.threshold:
                return self._pack("local", local[:top_k], t0, local_score=best, background_pending=False)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return self._pack("semantic_cache", cached, t0, local_score=best)
            scheduled = self._schedule_semantic(cache_key, query, top_k, channels)
            route = "local_background_scheduled" if scheduled else "local_background_pending"
            return self._pack(route, local[:top_k], t0, local_score=best, background_pending=True)

        if mode in {"auto", "balanced"} and best >= self.threshold:
            return self._pack("local", local[:top_k], t0, local_score=best)

        cached = self.cache.get(cache_key)
        if cached is not None:
            return self._pack("semantic_cache", cached, t0, local_score=best)

        try:
            semantic = self.semantic_client.retrieve(query, top_k=top_k, channels=channels)
            self.cache.set(cache_key, semantic)
            return self._pack("semantic", semantic, t0, local_score=best, local_candidates=local[:3])
        except Exception as exc:  # pragma: no cover - exercised by integration callers
            return self._pack("local_fallback", local[:top_k], t0, local_score=best, error=repr(exc))

    @staticmethod
    def _cache_key(query: str, top_k: int, channels: Optional[list[str]]) -> str:
        return json.dumps({"q": query, "k": top_k, "c": channels or []}, sort_keys=True, ensure_ascii=False)

    def _schedule_semantic(self, cache_key: str, query: str, top_k: int, channels: Optional[list[str]]) -> bool:
        with self._pending_lock:
            existing = self._pending.get(cache_key)
            if existing is not None and not existing.done():
                return False
            future = self._executor.submit(self._semantic_fill, cache_key, query, top_k, channels)
            self._pending[cache_key] = future
            self.background_submitted += 1
            future.add_done_callback(lambda completed, key=cache_key: self._mark_background_done(key, completed))
            return True

    def _semantic_fill(self, cache_key: str, query: str, top_k: int, channels: Optional[list[str]]) -> list[dict[str, Any]]:
        if self.background_start_delay_s:
            time.sleep(self.background_start_delay_s)
        semantic = self.semantic_client.retrieve(query, top_k=top_k, channels=channels)
        self.cache.set(cache_key, semantic)
        return semantic

    def _mark_background_done(self, cache_key: str, future: Future) -> None:
        with self._pending_lock:
            self._pending.pop(cache_key, None)
            self.background_completed += 1
            if future.exception() is not None:
                self.background_errors += 1

    def wait_background(self, timeout: Optional[float] = None) -> bool:
        with self._pending_lock:
            futures = list(self._pending.values())
        if not futures:
            return True
        _, not_done = wait(futures, timeout=timeout)
        return not not_done

    def background_stats(self) -> dict[str, Any]:
        with self._pending_lock:
            pending = sum(1 for future in self._pending.values() if not future.done())
        return {
            "submitted": self.background_submitted,
            "completed": self.background_completed,
            "errors": self.background_errors,
            "pending": pending,
            "cache_items": len(self.cache),
        }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    @staticmethod
    def _pack(route: str, results: list[dict[str, Any]], t0: int, **extra: Any) -> dict[str, Any]:
        return {
            "route": route,
            "count": len(results),
            "latency_ms": (time.perf_counter_ns() - t0) / 1e6,
            "results": results,
            **extra,
        }
