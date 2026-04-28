"""
memovex — Redis Cache Layer.

Caches recency scores and prefetch results.
Gracefully degrades if redis is not installed or not reachable.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    logger.warning("redis not installed — recency cache disabled")

_DEFAULT_TTL = 3600  # 1 hour


class RedisCache:
    """
    Thin Redis wrapper for recency and prefetch caching.

    The `namespace` parameter (e.g. "claude" or "hermes") prefixes all keys
    so multiple agents can share the same Redis instance without collision.
    All methods are no-ops when Redis is unavailable.
    """

    def __init__(self, host: str = "localhost", port: int = 6379,
                 db: int = 0, ttl: int = _DEFAULT_TTL,
                 namespace: str = "default"):
        self._host = host
        self._port = port
        self._db = db
        self._ttl = ttl
        self._ns = namespace          # agent namespace prefix
        self._client: Optional[Any] = None

    def _key(self, suffix: str) -> str:
        return f"memovex:{self._ns}:{suffix}"

    def connect(self) -> bool:
        if not _REDIS_AVAILABLE:
            return False
        try:
            self._client = redis.Redis(
                host=self._host, port=self._port, db=self._db,
                socket_connect_timeout=2, decode_responses=True,
            )
            self._client.ping()
            logger.info("Redis connected at %s:%s", self._host, self._port)
            return True
        except Exception as e:
            logger.warning("Redis unavailable: %s — caching disabled", e)
            self._client = None
            return False

    @property
    def available(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------
    # Recency tracking
    # ------------------------------------------------------------------

    def record_access(self, memory_id: str) -> None:
        if not self.available:
            return
        try:
            self._client.setex(self._key(f"access:{memory_id}"),
                               self._ttl, str(time.time()))
        except Exception:
            pass

    def last_access(self, memory_id: str) -> Optional[float]:
        if not self.available:
            return None
        try:
            val = self._client.get(self._key(f"access:{memory_id}"))
            return float(val) if val else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Prefetch cache
    # ------------------------------------------------------------------

    def cache_prefetch(self, query_key: str, result: str,
                       ttl: Optional[int] = None) -> None:
        if not self.available:
            return
        try:
            self._client.setex(self._key(f"prefetch:{query_key}"),
                               ttl or self._ttl, result)
        except Exception:
            pass

    def get_prefetch(self, query_key: str) -> Optional[str]:
        if not self.available:
            return None
        try:
            return self._client.get(self._key(f"prefetch:{query_key}"))
        except Exception:
            return None

    def invalidate_prefetch(self, query_key: str) -> None:
        if not self.available:
            return
        try:
            self._client.delete(self._key(f"prefetch:{query_key}"))
        except Exception:
            pass

    def flush_namespace(self) -> int:
        """Delete all keys belonging to this agent namespace."""
        if not self.available:
            return 0
        try:
            pattern = self._key("*")
            keys = self._client.keys(pattern)
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception:
            return 0

    def disconnect(self) -> None:
        self._client = None
