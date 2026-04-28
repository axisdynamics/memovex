"""
memovex — REST API (FastAPI).

Exposes a multi-agent memory API. Each agent has a fully isolated
namespace sharing the same Redis and Qdrant infrastructure.

Routes:
    GET  /health
    GET  /api/{agent_id}/stats
    POST /api/{agent_id}/store
    POST /api/{agent_id}/retrieve
    POST /api/{agent_id}/prefetch
    POST /api/{agent_id}/context        (alias of /prefetch)
    POST /api/{agent_id}/corroborate
    POST /api/{agent_id}/store_chain
    GET  /api/{agent_id}/wisdom
    GET  /api/{agent_id}/graph/stats
    POST /api/{agent_id}/snapshot       (force snapshot save)

Supported agents: claude, hermes, openclaw.

Persistence (env var MEMOVEX_PERSISTENCE_ENABLED, default "true"):
    When enabled, every agent loads its snapshot at first use and saves
    on FastAPI shutdown. Snapshots live in MEMOVEX_SNAPSHOT_DIR (default
    ./data/snapshots) at <dir>/<agent_id>_snapshot.json. Restarting the
    API restores all memories.

Start with:
    uvicorn memovex.api:app --host 0.0.0.0 --port 7914 --reload
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("memovex.api")

try:
    from fastapi import FastAPI, HTTPException, Path as PathParam, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    raise ImportError(
        "fastapi and pydantic are required for the API server.\n"
        "Install with: pip install fastapi uvicorn"
    )

from .core.memory_bank import MemoVexOrchestrator
from .core.types import MemoryType

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)

_ALLOWED_AGENTS = {"claude", "hermes", "openclaw"}
# Path-parameter constraint: lowercase letters, digits, underscore, dash.
_AGENT_ID_PATTERN = r"^[a-z][a-z0-9_-]{0,31}$"

_QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
_QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
_REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
_REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))
_EMBEDDINGS  = os.getenv("EMBEDDINGS_ENABLED", "false").lower() == "true"
_CHROMA_ENABLED     = os.getenv("CHROMA_ENABLED", "false").lower() == "true"
_CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

_PERSISTENCE_ENABLED = os.getenv("MEMOVEX_PERSISTENCE_ENABLED", "true").lower() == "true"
_SNAPSHOT_DIR = Path(os.getenv("MEMOVEX_SNAPSHOT_DIR", "./data/snapshots"))

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="memovex API",
    version="1.1.0",
    description="Multi-agent memory system — claude, hermes and openclaw isolated namespaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Agent registry — one orchestrator per agent_id, lazy-initialized
# ---------------------------------------------------------------------------

_agents: Dict[str, MemoVexOrchestrator] = {}


def _snapshot_path(agent_id: str) -> Path:
    return _SNAPSHOT_DIR / f"{agent_id}_snapshot.json"


def get_agent(agent_id: str) -> MemoVexOrchestrator:
    """Return the orchestrator for ``agent_id``, creating it on first use.

    Raises ``HTTPException(404)`` for any agent not in ``_ALLOWED_AGENTS``.
    On first use the snapshot is loaded if persistence is enabled.
    """
    if agent_id not in _ALLOWED_AGENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown agent '{agent_id}'. Allowed: {sorted(_ALLOWED_AGENTS)}",
        )
    bank = _agents.get(agent_id)
    if bank is not None:
        return bank

    bank = MemoVexOrchestrator(
        agent_id=agent_id,
        embeddings_enabled=_EMBEDDINGS,
    )
    bank.initialize()
    qdrant_ok = bank.connect_qdrant(host=_QDRANT_HOST, port=_QDRANT_PORT)
    redis_ok  = bank.connect_redis(host=_REDIS_HOST, port=_REDIS_PORT)
    chroma_ok = False
    if _CHROMA_ENABLED:
        chroma_dir = f"{_CHROMA_PERSIST_DIR}/{agent_id}"
        chroma_ok = bank.connect_chroma(persist_directory=chroma_dir)
    logger.info(
        "agent=%s initialized qdrant=%s redis=%s chroma=%s embeddings=%s",
        agent_id, qdrant_ok, redis_ok, chroma_ok, _EMBEDDINGS,
    )

    if _PERSISTENCE_ENABLED:
        snap = _snapshot_path(agent_id)
        try:
            n = bank.load_snapshot(str(snap))
            if n:
                logger.info("agent=%s snapshot_loaded count=%d path=%s", agent_id, n, snap)
            else:
                logger.info("agent=%s snapshot_empty path=%s", agent_id, snap)
        except Exception as exc:
            logger.warning("agent=%s snapshot_load_failed path=%s err=%s", agent_id, snap, exc)

    _agents[agent_id] = bank
    return bank


@app.on_event("shutdown")
async def _shutdown() -> None:
    for agent_id, bank in _agents.items():
        if _PERSISTENCE_ENABLED:
            snap = _snapshot_path(agent_id)
            try:
                n = bank.save_snapshot(str(snap))
                logger.info("agent=%s snapshot_saved count=%d path=%s", agent_id, n, snap)
            except Exception as exc:
                logger.error("agent=%s snapshot_save_failed path=%s err=%s",
                             agent_id, snap, exc)
        try:
            bank.shutdown()
        except Exception as exc:
            logger.warning("agent=%s shutdown_error err=%s", agent_id, exc)


# ---------------------------------------------------------------------------
# Generic error handler — never leak unhandled exceptions on public routes
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        # Let FastAPI handle declared HTTP errors normally.
        raise exc
    logger.exception("unhandled_exception path=%s err=%s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error", "type": exc.__class__.__name__},
    )


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

# Tunable but sensible defaults — keep payloads small enough to log safely.
_MAX_TEXT_LEN = 10_000
_MAX_QUERY_LEN = 2_000
_MAX_LIST_LEN = 64


class StoreRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=_MAX_TEXT_LEN)
    memory_type: str = "episodic"
    entities: List[str] = Field(default_factory=list, max_length=_MAX_LIST_LEN)
    tags: List[str] = Field(default_factory=list, max_length=_MAX_LIST_LEN)
    session_id: Optional[str] = Field(None, max_length=128)
    confidence: float = Field(0.7, ge=0.0, le=1.0)
    salience: float = Field(0.5, ge=0.0, le=1.0)


class StoreResponse(BaseModel):
    memory_id: str
    agent_id: str


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=_MAX_QUERY_LEN)
    top_k: int = Field(5, ge=1, le=50)
    channels: Optional[List[str]] = Field(None, max_length=_MAX_LIST_LEN)
    renormalize: bool = True


class MemoryResult(BaseModel):
    memory_id: str
    text: str
    memory_type: str
    total_score: float
    channel_scores: Dict[str, float]
    provider: str
    confidence: float
    salience: float


class RetrieveResponse(BaseModel):
    agent_id: str
    query: str
    results: List[MemoryResult]


class PrefetchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=_MAX_QUERY_LEN)
    max_tokens: int = Field(1000, ge=100, le=8000)


class PrefetchResponse(BaseModel):
    agent_id: str
    context: str


class ChainHop(BaseModel):
    source: str = Field(..., min_length=1, max_length=256)
    via: str = Field(..., max_length=256)
    target: str = Field(..., min_length=1, max_length=256)


class StoreChainRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=_MAX_TEXT_LEN)
    hops: List[ChainHop] = Field(..., min_length=1, max_length=_MAX_LIST_LEN)
    entities: List[str] = Field(default_factory=list, max_length=_MAX_LIST_LEN)
    confidence: float = Field(0.7, ge=0.0, le=1.0)


class CorroborateRequest(BaseModel):
    memory_id: str = Field(..., min_length=1, max_length=128)
    delta_confidence: float = Field(0.05, ge=0.0, le=0.5)


class SnapshotResponse(BaseModel):
    agent_id: str
    saved: int
    path: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": "1.1.0",
        "agents_active": sorted(_agents.keys()),
        "allowed_agents": sorted(_ALLOWED_AGENTS),
        "persistence_enabled": _PERSISTENCE_ENABLED,
        "snapshot_dir": str(_SNAPSHOT_DIR),
        "embeddings_enabled": _EMBEDDINGS,
    }


@app.get("/api/{agent_id}/stats")
async def stats(agent_id: str = PathParam(..., pattern=_AGENT_ID_PATTERN)):
    bank = get_agent(agent_id)
    return {"agent_id": agent_id, **bank.stats()}


@app.post("/api/{agent_id}/store", response_model=StoreResponse)
async def store(
    body: StoreRequest,
    agent_id: str = PathParam(..., pattern=_AGENT_ID_PATTERN),
):
    bank = get_agent(agent_id)
    try:
        mt = MemoryType(body.memory_type)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown memory_type '{body.memory_type}'. "
                   f"Allowed: {[t.value for t in MemoryType]}",
        )
    mid = bank.store(
        text=body.text,
        memory_type=mt,
        entities=set(body.entities),
        tags=set(body.tags),
        session_id=body.session_id,
        confidence=body.confidence,
        salience=body.salience,
    )
    logger.info("agent=%s op=store memory_id=%s text_len=%d entities=%d tags=%d",
                agent_id, mid, len(body.text), len(body.entities), len(body.tags))
    return StoreResponse(memory_id=mid, agent_id=agent_id)


@app.post("/api/{agent_id}/retrieve", response_model=RetrieveResponse)
async def retrieve(
    body: RetrieveRequest,
    agent_id: str = PathParam(..., pattern=_AGENT_ID_PATTERN),
):
    bank = get_agent(agent_id)
    results = bank.retrieve(
        body.query,
        top_k=body.top_k,
        channels=body.channels,
        renormalize=body.renormalize,
    )
    logger.info("agent=%s op=retrieve top_k=%d channels=%s hits=%d",
                agent_id, body.top_k, body.channels, len(results))
    return RetrieveResponse(
        agent_id=agent_id,
        query=body.query,
        results=[
            MemoryResult(
                memory_id=r.memory.memory_id,
                text=r.memory.text,
                memory_type=r.memory.memory_type.value,
                total_score=round(r.total_score, 4),
                channel_scores={k: round(v, 4) for k, v in r.channel_scores.items()},
                provider=r.provider,
                confidence=r.memory.confidence,
                salience=r.memory.salience,
            )
            for r in results
        ],
    )


@app.post("/api/{agent_id}/prefetch", response_model=PrefetchResponse)
async def prefetch(
    body: PrefetchRequest,
    agent_id: str = PathParam(..., pattern=_AGENT_ID_PATTERN),
):
    bank = get_agent(agent_id)
    context = bank.prefetch(body.query, max_tokens=body.max_tokens)
    return PrefetchResponse(agent_id=agent_id, context=context)


# Alias of /prefetch — some agents call this "context".
@app.post("/api/{agent_id}/context", response_model=PrefetchResponse)
async def context(
    body: PrefetchRequest,
    agent_id: str = PathParam(..., pattern=_AGENT_ID_PATTERN),
):
    return await prefetch(body, agent_id=agent_id)


@app.post("/api/{agent_id}/store_chain", response_model=StoreResponse)
async def store_chain(
    body: StoreChainRequest,
    agent_id: str = PathParam(..., pattern=_AGENT_ID_PATTERN),
):
    bank = get_agent(agent_id)
    mid = bank.store_reasoning_chain(
        text=body.text,
        hops=[h.model_dump() for h in body.hops],
        entities=set(body.entities),
        confidence=body.confidence,
    )
    logger.info("agent=%s op=store_chain memory_id=%s hops=%d",
                agent_id, mid, len(body.hops))
    return StoreResponse(memory_id=mid, agent_id=agent_id)


@app.post("/api/{agent_id}/corroborate")
async def corroborate(
    body: CorroborateRequest,
    agent_id: str = PathParam(..., pattern=_AGENT_ID_PATTERN),
):
    bank = get_agent(agent_id)
    bank.corroborate(body.memory_id, delta_confidence=body.delta_confidence)
    level = bank._wisdom_store.get_level(body.memory_id)
    return {
        "agent_id": agent_id,
        "memory_id": body.memory_id,
        "wisdom_level": level.value if level else None,
    }


@app.get("/api/{agent_id}/wisdom")
async def wisdom(agent_id: str = PathParam(..., pattern=_AGENT_ID_PATTERN)):
    bank = get_agent(agent_id)
    entries = bank._wisdom_store.list_wisdom()
    return {
        "agent_id": agent_id,
        "summary": bank.wisdom_summary(),
        "wisdom_memories": [
            {"memory_id": e.memory_id,
             "confidence": e.confidence,
             "evidence_count": e.evidence_count,
             "notes": e.notes}
            for e in entries
        ],
    }


@app.get("/api/{agent_id}/graph/stats")
async def graph_stats(agent_id: str = PathParam(..., pattern=_AGENT_ID_PATTERN)):
    bank = get_agent(agent_id)
    return {"agent_id": agent_id, **bank.graph_stats()}


@app.post("/api/{agent_id}/snapshot", response_model=SnapshotResponse)
async def snapshot(agent_id: str = PathParam(..., pattern=_AGENT_ID_PATTERN)):
    """Force a snapshot save now. Useful before a planned restart."""
    bank = get_agent(agent_id)
    snap = _snapshot_path(agent_id)
    try:
        n = bank.save_snapshot(str(snap))
    except Exception as exc:
        logger.error("agent=%s snapshot_save_failed err=%s", agent_id, exc)
        raise HTTPException(status_code=500, detail=f"snapshot save failed: {exc}")
    logger.info("agent=%s op=snapshot saved=%d path=%s", agent_id, n, snap)
    return SnapshotResponse(agent_id=agent_id, saved=n, path=str(snap))
