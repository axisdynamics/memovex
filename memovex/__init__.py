"""memovex — Multi-channel resonant memory framework for LLM agents.

Public API:
    MemoVexOrchestrator   — central per-agent orchestrator
    MemoryType            — memory record taxonomy enum
    ChannelType           — retrieval channel enum
    Memory                — memory dataclass
    RetrievalResult       — multi-channel retrieval result
    DEFAULT_CHANNEL_WEIGHTS  — default scoring weights
    WisdomLevel           — curation pipeline level enum

Convenience factories (lazy-imported on first use to keep this module light):
    create_claude_memory      — agent_id='claude'
    create_hermes_memory      — agent_id='hermes'  (returns HermesMemoryPlugin)
    create_openclaw_memory    — agent_id='openclaw'

Example:
    from memovex import MemoVexOrchestrator, MemoryType
    bank = MemoVexOrchestrator(agent_id='demo')
    bank.initialize()
    mid = bank.store('hola mundo', memory_type=MemoryType.EPISODIC)
    bank.shutdown()
"""

from .core.memory_bank import MemoVexOrchestrator
from .core.types import (
    ChannelType,
    Memory,
    MemoryType,
    QueryFeatures,
    RetrievalResult,
)
from .core.resonance_engine import (
    DEFAULT_CHANNEL_WEIGHTS,
    MemoryStore,
    ResonanceEngine,
)
from .core.wisdom_store import WisdomLevel, WisdomStore

__version__ = "1.1.0"

__all__ = [
    "MemoVexOrchestrator",
    "MemoryType",
    "ChannelType",
    "Memory",
    "QueryFeatures",
    "RetrievalResult",
    "DEFAULT_CHANNEL_WEIGHTS",
    "MemoryStore",
    "ResonanceEngine",
    "WisdomLevel",
    "WisdomStore",
    "__version__",
]


# Lazy-loaded plugin factories — importing the plugin module pulls in
# the orchestrator and (optionally) starts a background homeostasis thread,
# so we wait until the caller actually asks for them.
def __getattr__(name):
    if name == "create_claude_memory":
        from .plugins.claude_plugin import create_claude_memory
        return create_claude_memory
    if name == "create_hermes_memory":
        from .plugins.hermes_plugin import create_hermes_memory
        return create_hermes_memory
    if name == "create_openclaw_memory":
        from .plugins.openclaw_plugin import create_openclaw_memory
        return create_openclaw_memory
    if name == "HermesMemoryPlugin":
        from .plugins.hermes_plugin import HermesMemoryPlugin
        return HermesMemoryPlugin
    raise AttributeError(f"module 'memovex' has no attribute {name!r}")
