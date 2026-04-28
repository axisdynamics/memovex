#!/usr/bin/env python3
"""
Seed the Claude Code snapshot with initial project context memories.

Run once (or re-run to add more seeds):
    python3 scripts/seed_claude_memory.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from memovex.plugins.claude_plugin import create_claude_memory, save_claude_memory
from memovex.core.types import MemoryType


SEEDS = [
    # Project identity
    {
        "text": "[Claude] memovex es un framework de memoria multi-canal para agentes LLM. "
                "Usa 11 canales ponderados (semantic 0.32, entity 0.13, graph_traversal 0.12, wisdom 0.10, "
                "temporal 0.08, recency 0.06, reasoning_chain 0.05, symbolic 0.05, procedural 0.05, …) "
                "y almacena memorias en una MemoryStore local con índices secundarios.",
        "memory_type": MemoryType.SEMANTIC,
        "tags": {"memovex", "architecture", "project"},
        "confidence": 0.9,
        "salience": 0.8,
    },
    # Multi-agent isolation
    {
        "text": "[Claude] Cada agente tiene su propio namespace: colección Qdrant memovex_{agent_id}, "
                "prefijo Redis memovex:{agent_id}:*, e instancia MemoryStore separada. "
                "Agentes soportados: 'claude', 'hermes', 'openclaw'.",
        "memory_type": MemoryType.SEMANTIC,
        "tags": {"memovex", "multi-agent", "architecture"},
        "confidence": 0.9,
        "salience": 0.8,
    },
    # Hooks wiring
    {
        "text": "[Claude] Los hooks de Claude Code están configurados en .claude/settings.json. "
                "UserPromptSubmit → memory_inject.py (carga snapshot, inyecta contexto via additionalContext). "
                "Stop → memory_store.py (almacena último turno, guarda snapshot).",
        "memory_type": MemoryType.PROCEDURAL,
        "tags": {"memovex", "hooks", "claude-code"},
        "confidence": 0.9,
        "salience": 0.85,
    },
    # Snapshot persistence
    {
        "text": "[Claude] La persistencia cross-process se logra via snapshot JSON en "
                "~/.claude/memovex/claude_snapshot.json. Cada proceso hook es independiente; "
                "el snapshot es el puente entre invocaciones.",
        "memory_type": MemoryType.PROCEDURAL,
        "tags": {"memovex", "snapshot", "hooks"},
        "confidence": 0.9,
        "salience": 0.8,
    },
    # Docker setup
    {
        "text": "[Claude] Los contenedores Docker existentes son resonant-qdrant (puerto 6333/6334) "
                "y resonant-redis (6379) en la red v1_default (172.30.0.0/16). "
                "El docker-compose del framework los referencia como red externa.",
        "memory_type": MemoryType.SEMANTIC,
        "tags": {"docker", "infrastructure", "memovex"},
        "confidence": 0.85,
        "salience": 0.7,
    },
    # WisdomStore pipeline
    {
        "text": "[Claude] WisdomStore implementa un pipeline de curación RAW→PROCESSED→CURATED→WISDOM. "
                "Umbrales: PROCESSED ≥0.40 confianza, CURATED ≥0.60 + 1 evidencia, "
                "WISDOM ≥0.80 confianza + ≥0.70 salience + 2 evidencias.",
        "memory_type": MemoryType.SEMANTIC,
        "tags": {"memovex", "wisdom", "architecture"},
        "confidence": 0.85,
        "salience": 0.75,
    },
    # base64 clarification
    {
        "text": "[Claude] base64_resonance fue eliminado como canal de scoring porque es funcionalmente "
                "idéntico al overlap de tokens. Se conserva como utilidad de indexado cross-provider "
                "pero no contribuye al score de resonancia.",
        "memory_type": MemoryType.SEMANTIC,
        "tags": {"memovex", "architecture", "base64"},
        "confidence": 0.9,
        "salience": 0.7,
    },
]


def main():
    print("Seeding Claude Code memory snapshot…")
    bank = create_claude_memory(embeddings_enabled=False)

    for seed in SEEDS:
        mid = bank.store(**seed)
        bank.corroborate(mid)  # give seeds initial evidence
        print(f"  + stored: {seed['text'][:80]}…")

    n = save_claude_memory(bank)
    print(f"\nSnapshot saved: {n} memories → ~/.claude/memovex/claude_snapshot.json")
    bank.shutdown()


if __name__ == "__main__":
    main()
