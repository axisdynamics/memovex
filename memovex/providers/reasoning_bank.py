"""
memovex — ReasoningBank Integration.

Integrates reasoning chain storage, multi-hop query parsing,
and graph traversal with symbolic resonance guidance.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ReasoningBankAdapter:
    """
    ReasoningBank integration for memovex.

    Provides:
    - Multi-hop query parsing (decomposition into reasoning steps)
    - Reasoning chain storage and retrieval
    - Graph traversal with symbolic resonance guidance
    - Confidence scoring via path redundancy

    This is a conceptual adapter based on Google Research's ReasoningBank
    benchmark methodology, adapted for online memory systems.
    """

    def __init__(self, memory_bank_orchestrator=None):
        self._orchestrator = memory_bank_orchestrator
        self._query_patterns = self._init_query_patterns()

    # -----------------------------------------------------------------------
    # Query Parsing: Decompose multi-hop questions
    # -----------------------------------------------------------------------

    def _init_query_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for multi-hop query decomposition."""
        return {
            "location": re.compile(
                r"(?:where|dónde|location|live|ubicad[oa])\s+(.+?)(?:\?|$)",
                re.IGNORECASE,
            ),
            "relation": re.compile(
                r"(?:who|qu[ie]|which|cu[áa]l|what|qu[ée])\s+(?:is|es|was|era)\s+(.+?)(?:\s+of\s+|\s+de\s+)(.+?)(?:\?|$)",
                re.IGNORECASE,
            ),
            "comparison": re.compile(
                r"(?:compare|compar[ae]|difference|diferencia|vs|versus)\s+(.+?)(?:\s+and\s+|\s+y\s+)(.+?)(?:\?|$)",
                re.IGNORECASE,
            ),
            "temporal": re.compile(
                r"(?:when|cu[áa]ndo|time|época|año|year)\s+(.+?)(?:\?|$)",
                re.IGNORECASE,
            ),
            "causal": re.compile(
                r"(?:why|por qu[ée]|reason|raz[óo]n|cause|causa)\s+(.+?)(?:\?|$)",
                re.IGNORECASE,
            ),
        }

    def parse_multi_hop(self, query: str) -> List[Dict[str, Any]]:
        """
        Decompose a multi-hop question into reasoning steps.

        Returns list of hops, where each hop is:
        {"type": str, "source": str, "target": str, "relation": str}
        """
        hops = []

        for hop_type, pattern in self._query_patterns.items():
            match = pattern.search(query)
            if not match:
                continue

            if hop_type == "location":
                # "Where does X live?" → Hop: X → lives_in → ?
                entity = match.group(1).strip()
                hops.append({
                    "type": "location",
                    "source": entity,
                    "target": "?",
                    "relation": "lives_in",
                })

            elif hop_type == "relation":
                # "What is X of Y?" → Hop: Y → has_property → X
                entity = match.group(2).strip()
                attribute = match.group(1).strip()
                hops.append({
                    "type": "relation",
                    "source": entity,
                    "target": attribute,
                    "relation": "has_attribute",
                })

            elif hop_type == "comparison":
                # "Compare X and Y" → Two hops: X and Y as separate entities
                entity_a = match.group(1).strip()
                entity_b = match.group(2).strip()
                hops.append({
                    "type": "comparison",
                    "source": entity_a,
                    "target": entity_b,
                    "relation": "compared_to",
                })

            elif hop_type == "temporal":
                # "When did X happen?" → Hop: X → occurred_at → ?
                entity = match.group(1).strip()
                hops.append({
                    "type": "temporal",
                    "source": entity,
                    "target": "?",
                    "relation": "occurred_at",
                })

            elif hop_type == "causal":
                # "Why did X happen?" → Hop: X → caused_by → ?
                entity = match.group(1).strip()
                hops.append({
                    "type": "causal",
                    "source": entity,
                    "target": "?",
                    "relation": "caused_by",
                })

        return hops

    # -----------------------------------------------------------------------
    # Graph Traversal with Symbolic Guidance
    # -----------------------------------------------------------------------

    def traverse_with_resonance(self, start_entity: str, query: str,
                                 max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Traverse the knowledge graph with symbolic resonance guidance.

        At each hop, base64 symbols from the query guide which edges to
        explore first. This is the key integration point between
        ReasoningBank-style graph traversal and VEX symbolic resonance.
        """
        if not self._orchestrator:
            return []

        from ..core.resonance_engine import text_to_symbols, compute_symbolic_resonance

        query_symbols = text_to_symbols(query)

        # Get all reasoning chains involving the start entity
        chains = self._orchestrator.get_reasoning_chains(start_entity)
        if not chains:
            return []

        # Score each chain by symbolic resonance
        scored_chains = []
        for chain in chains:
            chain_text = f"{' '.join(chain.graph_nodes)} {' '.join(str(e) for e in chain.graph_edges)}"
            chain_symbols = text_to_symbols(chain_text)
            resonance = compute_symbolic_resonance(query_symbols, chain_symbols)
            scored_chains.append((chain, resonance))

        # Sort by resonance (symbolic guidance)
        scored_chains.sort(key=lambda x: x[1], reverse=True)

        # Extract reasoning paths
        paths = []
        for chain, resonance in scored_chains[:max_depth]:
            for hop in chain.reasoning_hops:
                paths.append({
                    "hop": hop,
                    "resonance_score": resonance,
                    "confidence": chain.confidence,
                })

        return paths

    # -----------------------------------------------------------------------
    # Reasoning Chain Validation
    # -----------------------------------------------------------------------

    def validate_chain(self, chain: List[Dict[str, Any]],
                        confidence_threshold: float = 0.6) -> float:
        """
        Validate a reasoning chain using path redundancy.

        Higher score = more paths lead to the same conclusion.
        This mirrors ReasoningBank's multi-evidence evaluation.
        """
        if not chain:
            return 0.0

        # Count unique conclusions (target entities)
        conclusions = {}
        for hop in chain:
            target = hop.get("target", "")
            if target and target != "?":
                conclusions[target] = conclusions.get(target, 0) + 1

        if not conclusions:
            return 0.0

        # Score based on most-repeated conclusion
        max_redundancy = max(conclusions.values())
        total_hops = len(chain)

        return min(max_redundancy / total_hops, 1.0)
