"""
memovex — Graph Store (NetworkX).

Maintains an in-memory knowledge graph of entities and their relations,
built from reasoning chains stored in the orchestrator.  Used by the
graph_traversal scoring channel and by traverse_graph().
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    logger.warning("networkx not installed — graph_traversal channel degraded to entity overlap")


class GraphStore:
    """
    Directed, weighted knowledge graph backed by NetworkX.

    Nodes  — entity strings (lowercased)
    Edges  — (source, target, relation) triples with a weight attribute
    """

    def __init__(self) -> None:
        if _NX_AVAILABLE:
            self._graph: "nx.DiGraph" = nx.DiGraph()
        else:
            self._graph = None
        self._memory_index: Dict[str, Set[str]] = {}  # entity → memory_ids

    @property
    def available(self) -> bool:
        return _NX_AVAILABLE and self._graph is not None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_entity(self, entity: str) -> None:
        if not self.available:
            return
        self._graph.add_node(entity.lower())

    def add_relation(self, source: str, relation: str, target: str,
                     weight: float = 1.0, memory_id: Optional[str] = None) -> None:
        if not self.available:
            return
        s, t = source.lower(), target.lower()
        self._graph.add_node(s)
        self._graph.add_node(t)
        if self._graph.has_edge(s, t):
            existing = self._graph[s][t]
            existing["weight"] = max(existing["weight"], weight)
            existing.setdefault("relations", set()).add(relation)
            if memory_id:
                existing.setdefault("memory_ids", set()).add(memory_id)
        else:
            self._graph.add_edge(
                s, t,
                relation=relation,
                weight=weight,
                relations={relation},
                memory_ids={memory_id} if memory_id else set(),
            )

        # Index entity → memory_id
        if memory_id:
            self._memory_index.setdefault(s, set()).add(memory_id)
            self._memory_index.setdefault(t, set()).add(memory_id)

    def add_from_hops(self, hops: List[Dict], memory_id: str,
                      confidence: float = 0.7) -> None:
        """Bulk-add edges from a reasoning chain hop list."""
        for hop in hops:
            source = hop.get("source", "")
            target = hop.get("target", "")
            via = hop.get("via", "unknown")
            if source and target:
                self.add_relation(source, via, target,
                                  weight=confidence, memory_id=memory_id)

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def neighbors(self, entity: str, depth: int = 1) -> List[str]:
        """Return entity names reachable within `depth` hops."""
        if not self.available:
            return []
        start = entity.lower()
        if start not in self._graph:
            return []
        try:
            reachable = nx.single_source_shortest_path_length(
                self._graph, start, cutoff=depth
            )
            return [n for n in reachable if n != start]
        except nx.NetworkXError:
            return []

    def score_entity(self, query_entities: Set[str], max_depth: int = 2) -> float:
        """
        Fraction of query entities that have at least one reachable neighbor.

        Returns 1.0 if all query entities are connected in the graph,
        0.0 if none are.  Used by the graph_traversal scoring channel.
        """
        if not self.available or not query_entities:
            return 0.0
        q_lower = {e.lower() for e in query_entities}
        connected = sum(
            1 for e in q_lower if self.neighbors(e, depth=max_depth)
        )
        return connected / len(q_lower)

    def memory_ids_for_entities(self, entities: Set[str]) -> Set[str]:
        """Return all memory_ids associated with the given entities."""
        out: Set[str] = set()
        for e in entities:
            out |= self._memory_index.get(e.lower(), set())
        return out

    def get_paths(self, source: str, target: str,
                  max_depth: int = 3) -> List[List[str]]:
        """Find all simple paths from source to target."""
        if not self.available:
            return []
        s, t = source.lower(), target.lower()
        if s not in self._graph or t not in self._graph:
            return []
        try:
            paths = list(nx.all_simple_paths(
                self._graph, s, t, cutoff=max_depth
            ))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        return self._graph.number_of_nodes() if self.available else 0

    def edge_count(self) -> int:
        return self._graph.number_of_edges() if self.available else 0

    def stats(self) -> Dict:
        return {
            "nodes": self.node_count(),
            "edges": self.edge_count(),
            "networkx_available": _NX_AVAILABLE,
        }

    def remove_memory(self, memory_id: str) -> None:
        """Remove all edges/nodes associated with a memory_id."""
        for entity, mids in list(self._memory_index.items()):
            mids.discard(memory_id)
            if not mids:
                del self._memory_index[entity]
