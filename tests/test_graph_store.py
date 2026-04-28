"""Tests for GraphStore — entity graph and traversal."""

import pytest
from memovex.integrations.graph_store import GraphStore


@pytest.fixture
def graph():
    gs = GraphStore()
    return gs


class TestGraphStoreBasic:
    def test_add_entity(self, graph):
        graph.add_entity("Paris")
        assert graph.node_count() >= (1 if graph.available else 0)

    def test_add_relation(self, graph):
        graph.add_relation("Paris", "capital_de", "Francia")
        if graph.available:
            assert graph.edge_count() == 1

    def test_neighbors_direct(self, graph):
        graph.add_relation("Paris", "capital_de", "Francia")
        if graph.available:
            neighbors = graph.neighbors("Paris", depth=1)
            assert "francia" in neighbors

    def test_neighbors_two_hops(self, graph):
        graph.add_relation("Paris", "en", "Francia")
        graph.add_relation("Francia", "en", "Europa")
        if graph.available:
            neighbors = graph.neighbors("Paris", depth=2)
            assert "europa" in neighbors

    def test_neighbors_unknown_entity(self, graph):
        result = graph.neighbors("desconocido", depth=1)
        assert result == []

    def test_add_from_hops(self, graph):
        hops = [
            {"source": "Tokyo", "via": "capital_de", "target": "Japón"},
            {"source": "Japón", "via": "en", "target": "Asia"},
        ]
        graph.add_from_hops(hops, memory_id="m1", confidence=0.9)
        if graph.available:
            assert graph.edge_count() == 2


class TestGraphStoreMemoryIndex:
    def test_memory_ids_for_entities(self, graph):
        hops = [{"source": "Berlin", "via": "capital_de", "target": "Alemania"}]
        graph.add_from_hops(hops, memory_id="m-berlin")
        ids = graph.memory_ids_for_entities({"Berlin"})
        assert "m-berlin" in ids

    def test_remove_memory_cleans_index(self, graph):
        hops = [{"source": "Roma", "via": "capital_de", "target": "Italia"}]
        graph.add_from_hops(hops, memory_id="m-roma")
        graph.remove_memory("m-roma")
        ids = graph.memory_ids_for_entities({"Roma"})
        assert "m-roma" not in ids


class TestGraphStorePaths:
    def test_get_paths_direct(self, graph):
        graph.add_relation("A", "r", "B")
        if graph.available:
            paths = graph.get_paths("A", "B", max_depth=1)
            assert len(paths) >= 1

    def test_get_paths_no_path(self, graph):
        graph.add_relation("X", "r", "Y")
        paths = graph.get_paths("X", "Z", max_depth=2)
        assert paths == []


class TestGraphStoreStats:
    def test_stats_keys(self, graph):
        s = graph.stats()
        assert "nodes" in s
        assert "edges" in s
        assert "networkx_available" in s

    def test_score_entity_zero_empty(self, graph):
        score = graph.score_entity(set())
        assert score == 0.0

    def test_score_entity_connected(self, graph):
        if not graph.available:
            pytest.skip("networkx not installed")
        graph.add_relation("Alpha", "r", "Beta")
        graph.add_relation("Beta", "r", "Gamma")
        score = graph.score_entity({"Alpha"}, max_depth=2)
        assert score > 0
