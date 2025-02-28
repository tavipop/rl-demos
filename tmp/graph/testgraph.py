import unittest

from rl.graph import Graph


class TestGraph(unittest.TestCase):

    def setUp(self):
        """Set up a basic graph for testing."""
        self.graph = Graph()
        self.directed_graph = Graph(directed=True)

    def test_add_vertex(self):
        self.graph.add_vertex(1)
        self.assertIn(1, self.graph.graph)
        self.assertEqual(self.graph.graph[1], {})

    def test_add_edge_undirected(self):
        self.graph.add_vertex(1)
        self.graph.add_vertex(2)
        self.graph.add_edge(1, 2, 10)
        self.assertEqual(self.graph.graph[1][2], 10)
        self.assertEqual(self.graph.graph[2][1], 10)

    def test_add_edge_directed(self):
        self.directed_graph.add_vertex(1)
        self.directed_graph.add_vertex(2)
        self.directed_graph.add_edge(1, 2, 5)
        self.assertEqual(self.directed_graph.graph[1][2], 5)
        self.assertNotIn(1, self.directed_graph.graph[2])

    def test_add_edge_nonexistent_vertex(self):
        self.graph.add_vertex(1)
        with self.assertRaises(KeyError):
            self.graph.add_edge(1, 3, 10)

    def test_remove_edge_undirected(self):
        self.graph.add_vertex(1)
        self.graph.add_vertex(2)
        self.graph.add_edge(1, 2, 10)
        self.graph.remove_edge(1, 2)
        self.assertNotIn(2, self.graph.graph[1])
        self.assertNotIn(1, self.graph.graph[2])

    def test_remove_edge_directed(self):
        self.directed_graph.add_vertex(1)
        self.directed_graph.add_vertex(2)
        self.directed_graph.add_edge(1, 2, 5)
        self.directed_graph.remove_edge(1, 2)
        self.assertNotIn(2, self.directed_graph.graph[1])

    def test_remove_vertex(self):
        self.graph.add_vertex(1)
        self.graph.add_vertex(2)
        self.graph.add_edge(1, 2, 10)
        self.graph.remove_vertex(1)
        self.assertNotIn(1, self.graph.graph)
        self.assertNotIn(1, self.graph.graph[2])

    def test_get_adjacent_vertices(self):
        self.graph.add_vertex(1)
        self.graph.add_vertex(2)
        self.graph.add_edge(1, 2, 10)
        adj_vertices = self.graph.get_adjacent_vertices(1)
        self.assertEqual(adj_vertices, [2])

    def test_get_edge_weight(self):
        self.graph.add_vertex(1)
        self.graph.add_vertex(2)
        self.graph.add_edge(1, 2, 10)
        weight = self.graph._get_edge_weight(1, 2)
        self.assertEqual(weight, 10)
        non_existent_weight = self.graph._get_edge_weight(2, 1)
        self.assertEqual(non_existent_weight, float('inf'))

    def test_str_representation(self):
        self.graph.add_vertex(1)
        self.graph.add_vertex(2)
        self.graph.add_edge(1, 2, 10)
        expected_str = "{1: {2: 10}, 2: {1: 10}}"
        self.assertEqual(str(self.graph), expected_str)


if __name__ == '__main__':
    unittest.main()