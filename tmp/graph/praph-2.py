import random
import heapq
import os
import numpy as np
import itertools

import unittest


class Graph:
    def __init__(self, directed=False):
        """
        Initialize the Graph.

        Parameters:
        - directed (bool): Specifies whether the graph is directed. Default is False (undirected).

        Attributes:
        - graph (dict): A dictionary to store vertices and their adjacent vertices (with weights).
        - directed (bool): Indicates whether the graph is directed.
        """
        self.graph = {}
        self.directed = directed

    def add_vertex(self, vertex):
        """
        Add a vertex to the graph.

        Parameters:
        - vertex: The vertex to add. It must be hashable.

        Ensures that each vertex is represented in the graph dictionary as a key with an empty dictionary as its value.
        """
        if not isinstance(vertex, (int, str, tuple)):
            raise ValueError("Vertex must be a hashable type.")
        if vertex not in self.graph:
            self.graph[vertex] = {}

    def add_edge(self, src, dest, weight):
        """
        Add a weighted edge from src to dest. If the graph is undirected, also add from dest to src.

        Parameters:
        - src: The source vertex.
        - dest: The destination vertex.
        - weight: The weight of the edge.

        Prevents adding duplicate edges and ensures both vertices exist.
        """
        if src not in self.graph or dest not in self.graph:
            raise KeyError("Both vertices must exist in the graph.")
        if dest not in self.graph[src]:  # Check to prevent duplicate edges
            self.graph[src][dest] = weight
        if not self.directed and src not in self.graph[dest]:
            self.graph[dest][src] = weight

    def remove_edge(self, src, dest):
        """
        Remove an edge from src to dest. If the graph is undirected, also remove from dest to src.

        Parameters:
        - src: The source vertex.
        - dest: The destination vertex.
        """
        if src in self.graph and dest in self.graph[src]:
            del self.graph[src][dest]
        if not self.directed:
            if dest in self.graph and src in self.graph[dest]:
                del self.graph[dest][src]

    def remove_vertex(self, vertex):
        """
        Remove a vertex and all edges connected to it.

        Parameters:
        - vertex: The vertex to be removed.
        """
        if vertex in self.graph:
            # Remove any edges from other vertices to this one
            for adj in list(self.graph):
                if vertex in self.graph[adj]:
                    del self.graph[adj][vertex]
            # Remove the vertex entry itself
            del self.graph[vertex]

    def get_adjacent_vertices(self, vertex):
        """
        Get a list of vertices adjacent to the specified vertex.

        Parameters:
        - vertex: The vertex whose neighbors are to be retrieved.

        Returns:
        - List of adjacent vertices. Returns an empty list if vertex is not found.
        """
        return list(self.graph.get(vertex, {}).keys())

    def _get_edge_weight(self, src, dest):
        """
        Get the weight of the edge from src to dest.

        Parameters:
        - src: The source vertex.
        - dest: The destination vertex.

        Returns:
        - The weight of the edge. If the edge does not exist, returns infinity.
        """
        return self.graph[src].get(dest, float('inf'))

    def __str__(self):
        """
        Provide a string representation of the graph's adjacency list for easy printing and debugging.

        Returns:
        - A string representation of the graph dictionary.
        """
        return str(self.graph)


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


class Graph_Advanced(Graph):

    def shortest_path(self, start, end):
        """
        Calculate the shortest path from a starting node to an ending node in a sparse graph
        with potentially 10000s of nodes. Must run under 0.5 seconds and find the shortest distance between two nodes.

        Parameters:
        start: The starting node.
        end: The ending node.

        Returns:
        A tuple containing the total distance of the shortest path and a list of nodes representing that path.
        """
        # Your code here
        return dist, path

    def tsp_small_graph(self, start_vertex):
        """
        Solve the Travelling Salesman Problem for a small (~10 node) complete graph starting from a specified node.
        Required to find the optimal tour. Expect graphs with at most 10 nodes. Must run under 1 second.

        Parameters:
        start: The starting node.

        Returns:
        A tuple containing the total distance of the tour and a list of nodes representing the tour path.
        """
        # Your code here
        return dist, path

    def tsp_large_graph(self, start):
        """
        Solve the Travelling Salesman Problem for a large (~1000 node) complete graph starting from a specified node.
        No requirement to find the optimal tour. Must run under 0.5 second with a "pretty good" solution.

        Parameters:
        start: The starting node.

        Returns:
        A tuple containing the total distance of the tour and a list of nodes representing the tour path.
        """
        # Your code here
        return dist, path

    def tsp_medium_graph(self, start):
        """
        Solve the Travelling Salesman Problem for a medium (~300 node) complete graph starting from a specified node.
        Expected to perform better than tsp_large_graph. Must run under 1.5 seconds.

        Parameters:
        start: The starting node.

        Returns:
        A tuple containing the total distance of the tour and a list of nodes representing the tour path.
        """

        # Your code here
        return dist, path
