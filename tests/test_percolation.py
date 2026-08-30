import unittest

import numpy as np

from phaseflow.problems import (
    EdgeListGraph,
    bond_percolate,
    component_statistics,
    sample_erdos_renyi,
)
from phaseflow._accelerate import native_available


class PercolationTests(unittest.TestCase):
    def test_known_components(self):
        graph = EdgeListGraph.from_edges(6, [(0, 1), (1, 2), (3, 4)])
        statistics = component_statistics(graph, use_native=False)
        np.testing.assert_array_equal(statistics.sizes, [3, 2, 1])
        self.assertEqual(statistics.largest_size, 3)
        self.assertAlmostEqual(statistics.largest_fraction, 0.5)
        self.assertAlmostEqual(statistics.susceptibility, 5.0 / 3.0)

    def test_random_graph_edge_cases(self):
        empty = sample_erdos_renyi(8, 0.0, seed=7, use_native=False)
        complete = sample_erdos_renyi(8, 1.0, seed=7, use_native=False)
        self.assertEqual(empty.edge_count, 0)
        self.assertEqual(complete.edge_count, 28)

    def test_sampling_is_reproducible(self):
        left = sample_erdos_renyi(100, 0.02, seed=19, use_native=False)
        right = sample_erdos_renyi(100, 0.02, seed=19, use_native=False)
        np.testing.assert_array_equal(left.edges, right.edges)

    def test_bond_percolation_extremes(self):
        graph = EdgeListGraph.from_edges(4, [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(bond_percolate(graph, 0.0, seed=1).edge_count, 0)
        np.testing.assert_array_equal(bond_percolate(graph, 1.0, seed=1).edges, graph.edges)

    @unittest.skipUnless(native_available(), "native extension is not installed")
    def test_native_component_kernel(self):
        graph = EdgeListGraph.from_edges(6, [(0, 1), (1, 2), (3, 4)])
        statistics = component_statistics(graph, use_native=True)
        np.testing.assert_array_equal(statistics.sizes, [3, 2, 1])


if __name__ == "__main__":
    unittest.main()
