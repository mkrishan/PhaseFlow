import unittest

import numpy as np

from phaseflow.analysis import (
    PermutationAction,
    coarse_grain_graph,
    compare_discrete_to_flow,
    equivariance_error,
    estimate_exponential_rate,
    estimate_peak,
    euler_integrate,
    feature_correlation_graph,
    first_hitting_time,
    intervention_effect,
    orbit_partition,
    representation_percolation_curve,
    run_percolation_sweep,
    sequential_partition,
)
from phaseflow.problems import EdgeListGraph


class FlowAnalysisTests(unittest.TestCase):
    def test_euler_and_exponential_fit(self):
        trajectory = euler_integrate(lambda value: -value, np.array([1.0]), step_size=0.1, steps=20)
        self.assertEqual(trajectory.states.shape, (21, 1))
        fit = estimate_exponential_rate(trajectory.times, trajectory.states[:, 0])
        self.assertGreater(fit.decay_rate, 1.0)
        self.assertGreater(fit.r_squared, 0.999)
        hitting = first_hitting_time(trajectory.times, trajectory.states[:, 0], 0.5)
        self.assertIsNotNone(hitting)

    def test_discrete_flow_comparison_interpolates(self):
        flow_times = np.array([0.0, 0.5, 1.0])
        flow_states = np.column_stack((flow_times, 2.0 * flow_times))
        discrete_times = np.array([0.25, 0.75])
        discrete_states = np.column_stack((discrete_times, 2.0 * discrete_times))
        comparison = compare_discrete_to_flow(
            discrete_times, discrete_states, flow_times, flow_states
        )
        self.assertAlmostEqual(comparison.maximum_error, 0.0)


class StructuralAnalysisTests(unittest.TestCase):
    def test_coarse_graining_preserves_edge_weight(self):
        graph = EdgeListGraph.from_edges(4, [(0, 1), (1, 2), (2, 3)])
        coarse = coarse_grain_graph(graph, sequential_partition(4, 2))
        self.assertEqual(coarse.graph.node_count, 2)
        self.assertEqual(coarse.graph.edge_count, 1)
        self.assertAlmostEqual(coarse.internal_edge_weight, 2.0)
        self.assertAlmostEqual(float(coarse.edge_weights.sum()), 1.0)

    def test_orbits_and_equivariance(self):
        permutation = np.array([1, 0, 3, 2])
        labels = orbit_partition([permutation], 4)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])
        action = PermutationAction(permutation)
        error = equivariance_error(lambda value: 2.0 * value, action, np.arange(4.0))
        self.assertAlmostEqual(error, 0.0)

    def test_feature_graph_and_intervention(self):
        rng = np.random.default_rng(5)
        latent = rng.normal(size=200)
        activations = np.column_stack(
            (latent, latent + 0.01 * rng.normal(size=200), rng.normal(size=200))
        )
        feature_graph = feature_correlation_graph(activations, 0.9)
        self.assertIn([0, 1], feature_graph.graph.edges.tolist())
        complete_feature_graph = feature_correlation_graph(activations, 0.0)
        self.assertEqual(complete_feature_graph.graph.edge_count, 3)
        curve = representation_percolation_curve(activations, [0.2, 0.9])
        self.assertEqual(len(curve), 2)
        effect = intervention_effect(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))
        self.assertEqual(effect.changed_argmax_fraction, 1.0)

    def test_percolation_sweep_and_peak(self):
        result = run_percolation_sweep(
            [40], [0.0, 1.0], trials=2, seed=3, use_native=False
        )
        peak = estimate_peak(result, metric="largest_fraction", size=40)
        self.assertEqual(peak["control"], 1.0)


if __name__ == "__main__":
    unittest.main()
