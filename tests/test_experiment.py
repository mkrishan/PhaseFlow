import unittest

import numpy as np

from phaseflow import ExperimentRunner, RunConfig
from phaseflow.algorithms import RRR
from phaseflow.problems import FeasibilityProblem, IdentitySet, PointSet


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.problem = FeasibilityProblem(
            set_a=IdentitySet(),
            set_b=PointSet(np.zeros(2)),
            initial=np.array([2.0, -1.0]),
        )

    def test_rrr_tracks_iteration_oracle_and_flow_clocks(self):
        result = ExperimentRunner().run(
            self.problem,
            RRR(relaxation=0.25),
            RunConfig(max_steps=4, tolerance=None, seed=3),
        )
        np.testing.assert_allclose(result.final_state, np.array([2.0, -1.0]) * 0.75**4)
        self.assertEqual(result.final_event.clocks.iteration, 4)
        self.assertEqual(result.final_event.clocks.oracle_calls, 8)
        self.assertAlmostEqual(result.final_event.clocks.flow_time, 1.0)
        self.assertLess(result.final_event.metrics["residual"], result.events[0].metrics["residual"])

    def test_generic_tolerance_stops_the_run(self):
        result = ExperimentRunner().run(
            self.problem,
            RRR(relaxation=0.5),
            RunConfig(max_steps=100, tolerance=0.1),
        )
        self.assertEqual(result.stop_reason, "converged")
        self.assertLessEqual(result.final_event.metrics["residual"], 0.1)
        self.assertLess(result.final_event.step, 100)

    def test_final_state_is_recorded_with_sparse_recording(self):
        result = ExperimentRunner().run(
            self.problem,
            RRR(relaxation=0.1),
            RunConfig(max_steps=5, tolerance=None, record_every=3),
        )
        self.assertEqual([event.step for event in result.events], [0, 3, 5])
        self.assertEqual(len(result.events), len(result.states))


if __name__ == "__main__":
    unittest.main()

