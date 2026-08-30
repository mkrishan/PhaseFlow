import unittest

import numpy as np

from phaseflow.problems import AffineSet, BoxSet, FeasibilityProblem, IdentitySet, PointSet


class FeasibilityTests(unittest.TestCase):
    def test_affine_projection_satisfies_constraint(self):
        affine = AffineSet(np.array([[1.0, 1.0]]), np.array([1.0]))
        projected = affine.project(np.array([2.0, 2.0]))
        np.testing.assert_allclose(projected.sum(), 1.0, atol=1e-12)

    def test_box_projection_clips_coordinates(self):
        box = BoxSet(np.array([-1.0, 0.0]), np.array([1.0, 2.0]))
        np.testing.assert_allclose(box.project(np.array([-2.0, 3.0])), [-1.0, 2.0])

    def test_rrr_vector_field_for_point_target(self):
        problem = FeasibilityProblem(
            IdentitySet(),
            PointSet(np.zeros(2)),
            np.ones(2),
        )
        np.testing.assert_allclose(problem.vector_field(np.array([2.0, -3.0])), [-2.0, 3.0])


if __name__ == "__main__":
    unittest.main()

