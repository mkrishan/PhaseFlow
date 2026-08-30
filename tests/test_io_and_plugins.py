import tempfile
import unittest
from pathlib import Path

import numpy as np

from phaseflow import ExperimentRunner, RunConfig
from phaseflow.algorithms import RRR
from phaseflow.io import load_result_bundle, save_result_bundle
from phaseflow.model import PLUGIN_API_VERSION
from phaseflow.plugins import PluginManifest, validate_manifest
from phaseflow.problems import FeasibilityProblem, IdentitySet, PointSet


class PersistenceTests(unittest.TestCase):
    def test_result_bundle_round_trip(self):
        problem = FeasibilityProblem(
            IdentitySet(),
            PointSet(np.zeros(2)),
            np.array([1.0, -1.0]),
        )
        result = ExperimentRunner().run(problem, RRR(0.2), RunConfig(max_steps=4))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "run"
            save_result_bundle(result, str(destination))
            loaded = load_result_bundle(str(destination))
        self.assertEqual(loaded.manifest["algorithm"], "rrr")
        self.assertEqual(len(loaded.events), len(result.events))
        np.testing.assert_allclose(loaded.states[-1], result.final_state)


class PluginTests(unittest.TestCase):
    def test_manifest_version_validation(self):
        validate_manifest(PluginManifest(name="example", version="1.0"))
        with self.assertRaises(ValueError):
            validate_manifest(
                PluginManifest(name="example", version="1.0", api_version=PLUGIN_API_VERSION + "x")
            )


if __name__ == "__main__":
    unittest.main()

