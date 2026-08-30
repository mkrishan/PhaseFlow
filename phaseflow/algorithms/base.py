"""Base classes for PhaseFlow algorithms."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..model import Capability, Problem


class BaseAlgorithm:
    name = "algorithm"
    capabilities: Sequence[Capability] = ()

    def initialize(self, problem: Problem, state: Any, rng: np.random.Generator) -> Any:
        return None

    def observe(self, problem: Problem, state: Any, algorithm_state: Any) -> Mapping[str, float]:
        return {}

