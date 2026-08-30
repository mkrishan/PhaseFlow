"""Projection algorithms for two-set feasibility problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..model import Capability, StepOutcome
from ..problems.feasibility import FeasibilityProblem
from .base import BaseAlgorithm


def _require_feasibility(problem: Any) -> FeasibilityProblem:
    required = ("project_a", "project_b", "residual")
    if not all(hasattr(problem, attribute) for attribute in required):
        raise TypeError("projection algorithms require a FeasibilityProblem-compatible object")
    return problem


@dataclass(frozen=True)
class RRR(BaseAlgorithm):
    """Reflect-Reflect-Relax dynamics with an explicit flow-time clock."""

    relaxation: float = 0.1
    name: str = "rrr"
    capabilities: Sequence[Capability] = (
        Capability.PROJECTABLE,
        Capability.DISCRETE,
        Capability.FLOW_LIMIT,
    )

    def __post_init__(self) -> None:
        if self.relaxation <= 0.0:
            raise ValueError("relaxation must be positive")

    def step(self, problem, state, algorithm_state, rng) -> StepOutcome:
        feasibility = _require_feasibility(problem)
        candidate = np.asarray(state, dtype=float)
        projection_a = feasibility.project_a(candidate)
        reflected_a = 2.0 * projection_a - candidate
        direction = feasibility.project_b(reflected_a) - projection_a
        updated = candidate + self.relaxation * direction
        return StepOutcome(
            state=updated,
            algorithm_state=algorithm_state,
            metrics={
                "direction_norm": float(np.linalg.norm(direction)),
                "step_norm": float(np.linalg.norm(updated - candidate)),
            },
            clock_increments={
                "iteration": 1,
                "oracle_calls": 2,
                "flow_time": self.relaxation,
            },
        )

    def observe(self, problem, state, algorithm_state):
        feasibility = _require_feasibility(problem)
        return {"residual": feasibility.residual(np.asarray(state, dtype=float))}


@dataclass(frozen=True)
class AlternatingProjections(BaseAlgorithm):
    name: str = "alternating_projections"
    capabilities: Sequence[Capability] = (
        Capability.PROJECTABLE,
        Capability.DISCRETE,
    )

    def step(self, problem, state, algorithm_state, rng) -> StepOutcome:
        feasibility = _require_feasibility(problem)
        candidate = np.asarray(state, dtype=float)
        updated = feasibility.project_b(feasibility.project_a(candidate))
        return StepOutcome(
            state=updated,
            algorithm_state=algorithm_state,
            metrics={"step_norm": float(np.linalg.norm(updated - candidate))},
            clock_increments={"iteration": 1, "oracle_calls": 2},
        )

    def observe(self, problem, state, algorithm_state):
        feasibility = _require_feasibility(problem)
        return {"residual": feasibility.residual(np.asarray(state, dtype=float))}

