"""Projection-set primitives and two-set feasibility problems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence, Union

import numpy as np

from ..model import Capability

Array = np.ndarray
InitialState = Union[Array, Callable[[np.random.Generator], Array]]


class ProjectionSet:
    """Interface implemented by sets that provide a Euclidean projection."""

    def project(self, value: Array) -> Array:
        raise NotImplementedError


@dataclass(frozen=True)
class IdentitySet(ProjectionSet):
    """The full ambient space."""

    def project(self, value: Array) -> Array:
        return np.asarray(value, dtype=float).copy()


@dataclass(frozen=True)
class PointSet(ProjectionSet):
    point: Array

    def __post_init__(self) -> None:
        point = np.asarray(self.point, dtype=float)
        if point.ndim != 1:
            raise ValueError("point must be one-dimensional")
        object.__setattr__(self, "point", point.copy())

    def project(self, value: Array) -> Array:
        candidate = np.asarray(value, dtype=float)
        if candidate.shape != self.point.shape:
            raise ValueError("value and point must have the same shape")
        return self.point.copy()


@dataclass(frozen=True)
class BoxSet(ProjectionSet):
    lower: Array
    upper: Array

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.ndim != 1 or lower.shape != upper.shape:
            raise ValueError("lower and upper must be equally shaped vectors")
        if np.any(lower > upper):
            raise ValueError("lower bounds cannot exceed upper bounds")
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())

    def project(self, value: Array) -> Array:
        candidate = np.asarray(value, dtype=float)
        if candidate.shape != self.lower.shape:
            raise ValueError("value and box bounds must have the same shape")
        return np.clip(candidate, self.lower, self.upper)


@dataclass
class AffineSet(ProjectionSet):
    """The affine set ``matrix @ x = target``."""

    matrix: Array
    target: Array
    rcond: float = 1e-12
    _correction: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)
        self.target = np.asarray(self.target, dtype=float)
        if self.matrix.ndim != 2 or self.target.ndim != 1:
            raise ValueError("matrix must be two-dimensional and target one-dimensional")
        if self.matrix.shape[0] != self.target.shape[0]:
            raise ValueError("matrix row count must equal target length")
        gram = self.matrix @ self.matrix.T
        self._correction = self.matrix.T @ np.linalg.pinv(gram, rcond=self.rcond)

    def project(self, value: Array) -> Array:
        candidate = np.asarray(value, dtype=float)
        if candidate.ndim != 1 or candidate.shape[0] != self.matrix.shape[1]:
            raise ValueError("value dimension does not match affine set")
        return candidate - self._correction @ (self.matrix @ candidate - self.target)


@dataclass
class FeasibilityProblem:
    """Find a point in the intersection of two projectable sets."""

    set_a: ProjectionSet
    set_b: ProjectionSet
    initial: InitialState
    name: str = "two_set_feasibility"
    capabilities: Sequence[Capability] = (
        Capability.PROJECTABLE,
        Capability.DISCRETE,
        Capability.FLOW_LIMIT,
    )

    def initial_state(self, rng: np.random.Generator) -> Array:
        value = self.initial(rng) if callable(self.initial) else self.initial
        state = np.asarray(value, dtype=float)
        if state.ndim != 1:
            raise ValueError("initial state must be a vector")
        return state.copy()

    def project_a(self, value: Array) -> Array:
        return self.set_a.project(value)

    def project_b(self, value: Array) -> Array:
        return self.set_b.project(value)

    def vector_field(self, value: Array) -> Array:
        candidate = np.asarray(value, dtype=float)
        projection_a = self.project_a(candidate)
        reflected_a = 2.0 * projection_a - candidate
        return self.project_b(reflected_a) - projection_a

    def residual(self, value: Array) -> float:
        return float(np.linalg.norm(self.vector_field(value)))

    def observe(self, state: Array):
        candidate = np.asarray(state, dtype=float)
        return {
            "state_norm": float(np.linalg.norm(candidate)),
            "residual": self.residual(candidate),
        }

