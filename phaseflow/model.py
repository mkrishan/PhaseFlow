"""Stable data structures shared by PhaseFlow experiments and plugins."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = "1.0"
PLUGIN_API_VERSION = "0.1"


class Capability(str, Enum):
    """Capabilities that problems, algorithms, and plugins may declare."""

    DISCRETE = "discrete"
    DIFFERENTIABLE = "differentiable"
    FLOW_LIMIT = "flow_limit"
    INTERPRETABLE = "interpretable"
    LOCALLY_CONSTRAINED = "locally_constrained"
    PROJECTABLE = "projectable"
    STOCHASTIC = "stochastic"
    SYMMETRIC = "symmetric"
    COARSE_GRAINABLE = "coarse_grainable"


@dataclass(frozen=True)
class ClockSnapshot:
    """A snapshot of the standard and domain-specific experiment clocks."""

    iteration: int = 0
    oracle_calls: int = 0
    flow_time: float = 0.0
    wall_time: float = 0.0
    extras: Mapping[str, float] = field(default_factory=dict)

    def advanced(
        self,
        increments: Optional[Mapping[str, float]] = None,
        *,
        wall_time: Optional[float] = None,
    ) -> "ClockSnapshot":
        values = dict(increments or {})
        iteration = self.iteration + int(values.pop("iteration", 1))
        oracle_calls = self.oracle_calls + int(values.pop("oracle_calls", 0))
        flow_time = self.flow_time + float(values.pop("flow_time", 0.0))
        extras = dict(self.extras)
        for key, value in values.items():
            extras[key] = extras.get(key, 0.0) + float(value)
        return ClockSnapshot(
            iteration=iteration,
            oracle_calls=oracle_calls,
            flow_time=flow_time,
            wall_time=self.wall_time if wall_time is None else float(wall_time),
            extras=extras,
        )

    def as_dict(self) -> Dict[str, float]:
        values: Dict[str, float] = {
            "iteration": self.iteration,
            "oracle_calls": self.oracle_calls,
            "flow_time": self.flow_time,
            "wall_time": self.wall_time,
        }
        values.update(self.extras)
        return values


@dataclass(frozen=True)
class Event:
    """One recorded observation of an algorithm trajectory."""

    step: int
    clocks: ClockSnapshot
    metrics: Mapping[str, float]
    tags: Tuple[str, ...] = ()


@dataclass
class StepOutcome:
    """Result of one algorithm update."""

    state: Any
    algorithm_state: Any = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    clock_increments: Mapping[str, float] = field(default_factory=dict)
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ExperimentSpec:
    """Portable, language-neutral description of an experiment."""

    identifier: str
    problem: str
    algorithm: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0
    clocks: Tuple[str, ...] = ("iteration", "oracle_calls", "flow_time", "wall_time")
    analyses: Tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunConfig:
    """Controls trajectory recording and generic convergence stopping."""

    max_steps: int
    seed: int = 0
    tolerance: Optional[float] = None
    convergence_metric: str = "residual"
    record_every: int = 1

    def __post_init__(self) -> None:
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if self.record_every <= 0:
            raise ValueError("record_every must be positive")
        if self.tolerance is not None and self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")


@dataclass
class RunResult:
    """Recorded output of an experiment run."""

    problem_name: str
    algorithm_name: str
    config: RunConfig
    events: Sequence[Event]
    states: Sequence[Any]
    stop_reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def final_event(self) -> Event:
        if not self.events:
            raise RuntimeError("run contains no events")
        return self.events[-1]

    @property
    def final_state(self) -> Any:
        if not self.states:
            raise RuntimeError("run contains no states")
        return self.states[-1]

    def metric_series(self, name: str) -> np.ndarray:
        return np.asarray([event.metrics.get(name, np.nan) for event in self.events], dtype=float)

    def clock_series(self, name: str) -> np.ndarray:
        values = [event.clocks.as_dict().get(name, np.nan) for event in self.events]
        return np.asarray(values, dtype=float)


class Problem(Protocol):
    name: str
    capabilities: Sequence[Capability]

    def initial_state(self, rng: np.random.Generator) -> Any:
        ...

    def observe(self, state: Any) -> Mapping[str, float]:
        ...


class Algorithm(Protocol):
    name: str
    capabilities: Sequence[Capability]

    def initialize(self, problem: Problem, state: Any, rng: np.random.Generator) -> Any:
        ...

    def step(
        self,
        problem: Problem,
        state: Any,
        algorithm_state: Any,
        rng: np.random.Generator,
    ) -> StepOutcome:
        ...

    def observe(self, problem: Problem, state: Any, algorithm_state: Any) -> Mapping[str, float]:
        ...


class Ensemble(Protocol):
    name: str

    def sample(
        self,
        size: int,
        control: float,
        rng: np.random.Generator,
    ) -> Problem:
        ...


class Observable(Protocol):
    name: str

    def measure(self, problem: Problem, state: Any, clocks: ClockSnapshot) -> Mapping[str, float]:
        ...


class CoarseGrainer(Protocol):
    name: str

    def transform(self, value: Any, scale: float) -> Any:
        ...


class Intervention(Protocol):
    name: str

    def apply(self, problem: Problem, state: Any, rng: np.random.Generator) -> Any:
        ...


Observer = Callable[[Problem, Any, ClockSnapshot], Mapping[str, float]]
StopPredicate = Callable[[Event], bool]


def snapshot_state(state: Any) -> Any:
    """Copy a state without requiring every plugin to implement a copy hook."""

    if isinstance(state, np.ndarray):
        return state.copy()
    return copy.deepcopy(state)
