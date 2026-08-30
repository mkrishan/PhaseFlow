"""Continuous flow integration and discrete-to-flow comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class FlowTrajectory:
    times: np.ndarray
    states: np.ndarray


@dataclass(frozen=True)
class FlowComparison:
    times: np.ndarray
    errors: np.ndarray
    maximum_error: float
    root_mean_square_error: float


@dataclass(frozen=True)
class ExponentialFit:
    decay_rate: float
    log_amplitude: float
    r_squared: float
    sample_count: int


def euler_integrate(
    vector_field: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    *,
    step_size: float,
    steps: int,
) -> FlowTrajectory:
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    initial_state = np.asarray(initial, dtype=float)
    if initial_state.ndim != 1:
        raise ValueError("initial must be a vector")
    states = np.empty((steps + 1, initial_state.size), dtype=float)
    states[0] = initial_state
    for index in range(steps):
        derivative = np.asarray(vector_field(states[index]), dtype=float)
        if derivative.shape != initial_state.shape:
            raise ValueError("vector field output has the wrong shape")
        states[index + 1] = states[index] + step_size * derivative
    return FlowTrajectory(times=np.arange(steps + 1, dtype=float) * step_size, states=states)


def _interpolate_states(times: np.ndarray, states: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if states.ndim != 2 or states.shape[0] != times.size:
        raise ValueError("states must have shape (time_count, dimension)")
    if times.size < 2:
        raise ValueError("flow trajectory must contain at least two times")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if targets.size and (targets.min() < times[0] or targets.max() > times[-1]):
        raise ValueError("target times fall outside the flow trajectory")
    columns = [np.interp(targets, times, states[:, column]) for column in range(states.shape[1])]
    return np.column_stack(columns) if columns else np.empty((targets.size, 0))


def compare_discrete_to_flow(
    discrete_times: np.ndarray,
    discrete_states: np.ndarray,
    flow_times: np.ndarray,
    flow_states: np.ndarray,
) -> FlowComparison:
    discrete_times = np.asarray(discrete_times, dtype=float)
    discrete_states = np.asarray(discrete_states, dtype=float)
    flow_times = np.asarray(flow_times, dtype=float)
    flow_states = np.asarray(flow_states, dtype=float)
    if discrete_states.ndim != 2 or discrete_states.shape[0] != discrete_times.size:
        raise ValueError("discrete states and times are inconsistent")
    interpolated = _interpolate_states(flow_times, flow_states, discrete_times)
    if interpolated.shape != discrete_states.shape:
        raise ValueError("discrete and flow state dimensions differ")
    errors = np.linalg.norm(discrete_states - interpolated, axis=1)
    maximum = float(errors.max()) if errors.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(errors)))) if errors.size else 0.0
    return FlowComparison(
        times=discrete_times,
        errors=errors,
        maximum_error=maximum,
        root_mean_square_error=rms,
    )


def first_hitting_time(
    times: np.ndarray,
    values: np.ndarray,
    threshold: float,
    *,
    direction: str = "below",
) -> Optional[float]:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.ndim != 1 or values.shape != times.shape:
        raise ValueError("times and values must be equally shaped vectors")
    if direction == "below":
        matches = np.flatnonzero(values <= threshold)
    elif direction == "above":
        matches = np.flatnonzero(values >= threshold)
    else:
        raise ValueError("direction must be 'below' or 'above'")
    return None if matches.size == 0 else float(times[int(matches[0])])


def estimate_exponential_rate(times: np.ndarray, values: np.ndarray) -> ExponentialFit:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.ndim != 1 or values.shape != times.shape:
        raise ValueError("times and values must be equally shaped vectors")
    mask = np.isfinite(times) & np.isfinite(values) & (values > 0.0)
    if int(mask.sum()) < 2:
        raise ValueError("at least two finite positive observations are required")
    selected_times = times[mask]
    logarithms = np.log(values[mask])
    slope, intercept = np.polyfit(selected_times, logarithms, 1)
    predicted = slope * selected_times + intercept
    residual = float(np.square(logarithms - predicted).sum())
    total = float(np.square(logarithms - logarithms.mean()).sum())
    r_squared = 1.0 if total == 0.0 else 1.0 - residual / total
    return ExponentialFit(
        decay_rate=float(-slope),
        log_amplitude=float(intercept),
        r_squared=float(r_squared),
        sample_count=int(mask.sum()),
    )
