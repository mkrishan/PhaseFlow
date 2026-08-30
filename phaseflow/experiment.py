"""Experiment execution and multi-clock event recording."""

from __future__ import annotations

import time
from typing import Iterable, Mapping, Optional

import numpy as np

from .model import (
    Algorithm,
    ClockSnapshot,
    Event,
    Observer,
    Problem,
    RunConfig,
    RunResult,
    StopPredicate,
    snapshot_state,
)


class ExperimentRunner:
    """Runs an algorithm and records a common PhaseFlow trajectory."""

    def run(
        self,
        problem: Problem,
        algorithm: Algorithm,
        config: RunConfig,
        *,
        observers: Iterable[Observer] = (),
        stop_when: Optional[StopPredicate] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> RunResult:
        rng = np.random.default_rng(config.seed)
        state = problem.initial_state(rng)
        algorithm_state = algorithm.initialize(problem, state, rng)
        clocks = ClockSnapshot()
        events = []
        states = []
        started = time.perf_counter()

        def collect_metrics() -> Mapping[str, float]:
            values = dict(problem.observe(state))
            values.update(algorithm.observe(problem, state, algorithm_state))
            for observer in observers:
                values.update(observer(problem, state, clocks))
            return {key: float(value) for key, value in values.items()}

        def record(step: int, metrics: Mapping[str, float], tags=()) -> Event:
            event = Event(step=step, clocks=clocks, metrics=dict(metrics), tags=tuple(tags))
            events.append(event)
            states.append(snapshot_state(state))
            return event

        initial_event = record(0, collect_metrics(), ("initial",))
        if self._should_stop(initial_event, config, stop_when):
            return RunResult(
                problem_name=problem.name,
                algorithm_name=algorithm.name,
                config=config,
                events=tuple(events),
                states=tuple(states),
                stop_reason="converged",
                metadata=dict(metadata or {}),
            )

        stop_reason = "maximum_steps"
        last_event = initial_event
        executed_steps = 0
        for step in range(1, config.max_steps + 1):
            executed_steps = step
            outcome = algorithm.step(problem, state, algorithm_state, rng)
            state = outcome.state
            algorithm_state = outcome.algorithm_state
            clocks = clocks.advanced(
                outcome.clock_increments,
                wall_time=time.perf_counter() - started,
            )

            metrics = dict(problem.observe(state))
            metrics.update(algorithm.observe(problem, state, algorithm_state))
            metrics.update({key: float(value) for key, value in outcome.metrics.items()})
            for observer in observers:
                metrics.update(observer(problem, state, clocks))
            metrics = {key: float(value) for key, value in metrics.items()}

            should_record = step % config.record_every == 0 or step == config.max_steps
            candidate = Event(step=step, clocks=clocks, metrics=metrics, tags=outcome.tags)
            should_stop = self._should_stop(candidate, config, stop_when)
            if should_record or should_stop:
                last_event = record(step, metrics, outcome.tags)
            if should_stop:
                stop_reason = "converged"
                break

        if last_event.step != executed_steps:
            record(executed_steps, collect_metrics(), ("final",))

        return RunResult(
            problem_name=problem.name,
            algorithm_name=algorithm.name,
            config=config,
            events=tuple(events),
            states=tuple(states),
            stop_reason=stop_reason,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _should_stop(
        event: Event,
        config: RunConfig,
        stop_when: Optional[StopPredicate],
    ) -> bool:
        if stop_when is not None and stop_when(event):
            return True
        if config.tolerance is None:
            return False
        value = event.metrics.get(config.convergence_metric)
        return value is not None and np.isfinite(value) and value <= config.tolerance
