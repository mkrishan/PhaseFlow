"""Parameter sweeps and finite-size phase-transition summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..problems.percolation import percolation_observables, sample_erdos_renyi


@dataclass(frozen=True)
class SweepRecord:
    size: int
    control: float
    trial: int
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class PhaseSweepResult:
    records: Tuple[SweepRecord, ...]
    control_name: str
    model_name: str
    seed: int

    def metric_values(self, name: str) -> np.ndarray:
        return np.asarray([record.metrics.get(name, np.nan) for record in self.records], dtype=float)


def run_percolation_sweep(
    sizes: Sequence[int],
    probabilities: Sequence[float],
    *,
    trials: int,
    seed: int = 0,
    use_native: bool = True,
) -> PhaseSweepResult:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if any(size < 0 for size in sizes):
        raise ValueError("sizes must be non-negative")
    if any(not 0.0 <= probability <= 1.0 for probability in probabilities):
        raise ValueError("probabilities must be between zero and one")

    total = len(sizes) * len(probabilities) * trials
    child_seeds = np.random.SeedSequence(seed).spawn(total)
    records: List[SweepRecord] = []
    seed_index = 0
    for size in sizes:
        for probability in probabilities:
            for trial in range(trials):
                child_seed = int(child_seeds[seed_index].generate_state(1, dtype=np.uint64)[0])
                seed_index += 1
                graph = sample_erdos_renyi(
                    int(size),
                    float(probability),
                    seed=child_seed,
                    use_native=use_native,
                )
                records.append(
                    SweepRecord(
                        size=int(size),
                        control=float(probability),
                        trial=trial,
                        metrics=percolation_observables(graph, use_native=use_native),
                    )
                )
    return PhaseSweepResult(
        records=tuple(records),
        control_name="edge_probability",
        model_name="erdos_renyi",
        seed=seed,
    )


def summarize_sweep(
    result: PhaseSweepResult,
    metric: str,
) -> Tuple[Mapping[str, float], ...]:
    grouped: Dict[Tuple[int, float], List[float]] = {}
    for record in result.records:
        value = record.metrics.get(metric)
        if value is not None and np.isfinite(value):
            grouped.setdefault((record.size, record.control), []).append(float(value))

    summaries = []
    for (size, control), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        standard_error = 0.0 if array.size < 2 else float(array.std(ddof=1) / np.sqrt(array.size))
        summaries.append(
            {
                "size": float(size),
                "control": float(control),
                "mean": float(array.mean()),
                "standard_error": standard_error,
                "trials": float(array.size),
            }
        )
    return tuple(summaries)


def estimate_peak(
    result: PhaseSweepResult,
    metric: str = "susceptibility",
    *,
    size: Optional[int] = None,
) -> Mapping[str, float]:
    if size is None:
        if not result.records:
            raise ValueError("cannot estimate a peak from an empty sweep")
        size = max(record.size for record in result.records)
    summaries = [row for row in summarize_sweep(result, metric) if int(row["size"]) == size]
    if not summaries:
        raise ValueError(f"no observations for size {size}")
    return max(summaries, key=lambda row: row["mean"])

