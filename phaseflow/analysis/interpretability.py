"""Model-agnostic feature graphs and causal intervention measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np

from ..problems.percolation import EdgeListGraph, component_statistics


@dataclass(frozen=True)
class FeatureGraph:
    graph: EdgeListGraph
    association: np.ndarray
    threshold: float
    absolute: bool


@dataclass(frozen=True)
class InterventionEffect:
    mean_absolute_effect: float
    root_mean_square_effect: float
    relative_l2_effect: float
    cosine_similarity: float
    changed_argmax_fraction: float


def _association_matrix(activations: np.ndarray) -> np.ndarray:
    values = np.asarray(activations, dtype=float)
    if values.ndim != 2:
        raise ValueError("activations must have shape (observations, features)")
    if values.shape[0] < 2:
        raise ValueError("at least two observations are required")
    association = np.corrcoef(values, rowvar=False)
    association = np.atleast_2d(association)
    association = np.nan_to_num(association, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(association, 0.0)
    return association


def feature_correlation_graph(
    activations: np.ndarray,
    threshold: float,
    *,
    absolute: bool = True,
) -> FeatureGraph:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    association = _association_matrix(activations)
    scores = np.abs(association) if absolute else association
    left, right = np.triu_indices(scores.shape[0], 1)
    selected = scores[left, right] >= threshold
    left, right = left[selected], right[selected]
    edges = np.column_stack((left, right)).astype(np.int64, copy=False)
    return FeatureGraph(
        graph=EdgeListGraph(node_count=association.shape[0], edges=edges),
        association=association,
        threshold=float(threshold),
        absolute=absolute,
    )


def representation_percolation_curve(
    activations: np.ndarray,
    thresholds: Sequence[float],
    *,
    absolute: bool = True,
) -> Tuple[Mapping[str, float], ...]:
    association = _association_matrix(activations)
    scores = np.abs(association) if absolute else association
    rows = []
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("thresholds must be between zero and one")
        left, right = np.triu_indices(scores.shape[0], 1)
        selected = scores[left, right] >= threshold
        left, right = left[selected], right[selected]
        edges = np.column_stack((left, right)).astype(np.int64, copy=False)
        graph = EdgeListGraph(node_count=association.shape[0], edges=edges)
        statistics = component_statistics(graph)
        rows.append(
            {
                "threshold": float(threshold),
                "edge_count": float(graph.edge_count),
                "largest_fraction": statistics.largest_fraction,
                "susceptibility": statistics.susceptibility,
                "component_count": float(statistics.component_count),
            }
        )
    return tuple(rows)


def intervention_effect(baseline: np.ndarray, intervened: np.ndarray) -> InterventionEffect:
    baseline = np.asarray(baseline, dtype=float)
    intervened = np.asarray(intervened, dtype=float)
    if baseline.shape != intervened.shape:
        raise ValueError("baseline and intervened outputs must have the same shape")
    if baseline.size == 0:
        raise ValueError("outputs cannot be empty")
    difference = intervened - baseline
    baseline_norm = float(np.linalg.norm(baseline))
    intervened_norm = float(np.linalg.norm(intervened))
    denominator = baseline_norm * intervened_norm
    cosine = 1.0 if denominator == 0.0 else float(np.vdot(baseline, intervened).real / denominator)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    relative = float(np.linalg.norm(difference) / max(baseline_norm, np.finfo(float).eps))
    changed = 0.0
    if baseline.ndim >= 2 and baseline.shape[-1] > 1:
        changed = float(np.mean(np.argmax(baseline, axis=-1) != np.argmax(intervened, axis=-1)))
    return InterventionEffect(
        mean_absolute_effect=float(np.mean(np.abs(difference))),
        root_mean_square_effect=float(np.sqrt(np.mean(np.square(difference)))),
        relative_l2_effect=relative,
        cosine_similarity=cosine,
        changed_argmax_fraction=changed,
    )
