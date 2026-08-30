"""Graph coarse-graining and renormalization trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ..problems.percolation import EdgeListGraph


@dataclass(frozen=True)
class CoarseGraph:
    graph: EdgeListGraph
    assignment: np.ndarray
    node_weights: np.ndarray
    edge_weights: np.ndarray
    internal_edge_weight: float


@dataclass(frozen=True)
class RenormalizationLevel:
    level: int
    graph: EdgeListGraph
    node_weights: np.ndarray
    edge_weights: np.ndarray
    internal_edge_weight: float


def sequential_partition(node_count: int, block_size: int) -> np.ndarray:
    if node_count < 0:
        raise ValueError("node_count must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return np.arange(node_count, dtype=np.int64) // block_size


def coarse_grain_graph(
    graph: EdgeListGraph,
    assignment: np.ndarray,
    *,
    node_weights: Optional[np.ndarray] = None,
    edge_weights: Optional[np.ndarray] = None,
) -> CoarseGraph:
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.shape != (graph.node_count,):
        raise ValueError("assignment must have one label per node")
    if np.any(assignment < 0):
        raise ValueError("assignment labels must be non-negative")
    _, compact = np.unique(assignment, return_inverse=True)
    coarse_count = int(compact.max() + 1) if compact.size else 0

    if node_weights is None:
        node_weights = np.ones(graph.node_count, dtype=float)
    node_weights = np.asarray(node_weights, dtype=float)
    if node_weights.shape != (graph.node_count,):
        raise ValueError("node_weights must have one value per node")
    coarse_node_weights = np.bincount(compact, weights=node_weights, minlength=coarse_count)

    if edge_weights is None:
        edge_weights = np.ones(graph.edge_count, dtype=float)
    edge_weights = np.asarray(edge_weights, dtype=float)
    if edge_weights.shape != (graph.edge_count,):
        raise ValueError("edge_weights must have one value per edge")

    if graph.edge_count == 0:
        coarse_edges = np.empty((0, 2), dtype=np.int64)
        coarse_edge_weights = np.empty(0, dtype=float)
        internal = 0.0
    else:
        endpoints = compact[graph.edges]
        endpoints = np.sort(endpoints, axis=1)
        internal_mask = endpoints[:, 0] == endpoints[:, 1]
        internal = float(edge_weights[internal_mask].sum())
        external = endpoints[~internal_mask]
        external_weights = edge_weights[~internal_mask]
        if external.size == 0:
            coarse_edges = np.empty((0, 2), dtype=np.int64)
            coarse_edge_weights = np.empty(0, dtype=float)
        else:
            coarse_edges, inverse = np.unique(external, axis=0, return_inverse=True)
            coarse_edge_weights = np.bincount(inverse, weights=external_weights)

    return CoarseGraph(
        graph=EdgeListGraph(node_count=coarse_count, edges=coarse_edges),
        assignment=compact.astype(np.int64),
        node_weights=coarse_node_weights,
        edge_weights=np.asarray(coarse_edge_weights, dtype=float),
        internal_edge_weight=internal,
    )


def renormalization_trajectory(
    graph: EdgeListGraph,
    block_sizes: Sequence[int],
) -> Tuple[RenormalizationLevel, ...]:
    levels = [
        RenormalizationLevel(
            level=0,
            graph=graph,
            node_weights=np.ones(graph.node_count, dtype=float),
            edge_weights=np.ones(graph.edge_count, dtype=float),
            internal_edge_weight=0.0,
        )
    ]
    current_graph = graph
    current_nodes = levels[0].node_weights
    current_edges = levels[0].edge_weights
    cumulative_internal = 0.0
    for level, block_size in enumerate(block_sizes, start=1):
        assignment = sequential_partition(current_graph.node_count, int(block_size))
        coarse = coarse_grain_graph(
            current_graph,
            assignment,
            node_weights=current_nodes,
            edge_weights=current_edges,
        )
        cumulative_internal += coarse.internal_edge_weight
        levels.append(
            RenormalizationLevel(
                level=level,
                graph=coarse.graph,
                node_weights=coarse.node_weights,
                edge_weights=coarse.edge_weights,
                internal_edge_weight=cumulative_internal,
            )
        )
        current_graph = coarse.graph
        current_nodes = coarse.node_weights
        current_edges = coarse.edge_weights
    return tuple(levels)

