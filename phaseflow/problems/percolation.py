"""Sparse graph models and percolation observables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from .._accelerate import native_available, native_component_statistics, native_erdos_renyi


@dataclass(frozen=True)
class EdgeListGraph:
    node_count: int
    edges: np.ndarray

    def __post_init__(self) -> None:
        if self.node_count < 0:
            raise ValueError("node_count must be non-negative")
        edges = np.asarray(self.edges, dtype=np.int64)
        if edges.size == 0:
            edges = np.empty((0, 2), dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must have shape (edge_count, 2)")
        if np.any(edges < 0) or np.any(edges >= self.node_count):
            raise ValueError("edge endpoint is outside the graph")
        edges = np.sort(edges, axis=1)
        edges = edges[edges[:, 0] != edges[:, 1]]
        if len(edges):
            edges = np.unique(edges, axis=0)
        object.__setattr__(self, "edges", edges)

    @classmethod
    def from_edges(cls, node_count: int, edges: Iterable[Tuple[int, int]]) -> "EdgeListGraph":
        materialized = list(edges)
        array = np.asarray(materialized, dtype=np.int64)
        if not materialized:
            array = np.empty((0, 2), dtype=np.int64)
        return cls(node_count=node_count, edges=array)

    @property
    def edge_count(self) -> int:
        return int(self.edges.shape[0])

    def degrees(self) -> np.ndarray:
        result = np.zeros(self.node_count, dtype=np.int64)
        if self.edge_count:
            np.add.at(result, self.edges[:, 0], 1)
            np.add.at(result, self.edges[:, 1], 1)
        return result


@dataclass(frozen=True)
class ComponentStatistics:
    labels: np.ndarray
    sizes: np.ndarray
    largest_size: int
    largest_fraction: float
    susceptibility: float
    component_count: int


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.weight = np.ones(size, dtype=np.int64)

    def find(self, value: int) -> int:
        root = value
        while root != self.parent[root]:
            root = int(self.parent[root])
        while value != root:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.weight[left_root] < self.weight[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.weight[left_root] += self.weight[right_root]


def _statistics_from_labels(labels: np.ndarray) -> ComponentStatistics:
    node_count = int(labels.size)
    if node_count == 0:
        return ComponentStatistics(
            labels=np.empty(0, dtype=np.int64),
            sizes=np.empty(0, dtype=np.int64),
            largest_size=0,
            largest_fraction=0.0,
            susceptibility=0.0,
            component_count=0,
        )
    _, compact = np.unique(labels, return_inverse=True)
    sizes = np.bincount(compact).astype(np.int64)
    sizes = np.sort(sizes)[::-1]
    largest = int(sizes[0])
    remaining = sizes[1:]
    denominator = int(remaining.sum())
    susceptibility = 0.0 if denominator == 0 else float(np.square(remaining).sum() / denominator)
    return ComponentStatistics(
        labels=compact.astype(np.int64),
        sizes=sizes,
        largest_size=largest,
        largest_fraction=float(largest / node_count),
        susceptibility=susceptibility,
        component_count=int(sizes.size),
    )


def component_statistics(graph: EdgeListGraph, *, use_native: bool = True) -> ComponentStatistics:
    if use_native and native_available():
        labels = native_component_statistics(graph.node_count, graph.edges)
        return _statistics_from_labels(labels)
    union_find = _UnionFind(graph.node_count)
    for left, right in graph.edges:
        union_find.union(int(left), int(right))
    labels = np.fromiter(
        (union_find.find(index) for index in range(graph.node_count)),
        dtype=np.int64,
        count=graph.node_count,
    )
    return _statistics_from_labels(labels)


def _python_erdos_renyi(node_count: int, probability: float, seed: Optional[int]) -> np.ndarray:
    if probability == 0.0 or node_count < 2:
        return np.empty((0, 2), dtype=np.int64)
    if probability == 1.0:
        left, right = np.triu_indices(node_count, 1)
        return np.column_stack((left, right)).astype(np.int64, copy=False)

    rng = np.random.default_rng(seed)
    log_q = np.log1p(-probability)
    edges = []
    right = 1
    left = -1
    while right < node_count:
        draw = float(rng.random())
        left += 1 + int(np.log1p(-draw) / log_q)
        while left >= right and right < node_count:
            left -= right
            right += 1
        if right < node_count:
            edges.append((left, right))
    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64)


def sample_erdos_renyi(
    node_count: int,
    probability: float,
    *,
    seed: Optional[int] = None,
    use_native: bool = True,
) -> EdgeListGraph:
    if node_count < 0:
        raise ValueError("node_count must be non-negative")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    if use_native and native_available() and seed is not None:
        edges = native_erdos_renyi(node_count, probability, seed)
    else:
        edges = _python_erdos_renyi(node_count, probability, seed)
    return EdgeListGraph(node_count=node_count, edges=edges)


def bond_percolate(
    graph: EdgeListGraph,
    retention_probability: float,
    *,
    seed: Optional[int] = None,
) -> EdgeListGraph:
    if not 0.0 <= retention_probability <= 1.0:
        raise ValueError("retention_probability must be between zero and one")
    rng = np.random.default_rng(seed)
    mask = rng.random(graph.edge_count) < retention_probability
    return EdgeListGraph(node_count=graph.node_count, edges=graph.edges[mask])


def percolation_observables(graph: EdgeListGraph, *, use_native: bool = True):
    statistics = component_statistics(graph, use_native=use_native)
    mean_degree = 0.0 if graph.node_count == 0 else 2.0 * graph.edge_count / graph.node_count
    return {
        "node_count": float(graph.node_count),
        "edge_count": float(graph.edge_count),
        "mean_degree": float(mean_degree),
        "component_count": float(statistics.component_count),
        "largest_component": float(statistics.largest_size),
        "largest_fraction": statistics.largest_fraction,
        "susceptibility": statistics.susceptibility,
    }

