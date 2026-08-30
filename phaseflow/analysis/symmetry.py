"""Finite permutation actions, orbit reduction, and equivariance checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np

from ..problems.percolation import EdgeListGraph
from .rg import CoarseGraph, coarse_grain_graph


@dataclass(frozen=True)
class PermutationAction:
    permutation: np.ndarray

    def __post_init__(self) -> None:
        permutation = np.asarray(self.permutation, dtype=np.int64)
        if permutation.ndim != 1 or not np.array_equal(np.sort(permutation), np.arange(permutation.size)):
            raise ValueError("permutation must contain each index exactly once")
        object.__setattr__(self, "permutation", permutation)

    def apply(self, value: np.ndarray) -> np.ndarray:
        candidate = np.asarray(value)
        if candidate.shape[0] != self.permutation.size:
            raise ValueError("value leading dimension does not match permutation")
        return candidate[self.permutation]


def orbit_partition(permutations: Iterable[np.ndarray], size: int) -> np.ndarray:
    parent = np.arange(size, dtype=np.int64)

    def find(value: int) -> int:
        while value != parent[value]:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for values in permutations:
        action = PermutationAction(values)
        if action.permutation.size != size:
            raise ValueError("all permutations must have the requested size")
        for source, target in enumerate(action.permutation):
            union(source, int(target))
    roots = np.fromiter((find(index) for index in range(size)), dtype=np.int64, count=size)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def equivariance_error(
    function: Callable[[np.ndarray], np.ndarray],
    action: PermutationAction,
    value: np.ndarray,
) -> float:
    candidate = np.asarray(value, dtype=float)
    transformed_output = np.asarray(function(action.apply(candidate)), dtype=float)
    expected = action.apply(np.asarray(function(candidate), dtype=float))
    return float(np.linalg.norm(transformed_output - expected))


def quotient_graph(graph: EdgeListGraph, orbit_labels: np.ndarray) -> CoarseGraph:
    return coarse_grain_graph(graph, orbit_labels)

