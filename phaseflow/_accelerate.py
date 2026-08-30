"""Optional native acceleration with a transparent Python fallback."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import _phaseflow_native as _native
except ImportError:  # pragma: no cover - exercised when native wheels are built
    _native = None


def native_available() -> bool:
    return _native is not None


def backend_name() -> str:
    return "rust" if native_available() else "python"


def native_component_statistics(node_count: int, edges: np.ndarray) -> np.ndarray:
    if _native is None:
        raise RuntimeError("PhaseFlow native extension is not installed")
    values = [(int(left), int(right)) for left, right in np.asarray(edges)]
    labels = _native.component_labels(node_count, values)
    return np.asarray(labels, dtype=np.int64)


def native_erdos_renyi(node_count: int, probability: float, seed: int) -> np.ndarray:
    if _native is None:
        raise RuntimeError("PhaseFlow native extension is not installed")
    edges = _native.erdos_renyi_edges(node_count, probability, int(seed))
    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64)
