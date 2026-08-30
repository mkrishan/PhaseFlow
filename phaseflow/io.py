"""Portable on-disk result bundles for PhaseFlow runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .model import ClockSnapshot, Event, RunConfig, RunResult


@dataclass(frozen=True)
class LoadedBundle:
    manifest: Mapping[str, Any]
    events: Sequence[Event]
    states: np.ndarray


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_result_bundle(result: RunResult, path: str, *, overwrite: bool = False) -> Path:
    destination = Path(path)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"result directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    arrays = [np.asarray(state) for state in result.states]
    try:
        states = np.stack(arrays)
    except ValueError as error:
        raise TypeError("portable bundles currently require equally shaped numeric states") from error
    if states.dtype == object:
        raise TypeError("portable bundles do not support object-valued states")

    manifest: Dict[str, Any] = {
        "schema_version": result.schema_version,
        "problem": result.problem_name,
        "algorithm": result.algorithm_name,
        "config": asdict(result.config),
        "stop_reason": result.stop_reason,
        "metadata": _jsonable(result.metadata),
        "event_count": len(result.events),
        "state_shape": list(states.shape),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (destination / "events.jsonl").open("w", encoding="utf-8") as stream:
        for event in result.events:
            row = {
                "step": event.step,
                "clocks": event.clocks.as_dict(),
                "metrics": _jsonable(event.metrics),
                "tags": list(event.tags),
            }
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    np.save(destination / "states.npy", states, allow_pickle=False)
    return destination


def load_result_bundle(path: str) -> LoadedBundle:
    source = Path(path)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    events = []
    with (source / "events.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            clock_values = dict(row["clocks"])
            extras = {
                key: value
                for key, value in clock_values.items()
                if key not in {"iteration", "oracle_calls", "flow_time", "wall_time"}
            }
            clocks = ClockSnapshot(
                iteration=int(clock_values.get("iteration", 0)),
                oracle_calls=int(clock_values.get("oracle_calls", 0)),
                flow_time=float(clock_values.get("flow_time", 0.0)),
                wall_time=float(clock_values.get("wall_time", 0.0)),
                extras=extras,
            )
            events.append(
                Event(
                    step=int(row["step"]),
                    clocks=clocks,
                    metrics={key: float(value) for key, value in row["metrics"].items()},
                    tags=tuple(row.get("tags", ())),
                )
            )
    states = np.load(source / "states.npy", allow_pickle=False)
    return LoadedBundle(manifest=manifest, events=tuple(events), states=states)

