# PhaseFlow

PhaseFlow is a Python-first research platform for studying multiscale dynamical
phase transitions in algorithms and learned systems. It combines discrete
algorithm trajectories, continuous flow limits, percolation observables,
coarse-graining, and causal intervention measurements in one experiment model.

PhaseFlow is currently an alpha release. The first release includes:

- a versioned experiment and multi-clock trajectory model;
- Reflect-Reflect-Relax (RRR) feasibility dynamics;
- efficient Erdős–Rényi and bond-percolation experiments;
- component, susceptibility, parameter-sweep, and threshold analysis;
- Euler flow integration, hitting times, and discrete-to-flow comparison;
- graph coarse-graining and renormalization trajectories;
- feature-graph and intervention-effect utilities for learned representations;
- portable result bundles and a small command-line interface;
- an optional Rust acceleration layer with a pure-Python fallback.

## Installation

For local development:

```bash
python -m pip install -e .
```

The Python package works without the Rust extension. See
[`docs/native-core.md`](docs/native-core.md) for optional native builds.

## Quick start

Run the built-in demonstrations:

```bash
phaseflow demo rrr --steps 80 --output run-rrr
phaseflow demo percolation --nodes 1000 --probability 0.001 --trials 20
```

Or use the Python API:

```python
import numpy as np

from phaseflow import ExperimentRunner, RunConfig
from phaseflow.algorithms import RRR
from phaseflow.problems import FeasibilityProblem, IdentitySet, PointSet

problem = FeasibilityProblem(
    set_a=IdentitySet(),
    set_b=PointSet(np.zeros(2)),
    initial=np.array([2.0, -1.0]),
)
result = ExperimentRunner().run(
    problem,
    RRR(relaxation=0.1),
    RunConfig(max_steps=100, tolerance=1e-8, seed=7),
)
print(result.final_event.metrics["residual"])
```

## Design principle

PhaseFlow keeps the stable core deliberately small. Problems, algorithms,
observables, coarse-grainers, and interventions are independent components.
New research modules can be distributed as Python plugins without modifying the
main package.

## Citation

The flow-limit design is based on:

Manish Krishan Lal, *The Flow-Limit of Reflect-Reflect-Relax: Existence,
Stability, and Discrete-Time Behavior*, arXiv:2512.23843, 2025.

## License

Copyright 2025 Manish Krishan Lal.

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE).

