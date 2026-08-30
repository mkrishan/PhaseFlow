# PhaseFlow architecture

PhaseFlow is organized around one invariant: every algorithm produces a
trajectory of states, clocks, metrics, and events that can be analyzed without
knowing the algorithm's implementation language.

## Layers

1. **Problem layer**
   Defines initial states, local structure, projections, constraints, and
   directly measurable quantities.

2. **Dynamics layer**
   Defines initialization and one algorithm update. An update returns a new
   state, internal algorithm state, metric values, clock increments, and event
   tags.

3. **Experiment layer**
   Owns deterministic random-number generation, stopping, state snapshots,
   observers, and the unified event stream.

4. **Analysis layer**
   Provides flow integration, trajectory comparison, parameter sweeps,
   transition estimates, graph coarse-graining, symmetry reduction, feature
   graphs, and intervention measurements.

5. **Backend layer**
   The Python implementation is always available. Rust accelerates stable,
   data-oriented kernels. A C-compatible interface gives Julia and other
   languages access to the same native implementation.

6. **Extension layer**
   Python package entry points discover separately distributed plugins. Plugin
   manifests declare their version, API version, capabilities, license, and
   optional citation.

## Stable concepts

The initial stable concepts are deliberately small:

- `ExperimentSpec`
- `RunConfig`
- `ClockSnapshot`
- `Event`
- `StepOutcome`
- `RunResult`
- `Problem`
- `Algorithm`

The scientific APIs will evolve during the alpha series, but saved result
bundles carry an explicit schema version so readers can migrate older runs.

## Multi-clock dynamics

The standard clocks are:

- `iteration`
- `oracle_calls`
- `flow_time`
- `wall_time`

Algorithms may add clocks such as training steps, processed tokens, layer
depth, sampling time, or coarse-graining scale. Extra clocks remain part of the
same `ClockSnapshot` and are serialized with every event.

## Native boundary

Rust's internal ABI is not a public contract. The stable native boundary is the
C header in `native/rust/phaseflow-ffi/include/phaseflow.h`. Python normally uses the
optional native extension, while Julia calls the shared library through its
native foreign-function interface.

In-memory callbacks from Python are intentionally kept outside tight Rust
loops. Native kernels receive arrays, edge lists, scalar parameters, and seeds;
they return arrays or compact statistics.

## Scientific origin

The explicit flow-time clock and the discrete-to-flow comparison design are
based on Manish Krishan Lal, *The Flow-Limit of Reflect-Reflect-Relax:
Existence, Stability, and Discrete-Time Behavior*, arXiv:2512.23843, 2025.
