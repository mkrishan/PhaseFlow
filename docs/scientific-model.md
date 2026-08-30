# Scientific model

PhaseFlow treats a computational experiment as six composable objects:

```text
ensemble + problem + dynamics + clocks + observables + interventions
```

## Ensembles and control parameters

An ensemble generates instances at a system size and control parameter. A phase
sweep repeats the experiment over sizes, control values, and deterministic
child seeds. This structure supports estimates of transition locations,
finite-size effects, susceptibility peaks, and changes in algorithmic behavior.

## Discrete and continuous dynamics

An algorithm produces discrete updates. When a flow description is available,
the analysis layer can integrate its vector field and compare the two
trajectories on the algorithm's flow-time clock.

PhaseFlow separates iteration count, oracle calls, flow time, and wall time.
This prevents a numerical implementation detail from being confused with the
underlying dynamical scale.

## Percolation observables

The initial graph observables are:

- number of components;
- largest-component size and fraction;
- susceptibility excluding the largest component;
- edge count and mean degree.

The sparse random-graph generator uses geometric skipping rather than examining
all possible edges when the graph is sparse.

## Coarse-graining

A coarse-grainer maps fine nodes to block labels. PhaseFlow records coarse node
weights, external edge weights, and internalized edge weight. Repeating the
operation produces a renormalization trajectory whose levels can be analyzed
with the same event and observable concepts used for algorithmic trajectories.

## Learned representations

The first interpretability interface is model-agnostic. It consumes activation
matrices or model outputs and provides:

- thresholded feature-association graphs;
- representation percolation curves;
- multiscale graph summaries;
- output differences after a controlled intervention;
- argmax changes, relative effects, and similarity measurements.

Probe or correlation results should be treated as descriptive. Causal claims
should be supported by explicit interventions and stability checks across
seeds, thresholds, and model instances.

## Symmetry

Finite permutation actions support orbit discovery, quotient graphs, and direct
equivariance-error measurements. More specialized algebraic structures can be
implemented as plugins without enlarging the core data model.

