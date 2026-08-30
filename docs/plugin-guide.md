# Plugin guide

A PhaseFlow plugin is a separately installable Python package that registers an
entry point in the `phaseflow.plugins` group.

## Minimal component

```python
from phaseflow import Capability
from phaseflow.plugins import PluginManifest


class ExamplePlugin:
    manifest = PluginManifest(
        name="example",
        version="0.1.0",
        capabilities=(Capability.STOCHASTIC,),
        description="An example PhaseFlow extension.",
        license="Apache-2.0",
    )


plugin = ExamplePlugin()
```

Register it in the plugin package's `pyproject.toml`:

```toml
[project.entry-points."phaseflow.plugins"]
example = "example_package:plugin"
```

## Contribution contract

A problem plugin should provide:

- `name` and `capabilities`;
- `initial_state(rng)`;
- `observe(state)`.

An algorithm plugin should provide:

- `name` and `capabilities`;
- `initialize(problem, state, rng)`;
- `step(problem, state, algorithm_state, rng)`;
- `observe(problem, state, algorithm_state)`.

Each algorithm step returns `StepOutcome`. Custom clocks are added through
`clock_increments`, and discrete events can be marked with short `tags`.

## Expectations

Plugins should include deterministic seed handling, a reproducible example,
small-instance correctness tests, capability declarations, dependency notes,
license metadata, and complexity notes for expensive operations.

The plugin API is versioned independently from the saved-result schema.
PhaseFlow rejects a plugin targeting a different API version instead of loading
it with uncertain behavior.

