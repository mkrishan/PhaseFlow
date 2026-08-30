"""Compare RRR trajectories with a finely integrated flow trajectory."""

import numpy as np

from phaseflow import ExperimentRunner, RunConfig
from phaseflow.algorithms import RRR
from phaseflow.analysis import compare_discrete_to_flow, euler_integrate
from phaseflow.problems import FeasibilityProblem, IdentitySet, PointSet


problem = FeasibilityProblem(
    set_a=IdentitySet(),
    set_b=PointSet(np.zeros(2)),
    initial=np.array([2.0, -1.0]),
)

flow = euler_integrate(problem.vector_field, problem.initial, step_size=0.002, steps=2100)

for relaxation in (0.2, 0.1, 0.05):
    steps = int(4.0 / relaxation)
    run = ExperimentRunner().run(
        problem,
        RRR(relaxation),
        RunConfig(max_steps=steps, record_every=1),
    )
    comparison = compare_discrete_to_flow(
        run.clock_series("flow_time"),
        np.stack(run.states),
        flow.times,
        flow.states,
    )
    print(
        f"relaxation={relaxation:.3f} "
        f"maximum_error={comparison.maximum_error:.6f} "
        f"final_residual={run.final_event.metrics['residual']:.6f}"
    )
