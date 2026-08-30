"""Command-line entry point for small reproducible PhaseFlow experiments."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

import numpy as np

from . import SCHEMA_VERSION, __version__
from ._accelerate import backend_name
from .algorithms import RRR
from .analysis import run_percolation_sweep, summarize_sweep
from .experiment import ExperimentRunner
from .io import save_result_bundle
from .model import RunConfig
from .problems import FeasibilityProblem, IdentitySet, PointSet


def _rrr_demo(arguments: argparse.Namespace) -> int:
    initial = np.asarray(arguments.initial, dtype=float)
    problem = FeasibilityProblem(
        set_a=IdentitySet(),
        set_b=PointSet(np.zeros_like(initial)),
        initial=initial,
    )
    result = ExperimentRunner().run(
        problem,
        RRR(relaxation=arguments.relaxation),
        RunConfig(
            max_steps=arguments.steps,
            seed=arguments.seed,
            tolerance=arguments.tolerance,
            record_every=arguments.record_every,
        ),
        metadata={"demo": "rrr"},
    )
    if arguments.output:
        save_result_bundle(result, arguments.output, overwrite=arguments.overwrite)
    summary = {
        "algorithm": result.algorithm_name,
        "backend": backend_name(),
        "final_flow_time": result.final_event.clocks.flow_time,
        "final_residual": result.final_event.metrics["residual"],
        "iterations": result.final_event.clocks.iteration,
        "output": arguments.output,
        "stop_reason": result.stop_reason,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _percolation_demo(arguments: argparse.Namespace) -> int:
    result = run_percolation_sweep(
        [arguments.nodes],
        [arguments.probability],
        trials=arguments.trials,
        seed=arguments.seed,
    )
    summary = summarize_sweep(result, "largest_fraction")[0]
    summary = {
        "backend": backend_name(),
        "model": result.model_name,
        "nodes": arguments.nodes,
        "probability": arguments.probability,
        "largest_fraction_mean": summary["mean"],
        "largest_fraction_standard_error": summary["standard_error"],
        "trials": arguments.trials,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _info(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "backend": backend_name(),
                "name": "PhaseFlow",
                "schema_version": SCHEMA_VERSION,
                "version": __version__,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phaseflow", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info", help="show installation and backend information")
    info.set_defaults(handler=_info)

    demo = commands.add_parser("demo", help="run a built-in experiment")
    demos = demo.add_subparsers(dest="demo", required=True)

    rrr = demos.add_parser("rrr", help="run a simple RRR flow-limit experiment")
    rrr.add_argument("--steps", type=int, default=80)
    rrr.add_argument("--relaxation", type=float, default=0.1)
    rrr.add_argument("--initial", type=float, nargs="+", default=[2.0, -1.0])
    rrr.add_argument("--seed", type=int, default=0)
    rrr.add_argument("--tolerance", type=float, default=1e-8)
    rrr.add_argument("--record-every", type=int, default=1)
    rrr.add_argument("--output")
    rrr.add_argument("--overwrite", action="store_true")
    rrr.set_defaults(handler=_rrr_demo)

    percolation = demos.add_parser("percolation", help="sample sparse random graphs")
    percolation.add_argument("--nodes", type=int, default=1000)
    percolation.add_argument("--probability", type=float, default=0.001)
    percolation.add_argument("--trials", type=int, default=20)
    percolation.add_argument("--seed", type=int, default=0)
    percolation.set_defaults(handler=_percolation_demo)
    return parser


def main(argv: Sequence[str] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    return int(arguments.handler(arguments))

