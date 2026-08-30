# PhaseFlow.jl

This package is the native Julia interface to PhaseFlow's Rust computational
core. It currently exposes the stable graph-kernel ABI while the higher-level
experiment API develops alongside the Python package.

Build `phaseflow-ffi`, set `PHASEFLOW_LIBRARY` to the resulting shared-library
path if it is not on the system library path, and then activate this package.

Julia edge endpoints are one-based. Conversion to and from the zero-based native
representation is handled by the package.

PhaseFlow.jl is licensed under Apache-2.0.

