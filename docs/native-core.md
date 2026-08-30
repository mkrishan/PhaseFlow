# Native core

PhaseFlow works as a normal Python package without compilation. The optional
native implementation accelerates sparse graph generation and connected
components while preserving the same Python API.

The Rust workspace contains:

- `phaseflow-core`: dependency-light graph and percolation kernels;
- `phaseflow-ffi`: a versioned C-compatible interface for Julia and other
  native clients;
- `phaseflow-python`: optional Python bindings.

## Python native build

Install a current Rust toolchain and `maturin`, then build from
`native/rust/phaseflow-python`:

```bash
maturin develop --release
```

`phaseflow info` reports `rust` when the extension is available and `python`
when PhaseFlow is using its portable fallback.

## Julia integration

Build the C-compatible library from the `native/rust` directory:

```bash
cargo build --release -p phaseflow-ffi
```

Julia can load the resulting shared library with `@ccall`. The public C
declarations are in `native/rust/phaseflow-ffi/include/phaseflow.h`. The ABI begins at
version 1 and is independent of Rust's internal ABI.
