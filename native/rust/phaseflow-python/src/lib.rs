use phaseflow_core::{component_labels as core_component_labels, erdos_renyi_edges as core_edges};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn component_labels(node_count: usize, edges: Vec<(usize, usize)>) -> PyResult<Vec<usize>> {
    core_component_labels(node_count, &edges)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn erdos_renyi_edges(
    node_count: usize,
    probability: f64,
    seed: u64,
) -> PyResult<Vec<(usize, usize)>> {
    core_edges(node_count, probability, seed)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn native_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _phaseflow_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(component_labels, module)?)?;
    module.add_function(wrap_pyfunction!(erdos_renyi_edges, module)?)?;
    module.add_function(wrap_pyfunction!(native_version, module)?)?;
    Ok(())
}
