//! C-compatible interface for Julia and other native-language clients.

use phaseflow_core::{component_labels, erdos_renyi_edges};
use std::slice;

pub const PHASEFLOW_OK: i32 = 0;
pub const PHASEFLOW_INVALID_ARGUMENT: i32 = 1;
pub const PHASEFLOW_COMPUTATION_ERROR: i32 = 2;

#[repr(C)]
pub struct PhaseFlowEdgeBuffer {
    pub data: *mut usize,
    pub edge_count: usize,
    pub capacity_values: usize,
}

#[no_mangle]
pub extern "C" fn phaseflow_abi_version() -> u32 {
    1
}

#[no_mangle]
pub unsafe extern "C" fn phaseflow_component_labels(
    node_count: usize,
    edge_values: *const usize,
    edge_count: usize,
    output_labels: *mut usize,
) -> i32 {
    if (edge_count > 0 && edge_values.is_null()) || (node_count > 0 && output_labels.is_null()) {
        return PHASEFLOW_INVALID_ARGUMENT;
    }
    let flat_edges = if edge_count == 0 {
        &[][..]
    } else {
        slice::from_raw_parts(edge_values, edge_count.saturating_mul(2))
    };
    let edges: Vec<(usize, usize)> = flat_edges
        .chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .collect();
    let labels = match component_labels(node_count, &edges) {
        Ok(labels) => labels,
        Err(_) => return PHASEFLOW_COMPUTATION_ERROR,
    };
    if node_count > 0 {
        let output = slice::from_raw_parts_mut(output_labels, node_count);
        output.copy_from_slice(&labels);
    }
    PHASEFLOW_OK
}

#[no_mangle]
pub unsafe extern "C" fn phaseflow_erdos_renyi_edges(
    node_count: usize,
    probability: f64,
    seed: u64,
    output: *mut PhaseFlowEdgeBuffer,
) -> i32 {
    if output.is_null() {
        return PHASEFLOW_INVALID_ARGUMENT;
    }
    let edges = match erdos_renyi_edges(node_count, probability, seed) {
        Ok(edges) => edges,
        Err(_) => return PHASEFLOW_COMPUTATION_ERROR,
    };
    let edge_count = edges.len();
    let mut values = Vec::with_capacity(edge_count.saturating_mul(2));
    for (left, right) in edges {
        values.push(left);
        values.push(right);
    }
    let buffer = PhaseFlowEdgeBuffer {
        data: values.as_mut_ptr(),
        edge_count,
        capacity_values: values.capacity(),
    };
    std::mem::forget(values);
    output.write(buffer);
    PHASEFLOW_OK
}

#[no_mangle]
pub unsafe extern "C" fn phaseflow_free_edge_buffer(buffer: *mut PhaseFlowEdgeBuffer) {
    if buffer.is_null() {
        return;
    }
    let buffer = &mut *buffer;
    if !buffer.data.is_null() {
        let value_count = buffer.edge_count.saturating_mul(2);
        drop(Vec::from_raw_parts(
            buffer.data,
            value_count,
            buffer.capacity_values,
        ));
    }
    buffer.data = std::ptr::null_mut();
    buffer.edge_count = 0;
    buffer.capacity_values = 0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_labels_cross_the_c_boundary() {
        let edge_values = [0_usize, 1, 1, 2, 3, 4];
        let mut labels = [0_usize; 5];
        let status = unsafe {
            phaseflow_component_labels(
                5,
                edge_values.as_ptr(),
                edge_values.len() / 2,
                labels.as_mut_ptr(),
            )
        };
        assert_eq!(status, PHASEFLOW_OK);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn allocated_edge_buffers_are_released() {
        let mut buffer = PhaseFlowEdgeBuffer {
            data: std::ptr::null_mut(),
            edge_count: 0,
            capacity_values: 0,
        };
        let status = unsafe { phaseflow_erdos_renyi_edges(10, 1.0, 7, &mut buffer) };
        assert_eq!(status, PHASEFLOW_OK);
        assert_eq!(buffer.edge_count, 45);
        unsafe { phaseflow_free_edge_buffer(&mut buffer) };
        assert!(buffer.data.is_null());
        assert_eq!(buffer.edge_count, 0);
    }
}
