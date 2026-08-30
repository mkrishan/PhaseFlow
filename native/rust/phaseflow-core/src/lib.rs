//! Dependency-light native kernels for PhaseFlow.

use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    EndpointOutOfRange {
        edge_index: usize,
        endpoint: usize,
        node_count: usize,
    },
    InvalidProbability(f64),
}

impl Display for GraphError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EndpointOutOfRange {
                edge_index,
                endpoint,
                node_count,
            } => write!(
                formatter,
                "edge {edge_index} has endpoint {endpoint}, outside node count {node_count}"
            ),
            Self::InvalidProbability(value) => {
                write!(formatter, "probability must be in [0, 1], received {value}")
            }
        }
    }
}

impl Error for GraphError {}

#[derive(Debug, Clone)]
struct UnionFind {
    parent: Vec<usize>,
    weight: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            weight: vec![1; size],
        }
    }

    fn find(&mut self, mut value: usize) -> usize {
        let mut root = value;
        while root != self.parent[root] {
            root = self.parent[root];
        }
        while value != root {
            let parent = self.parent[value];
            self.parent[value] = root;
            value = parent;
        }
        root
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        if self.weight[left_root] < self.weight[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        self.weight[left_root] += self.weight[right_root];
    }
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn open_unit_interval(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        (((self.next_u64() >> 11) as f64) + 0.5) * SCALE
    }
}

fn validate_probability(probability: f64) -> Result<(), GraphError> {
    if probability.is_finite() && (0.0..=1.0).contains(&probability) {
        Ok(())
    } else {
        Err(GraphError::InvalidProbability(probability))
    }
}

pub fn component_labels(
    node_count: usize,
    edges: &[(usize, usize)],
) -> Result<Vec<usize>, GraphError> {
    let mut union_find = UnionFind::new(node_count);
    for (edge_index, &(left, right)) in edges.iter().enumerate() {
        for endpoint in [left, right] {
            if endpoint >= node_count {
                return Err(GraphError::EndpointOutOfRange {
                    edge_index,
                    endpoint,
                    node_count,
                });
            }
        }
        if left != right {
            union_find.union(left, right);
        }
    }
    Ok((0..node_count)
        .map(|index| union_find.find(index))
        .collect())
}

pub fn erdos_renyi_edges(
    node_count: usize,
    probability: f64,
    seed: u64,
) -> Result<Vec<(usize, usize)>, GraphError> {
    validate_probability(probability)?;
    if node_count < 2 || probability == 0.0 {
        return Ok(Vec::new());
    }
    if probability == 1.0 {
        let mut edges = Vec::with_capacity(node_count.saturating_mul(node_count - 1) / 2);
        for right in 1..node_count {
            for left in 0..right {
                edges.push((left, right));
            }
        }
        return Ok(edges);
    }

    let mut random = SplitMix64::new(seed);
    let log_q = (-probability).ln_1p();
    let mut edges = Vec::new();
    let mut right = 1_usize;
    let mut left = -1_i64;
    while right < node_count {
        let draw = random.open_unit_interval();
        let skip = ((1.0 - draw).ln() / log_q).floor() as i64;
        left += 1 + skip;
        while left >= right as i64 && right < node_count {
            left -= right as i64;
            right += 1;
        }
        if right < node_count {
            edges.push((left as usize, right));
        }
    }
    Ok(edges)
}

pub fn bond_percolate_edges(
    edges: &[(usize, usize)],
    retention_probability: f64,
    seed: u64,
) -> Result<Vec<(usize, usize)>, GraphError> {
    validate_probability(retention_probability)?;
    if retention_probability == 0.0 {
        return Ok(Vec::new());
    }
    if retention_probability == 1.0 {
        return Ok(edges.to_vec());
    }
    let mut random = SplitMix64::new(seed);
    Ok(edges
        .iter()
        .copied()
        .filter(|_| random.open_unit_interval() < retention_probability)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components_are_joined() {
        let labels = component_labels(5, &[(0, 1), (1, 2), (3, 4)]).unwrap();
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn complete_graph_has_expected_edge_count() {
        let edges = erdos_renyi_edges(10, 1.0, 7).unwrap();
        assert_eq!(edges.len(), 45);
    }

    #[test]
    fn sampling_is_reproducible() {
        let left = erdos_renyi_edges(100, 0.05, 11).unwrap();
        let right = erdos_renyi_edges(100, 0.05, 11).unwrap();
        assert_eq!(left, right);
    }
}
