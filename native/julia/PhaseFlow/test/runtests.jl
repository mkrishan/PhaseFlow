using Test
using PhaseFlow

@test abi_version() == 1

edges = [1 2; 2 3; 4 5]
labels = component_labels(5, edges)
@test labels[1] == labels[3]
@test labels[4] == labels[5]
@test labels[1] != labels[4]

random_edges = erdos_renyi_edges(10, 1.0, 7)
@test size(random_edges) == (45, 2)

