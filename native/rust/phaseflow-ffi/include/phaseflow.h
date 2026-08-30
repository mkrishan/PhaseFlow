#ifndef PHASEFLOW_H
#define PHASEFLOW_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    PHASEFLOW_OK = 0,
    PHASEFLOW_INVALID_ARGUMENT = 1,
    PHASEFLOW_COMPUTATION_ERROR = 2
};

typedef struct PhaseFlowEdgeBuffer {
    size_t *data;
    size_t edge_count;
    size_t capacity_values;
} PhaseFlowEdgeBuffer;

uint32_t phaseflow_abi_version(void);

int32_t phaseflow_component_labels(
    size_t node_count,
    const size_t *edge_values,
    size_t edge_count,
    size_t *output_labels
);

int32_t phaseflow_erdos_renyi_edges(
    size_t node_count,
    double probability,
    uint64_t seed,
    PhaseFlowEdgeBuffer *output
);

void phaseflow_free_edge_buffer(PhaseFlowEdgeBuffer *buffer);

#ifdef __cplusplus
}
#endif

#endif

