module PhaseFlow

export abi_version, component_labels, erdos_renyi_edges

const libphaseflow = get(ENV, "PHASEFLOW_LIBRARY", "libphaseflow_ffi")

const PHASEFLOW_OK = Cint(0)

struct EdgeBuffer
    data::Ptr{Csize_t}
    edge_count::Csize_t
    capacity_values::Csize_t
end

function abi_version()::UInt32
    ccall((:phaseflow_abi_version, libphaseflow), UInt32, ())
end

function component_labels(node_count::Integer, edges::AbstractMatrix{<:Integer})
    node_count < 0 && throw(ArgumentError("node_count must be non-negative"))
    size(edges, 2) == 2 || throw(ArgumentError("edges must have two columns"))
    any(edges .< 1) && throw(ArgumentError("Julia edge endpoints must be one-based"))
    native_edges = Vector{Csize_t}(undef, 2 * size(edges, 1))
    for row in axes(edges, 1)
        native_edges[2 * row - 1] = Csize_t(edges[row, 1] - 1)
        native_edges[2 * row] = Csize_t(edges[row, 2] - 1)
    end
    labels = Vector{Csize_t}(undef, node_count)
    status = ccall(
        (:phaseflow_component_labels, libphaseflow),
        Cint,
        (Csize_t, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
        node_count,
        native_edges,
        size(edges, 1),
        labels,
    )
    status == PHASEFLOW_OK || error("PhaseFlow component computation failed with status $status")
    return Int.(labels) .+ 1
end

function erdos_renyi_edges(node_count::Integer, probability::Real, seed::Integer=0)
    node_count < 0 && throw(ArgumentError("node_count must be non-negative"))
    buffer = Ref(EdgeBuffer(C_NULL, 0, 0))
    status = ccall(
        (:phaseflow_erdos_renyi_edges, libphaseflow),
        Cint,
        (Csize_t, Cdouble, UInt64, Ref{EdgeBuffer}),
        node_count,
        probability,
        seed,
        buffer,
    )
    status == PHASEFLOW_OK || error("PhaseFlow graph generation failed with status $status")
    try
        edge_count = Int(buffer[].edge_count)
        edge_count == 0 && return Matrix{Int}(undef, 0, 2)
        values = unsafe_wrap(Vector{Csize_t}, buffer[].data, 2 * edge_count; own=false)
        return permutedims(reshape(Int.(copy(values)) .+ 1, 2, edge_count))
    finally
        ccall((:phaseflow_free_edge_buffer, libphaseflow), Cvoid, (Ref{EdgeBuffer},), buffer)
    end
end

end
