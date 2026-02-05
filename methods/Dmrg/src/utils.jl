using ITensors
using ITensorMPS

const _PAULI_SYMBOLS = (:X, :Y, :Z)


####################################################################################
############# Utility functions for computation of expectation values ##############
####################################################################################

"""
    get_energy_variance(H::MPO, psi::MPS) -> Real

Compute the energy variance ⟨H²⟩ - ⟨H⟩² for a given Hamiltonian MPO `H` and state MPS `psi`.

# Arguments
- `H::MPO`: The Hamiltonian represented as a Matrix Product Operator.
- `psi::MPS`: A state represented as a Matrix Product State.

# Returns
- `Real`: The energy variance of the state with respect to the Hamiltonian.
"""
function get_energy_variance(H::MPO, psi::MPS)::Real
    H2 = inner(H,psi,H,psi)
    E = inner(psi',H,psi)

    # throw error if the imaginary part is significant
    if abs(imag(H2 - E^2)) > 1e-14
        error("Energy variance has significant imaginary part: $(imag(H2 - E^2))")
    end
    return real(H2-E^2)
end

# TODO: write them more general and then test them
function total_photon_number(sites::Vector{<:Index})::MPO
    os = OpSum()
    os += "N", length(sites) # assume last site is bosonic
    return MPO(os, sites)
end

function total_fermion_number(sites::Vector{<:Index})::MPO
    f_sites = sites[1:end-1] # assume last site is bosonic
    L = length(f_sites)
    os = OpSum()
    for j in 1:L
        os += "n", j
    end
    return MPO(os, sites)
end


###################################################################
############# Utility functions for MPO construction ##############
###################################################################

# @inline tells the compiler to inline the function, which can improve
# performance by avoiding function call overhead.
# It is typically used for small, frequently called functions.
@inline function _check_site_tags(sites::AbstractVector{<:Index}, tag::String)
    for s in sites
        hastags(s, tag) ||
            throw(ArgumentError("All site indices must be for $(tag)."))
    end
    return nothing
end

@inline function _check_site_tag(site::Index, tag::String)
    hastags(site, tag) ||
        throw(ArgumentError("Site index must be for $(tag)."))
    return nothing
end

"""
    fill_op!(T::ITensor, inds::Tuple, M::AbstractMatrix, prefactor::Real=1.0, local_dim::Int=2)

Helper function to fill an ITensor `T` with matrix elements from `M` at specified indices. This is
heavily used in the functions for manual MPO construction. This replaces calls like this
`W[l][1, 1, i, j] = id_f[i, j]`.


# Arguments:
- `T::ITensor`: The ITensor to be filled.
- `inds::Tuple`: A tuple of indices where the matrix elements will be placed.
- `M::AbstractMatrix`: The matrix containing the elements to be inserted into `T`.
- `prefactor::Real=1.0`: An optional prefactor to multiply each matrix element by before insertion.
- `local_dim::Int=2`: The local dimension of the site (default is 2 for spin-1/2 systems).

# Throws:
- `ArgumentError`: If the dimensions of `M` do not match `local_dim`.
"""
@inline function fill_op!(
    T::ITensor,
    inds::Tuple,
    M::AbstractMatrix,
    prefactor::Real=1.0,
    local_dim::Int=2
    )
    # check if dimensions of M match local_dim
    size(M, 1) == local_dim || size(M, 2) == local_dim ||
        throw(ArgumentError("Dimension of Matrix M is not equal to local_dim"))

    for i in 1:local_dim, j in 1:local_dim
        T[inds..., i, j] = prefactor * M[i, j]
    end
end


@inline function _check_pauli_symbol(pauli::Symbol)
    pauli ∈ _PAULI_SYMBOLS ||
        throw(ArgumentError("pauli must be :X, :Y, or :Z"))
    return nothing
end

@inline function _pauli_matrix(pauli::Symbol)
    _check_pauli_symbol(pauli)

    # '===' checks for both value and type equality
    if pauli === :X
        return ComplexF64[0 1; 1 0]
    elseif pauli === :Y
        return ComplexF64[0 -im; im 0]
    else # :Z
        return ComplexF64[1 0; 0 -1]
    end
end

nothing
