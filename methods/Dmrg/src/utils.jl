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
