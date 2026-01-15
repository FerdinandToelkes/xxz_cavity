using ITensors, ITensorMPS

const _PAULI_SYMBOLS = (:X, :Y, :Z)

# TODO: add comment on conventions used to construct MPOs and reference papers 




# the following functions are not used, but were merely used to 
# test how ITensors works


# helper functions -> TODO: to utils.jl later?

@inline function _check_spinhalf_sites(sites::AbstractVector{<:Index})
    for s in sites
        hastags(s, "S=1/2") ||
            throw(ArgumentError("All site indices must be for spin-1/2 particles"))
    end
    return nothing
end

@inline function _check_spinless_fermions_sites(sites::AbstractVector{<:Index})
    for s in sites
        hastags(s, "Fermion") ||
            throw(ArgumentError("All site indices must be for spinless fermions"))
    end
    return nothing
end

# @inline tells the compiler to inline the function, which can improve 
# performance by avoiding function call overhead.
# It is typically used for small, frequently called functions.
@inline function _pauli_matrix(pauli::Symbol)
    pauli ∈ _PAULI_SYMBOLS ||
        throw(ArgumentError("pauli must be :X, :Y, or :Z"))

    # '===' checks for both value and type equality
    if pauli === :X
        return ComplexF64[0 1; 1 0]
    elseif pauli === :Y
        return ComplexF64[0 -im; im 0]
    else # :Z
        return ComplexF64[1 0; 0 -1]
    end
end



function xxz_cavity_mpo()
    # Placeholder for future implementation
    throw(ErrorException("xxz_cavity_mpo not yet implemented"))
end

function xxz_cavity_manual_mpo()
    # Placeholder for future implementation
    throw(ErrorException("xxz_cavity_manual_mpo not yet implemented"))
end

"""
    heisenberg_mpo(sites::Vector{<:Index}, J::Real, Jz::Real) -> MPO

Construct the spin-1/2 Heisenberg (XXZ) Hamiltonian

    H = ∑_{j=1}^{L-1} [ J/2 (S⁺ⱼ S⁻ⱼ₊₁ + S⁻ⱼ S⁺ⱼ₊₁) + Jz Sᶻⱼ Sᶻⱼ₊₁ ]

using `OpSum`.

Throws an `ArgumentError` if fewer than two sites are provided or if
`sites` are not spin-1/2 indices.
"""
function heisenberg_mpo(
    sites::Vector{<:Index},
    J::Real,
    Jz::Real,
)::MPO
    _check_spinhalf_sites(sites)

    L = length(sites)
    L ≥ 2 || throw(ArgumentError("Need at least two sites"))

    os = OpSum()
    for j in 1:(L-1)
        os += Jz, "Sz", j, "Sz", j+1
        os += J/2, "S+", j, "S-", j+1
        os += J/2, "S-", j, "S+", j+1
    end

    return MPO(os, sites)
end

"""
    heisenberg_manual_mpo(sites::Vector{<:Index}, J::Real, Jz::Real) -> MPO

Construct the spin-1/2 Heisenberg (XXZ) Hamiltonian by explicit MPO
construction with bond dimension 5.

Throws an `ArgumentError` if fewer than two sites are provided or if
`sites` are not spin-1/2 indices.
"""
function heisenberg_manual_mpo(sites::Vector{<:Index}, J::Real, Jz::Real)::MPO
    _check_spinhalf_sites(sites)

    L = length(sites)
    L ≥ 2 || throw(ArgumentError("Need at least two sites")) 

    # Local operators
    id = ComplexF64[1 0; 0 1]
    S_z = 1/2 * ComplexF64[1 0; 0 -1]
    S_plus = ComplexF64[0 1; 0 0]
    S_minus = ComplexF64[0 0; 1 0]

    # Virtual bond indices (dimension 5), excluding boundaries
    links = [Index(5, "link,l=$i") for i in 1:(L-1)]
    W = Vector{ITensor}(undef, L) # undef: do not initialize yet

    # First site
    W[1] = ITensor(links[1], prime(sites[1]), sites[1])
    for i in 1:2, j in 1:2
        W[1][2, i, j] = J/2 * S_minus[i, j]
        W[1][3, i, j] = J/2 * S_plus[i, j]
        W[1][4, i, j] = Jz * S_z[i, j]
        W[1][5, i, j] = id[i, j]
    end

    # Bulk sites
    for n in 2:(L-1)
        W[n] = ITensor(links[n-1], links[n], prime(sites[n]), sites[n])
        for i in 1:2, j in 1:2
            W[n][1, 1, i, j] = id[i, j]
            W[n][2, 1, i, j] = S_plus[i, j]
            W[n][3, 1, i, j] = S_minus[i, j]
            W[n][4, 1, i, j] = S_z[i, j]

            W[n][5, 2, i, j] = J/2 * S_minus[i, j]
            W[n][5, 3, i, j] = J/2 * S_plus[i, j]
            W[n][5, 4, i, j] = Jz * S_z[i, j]
            W[n][5, 5, i, j] = id[i, j]
        end
    end

    # Last site
    W[L] = ITensor(links[L-1], prime(sites[L]), sites[L])
    for i in 1:2, j in 1:2
        W[L][1, i, j] = id[i, j]
        W[L][2, i, j] = S_plus[i, j]
        W[L][3, i, j] = S_minus[i, j]
        W[L][4, i, j] = S_z[i, j]
    end

    return MPO(W)
end

let 
    L = 3
    sites = siteinds("S=1/2", L)
    h = heisenberg_mpo(sites, 1.0, 1.0)
    # @show h
    # for i in 1:L
    #     W = h[i]
    #     @show W
    # end

    h_man = heisenberg_manual_mpo(sites, 1.0, 1.0)

    random_state = random_mps(sites; linkdims=10)
    @show inner(random_state', h, random_state)
    @show inner(random_state', h_man, random_state)
    
end


"""
    pauli_sum_mpo(sites::Vector{<:Index}, a::Real; pauli::Symbol = :X) -> MPO

Construct an MPO representing

    H = a * ∑_{j=1}^L σ_j

where σ_j is the Pauli matrix specified by `pauli` acting on sites with spin 1/2 particles.

Throw an `ArgumentError` if `pauli ∉ (:X, :Y, :Z)` or if `sites` are not spin-1/2 indices.
"""
function pauli_sum_mpo(
    sites::Vector{<:Index},
    a::Real;
    pauli::Symbol = :X,
)::MPO
    _check_spinhalf_sites(sites)
    pauli ∈ _PAULI_SYMBOLS ||
        throw(ArgumentError("pauli must be :X, :Y, or :Z"))
    
    L = length(sites)
    L ≥ 1 || throw(ArgumentError("Need at least one site"))

    os = OpSum()
    for j in 1:L
        os += a, String(pauli), j
    end

    return MPO(os, sites)
end

"""
    pauli_sum_manual_mpo(sites::Vector{<:Index}, a::Real; pauli::Symbol = :X) -> MPO

Construct an MPO representing

    H = a * ∑_{j=1}^L σ_j

where σ_j is the Pauli matrix specified by `pauli` acting on sites with spin 1/2 particles.

Throw an `ArgumentError` if `pauli ∉ (:X, :Y, :Z)` or if `sites` are not spin-1/2 indices.
"""
function pauli_sum_manual_mpo(
    sites::Vector{<:Index},
    a::Real;
    pauli::Symbol = :X,
)::MPO
    _check_spinhalf_sites(sites)

    σ = _pauli_matrix(pauli)
    id = ComplexF64[1 0; 0 1]

    L = length(sites)
    L ≥ 1 || throw(ArgumentError("Need at least one site"))

    # Virtual bond indices (dimension 2), excluding boundaries
    links = [Index(2, "link,l=$i") for i in 1:(L-1)]

    # allocate vector of length L to hold site tensors
    W = Vector{ITensor}(undef, L) # undef: do not initialize yet

    # First site
    W[1] = ITensor(links[1], prime(sites[1]), sites[1])
    for i in 1:2, j in 1:2
        W[1][1, i, j] = a * σ[i, j]
        W[1][2, i, j] = id[i, j]
    end

    # Bulk sites
    for n in 2:(L - 1)
        W[n] = ITensor(links[n - 1], links[n], prime(sites[n]), sites[n])
        for i in 1:2, j in 1:2
            W[n][1, 1, i, j] = id[i, j]
            W[n][2, 1, i, j] = a * σ[i, j]
            W[n][2, 2, i, j] = id[i, j]
        end
    end

    # Last site
    W[L] = ITensor(links[L - 1], prime(sites[L]), sites[L])
    for i in 1:2, j in 1:2
        W[L][1, i, j] = id[i, j]
        W[L][2, i, j] = a * σ[i, j]
    end

    return MPO(W)
end
