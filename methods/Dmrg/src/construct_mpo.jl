using ITensors
using ITensorMPS

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

@inline function _check_boson_sites(sites::AbstractVector{<:Index})
    for s in sites
        hastags(s, "Boson") ||
            throw(ArgumentError("All site indices must be for bosons"))
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

# exp() can't be done with OpSum, so we need to manually construct the MPO
function xxz_cavity_small_coupling(
    f_sites::Vector{<:Index},
    b_site::Index,
    t::Real=1.0,
    U::Real=1.0,
    g::Real=1.0,
    omega::Real=1.0
)::MPO
    _check_spinless_fermions_sites(f_sites)
    _check_boson_sites([b_site]) # wrap in vector for checking
    sites = vcat(f_sites, [b_site])

    L = length(f_sites)
    L ≥ 2 || throw(ArgumentError("Need at least two sites"))
    g = g / L

    os = OpSum()
    b = L + 1 # boson site index

    # construct fermionic part with Peirls phase
    for j in 1:(L-1)
        # zeroth order term
        os += -t, "c†", j, "c", j + 1
        os += +t, "c†", j + 1, "c", j # sign change due to anticommutation
        # first order term
        os += -t * (+im * g), "c†", j, "c", j + 1, "a†", b
        os += -t * (+im * g), "c†", j, "c", j + 1, "a", b
        os += +t * (-im * g), "c†", j + 1, "c", j, "a†", b
        os += +t * (-im * g), "c†", j + 1, "c", j, "a", b
        # interaction term
        os += U, "n", j, "n", j + 1
    end

    # add boson energy term
    os += omega, "n", L + 1
    return MPO(os, sites)
end


function xxz_cavity_manual(
    sites::Vector{<:Index}
)
    _check_spinless_fermions_sites(sites)
    # Placeholder for future implementation
    throw(ErrorException("xxz_cavity_manual not yet implemented"))
end

"""
    xxz(sites::Vector{<:Index}, t::Real=1.0, U::Real=1.0) -> MPO

Construct the spinless fermion XXZ Hamiltonian
```math
H = \\sum_{j=1}^{L-1} -t(c^\\dagger_j c_{j+1} + c_j c^\\dagger_{j+1}) + U n_j n_{j+1}
```
using `OpSum`. We denote the Hamiltonian acting on spinless fermions as the Heisenberg
Hamiltonian and the one acting on spins as the XXZ Hamiltonian.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices for spinless fermions.
- `t::Real=1.0`: Hopping amplitude.
- `U::Real=1.0`: Interaction strength.

# Returns
- `MPO`: The constructed MPO representing the spinless fermion XXZ Hamiltonian.

# Throws
- `ArgumentError`: If fewer than two sites are provided or if `sites` are not spinless
    fermion indices.
"""
function xxz(
    sites::Vector{<:Index},
    t::Real=1.0,
    U::Real=1.0
)::MPO
    _check_spinless_fermions_sites(sites)

    L = length(sites)
    L ≥ 2 || throw(ArgumentError("Need at least two sites"))

    os = OpSum()
    for j in 1:(L-1)
        os += -t, "c†", j, "c", j + 1
        os += -t, "c†", j + 1, "c", j # or +t, "c", j, "c†", j + 1 due to anticommutation
        os += U, "n", j, "n", j + 1
    end

    return MPO(os, sites)
end

"""
    xxz_manual(sites::Vector{<:Index}, t::Real=1.0, U::Real=1.0) -> MPO

Construct the spinless fermion XXZ Hamiltonian by explicit MPO construction
with bond dimension 5:
```math
H = \\sum_{j=1}^{L-1} -t(c^\\dagger_j c_{j+1} + c_j c^\\dagger_{j+1}) + U n_j n_{j+1}
```
We denote the Hamiltonian acting on spinless fermions as the Heisenberg
Hamiltonian and the one acting on spins as the XXZ Hamiltonian.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices for spinless fermions.
- `t::Real=1.0`: Hopping amplitude.
- `U::Real=1.0`: Interaction strength.

# Returns
- `MPO`: The constructed MPO representing the spinless fermion XXZ Hamiltonian.

# Throws
- `ArgumentError`: If fewer than two sites are provided or if `sites` are not
    spinless fermion indices.

# Notes
- One has to use the Jordan-Wigner transformation (JWT) to map the fermionic operators
  to bosonic operators when constructing the MPO manually such that the anticommutation
  relations are correctly accounted for.
- The JWT for spinless fermions is given by:
```math
\\begin{align*}
c_j &= Z_1 Z_2 ... Z_{j-1} * a_j \\\\
c^\\dagger_j &= Z_1 Z_2 ... Z_{j-1} * a^\\dagger_j \\, , \\quad \\text{where} \\\\
Z_k &= e^{i \\pi n_k} = I - 2 n_k \\, .
\\end{align*}
```
- The bosonic operators ``a^\\dagger_j, a_j`` commute whereas the fermionic operators anticommute.
- In this case the JWT strings cancel out for nearest-neighbor hopping and the transformed
  Hamiltonian in terms of bosonic operators then reads
```math
H = \\sum_{j=1}^{L-1} -t(a^\\dagger_j a_{j+1} + a_j a^\\dagger_{j+1}) + U n_j n_{j+1} \\, .
```
- For more details, see e.g. [this ITensor tutorial](https://itensor.org/docs.cgi?page=tutorials/fermions),
  [this paper](https://arxiv.org/pdf/1611.02498), or [this topic on itensor.discourse](https://itensor.discourse.group/t/manual-construction-of-nearest-neighbor-hopping-of-spinless-fermions-on-a-1d-chain/2567/5)
"""
function xxz_manual(
    sites::Vector{<:Index},
    t::Real=1.0,
    U::Real=1.0
)::MPO
    _check_spinless_fermions_sites(sites)

    L = length(sites)
    L ≥ 2 || throw(ArgumentError("Need at least two sites"))

    # Local operators
    id = ComplexF64[1 0; 0 1]
    n = ComplexF64[0 0; 0 1]
    a_dag = ComplexF64[0 0; 1 0] # 'bosonic' creation operator
    a = ComplexF64[0 1; 0 0] # 'bosonic' annihilation operator

    # Virtual bond indices (dimension 5), excluding boundaries
    links = [Index(5, "link,l=$i") for i in 1:(L - 1)]
    W = Vector{ITensor}(undef, L) # undef: do not initialize yet

    # First site
    W[1] = ITensor(links[1], prime(sites[1]), sites[1])
    for i in 1:2, j in 1:2
        W[1][1, i, j] = id[i, j]
        W[1][2, i, j] = -t * a_dag[i, j]
        W[1][3, i, j] = -t * a[i, j]
        W[1][4, i, j] = U * n[i, j]
    end

    # Bulk sites
    for l in 2:(L-1) # n is already used
        W[l] = ITensor(links[l-1], links[l], prime(sites[l]), sites[l])
        for i in 1:2, j in 1:2
            W[l][1, 1, i, j] = id[i, j]
            W[l][1, 2, i, j] = -t * a_dag[i, j]
            W[l][1, 3, i, j] = -t * a[i, j]
            W[l][1, 4, i, j] = U * n[i, j]

            W[l][2, 5, i, j] = a[i, j]
            W[l][3, 5, i, j] = a_dag[i, j]
            W[l][4, 5, i, j] = n[i, j]
            W[l][5, 5, i, j] = id[i, j]
        end
    end

    # last site
    W[L] = ITensor(links[L-1], prime(sites[L]), sites[L])
    for i in 1:2, j in 1:2
        W[L][2, i, j] = a[i, j]
        W[L][3, i, j] = a_dag[i, j]
        W[L][4, i, j] = n[i, j]
        W[L][5, i, j] = id[i, j]
    end
    return MPO(W)
end



"""
    heisenberg(sites::Vector{<:Index}, J::Real=1.0, Jz::Real=1.0) -> MPO

Construct the spin-1/2 Heisenberg (XXZ) Hamiltonian
```math
H = \\sum_{j=1}^{L-1} \\frac{J}{2} (S^+_j S^-_{j+1} + S^-_j S^+_{j+1}) + J_z S^z_j S^z_{j+1}
```
using `OpSum`. We denote the Hamiltonian acting on spins as the Heisenberg Hamiltonian
and the one acting on spinless fermions as the XXZ Hamiltonian.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices for spin-1/2 particles.
- `J::Real=1.0`: Exchange coupling constant in the XY plane.
- `Jz::Real=1.0`: Exchange coupling constant in the Z direction.

# Returns
- `MPO`: The constructed Heisenberg Hamiltonian as an MPO.

# Throws
- `ArgumentError`: If fewer than two sites are provided or if `sites`
    are not spin-1/2 indices.
"""
function heisenberg(
    sites::Vector{<:Index},
    J::Real=1.0,
    Jz::Real=1.0,
)::MPO
    _check_spinhalf_sites(sites)

    L = length(sites)
    L ≥ 2 || throw(ArgumentError("Need at least two sites"))

    os = OpSum()
    for j in 1:(L-1)
        os += Jz, "Sz", j, "Sz", j + 1
        os += J/2, "S+", j, "S-", j + 1
        os += J/2, "S-", j, "S+", j + 1
    end

    return MPO(os, sites)
end

"""
    heisenberg_manual(sites::Vector{<:Index}, J::Real=1.0, Jz::Real=1.0) -> MPO

Construct the spin-1/2 Heisenberg (XXZ) Hamiltonian
```math
H = \\sum_{j=1}^{L-1} \\frac{J}{2} (S^+_j S^-_{j+1} + S^-_j S^+_{j+1}) + J_z S^z_j S^z_{j+1}
```
by explicit MPO construction with bond dimension 5. We denote the Hamiltonian
acting on spins as the Heisenberg Hamiltonian and the one acting on spinless
fermions as the XXZ Hamiltonian.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices for spin-1/2 particles.
- `J::Real=1.0`: Exchange coupling constant in the XY plane.
- `Jz::Real=1.0`: Exchange coupling constant in the Z direction.

# Returns
- `MPO`: The constructed Heisenberg Hamiltonian as an MPO.

# Throws
- `ArgumentError`: If fewer than two sites are provided or if `sites`
    are not spin-1/2 indices.
"""
function heisenberg_manual(
    sites::Vector{<:Index},
    J::Real=1.0,
    Jz::Real=1.0
)::MPO
    _check_spinhalf_sites(sites)

    L = length(sites)
    L ≥ 2 || throw(ArgumentError("Need at least two sites"))

    # Local operators
    id = ComplexF64[1 0; 0 1]
    S_z = 1/2 * ComplexF64[1 0; 0 -1]
    S_plus = ComplexF64[0 1; 0 0]
    S_minus = ComplexF64[0 0; 1 0]

    # Virtual bond indices (dimension 5), excluding boundaries
    links = [Index(5, "link,l=$i") for i in 1:(L - 1)]
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


"""
    pauli_sum(sites::Vector{<:Index}, a::Real=1.0, pauli::Symbol=:X) -> MPO

Construct an MPO representing
```math
H = a * \\sum_{j=1}^L \\sigma_j \\, ,
```
where σ_j is the Pauli matrix specified by `pauli` acting on sites with spin 1/2 particles.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices for spin-1/2 particles.
- `a::Real=1.0`: Coefficient multiplying the sum.
- `pauli::Symbol=:X`: Symbol specifying which Pauli matrix to use (:X, :Y, or :Z).

# Returns
- `MPO`: The constructed MPO representing the Pauli sum.

# Throws
- `ArgumentError`: If `pauli ∉ (:X, :Y, :Z)` or if `sites` are not spin-1/2 indices.
"""
function pauli_sum(
    sites::Vector{<:Index},
    a::Real=1.0,
    pauli::Symbol=:X,
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
    pauli_sum_manual(sites::Vector{<:Index}, a::Real=1.0, pauli::Symbol=:X) -> MPO

Construct an MPO representing
```math
H = a * \\sum_{j=1}^L \\sigma_j \\, ,
```
where σ_j is the Pauli matrix specified by `pauli` acting on sites with spin 1/2 particles.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices for spin-1/2 particles.
- `a::Real=1.0`: Coefficient multiplying the sum.
- `pauli::Symbol=:X`: Symbol specifying which Pauli matrix to use (:X, :Y, or :Z).

# Returns
- `MPO`: The constructed MPO representing the Pauli sum.

# Throws
- `ArgumentError`: If `pauli ∉ (:X, :Y, :Z)` or if `sites` are not spin-1/2 indices.
"""
function pauli_sum_manual(
    sites::Vector{<:Index},
    a::Real=1.0,
    pauli::Symbol=:X,
)::MPO
    _check_spinhalf_sites(sites)

    σ = _pauli_matrix(pauli)
    id = ComplexF64[1 0; 0 1]

    L = length(sites)
    L ≥ 1 || throw(ArgumentError("Need at least one site"))

    # Virtual bond indices (dimension 2), excluding boundaries
    links = [Index(2, "link,l=$i") for i in 1:(L - 1)]

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

nothing
