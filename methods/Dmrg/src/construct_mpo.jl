using ITensors
using ITensorMPS
using LinearAlgebra

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


"""
    build_peierls_phase(g::Real, dim_ph::Int) -> Matrix{ComplexF64}

Construct the Peierls phase matrix ``\\exp(ig(a + a^\\dagger))`` for a bosonic
site with given dimension `dim_ph`

# Arguments
- `g::Real`: Coupling strength.
- `dim_ph::Int`: Dimension of the bosonic site.

# Returns
- `Matrix{ComplexF64}`: The Peierls phase matrix of size `dim_ph x dim_ph`.
"""
function build_peierls_phase(g::Real, dim_ph::Int)::Matrix{ComplexF64}
    # zeros on diagonal
    d = zeros(Float64, dim_ph)

    # off-diagonal entries: sqrt(1), …, sqrt(dim_ph-1)
    e = sqrt.(collect(1:dim_ph-1)) # i.e. from 1 to N_ph-1

    # diagonalize a + a^\dagger =: A which is tridiagonal in the number basis
    A = SymTridiagonal(d, e)
    eigenvals, eigenvecs = eigen(A)

    # write A = V D V^\dagger with D diagonal matrix of eigenvals and V matrix of eigenvecs
    phases = exp.(1im * g .* eigenvals) # .* element-wise multiplication
    # U = V * diag(phases) * V†
    U = eigenvecs * Diagonal(phases) * eigenvecs'

    return ComplexF64.(U) # ensure complex type
end

"""
    xxz_cavity(
        sites::Vector{<:Index},
        t::Real=1.0,
        U::Real=1.0,
        g::Real=1.0,
        omega::Real=1.0
    ) -> MPO

Construct the spinless fermion XXZ Hamiltonian coupled to a single bosonic mode
(cavity) using `OpSum`:
```math
H = \\sum_{j=1}^{L-1} -t \\left( e^{i\\frac{g}{\\sqrt{L}}(a + a^\\dagger)} c^\\dagger_j c_{j+1} + e^{-i\\frac{g}{\\sqrt{L}}(a + a^\\dagger)} c^\\dagger_{j+1} c_{j} \\right) + \\sum_{j=1}^{L-1} U n_j n_{j+1} + \\Omega N_{\\text{ph}}
```
using `OpSum`.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices, with the last index being the bosonic site.
- `t::Real=1.0`: Hopping amplitude.
- `U::Real=1.0`: Interaction strength.
- `g::Real=1.0`: Coupling strength between fermions and bosonic mode.
- `omega::Real=1.0`: Frequency of the bosonic mode.

# Returns
- `MPO`: The constructed MPO representing the XXZ Hamiltonian coupled to a cavity.

# Throws
- `ArgumentError`: If fewer than two sites are provided or if the site indices
    are not valid spinless fermion and boson indices.
"""
function xxz_cavity(
    sites::Vector{<:Index},
    t::Real=1.0,
    U::Real=1.0,
    g::Real=1.0,
    omega::Real=1.0
)::MPO
    # unpack sites and check their validity
    f_sites = sites[1:end-1]
    b_site = sites[end]
    _check_spinless_fermions_sites(f_sites)
    _check_boson_sites([b_site]) # wrap in vector for checking

    L = length(f_sites) # number of fermionic sites
    L ≥ 2 || throw(ArgumentError("Need at least two lattice sites"))
    g = g / sqrt(L)
    dim_ph = dim(b_site)

    peierls_phase = build_peierls_phase(g, dim_ph)

    os = OpSum()
    b = L + 1 # boson site index
    for j in 1:(L-1)
        # dressed hopping
        os += -t, peierls_phase, b, "c†", j, "c", j+1
        os += -t, peierls_phase', b, "c†", j+1, "c", j
        # interaction term
        os += U, "n", j, "n", j+1
    end

    # add boson energy term
    os += omega, "N", b

    return MPO(os, sites)
end



"""
    xxz_cavity_manual(
        sites::Vector{<:Index},
        t::Real=1.0,
        U::Real=1.0,
        g::Real=1.0,
        omega::Real=1.0
    ) -> MPO

Construct the spinless fermion XXZ Hamiltonian with light-matter coupling by explicit MPO
construction with bond dimension 8:
```math
H = \\sum_{j=1}^{L-1} -t\\left( e^{i\\frac{g}{\\sqrt{L}}(a + a^\\dagger)} c^\\dagger_j c_{j+1} + e^{-i\\frac{g}{\\sqrt{L}}(a + a^\\dagger)} c^\\dagger_{j+1} c_{j} \\right) + \\sum_{j=1}^{L-1} U n_j n_{j+1} + \\Omega N_{\\text{ph}} \\, .
```

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices, with the last index being the bosonic site.
- `t::Real=1.0`: Hopping amplitude.
- `U::Real=1.0`: Interaction strength.
- `g::Real=1.0`: Coupling strength between fermions and bosonic mode.
- `omega::Real=1.0`: Frequency of the bosonic mode.

# Returns
- `MPO`: The constructed MPO representing the XXZ Hamiltonian coupled to a cavity.

# Throws
- `ArgumentError`: If fewer than two sites are provided or if the site indices
    are not valid spinless fermion and boson indices.

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
H = \\sum_{j=1}^{L-1} -t(a^\\dagger_j a_{j+1} + a_j a^\\dagger_{j+1}) + \\sum_{j=1}^{L-1} U n_j n_{j+1} \\, .
```
- For more details, see e.g. [this ITensor tutorial](https://itensor.org/docs.cgi?page=tutorials/fermions),
  [this paper](https://arxiv.org/pdf/1611.02498), or [this topic on itensor.discourse](https://itensor.discourse.group/t/manual-construction-of-nearest-neighbor-hopping-of-spinless-fermions-on-a-1d-chain/2567/5)
"""
function xxz_cavity_manual(
    sites::Vector{<:Index},
    t::Real=1.0,
    U::Real=1.0,
    g::Real=1.0,
    omega::Real=1.0
)::MPO
    # unpack sites and check their validity
    f_sites = sites[1:end-1]
    b_site = sites[end]
    _check_spinless_fermions_sites(f_sites)
    _check_boson_sites([b_site]) # wrap in vector for checking

    L_mpo = length(sites) # number of sites in MPO (fermionic + bosonic)
    L = length(f_sites) # number of fermionic sites
    L ≥ 2 || throw(ArgumentError("Need at least two lattice sites"))

    # Local operators on fermionic sites
    id_f = ComplexF64[1 0; 0 1] # complex since we have complex phases later
    n_f = ComplexF64[0 0; 0 1]
    a_dag = ComplexF64[0 0; 1 0] # 'bosonic' creation operator from JWT
    a = ComplexF64[0 1; 0 0] # 'bosonic' annihilation operator from JWT

    # Local operators on bosonic site
    dim_ph = dim(b_site)
    g = g / sqrt(L)
    id_ph = Matrix{ComplexF64}(I, dim_ph, dim_ph)
    n_ph = Diagonal(ComplexF64.(0:dim_ph-1)) # number operator for bosons
    peierls_phase = build_peierls_phase(g, dim_ph)
    peierls_phase_conj = peierls_phase' # conjugate transpose

    # Virtual bond indices (dimension 8), excluding boundaries
    links = [Index(8, "link,l=$i") for i in 1:L_mpo]
    W = Vector{ITensor}(undef, L_mpo) # undef: do not initialize yet

    # First site (fermionic)
    W[1] = ITensor(links[1], prime(sites[1]), sites[1])
    for i in 1:2, j in 1:2
        W[1][1, i, j] = id_f[i, j]
        W[1][2, i, j] = U * n_f[i, j]
        W[1][3, i, j] = a_dag[i, j]
        W[1][4, i, j] = a[i, j]
    end

    # Bulk sites (fermionic)
    for l in 2:(L_mpo-1)
        W[l] = ITensor(links[l-1], links[l], prime(sites[l]), sites[l])
        for i in 1:2, j in 1:2
            W[l][1, 1, i, j] = id_f[i, j]
            W[l][1, 2, i, j] = U * n_f[i, j]
            W[l][1, 3, i, j] = a_dag[i, j]
            W[l][1, 4, i, j] = a[i, j]

            W[l][2, 5, i, j] = n_f[i, j]
            W[l][3, 6, i, j] = a[i, j]
            W[l][4, 7, i, j] = a_dag[i, j]

            W[l][5, 5, i, j] = id_f[i, j]
            W[l][6, 6, i, j] = id_f[i, j]
            W[l][7, 7, i, j] = id_f[i, j]
        end
    end

    # last site (bosonic)
    W[L_mpo] = ITensor(links[L_mpo-1], prime(sites[L_mpo]), sites[L_mpo])
    for i in 1:dim_ph, j in 1:dim_ph
        W[L_mpo][1, i, j] = omega * n_ph[i, j]
        W[L_mpo][5, i, j] = id_ph[i, j]
        W[L_mpo][6, i, j] = -t * peierls_phase[i, j]
        W[L_mpo][7, i, j] = -t * peierls_phase_conj[i, j]
    end
    return MPO(W)
end


"""
    xxz(sites::Vector{<:Index}, t::Real=1.0, U::Real=1.0) -> MPO

Construct the spinless fermion XXZ Hamiltonian
```math
H = \\sum_{j=1}^{L-1} -t(c^\\dagger_j c_{j+1} + c_j c^\\dagger_{j+1}) + \\sum_{j=1}^{L-1} U n_j n_{j+1}
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
function xxz(sites::Vector{<:Index}, t::Real=1.0, U::Real=1.0)::MPO
    _check_spinless_fermions_sites(sites)

    L = length(sites)
    L ≥ 2 || throw(ArgumentError("Need at least two lattice sites"))

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
H = \\sum_{j=1}^{L-1} -t(c^\\dagger_j c_{j+1} + c_j c^\\dagger_{j+1}) + \\sum_{j=1}^{L-1} U n_j n_{j+1} \\, .
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
H = \\sum_{j=1}^{L-1} -t(a^\\dagger_j a_{j+1} + a_j a^\\dagger_{j+1}) + \\sum_{j=1}^{L-1} U n_j n_{j+1} \\, .
```
- For more details, see e.g. [this ITensor tutorial](https://itensor.org/docs.cgi?page=tutorials/fermions),
  [this paper](https://arxiv.org/pdf/1611.02498), or [this topic on itensor.discourse](https://itensor.discourse.group/t/manual-construction-of-nearest-neighbor-hopping-of-spinless-fermions-on-a-1d-chain/2567/5)
"""
function xxz_manual(sites::Vector{<:Index}, t::Real=1.0, U::Real=1.0)::MPO
    _check_spinless_fermions_sites(sites)

    L = length(sites)
    L ≥ 2 || throw(ArgumentError("Need at least two lattice sites"))

    # Local operators
    id = [1.0 0.0; 0.0 1.0]
    n = [0.0 0.0; 0.0 1.0]
    a_dag = [0.0 0.0; 1.0 0.0] # 'bosonic' creation operator
    a = [0.0 1.0; 0.0 0.0] # 'bosonic' annihilation operator

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
function heisenberg(sites::Vector{<:Index}, J::Real=1.0, Jz::Real=1.0)::MPO
    _check_spinhalf_sites(sites)

    L = length(sites)
    L ≥ 2 || throw(ArgumentError("Need at least two lattice sites"))

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
function heisenberg_manual(sites::Vector{<:Index}, J::Real=1.0, Jz::Real=1.0)::MPO
    _check_spinhalf_sites(sites)

    L = length(sites)
    L ≥ 2 || throw(ArgumentError("Need at least two lattice sites"))

    # Local operators (ensure Float64 type for division)
    id = [1.0 0.0; 0.0 1.0]
    S_z = 0.5 .* [1.0 0.0; 0.0 -1.0]
    S_plus = [0.0 1.0; 0.0 0.0]
    S_minus = [0.0 0.0; 1.0 0.0]

    # Virtual bond indices (dimension 5), excluding boundaries
    links = [Index(5, "link,l=$i") for i in 1:(L - 1)]
    W = Vector{ITensor}(undef, L) # undef: do not initialize yet

    # First site
    W[1] = ITensor(links[1], prime(sites[1]), sites[1])
    for i in 1:2, j in 1:2
        W[1][2, i, j] = (J/2) * S_minus[i, j]
        W[1][3, i, j] = (J/2) * S_plus[i, j]
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

            W[n][5, 2, i, j] = (J/2) * S_minus[i, j]
            W[n][5, 3, i, j] = (J/2) * S_plus[i, j]
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
function pauli_sum(sites::Vector{<:Index}, a::Real=1.0, pauli::Symbol=:X)::MPO
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
function pauli_sum_manual(sites::Vector{<:Index}, a::Real=1.0, pauli::Symbol=:X,
)::MPO
    _check_spinhalf_sites(sites)

    σ = _pauli_matrix(pauli)
    id = ComplexF64[1 0; 0 1] # since we have complex numbers in pauli matrices

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
