using ITensors
using ITensorMPS
using LinearAlgebra


"""
    build_peierls_phase(g::Real, dim_ph::Int) -> Matrix{ComplexF64}

Construct the Peierls phase matrix ``\\exp(ig(a + a^\\dagger))`` for a photon
site with given dimension `dim_ph`

# Arguments
- `g::Real`: Coupling strength.
- `dim_ph::Int`: Dimension of the photon site.

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
        sites::Vector{<:Index};
        t::Real=1.0,
        U::Real=1.0,
        g::Real=1.0,
        omega::Real=1.0
    ) -> MPO

Construct the spinless fermion XXZ Hamiltonian coupled to a single photon mode
(cavity) using `OpSum`:
```math
H = \\sum_{j=1}^{L-1} -t \\left( e^{i\\frac{g}{\\sqrt{L}}(a + a^\\dagger)} c^\\dagger_j c_{j+1} + e^{-i\\frac{g}{\\sqrt{L}}(a + a^\\dagger)} c^\\dagger_{j+1} c_{j} \\right) + \\sum_{j=1}^{L-1} U n_j n_{j+1} + \\Omega N_{\\text{ph}}
```
using `OpSum`.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices, with the last index being the photon site.
- `t::Real=1.0`: Hopping amplitude.
- `U::Real=1.0`: Interaction strength.
- `g::Real=1.0`: Coupling strength between fermions and photon mode.
- `omega::Real=1.0`: Frequency of the photon mode.
# Returns
- `MPO`: The constructed MPO representing the XXZ Hamiltonian coupled to a cavity.

# Throws
- `ArgumentError`: If fewer than two sites are provided or if the site indices
    are not valid spinless fermion and photon indices.
"""
function xxz_cavity(
    sites::Vector{<:Index};
    pbc::Bool=false,
    t::Real=1.0,
    U::Real=1.0,
    g::Real=1.0,
    omega::Real=1.0
)::MPO
    # unpack sites and check their validity
    f_sites = sites[1:end-1]
    _check_site_tags(f_sites, "Fermion")
    _check_site_tag(sites[end], "Photon")

    L = length(f_sites) # number of fermionic sites
    L ≥ 2 || throw(ArgumentError("Need at least two lattice sites"))

    set_cavity_params!(g=g)
    os = OpSum()
    ph_ind = length(sites) # photon site index
    for j in 1:(L-1)
        # dressed hopping
        os += -t, "PeierlsPhase", ph_ind, "c†", j, "c", j+1
        os += -t, "PeierlsPhaseDag", ph_ind, "c†", j+1, "c", j
        # interaction term
        os += U, "n", j, "n", j+1
    end

    # add photon energy term
    os += omega, "N", ph_ind

    return MPO(os, sites)
end


# this function is outdated since we it does only work for the case where the total number
# of fermions is not conserved by DMRG. We keep it now for reference and maybe testing
# purposes but it should be removed in the future.
function xxz_cavity_no_qn_conservation(
    sites::Vector{<:Index};
    pbc::Bool=false,
    t::Real=1.0,
    U::Real=1.0,
    g::Real=1.0,
    omega::Real=1.0
)::MPO
    # unpack sites and check their validity
    f_sites = sites[1:end-1]
    ph_site = sites[end]
    _check_site_tags(f_sites, "Fermion")
    _check_site_tags([ph_site], "Photon")

    L = length(f_sites) # number of fermionic sites
    L ≥ 2 || throw(ArgumentError("Need at least two lattice sites"))

    P = build_peierls_phase(g, dim(ph_site))
    os = OpSum()
    ph_ind = L + 1 # photon site index
    for j in 1:(L-1)
        # dressed hopping
        os += -t, P, ph_ind, "c†", j, "c", j+1
        os += -t, P', ph_ind, "c†", j+1, "c", j
        # interaction term
        os += U, "n", j, "n", j+1
    end

    # add photon energy term
    os += omega, "N", ph_ind

    return MPO(os, sites)
end


"""
    xxz_cavity_manual(
        sites::Vector{<:Index};
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
The construction is based on the finite state machine (FSM) approach and details can
be found in my notes on GitHub.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices, with the last index being the photon site.
- `t::Real=1.0`: Hopping amplitude.
- `U::Real=1.0`: Interaction strength.
- `g::Real=1.0`: Coupling strength between fermions and photon mode.
- `omega::Real=1.0`: Frequency of the photon mode.

# Returns
- `MPO`: The constructed MPO representing the XXZ Hamiltonian coupled to a cavity.

# Throws
- `ArgumentError`: If fewer than two sites are provided or if the site indices
    are not valid spinless fermion and photon indices.

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
    sites::Vector{<:Index};
    pbc::Bool=false,
    t::Real=1.0,
    U::Real=1.0,
    g::Real=1.0,
    omega::Real=1.0
)::MPO
    # unpack sites and check their validity
    f_sites = sites[1:end-1]
    ph_site = sites[end]
    _check_site_tags(f_sites, "Fermion")
    _check_site_tag(ph_site, "Photon")

    L_mpo = length(sites) # number of sites in MPO (fermion sites + photon site)
    L = length(f_sites) # number of fermionic sites
    L ≥ 2 || throw(ArgumentError("Need at least two lattice sites"))

    # Local operators on fermionic sites
    id_f = ComplexF64[1 0; 0 1] # complex since we have complex phases later
    n_f = ComplexF64[0 0; 0 1]
    a_dag = ComplexF64[0 0; 1 0] # 'bosonic' creation operator from JWT
    a = ComplexF64[0 1; 0 0] # 'bosonic' annihilation operator from JWT

    # Local operators on photon site
    dim_ph = dim(ph_site)
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

    # last site (photon)
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
    xxz(sites::Vector{<:Index}; pbc::Bool=false, t::Real=1.0, U::Real=1.0) -> MPO

Construct the spinless fermion XXZ Hamiltonian
```math
H = \\sum_{j=1}^{L-1} -t(c^\\dagger_j c_{j+1} + c_j c^\\dagger_{j+1}) + \\sum_{j=1}^{L-1} U n_j n_{j+1}
```
using `OpSum`. We denote the Hamiltonian acting on spinless fermions as the Heisenberg
Hamiltonian and the one acting on spins as the XXZ Hamiltonian.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices for spinless fermions.
- `pbc::Bool=false`: Whether to construct the Hamiltonian with periodic boundary conditions (PBC)
                     or open boundary conditions (OBC).
- `t::Real=1.0`: Hopping amplitude.
- `U::Real=1.0`: Interaction strength.

# Returns
- `MPO`: The constructed MPO representing the spinless fermion XXZ Hamiltonian.

# Throws
- `ArgumentError`: If fewer than two sites are provided or if `sites` are not spinless
    fermion indices.
"""
function xxz(sites::Vector{<:Index}; pbc::Bool=false, t::Real=1.0, U::Real=1.0)::MPO
    _check_site_tags(sites, "Fermion")

    L = length(sites)
    L ≥ 2 || throw(ArgumentError("Need at least two lattice sites"))
    pbc && L < 3 && throw(ArgumentError("Periodic boundary conditions not possible for L < 3"))

    os = OpSum()
    for j in 1:(L-1)
        os += -t, "c†", j, "c", j + 1
        os += -t, "c†", j + 1, "c", j # or +t, "c", j, "c†", j + 1 due to anticommutation
        os += U, "n", j, "n", j + 1
    end

    if pbc
        os += -t, "c†", L, "c", 1
        os += -t, "c†", 1, "c", L # or +t, "c", L, "c†", 1 due to anticommutation
        os += U, "n", L, "n", 1
    end

    return MPO(os, sites)
end

"""
    xxz_manual(sites::Vector{<:Index}; pbc::Bool=false, t::Real=1.0, U::Real=1.0) -> MPO

Construct the spinless fermion XXZ Hamiltonian by explicit MPO construction
with bond dimension 5:
```math
H = \\sum_{j=1}^{L-1} -t(c^\\dagger_j c_{j+1} + c_j c^\\dagger_{j+1}) + \\sum_{j=1}^{L-1} U n_j n_{j+1} \\, .
```
We denote the Hamiltonian acting on spinless fermions as the Heisenberg
Hamiltonian and the one acting on spins as the XXZ Hamiltonian. The construction is based
on the finite state machine (FSM) approach and details can be found in my notes on GitHub.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices for spinless fermions.
- `pbc::Bool=false`: Whether to construct the Hamiltonian with periodic boundary conditions (PBC)
                     or open boundary conditions (OBC).
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
function xxz_manual(sites::Vector{<:Index}; pbc::Bool=false, t::Real=1.0, U::Real=1.0)::MPO
    if pbc
        return xxz_manual_pbc(sites; t=t, U=U)
    else
        return xxz_manual_obc(sites; t=t, U=U)
    end
end


function xxz_manual_obc(sites::Vector{<:Index}; t::Real=1.0, U::Real=1.0)::MPO
    _check_site_tags(sites, "Fermion")

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
    fill_op!(W[1], (1,), id)
    fill_op!(W[1], (2,), a_dag, -t)
    fill_op!(W[1], (3,), a, -t)
    fill_op!(W[1], (4,), n, U)

    # Bulk sites
    for l in 2:(L-1) # n is already used
        W[l] = ITensor(links[l-1], links[l], prime(sites[l]), sites[l])
        fill_op!(W[l], (1, 1), id)
        fill_op!(W[l], (1, 2), a_dag, -t)
        fill_op!(W[l], (1, 3), a, -t)
        fill_op!(W[l], (1, 4), n, U)

        fill_op!(W[l], (2, 5), a)
        fill_op!(W[l], (3, 5), a_dag)
        fill_op!(W[l], (4, 5), n)
        fill_op!(W[l], (5, 5), id)
    end

    # last site
    W[L] = ITensor(links[L-1], prime(sites[L]), sites[L])
    fill_op!(W[L], (2,), a)
    fill_op!(W[L], (3,), a_dag)
    fill_op!(W[L], (4,), n)
    fill_op!(W[L], (5,), id)

    return MPO(W)
end

function xxz_manual_pbc(sites::Vector{<:Index}; t::Real=1.0, U::Real=1.0)::MPO
    _check_site_tags(sites, "Fermion")

    L = length(sites)
    L ≥ 3 || throw(ArgumentError("Need at least three lattice sites (PBC not possible for L < 3)"))

    # Local operators
    id = [1.0 0.0; 0.0 1.0]
    z = ComplexF64[1 0; 0 -1] # Jordan-Wigner string operator
    n = [0.0 0.0; 0.0 1.0]
    a_dag = [0.0 0.0; 1.0 0.0] # 'bosonic' creation operator
    a = [0.0 1.0; 0.0 0.0] # 'bosonic' annihilation operator

    # Virtual bond indices (dimension 8), excluding boundaries
    links = [Index(8, "link,l=$i") for i in 1:(L - 1)]
    W = Vector{ITensor}(undef, L) # undef: do not initialize yet

    # First site
    W[1] = ITensor(links[1], prime(sites[1]), sites[1])
    fill_op!(W[1], (1,), id)
    fill_op!(W[1], (2,), a_dag, -t)
    fill_op!(W[1], (3,), a, -t)
    fill_op!(W[1], (4,), n, U)
    fill_op!(W[1], (5,), a_dag*z, -t)
    fill_op!(W[1], (6,), z*a, -t)
    fill_op!(W[1], (7,), n, U)

    # Bulk sites
    for l in 2:(L-1) # n is already used
        W[l] = ITensor(links[l-1], links[l], prime(sites[l]), sites[l])
        fill_op!(W[l], (1, 1), id)
        fill_op!(W[l], (1, 2), a_dag, -t)
        fill_op!(W[l], (1, 3), a, -t)
        fill_op!(W[l], (1, 4), n, U)

        fill_op!(W[l], (2, 8), a)
        fill_op!(W[l], (3, 8), a_dag)
        fill_op!(W[l], (4, 8), n)

        fill_op!(W[l], (5, 5), z)
        fill_op!(W[l], (6, 6), z)

        fill_op!(W[l], (7, 7), id)
        fill_op!(W[l], (8, 8), id)
    end

    # last site
    W[L] = ITensor(links[L-1], prime(sites[L]), sites[L])
    fill_op!(W[L], (2,), a)
    fill_op!(W[L], (3,), a_dag)
    fill_op!(W[L], (4,), n)
    fill_op!(W[L], (5,), a)
    fill_op!(W[L], (6,), a_dag)
    fill_op!(W[L], (7,), n)
    fill_op!(W[L], (8,), id)

    return MPO(W)
end

#########################################################################
# The following functions are not used, but were merely used to
# test how ITensors works (especially OpSum vs. manual MPO construction).
#########################################################################

"""
    heisenberg(sites::Vector{<:Index}; pbc::Bool=false, J::Real=1.0, Jz::Real=1.0) -> MPO

Construct the spin-1/2 Heisenberg (XXZ) Hamiltonian
```math
H = \\sum_{j=1}^{L-1} \\frac{J}{2} (S^+_j S^-_{j+1} + S^-_j S^+_{j+1}) + J_z S^z_j S^z_{j+1}
```
using `OpSum`. We denote the Hamiltonian acting on spins as the Heisenberg Hamiltonian
and the one acting on spinless fermions as the XXZ Hamiltonian.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices for spin-1/2 particles.
- `pbc::Bool=false`: Whether to construct the Hamiltonian with periodic boundary conditions (PBC)
                     or open boundary conditions (OBC).
- `J::Real=1.0`: Exchange coupling constant in the XY plane.
- `Jz::Real=1.0`: Exchange coupling constant in the Z direction.

# Returns
- `MPO`: The constructed Heisenberg Hamiltonian as an MPO.

# Throws
- `ArgumentError`: If fewer than two sites are provided or if `sites`
    are not spin-1/2 indices.
"""
function heisenberg(sites::Vector{<:Index}; pbc::Bool=false, J::Real=1.0, Jz::Real=1.0)::MPO
    _check_site_tags(sites, "S=1/2")

    L = length(sites)
    L ≥ 2 || throw(ArgumentError("Need at least two lattice sites"))
    pbc && L < 3 && throw(ArgumentError("Periodic boundary conditions not possible for L < 3"))


    os = OpSum()
    for j in 1:(L-1)
        os += Jz, "Sz", j, "Sz", j + 1
        os += J/2, "S+", j, "S-", j + 1
        os += J/2, "S-", j, "S+", j + 1
    end

    # add terms at boundaries for periodic boundary conditions if L > 2
    if pbc && L > 2
        os += Jz, "Sz", L, "Sz", 1
        os += J/2, "S+", L, "S-", 1
        os += J/2, "S-", L, "S+", 1
    end

    return MPO(os, sites)
end


"""
    heisenberg_manual(sites::Vector{<:Index}; pbc::Bool=false, J::Real=1.0, Jz::Real=1.0) -> MPO

Construct the spin-1/2 Heisenberg (XXZ) Hamiltonian
```math
H = \\sum_{j=1}^{L-1} \\frac{J}{2} (S^+_j S^-_{j+1} + S^-_j S^+_{j+1}) + J_z S^z_j S^z_{j+1}
```
by explicit MPO construction with bond dimension 5. We denote the Hamiltonian
acting on spins as the Heisenberg Hamiltonian and the one acting on spinless
fermions as the XXZ Hamiltonian. The construction is based on the finite state
machine (FSM) approach and details can be found in my notes on GitHub.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices for spin-1/2 particles.
- `pbc::Bool=false`: Whether to construct the Hamiltonian with periodic boundary conditions (PBC)
                     or open boundary conditions (OBC).
- `J::Real=1.0`: Exchange coupling constant in the XY plane.
- `Jz::Real=1.0`: Exchange coupling constant in the Z direction.

# Returns
- `MPO`: The constructed Heisenberg Hamiltonian as an MPO.

# Throws
- `ArgumentError`: If fewer than two sites are provided or if `sites`
    are not spin-1/2 indices.
"""
function heisenberg_manual(sites::Vector{<:Index}; pbc::Bool=false, J::Real=1.0, Jz::Real=1.0)::MPO
    if pbc
        return heisenberg_manual_pbc(sites; J=J, Jz=Jz)
    else
        return heisenberg_manual_obc(sites; J=J, Jz=Jz)
    end
end


function heisenberg_manual_obc(sites::Vector{<:Index}; J::Real=1.0, Jz::Real=1.0)::MPO
    _check_site_tags(sites, "S=1/2")

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
    fill_op!(W[1], (2,), S_minus, J/2)
    fill_op!(W[1], (3,), S_plus, J/2)
    fill_op!(W[1], (4,), S_z, Jz)
    fill_op!(W[1], (5,), id)

    # Bulk sites
    for n in 2:(L-1)
        W[n] = ITensor(links[n-1], links[n], prime(sites[n]), sites[n])
        fill_op!(W[n], (1, 1), id)
        fill_op!(W[n], (2, 1), S_plus)
        fill_op!(W[n], (3, 1), S_minus)
        fill_op!(W[n], (4, 1), S_z)

        fill_op!(W[n], (5, 2), S_minus, J/2)
        fill_op!(W[n], (5, 3), S_plus, J/2)
        fill_op!(W[n], (5, 4), S_z, Jz)
        fill_op!(W[n], (5, 5), id)
    end

    # Last site
    W[L] = ITensor(links[L-1], prime(sites[L]), sites[L])
    fill_op!(W[L], (1,), id)
    fill_op!(W[L], (2,), S_plus)
    fill_op!(W[L], (3,), S_minus)
    fill_op!(W[L], (4,), S_z)

    return MPO(W)
end


function heisenberg_manual_pbc(sites::Vector{<:Index}; J::Real=1.0, Jz::Real=1.0)::MPO
    _check_site_tags(sites, "S=1/2")

    L = length(sites)
    L ≥ 3 || throw(ArgumentError("Need at least three lattice sites (PBC not possible for L < 3)"))

    # Local operators (ensure Float64 type for division)
    id = [1.0 0.0; 0.0 1.0]
    S_z = 0.5 .* [1.0 0.0; 0.0 -1.0]
    S_plus = [0.0 1.0; 0.0 0.0]
    S_minus = [0.0 0.0; 1.0 0.0]

    # Virtual bond indices (dimension 8), excluding boundaries
    links = [Index(8, "link,l=$i") for i in 1:(L - 1)]
    W = Vector{ITensor}(undef, L) # undef: do not initialize yet

    # First site
    W[1] = ITensor(links[1], prime(sites[1]), sites[1])
    fill_op!(W[1], (1,), id)
    fill_op!(W[1], (2,), S_plus)
    fill_op!(W[1], (3,), S_minus)
    fill_op!(W[1], (4,), S_z)
    fill_op!(W[1], (5,), S_plus)
    fill_op!(W[1], (6,), S_minus)
    fill_op!(W[1], (7,), S_z)

    # Bulk sites
    for n in 2:(L-1)
        W[n] = ITensor(links[n-1], links[n], prime(sites[n]), sites[n])
        fill_op!(W[n], (1, 1), id)
        fill_op!(W[n], (1, 2), S_plus)
        fill_op!(W[n], (1, 3), S_minus)
        fill_op!(W[n], (1, 4), S_z)

        fill_op!(W[n], (2, 8), S_minus, J/2)
        fill_op!(W[n], (3, 8), S_plus, J/2)
        fill_op!(W[n], (4, 8), S_z, Jz)

        fill_op!(W[n], (5, 5), id)
        fill_op!(W[n], (6, 6), id)
        fill_op!(W[n], (7, 7), id)
        fill_op!(W[n], (8, 8), id)
    end

    # Last site
    W[L] = ITensor(links[L-1], prime(sites[L]), sites[L])
    fill_op!(W[L], (2,), S_minus, J/2)
    fill_op!(W[L], (3,), S_plus, J/2)
    fill_op!(W[L], (4,), S_z, Jz)
    fill_op!(W[L], (5,), S_minus, J/2)
    fill_op!(W[L], (6,), S_plus, J/2)
    fill_op!(W[L], (7,), S_z, Jz)
    fill_op!(W[L], (8,), id)

    return MPO(W)
end


"""
    pauli_sum(sites::Vector{<:Index}; a::Real=1.0, pauli::Symbol=:X) -> MPO

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
function pauli_sum(sites::Vector{<:Index}; a::Real=1.0, pauli::Symbol=:X)::MPO
    _check_site_tags(sites, "S=1/2")
    _check_pauli_symbol(pauli)

    L = length(sites)
    L ≥ 1 || throw(ArgumentError("Need at least one site"))

    os = OpSum()
    for j in 1:L
        os += a, String(pauli), j
    end

    return MPO(os, sites)
end


"""
    pauli_sum_manual(sites::Vector{<:Index}; a::Real=1.0, pauli::Symbol=:X) -> MPO

Construct an MPO representing
```math
H = a * \\sum_{j=1}^L \\sigma_j \\, ,
```
where σ_j is the Pauli matrix specified by `pauli` acting on sites with spin 1/2 particles.
The construction is based on the finite state machine (FSM) approach and details can
be found in my notes on GitHub.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices for spin-1/2 particles.
- `a::Real=1.0`: Coefficient multiplying the sum.
- `pauli::Symbol=:X`: Symbol specifying which Pauli matrix to use (:X, :Y, or :Z).

# Returns
- `MPO`: The constructed MPO representing the Pauli sum.

# Throws
- `ArgumentError`: If `pauli ∉ (:X, :Y, :Z)` or if `sites` are not spin-1/2 indices.
"""
function pauli_sum_manual(sites::Vector{<:Index}; a::Real=1.0, pauli::Symbol=:X,
)::MPO
    _check_site_tags(sites, "S=1/2")
    _check_pauli_symbol(pauli)

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
    fill_op!(W[1], (1,), σ, a)
    fill_op!(W[1], (2,), id)

    # Bulk sites
    for n in 2:(L - 1)
        W[n] = ITensor(links[n - 1], links[n], prime(sites[n]), sites[n])
        fill_op!(W[n], (1, 1), id)
        fill_op!(W[n], (2, 1), σ, a)
        fill_op!(W[n], (2, 2), id)
    end

    # Last site
    W[L] = ITensor(links[L - 1], prime(sites[L]), sites[L])
    fill_op!(W[L], (1,), id)
    fill_op!(W[L], (2,), σ, a)

    return MPO(W)
end

nothing
