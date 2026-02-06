using ITensors
using ITensorMPS
using LinearAlgebra
"""
    construct_mpo_manual.jl

# Overview

All functions included in this file are only for testing the MPO construction with the OpSum method.
The finite state machine (FSM) approach to MPO construction, which is used throughout this file,
was chosen, since it was the most intuitive way for me to construct MPOs manually.

# References

- My handwritten notes within the GitHub repository in which this file is also contained
- The paper Finite automata for caching in matrix product algorithms by Crosswhite and Bacon,
  which introduced the finite state machine (FSM) approach to MPO construction and on which
  the manual MPO construction in this file is based, DOI: 10.1103/PhysRevA.78.012356

"""

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

# Keywords
- `pbc::Bool=false`: Whether to use periodic boundary conditions.
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
    if pbc
        return xxz_cavity_manual_pbc(sites; t=t, U=U, g=g, omega=omega)
    else
        return xxz_cavity_manual_obc(sites; t=t, U=U, g=g, omega=omega)
    end
end

"""
# Notes

We utilized matrix product operator diagrams as introduced in the paper by Crosswhite and Bacon
to construct the MPO with the bosonic site, thus the structure of the finite state machine (FSM) is
a bit different from the one for the XXZ Hamiltonian without the cavity.

The states of the FSM are as follows:
- State 1: The ``U \\mathrm{id}_{\\mathrm{ph}}`` has been applied to photonic site.
- State 2: The ``-t \\exp(ig(a^\\dagger + a)/L)`` operator has been applied to photonic site.
- State 3: The ``-t \\exp(-ig(a^\\dagger + a)/L)`` operator has been applied to photonic site.
- State 4: The ``n`` operator has been applied on the site j
- State 5: The ``a^\\dagger`` operator has been applied on the site j
- State 6: The ``a`` operator has been applied on the site j
- State 7: Final state, where the operator strings have been completed and
           only the fermionic identity operator is applied on the remaining sites.

The transition for example from state 6 to state 7 can be done by applying ``a^\\dagger``.
"""
function xxz_cavity_manual_obc(
    sites::Vector{<:Index};
    t::Real=1.0,
    U::Real=1.0,
    g::Real=1.0,
    omega::Real=1.0
)::MPO
    # unpack sites and check their validity
    f_sites = sites[2:end]
    ph_site = sites[1]
    _check_site_tags(f_sites, "Fermion")
    _check_site_tag(ph_site, "Photon")

    L_mpo = length(sites) # number of sites in MPO (fermion sites + photon site)
    length(f_sites) ≥ 2 ||
        throw(ArgumentError("Need at least two lattice sites"))

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
    peierls_phase_dag = adjoint(peierls_phase) # conjugate transpose

    # Virtual bond indices (dimension 7), excluding boundaries
    links = [Index(7, "link,l=$i") for i in 1:L_mpo]
    W = Vector{ITensor}(undef, L_mpo) # undef: do not initialize yet

    # First site (photon)
    W[1] = ITensor(links[1], prime(sites[1]), sites[1])
    fill_op!(W[1], (1,), id_ph; prefactor=U, local_dim=dim_ph)
    fill_op!(W[1], (2,), peierls_phase; prefactor=-t, local_dim=dim_ph)
    fill_op!(W[1], (3,), peierls_phase_dag; prefactor=-t, local_dim=dim_ph)
    fill_op!(W[1], (7,), n_ph; prefactor=omega, local_dim=dim_ph)

    # Bulk sites (fermionic)
    for l in 2:(L_mpo-1)
        W[l] = ITensor(links[l-1], links[l], prime(sites[l]), sites[l])
        fill_op!(W[l], (1, 4), n_f)
        fill_op!(W[l], (2, 5), a_dag)
        fill_op!(W[l], (3, 6), a)

        fill_op!(W[l], (4, 7), n_f)
        fill_op!(W[l], (5, 7), a)
        fill_op!(W[l], (6, 7), a_dag)

        fill_op!(W[l], (1, 1), id_f)
        fill_op!(W[l], (2, 2), id_f)
        fill_op!(W[l], (3, 3), id_f)
        fill_op!(W[l], (7, 7), id_f)
    end

    # last site (fermionic)
    W[L_mpo] = ITensor(links[L_mpo-1], prime(sites[L_mpo]), sites[L_mpo])
    fill_op!(W[L_mpo], (4,), n_f)
    fill_op!(W[L_mpo], (5,), a)
    fill_op!(W[L_mpo], (6,), a_dag)
    fill_op!(W[L_mpo], (7,), id_f)
    return MPO(W)
end

"""
# Notes

We utilized matrix product operator diagrams as introduced in the paper by Crosswhite and Bacon
to construct the MPO with the bosonic site, thus the structure of the finite state machine (FSM) is
a bit different from the one for the XXZ Hamiltonian without the cavity. Moreover, this is also
a special case, since not all the bulk matrices are built the same way.

The states of the FSM are as follows:
- State 1: The ``U \\mathrm{id}_{\\mathrm{ph}}`` has been applied to photonic site.
- State 2: The ``-t \\exp(ig(a^\\dagger + a)/L)`` operator has been applied to photonic site.
- State 3: The ``-t \\exp(-ig(a^\\dagger + a)/L)`` operator has been applied to photonic site.
- State 4: The ``n`` operator has been applied on the first fermionic site
- State 5: The ``a^\\dagger \\cdot z`` operator has been applied on the first fermionic site
- State 6: The ``z \\cdot a`` operator has been applied on the site first fermionic site
- State 7: The ``n`` operator has been applied on the site j
- State 8: The ``a^\\dagger`` operator has been applied on the site j
- State 9: The ``a`` operator has been applied on the site j
- State 10: Final state, where the operator strings have been completed and
           only the fermionic identity operator is applied on the remaining sites.

The transition for example from state 1 to state 7 can be done by applying ``n`` or from
state 5 to itself can be done by applying the Jordan-Wigner string operator ``z``.
"""
function xxz_cavity_manual_pbc(
    sites::Vector{<:Index};
    t::Real=1.0,
    U::Real=1.0,
    g::Real=1.0,
    omega::Real=1.0
)::MPO
    # unpack sites and check their validity
    f_sites = sites[2:end]
    ph_site = sites[1]
    _check_site_tags(f_sites, "Fermion")
    _check_site_tag(ph_site, "Photon")

    L_mpo = length(sites) # number of sites in MPO (fermion sites + photon site)
    length(f_sites) ≥ 3 ||
        throw(ArgumentError("Need at least three lattice sites (PBC not possible for L < 3)"))

    # Local operators on fermionic sites
    id_f = ComplexF64[1 0; 0 1] # complex since we have complex phases later
    n_f = ComplexF64[0 0; 0 1]
    z = ComplexF64[1 0; 0 -1] # Jordan-Wigner string operator
    a_dag = ComplexF64[0 0; 1 0] # 'bosonic' creation operator from JWT
    a = ComplexF64[0 1; 0 0] # 'bosonic' annihilation operator from JWT

    # Local operators on photon site
    dim_ph = dim(ph_site)
    id_ph = Matrix{ComplexF64}(I, dim_ph, dim_ph)
    n_ph = Diagonal(ComplexF64.(0:dim_ph-1)) # number operator for bosons
    peierls_phase = build_peierls_phase(g, dim_ph)
    peierls_phase_dag = adjoint(peierls_phase) # conjugate transpose

    # Virtual bond indices (dimension 10), excluding boundaries
    links = [Index(10, "link,l=$i") for i in 1:L_mpo]
    W = Vector{ITensor}(undef, L_mpo) # undef: do not initialize yet

    # First site (photon)
    W[1] = ITensor(links[1], prime(sites[1]), sites[1])
    fill_op!(W[1], (1,), id_ph; prefactor=U, local_dim=dim_ph)
    fill_op!(W[1], (2,), peierls_phase; prefactor=-t, local_dim=dim_ph)
    fill_op!(W[1], (3,), peierls_phase_dag; prefactor=-t, local_dim=dim_ph)
    fill_op!(W[1], (10,), n_ph; prefactor=omega, local_dim=dim_ph)

    # Second site (special due to PBC combined with photon site)
    W[2] = ITensor(links[1], links[2], prime(sites[2]), sites[2])
    fill_op!(W[2], (1, 1), id_f)
    fill_op!(W[2], (10, 10), id_f)

    fill_op!(W[2], (1, 4), n_f)
    fill_op!(W[2], (1, 7), n_f)

    fill_op!(W[2], (2, 5), a_dag*z)
    fill_op!(W[2], (2, 8), a_dag)

    fill_op!(W[2], (3, 6), z*a)
    fill_op!(W[2], (3, 9), a)

    # Bulk sites (fermionic)
    for l in 3:(L_mpo-1)
        W[l] = ITensor(links[l-1], links[l], prime(sites[l]), sites[l])
        fill_op!(W[l], (1, 7), n_f)
        fill_op!(W[l], (2, 8), a_dag)
        fill_op!(W[l], (3, 9), a)

        fill_op!(W[l], (7, 10), n_f)
        fill_op!(W[l], (8, 10), a)
        fill_op!(W[l], (9, 10), a_dag)

        fill_op!(W[l], (1, 1), id_f)
        fill_op!(W[l], (4, 4), id_f)
        fill_op!(W[l], (5, 5), z)
        fill_op!(W[l], (6, 6), z)
        fill_op!(W[l], (10, 10), id_f)
    end

    # last site (fermionic)
    W[L_mpo] = ITensor(links[L_mpo-1], prime(sites[L_mpo]), sites[L_mpo])
    fill_op!(W[L_mpo], (4,), n_f)
    fill_op!(W[L_mpo], (5,), a)
    fill_op!(W[L_mpo], (6,), a_dag)
    fill_op!(W[L_mpo], (7,), n_f)
    fill_op!(W[L_mpo], (8,), a)
    fill_op!(W[L_mpo], (9,), a_dag)
    fill_op!(W[L_mpo], (10,), id_f)
    return MPO(W)
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

# Keywords
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

"""
# Notes

States of the underlying finite state machine (FSM) are as follows:
- State 1: Initial state, no operator has been applied yet.
- State 2: The ``a^\\dagger`` operator has been applied on the site j
- State 3: The ``a`` operator has been applied on the site j
- State 4: The ``n`` operator has been applied on the site j
- State 5: Final state, where the operator strings have been completed and
           only the identity operator is applied on the remaining sites.

The transition for example from state 3 to state 5 can be done by applying ``a^\\dagger``.
"""
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
    fill_op!(W[1], (2,), a_dag; prefactor=-t)
    fill_op!(W[1], (3,), a; prefactor=-t)
    fill_op!(W[1], (4,), n; prefactor=U)

    # Bulk sites
    for l in 2:(L-1) # n is already used
        W[l] = ITensor(links[l-1], links[l], prime(sites[l]), sites[l])
        fill_op!(W[l], (1, 1), id)
        fill_op!(W[l], (1, 2), a_dag; prefactor=-t)
        fill_op!(W[l], (1, 3), a; prefactor=-t)
        fill_op!(W[l], (1, 4), n; prefactor=U)

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

"""
# Notes

States of the underlying finite state machine (FSM) are as follows:
- State 1: Initial state, no operator has been applied yet.
- State 2: The ``a^\\dagger`` operator has been applied on the site j
- State 3: The ``a`` operator has been applied on the site j
- State 4: The ``n`` operator has been applied on the site j
- State 5: The ``a^\\dagger`` operator has been applied on the first site
- State 6: The ``a`` operator has been applied on the first site
- State 7: The ``n`` operator has been applied on the first site
- State 8: Final state, where the operator strings have been completed and
           only the identity operator is applied on the remaining sites.

The transition for example from state 4 to state 8 can be done by applying ``n``.
"""
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
    fill_op!(W[1], (2,), a_dag; prefactor=-t)
    fill_op!(W[1], (3,), a; prefactor=-t)
    fill_op!(W[1], (4,), n; prefactor=U)
    fill_op!(W[1], (5,), a_dag*z; prefactor=-t)
    fill_op!(W[1], (6,), z*a; prefactor=-t)
    fill_op!(W[1], (7,), n; prefactor=U)

    # Bulk sites
    for l in 2:(L-1) # n is already used
        W[l] = ITensor(links[l-1], links[l], prime(sites[l]), sites[l])
        fill_op!(W[l], (1, 1), id)
        fill_op!(W[l], (1, 2), a_dag; prefactor=-t)
        fill_op!(W[l], (1, 3), a; prefactor=-t)
        fill_op!(W[l], (1, 4), n; prefactor=U)

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

# Keywords
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

"""
# Notes

States of the underlying finite state machine (FSM) are as follows:
- State 1: Initial state, no operator has been applied yet.
- State 2: The ``S^+`` operator has been applied on the site j
- State 3: The ``S^-`` operator has been applied on the site j
- State 4: The ``S^z`` operator has been applied on the site j
- State 5: Final state, where the operator strings have been completed and
           only the identity operator is applied on the remaining sites.

This is written in the convention of denoting with (i, j) a transition from state j to state i, i.e.
everything is transposed (W[1] and W[L] are switched).
The transition for example from state 3 to state 5 can be done by applying ``(J/2) S^+``.
"""
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
    fill_op!(W[1], (2,), S_minus; prefactor=J/2)
    fill_op!(W[1], (3,), S_plus; prefactor=J/2)
    fill_op!(W[1], (4,), S_z; prefactor=Jz)
    fill_op!(W[1], (5,), id)

    # Bulk sites
    for n in 2:(L-1)
        W[n] = ITensor(links[n-1], links[n], prime(sites[n]), sites[n])
        fill_op!(W[n], (1, 1), id)
        fill_op!(W[n], (2, 1), S_plus)
        fill_op!(W[n], (3, 1), S_minus)
        fill_op!(W[n], (4, 1), S_z)

        fill_op!(W[n], (5, 2), S_minus; prefactor=J/2)
        fill_op!(W[n], (5, 3), S_plus; prefactor=J/2)
        fill_op!(W[n], (5, 4), S_z; prefactor=Jz)
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

"""
# Notes

States of the underlying finite state machine (FSM) are as follows:
- State 1: Initial state, no operator has been applied yet.
- State 2: The ``S^+`` operator has been applied on the site j
- State 3: The ``S^-`` operator has been applied on the site j
- State 4: The ``S^z`` operator has been applied on the site j
- State 5: The ``S^+`` operator has been applied on the first site
- State 6: The ``S^-`` operator has been applied on the first site
- State 7: The ``S^z`` operator has been applied on the first site
- State 8: Final state, where the operator strings have been completed and
           only the identity operator is applied on the remaining sites.

This is written in the convention of denoting with (i, j) a transition from state i to state j,
i.e. everything is transposed (sorry for inconsistency). The transition for example from
state 3 to state 8 can be done by applying ``(J/2) S^+``.
"""
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

        fill_op!(W[n], (2, 8), S_minus; prefactor=J/2)
        fill_op!(W[n], (3, 8), S_plus; prefactor=J/2)
        fill_op!(W[n], (4, 8), S_z; prefactor=Jz)

        fill_op!(W[n], (5, 5), id)
        fill_op!(W[n], (6, 6), id)
        fill_op!(W[n], (7, 7), id)
        fill_op!(W[n], (8, 8), id)
    end

    # Last site
    W[L] = ITensor(links[L-1], prime(sites[L]), sites[L])
    fill_op!(W[L], (2,), S_minus; prefactor=J/2)
    fill_op!(W[L], (3,), S_plus; prefactor=J/2)
    fill_op!(W[L], (4,), S_z; prefactor=Jz)
    fill_op!(W[L], (5,), S_minus; prefactor=J/2)
    fill_op!(W[L], (6,), S_plus; prefactor=J/2)
    fill_op!(W[L], (7,), S_z; prefactor=Jz)
    fill_op!(W[L], (8,), id)

    return MPO(W)
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

# Keywords
- `a::Real=1.0`: Coefficient multiplying the sum.
- `pauli::Symbol=:X`: Symbol specifying which Pauli matrix to use (:X, :Y, or :Z).

# Returns
- `MPO`: The constructed MPO representing the Pauli sum.

# Throws
- `ArgumentError`: If `pauli ∉ (:X, :Y, :Z)` or if `sites` are not spin-1/2 indices.

# Notes

States of the underlying finite state machine (FSM) are as follows:
- State 1: Initial state, no operator has been applied yet.
- State 2: Final state, where the Pauli operator has been applied on one site
           and only the identity operator is applied on the remaining sites.

This is written in the convention of denoting with (i, j) a transition from state j to state i,
i.e. everything is transposed (W[1] and W[L] are switched). The transition for example from
state 1 to state 2 can be done by applying the corresponding Pauli operator with prefactor `a`.
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
    fill_op!(W[1], (1,), σ; prefactor=a)
    fill_op!(W[1], (2,), id)

    # Bulk sites
    for n in 2:(L - 1)
        W[n] = ITensor(links[n - 1], links[n], prime(sites[n]), sites[n])
        fill_op!(W[n], (1, 1), id)
        fill_op!(W[n], (2, 1), σ; prefactor=a)
        fill_op!(W[n], (2, 2), id)
    end

    # Last site
    W[L] = ITensor(links[L - 1], prime(sites[L]), sites[L])
    fill_op!(W[L], (1,), id)
    fill_op!(W[L], (2,), σ; prefactor=a)

    return MPO(W)
end

nothing
