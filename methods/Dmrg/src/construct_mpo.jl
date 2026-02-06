using ITensors
using ITensorMPS
using LinearAlgebra




"""
    xxz_cavity(
        sites::Vector{<:Index};
        pbc::Bool=false,
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
    ph_ind = 1 # photon site index
    f_sites = sites[2:end]
    _check_site_tags(f_sites, "Fermion")
    _check_site_tag(sites[ph_ind], "Photon")

    L = length(f_sites) # number of fermionic sites
    L ≥ 2 || throw(ArgumentError("Need at least two lattice sites"))
    pbc && L < 3 && throw(ArgumentError("Periodic boundary conditions not possible for L < 3"))

    set_cavity_params!(g=g)
    os = OpSum()

    # there are L+1 sites in total, but we only loop over the fermionic sites and
    # exclude the last one here for open boundary conditions
    for j in (ph_ind+1):length(sites)-1
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
    ph_ind = 1 # photon site index
    ph_site = sites[ph_ind]
    f_sites = sites[2:end]
    _check_site_tags(f_sites, "Fermion")
    _check_site_tags([ph_site], "Photon")

    L = length(f_sites) # number of fermionic sites
    L ≥ 2 || throw(ArgumentError("Need at least two lattice sites"))
    pbc && L < 3 && throw(ArgumentError("Periodic boundary conditions not possible for L < 3"))

    P = build_peierls_phase(g, dim(ph_site))
    os = OpSum()

    # there are L+1 sites in total, but we only loop over the fermionic sites and
    # exclude the last one here for open boundary conditions
    for j in (ph_ind+1):length(sites)-1
        # dressed hopping
        os += -t, P, ph_ind, "c†", j, "c", j+1
        os += -t, P', ph_ind, "c†", j+1, "c", j
        # interaction term
        os += U, "n", j, "n", j+1
    end

    # add photon energy term
    os += omega, "N", ph_ind

    if pbc
        # add hopping terms connecting last and first fermionic site
        os += -t, P, ph_ind, "c†", length(sites), "c", ph_ind+1
        os += -t, P', ph_ind, "c†", ph_ind+1, "c", length(sites)
    end

    return MPO(os, sites)
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

# Keywords
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
    pauli_sum(sites::Vector{<:Index}; a::Real=1.0, pauli::Symbol=:X) -> MPO

Construct an MPO representing
```math
H = a * \\sum_{j=1}^L \\sigma_j \\, ,
```
where σ_j is the Pauli matrix specified by `pauli` acting on sites with spin 1/2 particles.

# Arguments
- `sites::Vector{<:Index}`: Vector of site indices for spin-1/2 particles.

# Keywords
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

nothing
