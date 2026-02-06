using ITensors
using ITensorMPS
using LinearAlgebra


"""
    photon.jl

Custom site type and operators for photonic degrees of freedom in fermion-cavity systems.

# Overview
This file defines a custom "Photon" site type for ITensor that represents quantized light modes
coupled to a system of particles (e.g. spinless fermions) on a lattice. The photon site is
essential for handling systems where:

- A lattice of L sites contains particles (e.g. fermions) with particle number conservation
- The system couples to a single cavity mode (photon field)
- Total particle number is conserved, but photon number is not
- The interaction is described by quantized Peierls phase factors

# Design Rationale

## Custom Site Type
The "Photon" site type uses a trivial quantum number structure (no QN conservation by default)
despite being coupled to a fermionic system that does conserve particle number. This is necessary
because ITensors' `OpSum` requires homogeneous quantum number structures across all sites: either
all sites conserve QNs or none do (at least in the general use case). Since we cannot mix
QN-conserving fermion sites with non-QN photon sites, the photon site must have a dummy QN structure.

## Global Parameters and Custom Operators
Custom operators like "PeierlsPhase" are defined with global parameter passing via `CAVITY_PARAMS`.
This workaround was necessary because ITensors does not provide a native mechanism for passing
arguments to custom operators when building `OpSum` expressions (at least I, a complete beginner
in Julia and ITensors did not find a better way). The coupling strength `g` is
stored globally and retrieved when operators are instantiated.

# Key Components

- `CAVITY_PARAMS`: Reference container holding coupling parameters (e.g., photon-fermion coupling)
- `set_cavity_params!()`: A setter function to update global cavity parameters
- `space()`: Hilbert space specification for photon sites
- `PeierlsPhase` and `PeierlsPhaseDag`: Operators implementing photon-assisted tunneling

# References

- Code on which this is based: https://github.com/ITensor/ITensors.jl/blob/main/src/lib/SiteTypes/src/sitetypes/boson.jl
- ITensors discourse forum: https://itensor.discourse.group/t/fermionic-hopping-dressed-with-quantized-peierls-phase-and-conservation-of-fermions/2574
"""

# Global parameter container
const CAVITY_PARAMS = Ref((g = 0.0,))

# Setter for cavity parameters
@inline set_cavity_params!(; g) = (CAVITY_PARAMS[] = (g = g,))

alias(::SiteType"Photon") = SiteType"Boson"()

# Key difference between this photon site and a standard boson site
"""
    space(::SiteType"Photon";
          dim = 2,
          conserve_qns = false,
          conserve_number = false,
          qnname_number = "dummy")

Create the Hilbert space for a site of type "Photon".

Optionally specify the conserved symmetries and their quantum number labels.
"""
function ITensors.space(
        ::SiteType"Photon";
        dim = 2,
        conserve_qns = false,
        conserve_number = conserve_qns,
        qnname_number = "dummy"
    )
    if conserve_number
        return [QN(qnname_number, 0) => dim]
    end
    return dim
end

############################################################################################
##### Everything as in boson.jl except for "Boson" -> "Photon" and prefix "ITensors." ######
############################################################################################

ITensors.val(vn::ValName, st::SiteType"Photon") = val(vn, alias(st))

# Everything as in boson.jl except for "Boson" -> "Photon" and prefix "ITensors."
function ITensors.state(sn::StateName, st::SiteType"Photon", s::Index; kwargs...)
    return state(sn, alias(st), s; kwargs...)
end

# Everything as in boson.jl except for "Boson" -> "Photon" and prefix "ITensors."
function ITensors.op(on::OpName, st::SiteType"Photon", ds::Int...; kwargs...)
    return op(on, alias(st), ds...; kwargs...)
end

# Everything as in boson.jl except for "Boson" -> "Photon" and prefix "ITensors."
function ITensors.op(on::OpName, st::SiteType"Photon", s1::Index, s_tail::Index...; kwargs...)
    rs = reverse((s1, s_tail...))
    ds = dim.(rs)
    opmat = op(on, st, ds...; kwargs...)
    return itensor(opmat, prime.(rs)..., dag.(rs)...)
end

############################################################################################
######## Definition of quantized Peierls phase matrix and its operators for OpSum ##########
############################################################################################

"""
    build_peierls_phase(g::Real, dim_ph::Int) -> Matrix{ComplexF64}

Construct the Peierls phase matrix ``\\exp(ig(a + a^\\dagger))`` for a photon
site with given dimension `dim_ph`

# Arguments
- `g::Real`: Coupling strength.
- `dim_ph::Int`: Dimension of the photon site.

# Returns
- `Matrix{ComplexF64}`: The Peierls phase matrix of size `dim_ph x dim_ph`.

# Throws
- `ArgumentError`: If `dim_ph` is less than 1.
"""
function build_peierls_phase(g::Real, dim_ph::Int)::Matrix{ComplexF64}
    dim_ph >= 1 || error("Photon site dimension must be at least 1")

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
    ITensors.op(::OpName"PeierlsPhase", ::SiteType"Photon", d::Int) -> Matrix{ComplexF64}

Construct the Peierls phase operator for a photon site of dimension `d` using the global
coupling parameter `g` from `CAVITY_PARAMS`.

# Arguments
- `d::Int`: Dimension of the photon site.

# Returns
- `Matrix{ComplexF64}`: The Peierls phase operator matrix of size `d x d`.
"""
function ITensors.op(::OpName"PeierlsPhase", ::SiteType"Photon", d::Int)
    p = CAVITY_PARAMS[]
    g = p.g
    mat = build_peierls_phase(g, d)
    return mat
end

"""
    ITensors.op(::OpName"PeierlsPhaseDag", ::SiteType"Photon", d::Int) -> Matrix{ComplexF64}

Construct the Hermitian adjoint of the Peierls phase operator for a photon site of
dimension `d` using the global coupling parameter `g` from `CAVITY_PARAMS`.

# Arguments
- `d::Int`: Dimension of the photon site.

# Returns
- `Matrix{ComplexF64}`: The Hermitian adjoint of the Peierls phase operator matrix of size `d x d`.
"""
function ITensors.op(::OpName"PeierlsPhaseDag", ::SiteType"Photon", d::Int)
    p = CAVITY_PARAMS[]
    g = p.g
    mat = build_peierls_phase(g, d)
    return adjoint(mat)
end

nothing
