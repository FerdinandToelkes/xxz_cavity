# using ITensors
# using ITensorMPS
# using LinearAlgebra


# alias(::SiteType"CavityMode") = SiteType"Boson"()

# function ITensors.space(::SiteType"CavityMode"; dim=2)
#     # Single dummy sector, so it does not change Fermion particle-number
#     return [QN("N", 0, 1) => dim]  # N = particle number label (dummy), dim states
# end

# function ITensors.op(on::OpName, st::SiteType"CavityMode", ds::Int...; kwargs...)
#     return op(on, alias(st), ds...; kwargs...)
# end

# function ITensors.op(on::OpName, st::SiteType"CavityMode", s1::Index, s_tail::Index...; kwargs...)
#     rs = reverse((s1, s_tail...))
#     ds = dim.(rs)
#     opmat = op(on, st, ds...; kwargs...)
#     return itensor(opmat, prime.(rs)..., dag.(rs)...)
# end

# function ITensors.op(::OpName"T", ::SiteType"CavityMode", d::Int...; kwargs...)
#     # Example: a T operator with some cavity-dependent structure
#     Tmat = zeros(d, d)
#     for n in 0:(d-1)
#         Tmat[n+1, n+1] = 1 + g*n/L  # just an example
#     end
#     return Tmat
# end


# function build_peierls_phase(g::Real, dim_ph::Int)::Matrix{ComplexF64}
#     # zeros on diagonal
#     d = zeros(Float64, dim_ph)

#     # off-diagonal entries: sqrt(1), …, sqrt(dim_ph-1)
#     e = sqrt.(collect(1:dim_ph-1)) # i.e. from 1 to N_ph-1

#     # diagonalize a + a^\dagger =: A which is tridiagonal in the number basis
#     A = SymTridiagonal(d, e)
#     eigenvals, eigenvecs = eigen(A)

#     # write A = V D V^\dagger with D diagonal matrix of eigenvals and V matrix of eigenvecs
#     phases = exp.(1im * g .* eigenvals) # .* element-wise multiplication
#     # U = V * diag(phases) * V†
#     U = eigenvecs * Diagonal(phases) * eigenvecs'

#     return ComplexF64.(U) # ensure complex type
# end

# function xxz_cavity(
#     sites::Vector{<:Index},
#     t::Real=1.0,
#     U::Real=1.0,
#     g::Real=1.0,
#     omega::Real=1.0
# )::MPO
#     # unpack sites and check their validity
#     f_sites = sites[1:end-1]
#     b_site = sites[end]

#     L = length(f_sites) # number of fermionic sites
#     L ≥ 2 || throw(ArgumentError("Need at least two lattice sites"))

#     P = build_peierls_phase(g, dim(b_site))
#     os = OpSum()
#     b = L + 1 # boson site index
#     for j in 1:(L-1)
#         # dressed hopping
#         # os += -t, P, b, "c†", j, "c", j+1
#         # os += -t, P', b, "c†", j+1, "c", j
#         os += -t, "c†", j, "c", j+1
#         os += -t, "c†", j+1, "c", j
#         # interaction term
#         os += U, "n", j, "n", j+1
#     end

#     # add boson energy term
#     n_ph = Diagonal(ComplexF64.(0:dim(b_site)-1))
#     # os += omega, n_ph, b
#     # os += omega, "N", b

#     return MPO(os, sites)
# end

# function main()
#     L = 3  # Number of sites
#     N_f = div(L, 2)  # half-filling
#     N_ph = 2  # Max number of photons
#     t = 1.0  # Hopping parameter
#     U = 2.0 * t  # On-site interaction
#     g = 0.5 * t / sqrt(L) # Light-matter coupling
#     omega = 1.0  # Cavity frequency
#     f_sites = siteinds("Fermion", L; conserve_qns=true)

#     n_max = 5
#     x = siteind("CavityMode"; dim=n_max+1)
#     T = op("T", x, n_max)

# end

# main()
# nothing

using ITensors, ITensorMPS

alias(::SiteType"CavityMode") = SiteType"Boson"()

function ITensors.space(::SiteType"CavityMode"; dim=2)
    # Single dummy sector, so it does not change Fermion particle-number
    return [QN("N", 0, 1) => dim]  # N = particle number label (dummy), dim states
end

function ITensors.op(on::OpName, st::SiteType"CavityMode", ds::Int...; kwargs...)
    return op(on, alias(st), ds...; kwargs...)
end

function ITensors.op(on::OpName, st::SiteType"CavityMode", s1::Index, s_tail::Index...; kwargs...)
    rs = reverse((s1, s_tail...))
    ds = dim.(rs)
    opmat = op(on, st, ds...; kwargs...)
    return itensor(opmat, prime.(rs)..., dag.(rs)...)
end

function ITensors.op(::OpName"T", ::SiteType"CavityMode", d::Int...; kwargs...)
    return [1 0 1;
        0 0 0;
        1 0 1]
end

let
    x = siteinds("CavityMode", 1; dim=3)
    # x = siteinds("Boson", 1; dim=3)
    #T = op("T", x)

    # psi = MPS([1, 0, 1], (x,))
    # println(psi[1])
    # psi = apply([T], psi)
    # println(psi[1])

    psi = productMPS(x, ["0"])
    # println(psi[1])
    # psi = apply([T], psi)
    # println(psi[1])

    # sites = siteinds("CavityMode", 1; dim=3)
    # os = OpSum()
    # os += 1.0, "T", 1
    # H = MPO(os, sites)
    # state = ["0"]
    # psi = MPS(sites, state)
    # psi = apply(H, psi)
    # println(psi[1])
end
