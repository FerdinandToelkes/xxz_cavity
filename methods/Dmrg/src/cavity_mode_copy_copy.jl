using Revise
using Dmrg

using ITensors
using ITensorMPS

using LinearAlgebra


# 04.02.2026
ITENSOR_SITE_TYPES = [
    "S=1/2",
    "S=½",
    "S=1",
    "Qubit",
    "Qudit",
    "Boson",
    "Fermion",
    "tJ",
    "Electron"
]

function ITensors.space(
    ::SiteType{S};
    dim=2,
    conserve_qns=false,
    conserve_number=conserve_qns,
    qnname_number="dummy"
) where {S}
    occursin("Photon", string(S)) || return invoke(ITensors.space, Tuple{SiteType}, st; kwargs...)


    return conserve_number ? [QN(qnname_number, 0) => dim] : dim
end

function ITensors.val(vn::ValName, st::SiteType{S}) where {S}
    occursin("Photon", string(S)) || return invoke(ITensors.val, Tuple{ValName, SiteType{S}}, vn, st)
    return val(vn, SiteType"Boson"())
end

function ITensors.state(sn::StateName, st::SiteType{S}, s::Index; kwargs...) where {S}
    occursin("Photon", string(S)) || return invoke(ITensors.state, Tuple{StateName, SiteType{S}, Index}, sn, st, s; kwargs...)
    return state(sn, SiteType"Boson"(), s; kwargs...)
end






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


function ITensors.op(::OpName"PeierlsPhase", ::SiteType{S}, d::Int) where {S}
    name = string(S)
    g_str = match(r"_g([0-9p]+)", name).captures[1]
    L_str = match(r"_L([0-9]+)", name).captures[1]

    g = parse(Float64, replace(g_str, "p" => "."))
    L = parse(Int, replace(L_str, "p" => "."))

    g = g / L
    mat = build_peierls_phase(g, d)
    return mat
end

function ITensors.op(::OpName"PeierlsPhase", st::SiteType{S}, s::Index; kwargs...) where {S}
    occursin("Photon", string(S)) ||
        error("PeierlsPhase only defined for Photon sites")

    d = dim(s)
    mat = op(OpName("PeierlsPhase"), st, d)
    return itensor(mat, prime(s), dag(s))
end


# one-body operators
# function ITensors.op(::OpName"Id", ::SiteType{S}, ds::Int...) where {S}
#     occursin("Photon", string(S)) ||
#         error("Id only defined for Photon sites")
#     d = prod(ds)
#     return Matrix(1.0I, d, d)
# end

# ITensors.op(::OpName"I", st::SiteType{S}, ds::Int...) where {S} =
#     op(OpName"Id"(), st, ds...)


# function ITensors.op(::OpName"Adag", ::SiteType{S}, d::Int) where {S}
#     occursin("Photon", string(S)) ||
#         error("Adag only defined for Photon sites")

#     mat = zeros(d, d)
#     for k in 1:(d-1)
#         mat[k + 1, k] = sqrt(k)
#     end
#     return mat
# end

# ITensors.op(::OpName"adag", st::SiteType{S}, d::Int) where {S} =
#     op(OpName"Adag"(), st, d)

# ITensors.op(::OpName"a†", st::SiteType{S}, d::Int) where {S} =
#     op(OpName"Adag"(), st, d)



# function ITensors.op(::OpName"A", ::SiteType{S}, d::Int) where {S}
#     occursin("Photon", string(S)) ||
#         error("A only defined for Photon sites")

#     mat = zeros(d, d)
#     for k in 1:(d-1)
#         mat[k, k + 1] = sqrt(k)
#     end
#     return mat
# end

# ITensors.op(::OpName"a", st::SiteType{S}, d::Int) where {S} =
#     op(OpName"A"(), st, d)


# function ITensors.op(::OpName"N_ph", ::SiteType{S}, d::Int) where {S}
#     occursin("Photon", string(S)) ||
#         error("N_ph only defined for Photon sites")

#     mat = zeros(d, d)
#     for k in 1:d
#         mat[k, k] = k - 1
#     end
#     return mat
# end

# ITensors.op(::OpName"n_ph", st::SiteType{S}, d::Int) where {S} =
#     op(OpName"N_ph"(), st, d)

# function ITensors.op(::OpName"PeierlsPhaseDag", ::SiteType{S}, d::Int) where {S}
#     name = string(S)
#     g_str = match(r"_g([0-9p]+)", name).captures[1]
#     L_str = match(r"_L([0-9]+)", name).captures[1]

#     g = parse(Float64, replace(g_str, "p" => "."))
#     L = parse(Int, replace(L_str, "p" => "."))

#     g = g / L
#     mat = build_peierls_phase(g, d)
#     return mat'
# end



function cavity_site_type(g::Real, L::Int)::String
    site_type = Symbol("Photon_g", replace(string(g), "."=>"p"), "_L", replace(string(L)))
    site_type = string(site_type)

    # check that string is small enough
    @assert length(site_type) <= 16 "Site type string too long for ITensors: $site_type with length $(length(site_type)) > 16"

    return site_type
end

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
    return H2-E^2
end

function total_photon_number(sites::Vector{<:Index})::MPO
    os = OpSum()
    os += "N_ph", length(sites) # assume last site is bosonic
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


using Random
function main_one()

    Random.seed!(1234)
    L = 4
    N = div(L, 2)
    n_max = 5
    g = 3.5
    conserve_qns = true
    # b_site = siteind("Photon", 1; dim=n_max+1, conserve_qns=conserve_qns)
    st = cavity_site_type(g, L)
    println("Cavity site type: ", st)
    b_site = siteinds(st, 1; dim=n_max+1, conserve_qns=conserve_qns)
    f_sites = siteinds("Fermion", L; conserve_qns=conserve_qns)
    sites = vcat(f_sites, b_site)

    # adag_a = op("PeierlsPhase", b_site[1])
    # @show adag_a

    # H = xxz_cavity_manual(sites)
    t = 1.0
    U = 0.0
    g = 1.0
    omega = 1.0
    H = xxz_cavity_dev(sites, t, U, g, omega)

    ##############################
    psi0 = MPS(sites)
    # f_states = [isodd(n) ? "1" : "0" for n=1:L]
    f_states = [n<=1 ? "0" : "1" for n=1:L]
    # f_psi0 = random_mps(f_sites, f_states)
    f_psi0 = MPS(f_sites, f_states)

    for i in 1:L
        psi0[i] = f_psi0[i]
    end

    # site L+1: random local state
    psi0[L+1] = randomITensor(sites[L+1])

    # normalize the full MPS
    normalize!(psi0)
    ##############################



    # run DMRG
    nsweeps = 10
    maxdim = [100, 200, 400, 800, 1600]
    cutoff = [1E-10]
    noise = [1E-6, 1E-7, 1E-8, 0.0] # help escape local minima

    energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff, noise=noise)

    # compute initial energy
    energy_is = inner(psi0', H, psi0)

    @show energy_is
    @show energy



    # apply photon number operator
    n_ph_op = total_photon_number(sites)
    n_ph_is = inner(psi0', n_ph_op, psi0)
    n_ph_gs = inner(psi', n_ph_op, psi)
    @show n_ph_is
    @show n_ph_gs

    # apply fermion number operator
    n_f_op = total_fermion_number(sites)
    n_f_is = inner(psi0', n_f_op, psi0)
    n_f_gs = inner(psi', n_f_op, psi)
    @show n_f_is
    @show n_f_gs


    # compute energy variance -> zero if psi is eigenstate
    var_gs = get_energy_variance(H, psi)
    var_is = get_energy_variance(H, psi0)
    @show var_is
    @show var_gs

end

main_one()
nothing



# alias(::SiteType"Photon") = SiteType"Boson"()

# Generic alias for all Photon_* types
# function alias(::SiteType{S}) where {S}
#     if occursin("Photon", string(S))
#         return SiteType"Boson"()
#     else
#        return SiteType"Boson"()
#     end
# end

#ITensors.val(vn::ValName, st::SiteType"Photon") = val(vn, alias(st))


# function ITensors.state(sn::StateName, st::SiteType"Photon", s::Index; kwargs...)
#     return state(sn, alias(st), s; kwargs...)
# end

# function ITensors.op(on::OpName, st::SiteType"Photon", ds::Int...; kwargs...)
#     return op(on, alias(st), ds...; kwargs...)
# end

# function ITensors.op(on::OpName, st::SiteType"Photon", s1::Index, s_tail::Index...; kwargs...)
#     rs = reverse((s1, s_tail...))
#     ds = dim.(rs)
#     opmat = op(on, st, ds...; kwargs...)
#     return itensor(opmat, prime.(rs)..., dag.(rs)...)
# end

# function ITensors.op(::OpName"Adag+A", ::SiteType{S}, d::Int) where {S}
#     name = string(S)
#     g_str = match(r"_g([0-9p]+)", name).captures[1]
#     L_str = match(r"_L([0-9]+)", name).captures[1]

#     g = parse(Float64, replace(g_str, "p" => "."))
#     L = parse(Int, replace(L_str, "p" => "."))

#     mat = zeros(d, d)
#     for k in 1:(d-1)
#         mat[k+1, k] = sqrt(k)
#         mat[k, k+1] = sqrt(k)
#     end
#     return g * mat + L * I # example: combine multiple params however you like
# end
