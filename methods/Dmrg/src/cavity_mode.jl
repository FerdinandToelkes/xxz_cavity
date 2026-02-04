using Revise
using Dmrg

using ITensors
using ITensorMPS

using LinearAlgebra

# Global parameter container
const CAVITY_PARAMS = Ref((g = 0.0,))

# Setter for cavity parameters
set_cavity_params!(; g) = (CAVITY_PARAMS[] = (g = g,))


alias(::SiteType"CavityMode") = SiteType"Boson"()

"""
    space(::SiteType"CavityMode";
          dim = 2,
          conserve_qns = false,
          conserve_number = false,
          qnname_number = "dummy")

Create the Hilbert space for a site of type "CavityMode".

Optionally specify the conserved symmetries and their quantum number labels.
"""
function ITensors.space(
        ::SiteType"CavityMode";
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


ITensors.val(vn::ValName, st::SiteType"CavityMode") = val(vn, alias(st))

function ITensors.state(sn::StateName, st::SiteType"CavityMode", s::Index; kwargs...)
    return state(sn, alias(st), s; kwargs...)
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



function ITensors.op(::OpName"Adag+A", ::SiteType"CavityMode", d::Int)
    p = CAVITY_PARAMS[]
    g = p.g
    mat = zeros(d, d)
    for k in 1:(d - 1)
        mat[k + 1, k] = sqrt(k)
        mat[k, k+1] = sqrt(k)
    end
    mat .*= g
    return mat
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

function ITensors.op(::OpName"PeierlsPhase", ::SiteType"CavityMode", d::Int)
    p = CAVITY_PARAMS[]
    g = p.g
    mat = build_peierls_phase(g, d)
    return mat
end

function ITensors.op(::OpName"PeierlsPhaseDag", ::SiteType"CavityMode", d::Int)
    p = CAVITY_PARAMS[]
    g = p.g
    mat = build_peierls_phase(g, d)
    return mat'
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


using Random
function main_one()
    # parse
    # g = parse(Float64, ARGS[1])

    g = 3.5

    set_cavity_params!(g=g)


    Random.seed!(1234)
    L = 4
    N = div(L, 2)
    n_max = 5
    conserve_qns = true
    b_site = siteinds("CavityMode", 1; dim=n_max+1, conserve_qns=conserve_qns)
    f_sites = siteinds("Fermion", L; conserve_qns=conserve_qns)
    sites = vcat(f_sites, b_site)


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



# from qudit
# one-body operators
# function ITensors.op(::OpName"Id", ::SiteType"CavityMode", ds::Int...)
#     d = prod(ds)
#     return Matrix(1.0I, d, d)
# end
#ITensors.op(::OpName"I", st::SiteType"CavityMode", ds::Int...) = op(OpName"Id"(), st, ds...)

# function ITensors.op(::OpName"Adag", ::SiteType"CavityMode", d::Int)
#     mat = zeros(d, d)
#     for k in 1:(d - 1)
#         mat[k + 1, k] = sqrt(k)
#     end
#     return mat
# end
#ITensors.op(::OpName"adag", st::SiteType"CavityMode", d::Int) = op(OpName"Adag"(), st, d)
#ITensors.op(::OpName"a†", st::SiteType"CavityMode", d::Int) = op(OpName"Adag"(), st, d)

# function ITensors.op(::OpName"A", ::SiteType"CavityMode", d::Int)
#     mat = zeros(d, d)
#     for k in 1:(d - 1)
#         mat[k, k + 1] = sqrt(k)
#     end
#     return mat
# end
#ITensors.op(::OpName"a", st::SiteType"CavityMode", d::Int) = op(OpName"A"(), st, d)
# function ITensors.op(::OpName"N", ::SiteType"CavityMode", d::Int)
#     mat = zeros(d, d)
#     for k in 1:d
#         mat[k, k] = k - 1
#     end
#     return mat
# end
#ITensors.op(::OpName"n", st::SiteType"CavityMode", d::Int) = op(OpName"N"(), st, d)

# ------------ own ---------------
# function ITensors.op(::OpName"Adag+A", ::SiteType"CavityMode", d::Int)
#     mat = zeros(d, d)
#     for k in 1:(d - 1)
#         mat[k + 1, k] = sqrt(k)
#         mat[k, k+1] = sqrt(k)
#     end
#     return mat
# end
# ITensors.op(::OpName"adag+a", st::SiteType"CavityMode", d::Int) = op(OpName"Adag+A"(), st, d)
# ITensors.op(::OpName"a†+a", st::SiteType"CavityMode", d::Int) = op(OpName"Adag+A"(), st, d)


# function ITensors.op(::OpName"B", ::SiteType"CavityMode", d::Int; g::Real)
#     mat = zeros(d, d)
#     for k in 1:(d - 1)
#         mat[k + 1, k] = g * sqrt(k)
#         mat[k, k + 1] = g * sqrt(k)
#     end
#     return mat
# end

# function ITensors.op(opname::OpName, ::SiteType"CavityMode", d::Int)
#     name = String(opname)

#     if startswith(name, "Exponent")
#         a_str = name[length("Exponent")+1:end]
#         a = Meta.parse(a_str) |> eval  # e.g. "1/2" → 0.5
#         mat = zeros(d, d)
#         for k in 1:(d - 1)
#             mat[k + 1, k] = a
#             mat[k, k + 1] = a
#         end
#         return mat
#     end

#     error("Unknown operator $name")
# end
