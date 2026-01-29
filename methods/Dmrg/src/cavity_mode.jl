using Dmrg

using ITensors
using ITensorMPS

using LinearAlgebra

alias(::SiteType"CavityMode") = SiteType"Boson"()

"""
    space(::SiteType"CavityMode";
          dim = 2,
          conserve_qns = false,
          conserve_number = false,
          qnname_number = "N_f")

Create the Hilbert space for a site of type "CavityMode".

Optionally specify the conserved symmetries and their quantum number labels.
"""
function ITensors.space(
        ::SiteType"CavityMode";
        dim = 2,
        conserve_qns = false,
        conserve_number = conserve_qns,
        qnname_number = "N_f",
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



function main_one()
    L = 4
    n_max = 5
    conserve_qns = false
    b_site = siteind("CavityMode", 1; dim=n_max+1, conserve_qns=conserve_qns)
    f_sites = siteinds("Fermion", L; conserve_qns=conserve_qns)
    sites = vcat(f_sites, [b_site])


    H = xxz_cavity_manual(sites)

    # B = op("B", b_site[1]; g=0.5)
    # @show B

    # id = op("Id", b_site[1])
    # a = op("a", b_site[1])
    # adag = op("adag", b_site[1])
    # n = op("n", b_site[1])
    # @show id, a, adag, n

    f_states = [isodd(n) ? "1" : "0" for n=1:L]
    b_state = ["0"]
    psi0 = MPS(sites, vcat(f_states, b_state))

    # os = OpSum()
    # for j in 1:L
    #     os += 1.0, "n", j
    # end

    # H = MPO(os, sites)

    # P = build_peierls_phase(0.5, dim(b_site[1]))
    # H[L+1] = ITensor(P, prime(b_site[1]), dag(b_site[1]))



    @show apply(H, psi0)
    @show inner(psi0', H, psi0)

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

# own
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
