using ITensors
using ITensorMPS
using LinearAlgebra

# Global parameter container
const CAVITY_PARAMS = Ref((g = 0.0,))

# Setter for cavity parameters
@inline set_cavity_params!(; g) = (CAVITY_PARAMS[] = (g = g,))


alias(::SiteType"Photon") = SiteType"Boson"()

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


ITensors.val(vn::ValName, st::SiteType"Photon") = val(vn, alias(st))


function ITensors.state(sn::StateName, st::SiteType"Photon", s::Index; kwargs...)
    return state(sn, alias(st), s; kwargs...)
end


function ITensors.op(on::OpName, st::SiteType"Photon", ds::Int...; kwargs...)
    return op(on, alias(st), ds...; kwargs...)
end


function ITensors.op(on::OpName, st::SiteType"Photon", s1::Index, s_tail::Index...; kwargs...)
    rs = reverse((s1, s_tail...))
    ds = dim.(rs)
    opmat = op(on, st, ds...; kwargs...)
    return itensor(opmat, prime.(rs)..., dag.(rs)...)
end


function ITensors.op(::OpName"PeierlsPhase", ::SiteType"Photon", d::Int)
    p = CAVITY_PARAMS[]
    g = p.g
    mat = build_peierls_phase(g, d)
    return mat
end

function ITensors.op(::OpName"PeierlsPhaseDag", ::SiteType"Photon", d::Int)
    p = CAVITY_PARAMS[]
    g = p.g
    mat = build_peierls_phase(g, d)
    return mat'
end

nothing
