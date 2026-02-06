using Dmrg

using ITensors
using ITensorMPS
using LinearAlgebra
using Random
using Test

# ------------------------------
# Helper functions
# ------------------------------
"""
    test_equivalence(args...) -> Nothing

Test equivalence of two MPOs by comparing expectation values on random MPS for given
sites. Note that the MPOs do not need to be identical, only their action on states must
be the same.
"""
function test_equivalence(
    sites::Vector{<:Index},
    H::MPO,
    H_manual::MPO;
    product_state::String="0",
    rng::AbstractRNG=Random.default_rng(),
    ntests::Int=3
)::Nothing
    # check Frobenius norm of difference between MPOs (gauge independent)
    @test isapprox(norm(H - H_manual), 0.0; atol=ATOL)

    # action on simple product state
    ψ_0 = productMPS(sites, fill(product_state, length(sites)))
    @test isapprox(inner(ψ_0', H, ψ_0), inner(ψ_0', H_manual, ψ_0); atol=ATOL)

    # action on random MPS states
    for _ in 1:ntests
        ψ = random_mps(rng, sites; linkdims=10)
        @test isapprox(inner(ψ', H, ψ), inner(ψ', H_manual, ψ); atol=ATOL)
    end
    return nothing
end

"""
    test_invalid_sites_errors(args...) -> Nothing

Ensure both MPO constructors throw ArgumentError for invalid sites. Note that we are doing sneaky
stuff here. We use that pbc=false is the default for the MPO constructors. This way, we can also
use this function for testing the pauli_sum functions, which don't have a pbc argument, without
having to write a separate helper function for that.
"""
function test_invalid_sites_errors(
    opsum_fn::Function,
    manual_fn::Function,
    invalid_sites::Tuple{Vararg{Vector{<:Index}}};
    pbc::Bool=false
)::Nothing
    for sites in invalid_sites
        if pbc
            @test_throws ArgumentError opsum_fn(sites; pbc=pbc)
            @test_throws ArgumentError manual_fn(sites; pbc=pbc)
        else
            @test_throws ArgumentError opsum_fn(sites)
            @test_throws ArgumentError manual_fn(sites)
        end
    end
    return nothing
end
# Tuple{Vararg{Vector{<:Index}}} is used to allow passing a variable number
# of vectors of indices



# Signature-specific helpers
"""
    test_xxz_cavity_equivalence(args...) -> Nothing

Test equivalence of XXZ cavity MPO constructed via OpSum vs manual construction.
"""
function test_xxz_cavity_equivalence(
    sites::Vector{<:Index};
    t::Real,
    U::Real,
    g::Real,
    omega::Real,
    rng::AbstractRNG
)
    H = Dmrg.xxz_cavity(sites; t=t, U=U, g=g, omega=omega)
    Hm = Dmrg.xxz_cavity_manual(sites; t=t, U=U, g=g, omega=omega)
    test_equivalence(sites, H, Hm; rng=rng, product_state="0")
    return nothing
end


"""
    test_xxz_equivalence(args...) -> Nothing

Test equivalence of XXZ MPO constructed via OpSum vs manual construction.
"""
function test_xxz_equivalence(
    sites::Vector{<:Index};
    pbc::Bool,
    t::Real,
    U::Real,
    rng::AbstractRNG
)
    H = Dmrg.xxz(sites; pbc=pbc, t=t, U=U)
    Hm = Dmrg.xxz_manual(sites; pbc=pbc, t=t, U=U)
    test_equivalence(sites, H, Hm; rng=rng, product_state="0")
    return nothing
end

"""
    test_heisenberg_equivalence(args...) -> Nothing

Test equivalence of Heisenberg MPO constructed via OpSum vs manual construction.
"""
function test_heisenberg_equivalence(
    sites::Vector{<:Index};
    pbc::Bool,
    J::Real,
    Jz::Real,
    rng::AbstractRNG
)
    H = Dmrg.heisenberg(sites; pbc=pbc, J=J, Jz=Jz)
    Hm = Dmrg.heisenberg_manual(sites; pbc=pbc, J=J, Jz=Jz)
    test_equivalence(sites, H, Hm; rng=rng, product_state="Up")
    return nothing
end

"""
    test_pauli_equivalence(args...) -> Nothing

Test equivalence of Pauli MPO constructed via OpSum vs manual construction.
"""
function test_pauli_equivalence(
    sites::Vector{<:Index};
    a::Real,
    pauli::Symbol,
    rng::AbstractRNG
)
    H = Dmrg.pauli_sum(sites; a=a, pauli=pauli)
    Hm = Dmrg.pauli_sum_manual(sites; a=a, pauli=pauli)
    test_equivalence(sites, H, Hm; rng=rng, product_state="Up")
    return nothing
end

# For comparison to pen and paper
"""
    expected_peierls_phase(g::Real, N_ph::Int) -> Matrix{ComplexF64}

Helper function to compute expected Peierls phase matrix for small N_ph.
See tablet notes for computation of exp(ig(a + a^+)) matrices.

# Arguments:
- `g::Real`: Coupling strength.
- `N_ph::Int`: Maximum number of photons (dimension of bosonic site minus one).

# Returns:
- `Matrix{ComplexF64}`: The expected Peierls phase matrix.

# Throws:
- `ArgumentError`: If `N_ph` is not supported.
"""
function expected_peierls_phase(g::Real, N_ph::Int)::Matrix{ComplexF64}
    if N_ph == 1
        return [
            cos(g) 1im*sin(g);
            1im*sin(g) cos(g)
        ]

    elseif N_ph == 2
        s3 = sqrt(3)
        return (1/3)*[
            cos(s3*g)+2  1im*s3*sin(s3*g)    sqrt(2)*(cos(s3*g)-1);
            1im*s3*sin(s3*g)  3*cos(s3*g)    1im*sqrt(6)*sin(s3*g);
            sqrt(2)*(cos(s3*g)-1) 1im*sqrt(6)*sin(s3*g) 2*cos(s3*g) + 1
        ]
    else
        throw(ArgumentError("Unsupported photon number"))
    end
end

@testset "Peierls phase matrix construction" begin
    g_values = [0, 1, π/2, 2, π, 4, 3*π/2]
    N_ph_values = [1, 2]

    for g in g_values, N_ph in N_ph_values
        dim_ph = N_ph + 1
        peierls_phase = Dmrg.build_peierls_phase(g, dim_ph)
        expected_phase = expected_peierls_phase(g, N_ph)
        @test isapprox(peierls_phase, expected_phase; atol=ATOL)
    end
end

# ------------------------------
# XXZ cavity MPO tests
# ------------------------------

@testset "XXZ cavity MPO: OpSum vs manual construction" begin
    # test setup
    rng = MersenneTwister(1234)
    Ls = (2, 3, 4)
    ts = (1.0, -0.5)
    Us = (2.0, -1.0)
    gs = (0.5, -0.3)
    omegas = (1.0, 2.0)
    dim_phs = (2, 10, 20)
    pbcs = (false) #true

    @testset "valid constructions" begin
        for L in Ls, t in ts, U in Us, g in gs, omega in omegas, dim_ph in dim_phs, pbc in pbcs
            @testset "L=$L, t=$t, U=$U, g=$g, omega=$omega, dim_ph=$dim_ph, pbc=$pbc" begin
                f_sites = siteinds("Fermion", L)
                ph_site = siteind("Photon", 1; dim=dim_ph)
                sites = vcat(f_sites, [ph_site])
                test_xxz_cavity_equivalence(sites; t=t, U=U, g=g, omega=omega, rng=rng)
            end
        end
    end

    @testset "invalid sites" begin
        invalid_sites = (
            [siteinds("Fermion", 1); siteind("Photon", 1; dim=2)], # too short chain
            [siteinds("S=1/2", 3); siteind("Photon", 1; dim=2)], # wrong particle type
            [siteinds("Fermion", 3); siteind("Fermion", 1)], # wrong particle
            siteinds("Fermion", 2) # missing bosonic site
        )
        println(typeof(invalid_sites))
        test_invalid_sites_errors(
            Dmrg.xxz_cavity,
            Dmrg.xxz_cavity_manual,
            invalid_sites,
        )
    end

    @testset "zero-coupling sanity check" begin
        f_sites = siteinds("Fermion", 4)
        ph_site = siteind("Photon", 1; dim=2) # otherwise error in MPO construction
        sites = vcat(f_sites, ph_site)
        ψ = random_mps(rng, sites; linkdims = 5)
        H = Dmrg.xxz_cavity(sites; t=0.0, U=0.0, g=0.0, omega=0.0)
        @test isapprox(inner(ψ', H, ψ), 0.0; atol = ATOL)
    end
end


# # ------------------------------
# # XXZ MPO tests
# # ------------------------------

# @testset "XXZ MPO: OpSum vs manual construction" begin
#     # test setup
#     rng = MersenneTwister(1234)
#     Ls = (6)
#     ts = (1.0)
#     Us = (-1.0)
#     pbcs = (false, true)

#     @testset "valid constructions" begin
#         for L in Ls, t in ts, U in Us, pbc in pbcs
#             @testset "L=$L, t=$t, U=$U, pbc=$pbc" begin
#                 sites = siteinds("Fermion", L)
#                 test_xxz_equivalence(sites; pbc=pbc, t=t, U=U, rng=rng)
#             end
#         end
#     end

#     @testset "invalid sites" begin
#         # wrong particle type & too short chains
#         test_invalid_sites_errors(
#             Dmrg.xxz,
#             Dmrg.xxz_manual,
#             (siteinds("S=1/2", 3), siteinds("Fermion", 1),)
#         )
#         # too short chain for periodic boundary conditions
#         test_invalid_sites_errors(
#             Dmrg.xxz,
#             Dmrg.xxz_manual,
#             (siteinds("Fermion", 2),); # , since function is defined for variable number of sites
#             pbc=true
#         )
#     end

#     @testset "zero-coupling sanity check" begin
#         sites = siteinds("Fermion", 4)
#         ψ = random_mps(rng, sites; linkdims=5)
#         H = Dmrg.xxz(sites; t=0.0, U=0.0, pbc=false)
#         @test isapprox(inner(ψ', H, ψ), 0.0; atol=ATOL)
#     end

# end

# # ------------------------------
# # Heisenberg MPO tests
# # ------------------------------

# @testset "Heisenberg MPO: OpSum vs manual construction" begin
#     # test setup
#     rng = MersenneTwister(1234)
#     Ls = (7)
#     Js = (2.5)
#     Jzs = (1.0)
#     pbcs = (false, true)

#     @testset "valid constructions" begin
#         for L in Ls, J in Js, Jz in Jzs, pbc in pbcs
#             @testset "L=$L, J=$J, Jz=$Jz, pbc=$pbc" begin
#                 sites = siteinds("S=1/2", L)
#                 test_heisenberg_equivalence(sites; pbc=pbc, J=J, Jz=Jz, rng=rng)
#             end
#         end
#     end

#     @testset "invalid sites" begin
#         # wrong spin & too short chain
#         test_invalid_sites_errors(
#             Dmrg.heisenberg,
#             Dmrg.heisenberg_manual,
#             (siteinds("S=1", 3), siteinds("S=1/2", 1))
#         )
#         # too short chain for periodic boundary conditions
#         test_invalid_sites_errors(
#             Dmrg.heisenberg,
#             Dmrg.heisenberg_manual,
#             (siteinds("S=1/2", 2),); # , since function is defined for variable number of sites
#             pbc=true
#         )
#     end

#     @testset "zero-coupling sanity check" begin
#         sites = siteinds("S=1/2", 4)
#         ψ = random_mps(rng, sites; linkdims=5)
#         H = Dmrg.heisenberg(sites; pbc=false, J=0.0, Jz=0.0)
#         @test isapprox(inner(ψ', H, ψ), 0.0; atol=ATOL)
#     end
# end

# # ------------------------------
# # Pauli MPO tests
# # ------------------------------

# @testset "Pauli sum MPO: OpSum vs manual construction" begin
#     # test setup
#     rng = MersenneTwister(1234)
#     Ls = (5)
#     as = (-1.2)
#     paulis = (:X, :Y, :Z)

#     @testset "valid constructions" begin
#         for L in Ls, a in as, pauli in paulis
#             @testset "L=$L, a=$a, pauli=$pauli" begin
#                 sites = siteinds("S=1/2", L)
#                 test_pauli_equivalence(sites; a=a, pauli=pauli, rng=rng)
#             end
#         end
#     end

#     @testset "invalid sites" begin
#         # wrong spin & too short chain
#         test_invalid_sites_errors(
#             Dmrg.pauli_sum,
#             Dmrg.pauli_sum_manual,
#             (siteinds("S=1", 3), siteinds("S=1/2", 0))
#         )
#     end

#     @testset "invalid pauli symbol" begin
#         sites = siteinds("S=1/2", 3)
#         @test_throws ArgumentError Dmrg.pauli_sum(sites; a=1.0, pauli=:A)
#         @test_throws ArgumentError Dmrg.pauli_sum_manual(sites; a=1.0, pauli=:A)
#     end
# end

nothing
