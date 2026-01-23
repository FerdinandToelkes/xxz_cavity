using Dmrg
using ITensors
using ITensorMPS
using Random
using Test

const ATOL = 1e-12  # absolute tolerance for inner product comparisons

# ------------------------------
# Helper functions
# ------------------------------
"""
    test_equivalence(args...) -> Nothing

Test equivalence of two MPOs by comparing expectation values on
random MPS for given sites.
"""
function test_equivalence(
    sites::Vector{<:Index},
    H::MPO,
    H_manual::MPO;
    product_state::String="Up",
    rng::AbstractRNG=Random.default_rng(),
    ntests::Int=3
)::Nothing
    # deterministic product state
    ψ_0 = productMPS(sites, fill(product_state, length(sites)))
    @test isapprox(inner(ψ_0', H, ψ_0), inner(ψ_0', H_manual, ψ_0); atol=ATOL)

    # random MPS states
    for _ in 1:ntests
        ψ = random_mps(rng, sites; linkdims=10)
        @test isapprox(inner(ψ', H, ψ), inner(ψ', H_manual, ψ); atol=ATOL)
    end
    return nothing
end

"""
    test_invalid_sites_errors(args...) -> Nothing

Ensure both MPO constructors throw ArgumentError for invalid sites.
"""
function test_invalid_sites_errors(
    opsum_fn::Function,
    manual_fn::Function,
    invalid_sites::Tuple{Vararg{Vector{<:Index}}}
)::Nothing
    for sites in invalid_sites
        @test_throws ArgumentError opsum_fn(sites)
        @test_throws ArgumentError manual_fn(sites)
    end
    return nothing
end
# Tuple{Vararg{Vector{<:Index}}} is used to allow passing a variable number
# of vectors of indices



# Signature-specific helpers
"""
    test_xxz_equivalence(args...) -> Nothing

Test equivalence of XXZ MPO constructed via OpSum vs manual construction.
"""
function test_xxz_equivalence(
    sites::Vector{<:Index},
    t::Real,
    U::Real,
    rng::AbstractRNG
)
    H = Dmrg.xxz(sites, t, U)
    Hm = Dmrg.xxz_manual(sites, t, U)
    test_equivalence(sites, H, Hm; rng=rng, product_state="0")
    return nothing
end

"""
    test_heisenberg_equivalence(args...) -> Nothing

Test equivalence of Heisenberg MPO constructed via OpSum vs manual construction.
"""
function test_heisenberg_equivalence(
    sites::Vector{<:Index},
    J::Real,
    Jz::Real,
    rng::AbstractRNG
)
    H = Dmrg.heisenberg(sites, J, Jz)
    Hm = Dmrg.heisenberg_manual(sites, J, Jz)
    test_equivalence(sites, H, Hm; rng=rng, product_state="Up")
    return nothing
end

"""
    test_pauli_equivalence(args...) -> Nothing

Test equivalence of Pauli MPO constructed via OpSum vs manual construction.
"""
function test_pauli_equivalence(
    sites::Vector{<:Index},
    a::Real,
    pauli::Symbol,
    rng::AbstractRNG
)
    H = Dmrg.pauli_sum(sites, a, pauli)
    Hm = Dmrg.pauli_sum_manual(sites, a, pauli)
    test_equivalence(sites, H, Hm; rng=rng, product_state="Up")
    return nothing
end

# ------------------------------
# XXZ MPO tests
# ------------------------------

@testset "XXZ MPO: OpSum vs manual construction" begin
    rng = MersenneTwister(1234)

    @testset "valid constructions" begin
        for L in (2, 4, 6), t in (1.0, -0.5), U in (2.0, -1.0)
            @testset "L=$L, t=$t, U=$U" begin
                sites = siteinds("Fermion", L)
                test_xxz_equivalence(sites, t, U, rng)
            end
        end
    end

    @testset "invalid sites" begin
        # wrong particle type & too short chain
        test_invalid_sites_errors(
            Dmrg.xxz,
            Dmrg.xxz_manual,
            (siteinds("S=1/2", 3), siteinds("Fermion", 1))
        )
    end

    @testset "zero-coupling sanity check" begin
        sites = siteinds("Fermion", 4)
        ψ = random_mps(rng, sites; linkdims = 5)
        H = Dmrg.xxz(sites, 0.0, 0.0)
        @test isapprox(inner(ψ', H, ψ), 0.0; atol = ATOL)
    end
end

# ------------------------------
# Heisenberg MPO tests
# ------------------------------

@testset "Heisenberg MPO: OpSum vs manual construction" begin
    rng = MersenneTwister(1234)

    @testset "valid constructions" begin
        for L in (2, 4, 6), J in (2.5, -0.2), Jz in (1.0, -1.0)
            @testset "L=$L, J=$J, Jz=$Jz" begin
                sites = siteinds("S=1/2", L)
                test_heisenberg_equivalence(sites, J, Jz, rng)
            end
        end
    end

    @testset "invalid sites" begin
        # wrong spin & too short chain
        test_invalid_sites_errors(
            Dmrg.heisenberg,
            Dmrg.heisenberg_manual,
            (siteinds("S=1", 3),siteinds("S=1/2", 1))
        )
    end

    @testset "zero-coupling sanity check" begin
        sites = siteinds("S=1/2", 4)
        ψ = random_mps(rng, sites; linkdims = 5)
        H = Dmrg.heisenberg(sites, 0.0, 0.0)
        @test isapprox(inner(ψ', H, ψ), 0.0; atol = ATOL)
    end
end

# ------------------------------
# Pauli MPO tests
# ------------------------------

@testset "Pauli sum MPO: OpSum vs manual construction" begin
    # fixed RNG for reproducibility
    rng = MersenneTwister(1234)

    @testset "valid constructions" begin
        for L in (2, 3, 5), a in (0.7, -1.2), pauli in (:X, :Y, :Z)
            @testset "L=$L, a=$a, pauli=$pauli" begin
                sites = siteinds("S=1/2", L)
                test_pauli_equivalence(sites, a, pauli, rng)
            end
        end
    end

    @testset "invalid sites" begin
        # wrong spin & too short chain
        test_invalid_sites_errors(
            Dmrg.pauli_sum,
            Dmrg.pauli_sum_manual,
            (siteinds("S=1", 3), siteinds("S=1/2", 0))
        )
    end

    @testset "invalid pauli symbol" begin
        sites = siteinds("S=1/2", 3)
        @test_throws ArgumentError Dmrg.pauli_sum(sites, 1.0, :A)
        @test_throws ArgumentError Dmrg.pauli_sum_manual(sites, 1.0, :A)
    end
end
