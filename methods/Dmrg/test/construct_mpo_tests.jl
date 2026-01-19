using Dmrg
using Test
using ITensors, ITensorMPS
using Random

const ATOL = 1e-12  # absolute tolerance for inner product comparisons

# ------------------------------
# Helper functions
# ------------------------------
"""
    test_mpo_equivalence(sites::AbstractVector{<:Index}, H::MPO, H_manual::MPO, rng::AbstractRNG=Random.default_rng(), ntests::Int=3)::Nothing

Test equivalence of two MPOs by comparing expectation values on
random MPS for given sites.
"""
function test_mpo_equivalence(
    sites::AbstractVector{<:Index},
    H::MPO,
    H_manual::MPO,
    rng::AbstractRNG=Random.default_rng(),
    ntests::Int=3
)::Nothing
    # deterministic product state
    ψ_0 = productMPS(sites, fill("Up", length(sites)))
    @test isapprox(inner(ψ_0', H, ψ_0), inner(ψ_0', H_manual, ψ_0); atol=ATOL)

    # random MPS states
    for _ in 1:ntests
        ψ = random_mps(rng, sites; linkdims=10)
        @test isapprox(inner(ψ', H, ψ), inner(ψ', H_manual, ψ); atol=ATOL)
    end
    return nothing
end

"""
    test_invalid_sites_errors(mpo_fn::Function, manual_fn::Function; invalid_sites::AbstractVector{Vector{<:Index}})::Nothing

Ensure both MPO constructors throw ArgumentError for invalid sites.
"""
function test_invalid_sites_errors(
    opsum_mpo_fn::Function,
    manual_mpo_fn::Function,
    invalid_sites::Tuple{Vararg{AbstractVector{<:Index}}}
)::Nothing
    for sites in invalid_sites
        @test_throws ArgumentError opsum_mpo_fn(sites)
        @test_throws ArgumentError manual_mpo_fn(sites)
    end
    return nothing
end
# Tuple{Vararg{AbstractVector{<:Index}}} is used to allow passing a variable number of vectors of indices



# Signature-specific helpers
"""
    test_pauli_mpo_equivalence(sites::AbstractVector{<:Index}, a::Real, pauli::Symbol, rng::AbstractRNG)

Test equivalence of Pauli MPO constructed via OpSum vs manual construction.
"""
function test_pauli_mpo_equivalence(
    sites::AbstractVector{<:Index},
    a::Real,
    pauli::Symbol,
    rng::AbstractRNG
)
    H = Dmrg.pauli_sum_mpo(sites, a, pauli)
    Hm = Dmrg.pauli_sum_manual_mpo(sites, a, pauli)
    test_mpo_equivalence(sites, H, Hm, rng)
end

"""
    test_heisenberg_mpo_equivalence(sites::AbstractVector{<:Index}, J::Real, Jz::Real, rng::AbstractRNG)

Test equivalence of Heisenberg MPO constructed via OpSum vs manual construction.
"""
function test_heisenberg_mpo_equivalence(
    sites::AbstractVector{<:Index},
    J::Real,
    Jz::Real,
    rng::AbstractRNG
)
    H = Dmrg.heisenberg_mpo(sites, J, Jz)
    Hm = Dmrg.heisenberg_manual_mpo(sites, J, Jz)
    test_mpo_equivalence(sites, H, Hm, rng)
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
                test_pauli_mpo_equivalence(sites, a, pauli, rng)
            end
        end
    end

    @testset "invalid sites" begin
        # wrong spin & too short chain
        test_invalid_sites_errors(
            Dmrg.pauli_sum_mpo,
            Dmrg.pauli_sum_manual_mpo,
            (siteinds("S=1", 3), siteinds("S=1/2", 0))
        )
    end

    @testset "invalid pauli symbol" begin
        sites = siteinds("S=1/2", 3)
        @test_throws ArgumentError Dmrg.pauli_sum_mpo(sites, 1.0, :A)
        @test_throws ArgumentError Dmrg.pauli_sum_manual_mpo(sites, 1.0, :A)
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
                test_heisenberg_mpo_equivalence(sites, J, Jz, rng)
            end
        end
    end

    @testset "invalid sites" begin
        # wrong spin & too short chain
        test_invalid_sites_errors(
            Dmrg.heisenberg_mpo,
            Dmrg.heisenberg_manual_mpo,
            (siteinds("S=1", 3),siteinds("S=1/2", 1))
        )
    end

    @testset "zero-coupling sanity check" begin
        sites = siteinds("S=1/2", 4)
        ψ = random_mps(rng, sites; linkdims = 5)
        H = Dmrg.heisenberg_mpo(sites, 0.0, 0.0)
        @test isapprox(inner(ψ', H, ψ), 0.0; atol = ATOL)
    end
end




# @testset "Pauli sum MPO: OpSum vs manual construction" begin
#     Random.seed!(1234)

#     # test intended use for various system sizes and coefficients
#     for L in (2, 3, 5), a in (0.7, -1.2), pauli in (:X, :Y, :Z)
#         sites = siteinds("S=1/2", L)
#         H = Dmrg.pauli_sum_mpo(sites, a, pauli)
#         H_manual = Dmrg.pauli_sum_manual_mpo(sites, a, pauli)

#         test_mpo_equivalence(sites, H, H_manual)
#     end

#     # test error handling for invalid sites (wrong spin & too short chain)
#     invalid_sites = (siteinds("S=1", 3), siteinds("S=1/2", 0))
#     test_invalid_sites_errors(Dmrg.pauli_sum_mpo, Dmrg.pauli_sum_manual_mpo, invalid_sites)
#     # test error handling for invalid pauli symbol
#     @test_throws ArgumentError Dmrg.pauli_sum_manual_mpo(siteinds("S=1/2", 3), 1.0, :A)
#     @test_throws ArgumentError Dmrg.pauli_sum_mpo(siteinds("S=1/2", 3), 1.0, :A)
   
# end


# @testset "Heisenberg MPO: OpSum vs manual construction" begin
#     Random.seed!(1234)

#     # test intended use for various system sizes and coefficients
#     for L in (2, 4, 6), J in (2.5, -0.2), Jz in (1.0, -1.0)
#         sites = siteinds("S=1/2", L)
#         H = Dmrg.heisenberg_mpo(sites, J, Jz)
#         H_manual = Dmrg.heisenberg_manual_mpo(sites, J, Jz)

#         test_mpo_equivalence(sites, H, H_manual)
#     end

#     # test error handling for invalid sites
#     invalid_sites = (siteinds("S=1", 3), siteinds("S=1/2", 1))
#     test_invalid_sites_errors(Dmrg.heisenberg_mpo, Dmrg.heisenberg_manual_mpo, invalid_sites)
# end


