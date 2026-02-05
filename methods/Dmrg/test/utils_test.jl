using Dmrg

using ITensors
using ITensorMPS
using Random
using Test

const ATOL = 1e-12  # absolute tolerance numerical comparisons


@testset "Energy Variance Tests" begin
    # Initialize random number generator
    rng = Random.MersenneTwister(1234)

    # Create a simple spin-1/2 chain
    N = 4
    sites = siteinds("S=1/2", N)

    # Create a simple Heisenberg Hamiltonian
    os = OpSum()
    for j in 1:(N-1)
        os += 0.5, "S+", j, "S-", j + 1
        os += 0.5, "S-", j, "S+", j + 1
        os += "Sz", j, "Sz", j + 1
    end
    H = MPO(os, sites)

    # Create a random MPS
    psi = random_mps(rng,sites; linkdims=10)

    # Test that energy variance is real and non-negative
    var = Dmrg.get_energy_variance(H, psi)
    @test isreal(var)
    @test var >= -ATOL  # allow small numerical errors
end

# TODO write nice tests, if these functions are nicely implemented
# @testset "Total Photon Number Tests" begin
#     N = 4
#     sites = siteinds("Boson", N; dim=3)

#     mpo = Dmrg.total_photon_number(sites)
#     @test length(mpo) == N
# end

# @testset "Total Fermion Number Tests" begin
#     N = 5
#     f_sites = siteinds("Fermion", N - 1)
#     b_sites = siteinds("Boson", 1; dim=3)
#     sites = vcat(f_sites, b_sites)

#     mpo = Dmrg.total_fermion_number(sites)
#     @test length(mpo) == N
# end

@testset "Check Site Tags Tests" begin
    sites = siteinds("S=1/2", 4)

    # Test with correct tags
    @test Dmrg._check_site_tags(sites, "S=1/2") === nothing

    # Test with incorrect tags
    @test_throws ArgumentError Dmrg._check_site_tags(sites, "Fermion")
end

@testset "Check Site Tag Tests" begin
    site = siteind("S=1", 1)

    # Test with correct tag
    @test Dmrg._check_site_tag(site, "S=1") === nothing

    # Test with incorrect tag
    @test_throws ArgumentError Dmrg._check_site_tag(site, "Boson")
end

@testset "Fill Operator Tests" begin
    i, j, k, l = Index(2), Index(2), Index(2), Index(2)
    T = ITensor(i, j, k, l)
    M = ComplexF64[1 2; 3 4]

    Dmrg.fill_op!(T, (1, 1), M, 1.0)
    @test T[1, 1, 1, 1] == 1
    @test T[1, 1, 1, 2] == 2
    @test T[1, 1, 2, 1] == 3
    @test T[1, 1, 2, 2] == 4
end

@testset "Fill Operator with Prefactor Tests" begin
    i, j, k, l = Index(2), Index(2), Index(2), Index(2)
    T = ITensor(i, j, k, l)
    M = ComplexF64[1 2; 3 4]

    Dmrg.fill_op!(T, (2, 1), M, 0.5)
    @test T[2, 1, 1, 1] == 0.5
    @test T[2, 1, 1, 2] == 1.0
    @test T[2, 1, 2, 1] == 1.5
    @test T[2, 1, 2, 2] == 2.0
end

@testset "Fill Operator Error Handling Tests" begin
    i, j, k, l = Index(2), Index(2), Index(2), Index(2)
    T = ITensor(i, j, k, l)
    M_wrong = ComplexF64[1 2 3; 4 5 6; 7 8 9]  # 3x3 matrix

    @test_throws ArgumentError  Dmrg.fill_op!(T, (1, 1), M_wrong, 1.0, 2)
end

@testset "Check Pauli Symbol Tests" begin
    # Test valid symbols
    @test Dmrg._check_pauli_symbol(:X) === nothing
    @test Dmrg._check_pauli_symbol(:Y) === nothing
    @test Dmrg._check_pauli_symbol(:Z) === nothing

    # Test invalid symbols
    @test_throws ArgumentError Dmrg._check_pauli_symbol(:A)
    @test_throws ArgumentError Dmrg._check_pauli_symbol(:P)
end

@testset "Pauli Matrix Tests" begin
    X = Dmrg._pauli_matrix(:X)
    Y = Dmrg._pauli_matrix(:Y)
    Z = Dmrg._pauli_matrix(:Z)

    @test X == ComplexF64[0 1; 1 0]
    @test Y == ComplexF64[0 -im; im 0]
    @test Z == ComplexF64[1 0; 0 -1]

    # Test invalid pauli symbol
    @test_throws ArgumentError Dmrg._pauli_matrix(:A)
end
