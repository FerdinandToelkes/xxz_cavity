using Dmrg
using Test
using ITensors, ITensorMPS
using Random

@testset "Pauli sum MPO: OpSum vs manual construction" begin
    Random.seed!(1234)

    # test intended use for various system sizes and coefficients
    for L in (2, 3, 5), coeff in (0.7, -1.2), pauli in (:X, :Y, :Z)
        sites = siteinds("S=1/2", L)
        H = Dmrg.pauli_sum_mpo(sites, coeff; pauli=pauli)
        H_manual = Dmrg.pauli_sum_manual_mpo(sites, coeff; pauli=pauli)

        for _ in 1:3
            ψ = random_mps(sites; linkdims=10)
            @test isapprox(
                inner(ψ', H, ψ),
                inner(ψ', H_manual, ψ);
                atol=1e-12
            )
        end
    end

    # test error handling for invalid sites
    @test_throws ArgumentError Dmrg.pauli_sum_manual_mpo(siteinds("S=1", 3), 1.0)
    @test_throws ArgumentError Dmrg.pauli_sum_mpo(siteinds("S=1", 3), 1.0)
    # test error handling for invalid pauli symbol
    @test_throws ArgumentError Dmrg.pauli_sum_manual_mpo(siteinds("S=1/2", 3), 1.0; pauli=:A)
    @test_throws ArgumentError Dmrg.pauli_sum_mpo(siteinds("S=1/2", 3), 1.0; pauli=:A)
    # test error handling for too short chains
    @test_throws ArgumentError Dmrg.pauli_sum_manual_mpo(siteinds("S=1/2", 0), 1.0)
    @test_throws ArgumentError Dmrg.pauli_sum_mpo(siteinds("S=1/2", 0), 1.0)
end


@testset "Heisenberg MPO: OpSum vs manual construction" begin
    Random.seed!(1234)

    # test intended use for various system sizes and coefficients
    for L in (2, 4, 6), J in (2.5, -0.2), Jz in (1.0, -1.0)
        sites = siteinds("S=1/2", L)
        H = Dmrg.heisenberg_mpo(sites, J, Jz)
        H_manual = Dmrg.heisenberg_manual_mpo(sites, J, Jz)

        for _ in 1:3
            ψ = random_mps(sites; linkdims=10)
            @test isapprox(
                inner(ψ', H, ψ),
                inner(ψ', H_manual, ψ);
                atol=1e-12
            )
        end
    end

    # test error handling for invalid sites
    @test_throws ArgumentError Dmrg.heisenberg_manual_mpo(siteinds("S=1", 3), 1.0, 1.0)
    @test_throws ArgumentError Dmrg.heisenberg_mpo(siteinds("S=1", 3), 1.0, 1.0)
    # test error handling for too short chains
    @test_throws ArgumentError Dmrg.heisenberg_manual_mpo(siteinds("S=1/2", 1), 1.0, 1.0)
    @test_throws ArgumentError Dmrg.heisenberg_mpo(siteinds("S=1/2", 1), 1.0, 1.0)
end
