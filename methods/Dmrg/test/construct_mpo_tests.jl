using Dmrg
using Test
using ITensors, ITensorMPS
using Random

@testset "Pauli sum MPO: OpSum vs manual construction" begin
    Random.seed!(1234)

    # test intended use for various system sizes and coefficients
    for L in (2, 3, 5), coeff in (0.7, -1.2)
        sites = siteinds("S=1/2", L)
        H = Dmrg.pauli_sum_mpo(sites, coeff)
        H_manual = Dmrg.pauli_sum_manual_mpo(sites, coeff)

        for _ in 1:3
            ψ = random_mps(sites; linkdims=10)
            @test isapprox(
                inner(ψ', H, ψ),
                inner(ψ', H_manual, ψ);
                atol=1e-12
            )
        end
    end

    # test error handling for invalid inputs
    @test_throws ArgumentError Dmrg.pauli_sum_manual_mpo(siteinds("S=1", 3), 1.0)
    @test_throws ArgumentError Dmrg.pauli_sum_mpo(siteinds("S=1", 3), 1.0)
    @test_throws ArgumentError Dmrg.pauli_sum_manual_mpo(siteinds("S=1/2", 3), 1.0; pauli=:A)
    @test_throws ArgumentError Dmrg.pauli_sum_mpo(siteinds("S=1/2", 3), 1.0; pauli=:A)
    @test_throws ArgumentError Dmrg.pauli_sum_manual_mpo(siteinds("S=1/2", 0), 1.0)
    @test_throws ArgumentError Dmrg.pauli_sum_mpo(siteinds("S=1/2", 0), 1.0)
end

