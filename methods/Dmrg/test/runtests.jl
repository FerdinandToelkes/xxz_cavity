using Dmrg
using Test


@testset "Dmrg tests" begin

    @testset "Utils tests" begin
        include("utils_test.jl")
    end

    @testset "Construct MPO tests" begin
        include("construct_mpo_tests.jl")
    end


end
