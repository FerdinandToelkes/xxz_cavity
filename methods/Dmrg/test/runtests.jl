using Dmrg
using Test

# define absolute tolerance for numerical comparisons
const ATOL = 1e-10

@testset "Dmrg tests" begin

    @testset "Utils tests" begin
        include("utils_test.jl")
    end

    @testset "Construct MPO tests" begin
        include("construct_mpo_tests.jl")
    end

end

nothing
