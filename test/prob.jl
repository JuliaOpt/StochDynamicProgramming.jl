################################################################################
# Test probability functions
################################################################################
using Base.Test, StochDynamicProgramming

@testset "Probability functions" begin
    support = [1, 2, 3]
    proba = [.2 .5 .3]

    # test reshaping_noise
    @test typeof(StochDynamicProgramming.reshaping_noise(support, proba))==Tuple{Array{Int64,2},Array{Float64,1}}
    @test typeof(StochDynamicProgramming.reshaping_noise([1 2 3], proba))==Tuple{Array{Int64,2},Array{Float64,1}}
    
    law = NoiseLaw(support, proba)
    @test typeof(law) == NoiseLaw
    @test law.supportSize == 3
    @test_throws ErrorException NoiseLaw(support, [proba 0.1])



    dims = (2, 2, 1)
    scenarios = simulate_scenarios([law, law], 2)
    @test typeof(scenarios) == Array{Float64, 3}
    @test size(scenarios) == (2, 2, 1)

    # test product of noiselaws:
    support2 = [4, 5, 6]
    proba2 = [.3 .3 .4]
    law2 = NoiseLaw(support2, proba2)
    law3 = StochDynamicProgramming.noiselaw_product(law, law2)
    @test law3.supportSize == law.supportSize*law2.supportSize
    @test law3.proba == vec(proba' * proba2)
    @test size(law3.support)[1] == size(law.support)[1] + size(law2.support)[1]
    @test law3.support[:, 1] == [1., 4.]

    # Test product of three noiselaws:
    StochDynamicProgramming.noiselaw_product(law, law2, law)

    # Test sampling:
    samp = StochDynamicProgramming.sampling([law, law2, law3], 1)
end

