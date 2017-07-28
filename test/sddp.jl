################################################################################
# Test SDDP functions
################################################################################


include("framework.jl")
using Base.Test

# Test SDDP with a one dimensional stock:
@testset "SDDP algorithm: 1D case" begin
    solver = ClpSolver()

    # test reshaping of bounds
    @test StochDynamicProgramming.test_and_reshape_bounds(u_bounds,2,2,"control") == [(0.0,7.0) (0.0,7.0);(0.0,Inf) (0.0,Inf)]
    @test_throws ErrorException StochDynamicProgramming.test_and_reshape_bounds(u_bounds,1,2,"control")

    sddp_costs = 0

    @testset "Linear cost" begin
        # Compute bellman functions with SDDP:
        sddp = solve_SDDP(model, param, 0)
        @test isa(sddp, SDDPInterface)
        @test typeof(sddp.bellmanfunctions) == Vector{StochDynamicProgramming.PolyhedralFunction}
        @test typeof(sddp.solverinterface) == Vector{JuMP.Model}
        @test length(sddp.solverinterface) == n_stages - 1
        @test length(sddp.bellmanfunctions) == n_stages

        V = sddp.bellmanfunctions
        # Test if the first subgradient has the same dimension as state:
        @test size(V[1].lambdas, 2) == model.dimStates
        @test V[1].numCuts <= n_scenarios*max_iterations + n_scenarios
        @test size(V[1].lambdas, 1) == V[1].numCuts

        # Test upper bounds estimation with Monte-Carlo:
        n_simulations = 100
        upb = StochDynamicProgramming.estimate_upper_bound(model,
                                                           param,
                                                           V,
                                                           sddp.solverinterface,
                                                           n_simulations)[1]
        @test typeof(upb) == Float64

        pbs = sddp.solverinterface
        sddp_costs, stocks = forward_simulations(model, param, pbs,
                                                 noise_scenarios)
        # Test error if scenarios are not given in the right shape:
        @test_throws BoundsError forward_simulations(model, param, pbs, [1.])

        # Test computation of optimal control:
        aleas = noise_scenarios[1, 1, :]
        opt = StochDynamicProgramming.get_control(model, param,
                                                  sddp.solverinterface,
                                                  1, model.initialState, aleas)
        @test typeof(opt) == Vector{Float64}

        # Test display:
        param.compute_ub = 0
        sddp = solve_SDDP(model, param, V, 1)
    end

    @testset "Value functions calculation" begin
        V0 = StochDynamicProgramming.get_lower_bound(model, param, V)
        @test isa(V0, Float64)
    end

    @testset "Hotstart" begin
        # Test hot start with previously computed value functions:
        sddp = solve_SDDP(model, param, V, 0)
        @test isa(sddp, SDDPInterface)
        # Test if costs are roughly the same:
        sddp_costs2, stocks = forward_simulations(model, param,
                                                  sddp.solverinterface, noise_scenarios)
        @test mean(sddp_costs) ≈ mean(sddp_costs2)
    end

    # FIXME : correct solverQP
    # @testset "Quadratic regularization" begin
    #     param.SOLVER = solverQP
    #     regularization=SDDPRegularization(1., .99)
    #     sddp = solve_SDDP(model, param, 0,regularization=regularization)
    #     V0 = StochDynamicProgramming.get_lower_bound(sddp)
    #     #@test isa(V0, Float64)
    #
    #     @test_throws ErrorException solve_SDDP(model, param2, 0,
    #                                            regularization=SDDPRegularization(1., .99))
    # end

    # Test definition of final cost with a JuMP.Model:
    @testset "Final cost" begin
        function fcost(model, m)
            alpha = getindex(m, :alpha)
            @constraint(m, alpha == 0.)
        end
        # Store final cost in model:
        model.finalCost = fcost
        solve_SDDP(model, param, 0)
        solve_SDDP(model, param, V, 0)
    end

    @testset "Piecewise linear cost" begin
        # Test Piecewise linear costs:
        model = StochDynamicProgramming.LinearSPModel(n_stages,
                                                      u_bounds, x0,
                                                      [cost],
                                                      dynamic, laws)
        set_state_bounds(model, x_bounds)
        sddp = solve_SDDP(model, param, 0)
    end

    @testset "SMIP" begin
        controlCat = [:Bin, :Cont]
        u_bounds = [(0., 1.), (0., Inf)]
        model2 = StochDynamicProgramming.LinearSPModel(n_stages,
                                                      u_bounds, x0,
                                                      cost,
                                                      dynamic, laws,
                                                      control_cat=controlCat)
        set_state_bounds(model2, x_bounds)
        param.MIPSOLVER = solverMILP
        solve_SDDP(model2, param, 0)
        #@test_throws ErrorException solve_SDDP(model2, param, 0)
    end

    @testset "Stopping criterion" begin
        # Compute upper bound every %% iterations:
        param.compute_ub = 1
        gap = .1
        sddp = solve_SDDP(model, param, V, 0)
        V0 = StochDynamicProgramming.get_lower_bound(model, param, sddp.bellmanfunctions)
        n_simulations = 1000
        upb = StochDynamicProgramming.estimate_upper_bound(model, param,
                                                           sddp.bellmanfunctions,
                                                           sddp.solverinterface,
                                                           n_simulations)[1]

        @test abs((V0 - upb))/V0 < gap
    end

    @testset "Dump" begin
        # Dump V in text file:
        StochDynamicProgramming.writecsv("dump.dat", V)
        # Get stored values:
        Vdump = StochDynamicProgramming.read_polyhedral_functions("dump.dat")

        @test V[1].numCuts == Vdump[1].numCuts
        @test V[1].betas == Vdump[1].betas
        @test V[1].lambdas == Vdump[1].lambdas
    end

    #= @testset "Compare parameters" begin =#
    #=     paramDDP = [param for i in 1:3] =#
    #=     scenarios = StochDynamicProgramming.simulate_scenarios(laws, 1000) =#
    #=     benchmark_parameters(model, paramDDP, scenarios, 12) =#
    #= end =#
end


# Test SDDP with a two-dimensional stock:
@testset "SDDP algorithm: 2D case" begin
    solver = solverLP

    # SDDP's tolerance:
    epsilon = .05
    # maximum number of iterations:
    max_iterations = 2
    # number of scenarios in forward and backward pass:
    n_scenarios = 10
    # number of aleas:
    n_aleas = 5
    # number of stages:
    n_stages = 2

    # define dynamic:
    function dynamic(t, x, u, w)
        return [x[1] - u[1] - u[2] + w[1], x[2] - u[4] - u[3] + u[1] + u[2]]
    end
    # define cost:
    function cost(t, x, u, w)
        return -u[1] - u[3]
    end

    # Generate probability laws:
    laws = Vector{NoiseLaw}(n_stages)
    proba = 1/n_aleas*ones(n_aleas)
    for t=1:n_stages
        laws[t] = NoiseLaw([0, 1, 3, 4, 6], proba)
    end

    # set initial position:
    x0 = [10., 10]
    # set bounds on state:
    x_bounds = [(0., 100.), (0, 100)]
    # set bounds on control:
    u_bounds = [(0., 7.), (0., Inf), (0., 7.), (0., Inf)]

    # Instantiate parameters of SDDP:
    param = StochDynamicProgramming.SDDPparameters(solver,
                                                    passnumber=n_scenarios,
                                                    gap=epsilon,
                                                    max_iterations=max_iterations)
    V = nothing
    @testset "Linear cost" begin
        # Instantiate a SDDP linear model:
        model = StochDynamicProgramming.LinearSPModel(n_stages,
                                                      u_bounds, x0,
                                                      cost,
                                                      dynamic, laws)
        set_state_bounds(model, x_bounds)


        # Compute bellman functions with SDDP:
        sddp = solve_SDDP(model, param, 0)
        @test isa(sddp, SDDPInterface)
        V = sddp.bellmanfunctions
        pbs = sddp.solverinterface
        stats = sddp.stats
        @test typeof(V) == Vector{StochDynamicProgramming.PolyhedralFunction}
        @test typeof(pbs) == Vector{JuMP.Model}
        @test typeof(stats) == StochDynamicProgramming.SDDPStat

        # Test if the first subgradient has the same dimension as state:
        @test length(V[1].lambdas[1, :]) == model.dimStates

        # Test upper bounds estimation with Monte-Carlo:
        n_simulations = 100
        upb = StochDynamicProgramming.estimate_upper_bound(model, param, V, pbs,
                                                           n_simulations)[1]
        @test typeof(upb) == Float64


        # Test a simulation upon given scenarios:
        noise_scenarios = simulate_scenarios(model.noises,n_simulations)

        sddp_costs, stocks = forward_simulations(model, param, pbs, noise_scenarios)

        # Compare sddp cost with those given by extensive formulation:
        ef_cost = StochDynamicProgramming.extensive_formulation(model,param)[1]
        @test typeof(ef_cost) == Float64

        @test mean(sddp_costs) ≈ ef_cost

    end


    @testset "Dump" begin
        # Dump V in text file:
        StochDynamicProgramming.writecsv("dump.dat", V)
        # Get stored values:
        Vdump = StochDynamicProgramming.read_polyhedral_functions("dump.dat")

        @test V[1].numCuts == Vdump[1].numCuts
        @test V[1].betas == Vdump[1].betas
        @test V[1].lambdas == Vdump[1].lambdas
    end
end
