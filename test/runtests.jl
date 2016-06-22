#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# run unit-tests
#############################################################################


using StochDynamicProgramming
using SDPutils
using Distributions, Clp, FactCheck, JuMP


# Test probability functions
facts("Probability functions") do
    support = [1, 2, 3]
    proba = [.2 .5 .3]

    law = NoiseLaw(support, proba)
    @fact typeof(law) --> NoiseLaw
    @fact law.supportSize --> 3

    dims = (2, 2, 1)
    scenarios = simulate_scenarios([law, law], 2)
    @fact typeof(scenarios) --> Array{Float64, 3}
    @fact size(scenarios) --> (2, 2, 1)

    # test product of noiselaws:
    support2 = [4, 5, 6]
    proba2 = [.3 .3 .4]
    law2 = NoiseLaw(support2, proba2)
    law3 = StochDynamicProgramming.noiselaw_product(law, law2)
    @fact law3.supportSize --> law.supportSize*law2.supportSize
    @fact law3.proba --> vec(proba' * proba2)
    @fact size(law3.support)[1] --> size(law.support)[1] + size(law2.support)[1]
    @fact law3.support[:, 1] --> [1., 4.]

    # Test product of three noiselaws:
    StochDynamicProgramming.noiselaw_product(law, law2, law)

    # Test sampling:
    samp = StochDynamicProgramming.sampling([law, law2, law3], 1)
end



facts("Utils functions") do
    # Test extraction of vector in array:
    arr = rand(4, 4, 2)
    v = StochDynamicProgramming.extract_vector_from_3Dmatrix(arr, 2, 1)
    @fact typeof(v) --> Vector{Float64}
    @fact size(v) --> (2,)
    @fact v --> vec(arr[2, 1,:])

    # Test upper bound calculation:
    cost = rand(10)
    upb = StochDynamicProgramming.upper_bound(cost)
    tol = sqrt(2) * erfinv(2*.975 - 1)
    @fact upb --> mean(cost) + tol*std(cost)/sqrt(length(cost))

    # Test stopping criterion:
    @fact StochDynamicProgramming.test_stopping_criterion(1., .999, 0.01) --> true
end


# Test SDDP with a one dimensional stock:
facts("SDDP algorithm: 1D case") do
    solver = ClpSolver()

    # SDDP's tolerance:
    epsilon = .05
    # maximum number of iterations:
    max_iterations = 2
    # number of scenarios in forward and backward pass:
    n_scenarios = 10
    # number of aleas:
    n_aleas = 5
    # number of stages:
    n_stages = 3

    # define dynamic:
    function dynamic(t, x, u, w)
        return [x[1] - u[1] - u[2] + w[1]]
    end
    # define cost:
    function cost(t, x, u, w)
        return -u[1]
    end

    # Generate probability laws:
    laws = Vector{NoiseLaw}(n_stages)
    proba = 1/n_aleas*ones(n_aleas)
    for t=1:n_stages
        laws[t] = NoiseLaw([0, 1, 3, 4, 6], proba)
    end

    # set initial position:
    x0 = [10.]
    # set bounds on state:
    x_bounds = [(0., 100.)]
    # set bounds on control:
    u_bounds = [(0., 7.), (0., Inf)]

    # Instantiate parameters of SDDP:
    params = StochDynamicProgramming.SDDPparameters(solver, n_scenarios,
    epsilon, max_iterations)

    V = nothing
    model = StochDynamicProgramming.LinearDynamicLinearCostSPmodel(n_stages, u_bounds,
                                                                   x0, cost, dynamic, laws)

    set_state_bounds(model, x_bounds)
    # Test error if bounds are not well specified:
    @fact_throws set_state_bounds(model, [(0,1), (0,1)])

    # Generate scenarios for forward simulations:
    noise_scenarios = simulate_scenarios(model.noises,params.forwardPassNumber)

    sddp_costs = 0

    context("Unsolvable extensive formulation") do
        model_ef = StochDynamicProgramming.LinearDynamicLinearCostSPmodel(n_stages, u_bounds,
                                                                   x0, cost, dynamic, laws)
        x_bounds_ef = [(-2., -1.)]
        set_state_bounds(model_ef, x_bounds_ef)
        @fact_throws extensive_formulation(model_ef, params)
    end

    context("Linear cost") do
        # Compute bellman functions with SDDP:
        V, pbs = solve_SDDP(model, params, 0)
        @fact typeof(V) --> Vector{StochDynamicProgramming.PolyhedralFunction}
        @fact typeof(pbs) --> Vector{JuMP.Model}
        @fact length(pbs) --> n_stages - 1
        @fact length(V) --> n_stages

        # Test if the first subgradient has the same dimension as state:
        @fact length(V[1].lambdas[1, :]) --> model.dimStates
        @fact V[1].numCuts --> n_scenarios*max_iterations + 1
        @fact length(V[1].lambdas[:, 1]) --> n_scenarios*max_iterations + 1

        # Test upper bounds estimation with Monte-Carlo:
        n_simulations = 100
        upb = StochDynamicProgramming.estimate_upper_bound(model, params, V, pbs,
        n_simulations)[1]
        @fact typeof(upb) --> Float64

        sddp_costs, stocks = forward_simulations(model, params, V, pbs, noise_scenarios)
        # Test error if scenarios are not given in the right shape:
        @fact_throws forward_simulations(model, params, V, pbs, [1.])

        # Compare sddp cost with those given by extensive formulation:
        ef_cost = StochDynamicProgramming.extensive_formulation(model,params)[1]
        @fact typeof(ef_cost) --> Float64

        # As SDDP result is suboptimal, cost must be greater than those of extensive formulation:
        @fact mean(sddp_costs) > ef_cost --> true

        # Test computation of optimal control:
        aleas = StochDynamicProgramming.extract_vector_from_3Dmatrix(noise_scenarios, 1, 1)
        opt = StochDynamicProgramming.get_control(model, params, pbs, 1, model.initialState, aleas)
        @fact typeof(opt) --> Vector{Float64}

        # Test display:
        StochDynamicProgramming.set_max_iterations(params, 1)
        V, pbs = solve_SDDP(model, params, 1, V)
    end

    context("Value functions calculation") do
        V0 = StochDynamicProgramming.get_lower_bound(model, params, V)
    end

    context("Hotstart") do
        # Test hot start with previously computed value functions:
        V, pbs = solve_SDDP(model, params, 0, V)
        # Test if costs are roughly the same:
        sddp_costs2, stocks = forward_simulations(model, params, V, pbs, noise_scenarios)
        @fact mean(sddp_costs) --> roughly(mean(sddp_costs2))
    end

    context("Cuts pruning") do
        v = V[1]
        vt = PolyhedralFunction([v.betas[1]; v.betas[1] - 1.], v.lambdas[[1,1],:],  2)
        StochDynamicProgramming.prune_cuts!(model, params, V)
        isactive1 = StochDynamicProgramming.is_cut_relevant(model, 1, vt, params.solver)
        isactive2 = StochDynamicProgramming.is_cut_relevant(model, 2, vt, params.solver)
        @fact isactive1 --> true
        @fact isactive2 --> false
    end

    # Test definition of final cost with a JuMP.Model:
    context("Final cost") do
        function fcost(model, m)
            alpha = getvariable(m, :alpha)
            @constraint(m, alpha == 0.)
        end
        # Store final cost in model:
        model.finalCost = fcost
        V, pbs = solve_SDDP(model, params, 0)
        V, pbs = solve_SDDP(model, params, 0, V)
    end

    context("Piecewise linear cost") do
        # Test Piecewise linear costs:
        model = StochDynamicProgramming.PiecewiseLinearCostSPmodel(n_stages,
        u_bounds, x0,
        [cost],
        dynamic, laws)
        set_state_bounds(model, x_bounds)
        V, pbs = solve_SDDP(model, params, false)
    end

    context("Dump") do
        # Dump V in text file:
        StochDynamicProgramming.dump_polyhedral_functions("dump.dat", V)
        # Get stored values:
        Vdump = StochDynamicProgramming.read_polyhedral_functions("dump.dat")

        @fact V[1].numCuts --> Vdump[1].numCuts
        @fact V[1].betas --> Vdump[1].betas
        @fact V[1].lambdas --> Vdump[1].lambdas
    end
end


# Test SDDP with a two-dimensional stock:
facts("SDDP algorithm: 2D case") do
    solver = ClpSolver()

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
    params = StochDynamicProgramming.SDDPparameters(solver, n_scenarios,
    epsilon, max_iterations)
    V = nothing
    context("Linear cost") do
        # Instantiate a SDDP linear model:
        model = StochDynamicProgramming.LinearDynamicLinearCostSPmodel(n_stages,
        u_bounds, x0,
        cost,
        dynamic, laws)
        set_state_bounds(model, x_bounds)


        # Compute bellman functions with SDDP:
        V, pbs = solve_SDDP(model, params, false)
        @fact typeof(V) --> Vector{StochDynamicProgramming.PolyhedralFunction}
        @fact typeof(pbs) --> Vector{JuMP.Model}

        # Test if the first subgradient has the same dimension as state:
        @fact length(V[1].lambdas[1, :]) --> model.dimStates

        # Test upper bounds estimation with Monte-Carlo:
        n_simulations = 100
        upb = StochDynamicProgramming.estimate_upper_bound(model, params, V, pbs,
        n_simulations)[1]
        @fact typeof(upb) --> Float64


        # Test a simulation upon given scenarios:
        noise_scenarios = simulate_scenarios(model.noises,n_simulations)

        sddp_costs, stocks = forward_simulations(model, params, V, pbs, noise_scenarios)

        # Compare sddp cost with those given by extensive formulation:
        ef_cost = StochDynamicProgramming.extensive_formulation(model,params)[1]
        @fact typeof(ef_cost) --> Float64

        @fact mean(sddp_costs) --> roughly(ef_cost)

    end


    context("Dump") do
        # Dump V in text file:
        StochDynamicProgramming.dump_polyhedral_functions("dump.dat", V)
        # Get stored values:
        Vdump = StochDynamicProgramming.read_polyhedral_functions("dump.dat")

        @fact V[1].numCuts --> Vdump[1].numCuts
        @fact V[1].betas --> Vdump[1].betas
        @fact V[1].lambdas --> Vdump[1].lambdas
    end
end


facts("Indexation for SDP") do

    bounds = [(0.1,10.0), (1.2, 4.0), (0.5, 2.0)]
    steps = [0.1, 0.05, 0.01]
    var = [0.4, 3.7, 1.9]
    vart = [0.42, 3.78, 1.932]

    ind = SDPutils.index_from_variable(var, bounds, steps)
    ind2 = SDPutils.real_index_from_variable(vart, bounds, steps)


    @fact ind --> (4,51,141)
    @fact ind2[1] --> roughly(4.2)
    @fact ind2[2] --> roughly(52.6)
    @fact ind2[3] --> roughly(144.2)


end


facts("SDP algorithm") do

    # Number of timesteps :
    TF = 3

    # Capacity of dams:
    VOLUME_MAX = 20
    VOLUME_MIN = 0

    # Specify the maximum flow of turbines:
    CONTROL_MAX = 10
    CONTROL_MIN = -10

    # Some statistics about aleas (water inflow):
    W_MAX = 5
    W_MIN = 0
    DW = 1

    T0 = 1

    # Define aleas' space:
    N_ALEAS = Int(round(Int, (W_MAX - W_MIN) / DW + 1))
    ALEAS = linspace(W_MIN, W_MAX, N_ALEAS);

    N_CONTROLS = 2;
    N_STATES = 2;
    N_NOISES = 1;

    infoStruct = "HD"

    COST = 66*2.7*(1 + .5*(rand(TF) - .5));

    # Define dynamic of the dam:
    function dynamic(t, x, u, w)
        return [x[1] + u[1] + w[1] - u[2], x[2] - u[1]]
    end

    # Define cost corresponding to each timestep:
    function cost_t(t, x, u, w)
        return COST[t] * (u[1])
    end

    function constraints(t, x, u, w)
        return (x[1]<=VOLUME_MAX)&(x[1]>=VOLUME_MIN)&(x[2]<=VOLUME_MAX)&(x[2]>=VOLUME_MIN)
    end

    function finalCostFunction(x)
        return 0.
    end

    """Build admissible scenarios for water inflow over the time horizon."""
    function build_scenarios(n_scenarios::Int64, N_STAGES)
        scenarios = zeros(n_scenarios, N_STAGES)

        for scen in 1:n_scenarios
            scenarios[scen, :] = (W_MAX-W_MIN)*rand(N_STAGES)+W_MIN
        end
        return scenarios
    end

    """Build probability distribution at each timestep based on N scenarios.
    Return a Vector{NoiseLaw}"""
    function generate_probability_laws(N_STAGES, N_SCENARIOS)
        aleas = zeros(N_SCENARIOS, N_STAGES, 1)
        aleas[:, :, 1] = build_scenarios(N_SCENARIOS, N_STAGES)

        laws = Vector{NoiseLaw}(N_STAGES)

        # uniform probabilities:
        proba = 1/N_SCENARIOS*ones(N_SCENARIOS)

        for t=1:N_STAGES
            aleas_t = reshape(aleas[:, t, :], N_SCENARIOS, 1)'
            laws[t] = NoiseLaw(aleas_t, proba)
        end

        return laws
    end

    N_SCENARIO = 10
    aleas = generate_probability_laws(TF-1, N_SCENARIO)

    x_bounds = [(VOLUME_MIN, VOLUME_MAX), (VOLUME_MIN, VOLUME_MAX)];
    u_bounds = [(CONTROL_MIN, CONTROL_MAX), (VOLUME_MIN, VOLUME_MAX)];

    x0 = [5, 0]

    alea_year = Array([7.0 7.0])

    aleas_scen = zeros(2, 1, 1)
    aleas_scen[:, 1, 1] = alea_year;

    stateSteps = [1,1];
    controlSteps = [1,1];
    monteCarloSize = 2;

    modelSDP = StochDynProgModel(TF, x_bounds, u_bounds,
                                    x0, cost_t,
                                    finalCostFunction, dynamic,
                                    constraints, aleas);

    paramsSDP = StochDynamicProgramming.SDPparameters(modelSDP, stateSteps,
                                                        controlSteps,
                                                        infoStruct,
                                                        "Exact");


        context("Compare StochDynProgModel constructors") do


            modelSDPPiecewise = StochDynamicProgramming.PiecewiseLinearCostSPmodel(TF,
            u_bounds, x0,
            [cost_t],
            dynamic, aleas)
            set_state_bounds(modelSDPPiecewise, x_bounds)

            modelSDPLinear = StochDynamicProgramming.LinearDynamicLinearCostSPmodel(TF,
            u_bounds, x0,
            cost_t,
            dynamic, aleas)

            set_state_bounds(modelSDPLinear, x_bounds)


            test_costs = true
            x = x0
            u = [1, 1]
            w = [4]

            convertedSDPmodel = StochDynamicProgramming.build_sdpmodel_from_spmodel(modelSDPPiecewise)

            set_state_bounds(modelSDPLinear, x_bounds)


            for t in 1:TF-1
                test_costs &= (modelSDPLinear.costFunctions(t,x,u,w)==modelSDP.costFunctions(t,x,u,w))
                test_costs &= (modelSDPPiecewise.costFunctions[1](t,x,u,w)==modelSDP.costFunctions(t,x,u,w))
                test_costs &= (modelSDPPiecewise.costFunctions[1](t,x,u,w)==convertedSDPmodel.costFunctions(t,x,u,w))
            end

            @fact test_costs --> true

            @fact convertedSDPmodel.constraints(1,x,u,w) --> true

        end

        context("Solve and simulate using SDP") do

            V_sdp = solve_DP(modelSDP, paramsSDP, false);

            @fact size(V_sdp) --> (paramsSDP.stateVariablesSizes..., TF)

            costs_sdp, stocks_sdp, controls_sdp = StochDynamicProgramming.sdp_forward_single_simulation(modelSDP,
                                                                                                        paramsSDP,
                                                                                                        aleas_scen, x0,
                                                                                                        V_sdp, true )


            costs_sdp2, stocks_sdp2, controls_sdp2 = StochDynamicProgramming.sdp_forward_simulation(modelSDP,
                                                                                                    paramsSDP,
                                                                                                    aleas_scen,
                                                                                                    V_sdp, true )

            @fact costs_sdp2[1] --> costs_sdp

            x = x0
            V_sdp2 = StochDynamicProgramming.sdp_compute_value_functions(modelSDP, paramsSDP, false);
            paramsSDP.infoStructure = "DH"
            V_sdp3 = StochDynamicProgramming.sdp_compute_value_functions(modelSDP, paramsSDP, false);
            paramsSDP.infoStructure = "HD"

            Vitp = StochDynamicProgramming.value_function_interpolation( modelSDP.dimStates, V_sdp, 1)
            Vitp2 = StochDynamicProgramming.value_function_interpolation( modelSDP.dimStates, V_sdp2, 1)
            Vitp3 = StochDynamicProgramming.value_function_interpolation( modelSDP.dimStates, V_sdp3, 1)

            v1 = Vitp[(1.1,1.1)...]
            v2 = Vitp2[(1.1,1.1)...]
            v3 = Vitp3[(1.1,1.1)...]

            @fact v1 --> v2
            @fact (v1<=v3) --> true

            paramsSDP.infoStructure = "DH"
            costs_sdp3, stocks_sdp3, controls_sdp3 = StochDynamicProgramming.sdp_forward_simulation(modelSDP,
                                                                                                    paramsSDP,
                                                                                                    aleas_scen,
                                                                                                    V_sdp3, true )
            paramsSDP.infoStructure = "HD"

            @fact costs_sdp3[1]>=costs_sdp2[1] --> true

            a,b = StochDynamicProgramming.generate_grid(modelSDP, paramsSDP)

            x_bounds = modelSDP.xlim
            x_steps = paramsSDP.stateSteps

            u_bounds = modelSDP.ulim
            u_steps = paramsSDP.controlSteps

            @fact length(collect(a)) --> (x_bounds[1][2]-x_bounds[1][1]+x_steps[1])*(x_bounds[2][2]-x_bounds[2][1]+x_steps[2])/(x_steps[1]*x_steps[2])
            @fact length(collect(b)) --> (u_bounds[1][2]-u_bounds[1][1]+u_steps[1])*(u_bounds[2][2]-u_bounds[2][1]+u_steps[2])/(u_steps[1]*u_steps[2])

            ind = SDPutils.index_from_variable(x, x_bounds, x_steps)
            @fact get_bellman_value(modelSDP, paramsSDP, V_sdp2) --> V_sdp2[ind...,1]

            @fact size(V_sdp) --> (paramsSDP.stateVariablesSizes..., TF)
            @fact V_sdp2[1,1,1] <= V_sdp3[1,1,1] --> true

            @fact size(stocks_sdp) --> (3,1,2)
            @fact size(controls_sdp) --> (2,1,2)

            state_ref = zeros(2)
            state_ref[1] = stocks_sdp[2,1,1]
            state_ref[2] = stocks_sdp[2,1,2]
            w = [4]

            @fact_throws get_control(modelSDP,paramsSDP,V_sdp3, 1, x)
            @fact (get_control(modelSDP,paramsSDP,V_sdp3, 1, x, w)[1] >= CONTROL_MIN) --> true
            @fact (get_control(modelSDP,paramsSDP,V_sdp3, 1, x, w)[1] <= CONTROL_MAX) --> true

            paramsSDP.infoStructure = "DH"
            @fact (get_control(modelSDP,paramsSDP,V_sdp3, 1, x)[1] >= CONTROL_MIN) --> true
            @fact (get_control(modelSDP,paramsSDP,V_sdp3, 1, x)[1] <= CONTROL_MAX) --> true

            @fact size(stocks_sdp) --> (3,1,2)
            @fact size(controls_sdp) --> (2,1,2)

        end

    end
