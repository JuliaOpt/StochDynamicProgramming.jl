################################################################################
# Test SDDP functions
################################################################################
using Base.Test, StochDynamicProgramming
using StochDynamicProgramming.SdpLoops

@testset "Indexation for SDP" begin

    bounds = [(0.1,10.0), (1.2, 4.0), (0.5, 2.0)]
    steps = [0.1, 0.05, 0.01]
    var = [0.4, 3.7, 1.9]
    vart = [0.42, 3.78, 1.932]

    ind = SdpLoops.index_from_variable(var, bounds, steps)
    ind2 = SdpLoops.real_index_from_variable(vart, bounds, steps)

    checkFalse = SdpLoops.is_next_state_feasible([0,1,2],3,bounds)
    checkTrue = SdpLoops.is_next_state_feasible([0.12,1.3,1.3],3,bounds)


    @test ind == (4,51,141)
    @test ind2[1] ≈ 4.2
    @test ind2[2] ≈ 52.6
    @test ind2[3] ≈ 144.2
    @test ~checkFalse
    @test checkTrue

end


@testset "SDP algorithm" begin

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
    cost_t(t, x, u, w) = COST[t] * (u[1])
    constraints(t, x, u, w) = true
    finalCostFunction(x) = 0.

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

    stateSteps = [2,2];
    controlSteps = [2,2];
    monteCarloSize = 2;

    modelSDP = StochDynProgModel(TF, x_bounds, u_bounds,
                                    x0, cost_t,
                                    finalCostFunction, dynamic,
                                    constraints, aleas);

    paramsSDP = StochDynamicProgramming.SDPparameters(modelSDP, stateSteps,
                                                        controlSteps,
                                                        "HD",
                                                        "Exact");


        @testset "Compare StochDynProgModel constructors" begin


            modelSDPPiecewise = StochDynamicProgramming.LinearSPModel(TF,
            u_bounds, x0,
            [cost_t],
            dynamic, aleas)
            set_state_bounds(modelSDPPiecewise, x_bounds)

            modelSDPLinear = StochDynamicProgramming.LinearSPModel(TF,
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

            @test test_costs

            @test convertedSDPmodel.constraints(1,x,u,w)

        end

        @testset "Solve and simulate using SDP" begin
            paramsSDP.infoStructure = "anything"
            solve_dp(modelSDP, paramsSDP, false);
            @test paramsSDP.infoStructure == "DH"
            paramsSDP.infoStructure = infoStruct

            V_sdp = solve_dp(modelSDP, paramsSDP, false);

            @test size(V_sdp) == (paramsSDP.stateVariablesSizes..., TF)


            costs_sdp2, stocks_sdp2, controls_sdp2 = StochDynamicProgramming.forward_simulations(modelSDP,
                                                                                                    paramsSDP,
                                                                                                    V_sdp,
                                                                                                    aleas_scen)

            x = x0
            V_sdp2 = StochDynamicProgramming.compute_value_functions_grid(modelSDP, paramsSDP, false);
            paramsSDP.infoStructure = "DH"
            V_sdp3 = StochDynamicProgramming.compute_value_functions_grid(modelSDP, paramsSDP, false);
            paramsSDP.infoStructure = "HD"

            Vitp = StochDynamicProgramming.value_function_interpolation( modelSDP.dimStates, V_sdp, 1)
            Vitp2 = StochDynamicProgramming.value_function_interpolation( modelSDP.dimStates, V_sdp2, 1)
            Vitp3 = StochDynamicProgramming.value_function_interpolation( modelSDP.dimStates, V_sdp3, 1)

            v1 = Vitp[(1.1,1.1)...]
            v2 = Vitp2[(1.1,1.1)...]
            v3 = Vitp3[(1.1,1.1)...]

            @test v1 == v2
            @test v1 <= v3

            paramsSDP.infoStructure = "DH"
            costs_sdp3, stocks_sdp3, controls_sdp3 = StochDynamicProgramming.forward_simulations(modelSDP,
                                                                                                    paramsSDP,
                                                                                                    V_sdp3,
                                                                                                    aleas_scen)
            paramsSDP.infoStructure = "HD"

            @test costs_sdp3[1] >= costs_sdp2[1]

            a = StochDynamicProgramming.generate_state_grid(modelSDP, paramsSDP)
            b = StochDynamicProgramming.generate_control_grid(modelSDP, paramsSDP)

            x_bounds = modelSDP.xlim
            x_steps = paramsSDP.stateSteps

            u_bounds = modelSDP.ulim
            u_steps = paramsSDP.controlSteps

            @test length(collect(a)) == (x_bounds[1][2]-x_bounds[1][1]+x_steps[1])*(x_bounds[2][2]-x_bounds[2][1]+x_steps[2])/(x_steps[1]*x_steps[2])
            @test length(collect(b)) == (u_bounds[1][2]-u_bounds[1][1]+u_steps[1])*(u_bounds[2][2]-u_bounds[2][1]+u_steps[2])/(u_steps[1]*u_steps[2])

            #modelSDP.initialState = [xi[1] for xi in x_bounds]
            modelSDP.initialState = x0
            ind = SdpLoops.index_from_variable(modelSDP.initialState, x_bounds, x_steps)
            #FIXME following test is failing
            #@test get_bellman_value(modelSDP, paramsSDP, V_sdp2) == V_sdp2[ind...,1]
            #modelSDP.initialState = x0

            @test size(V_sdp) == (paramsSDP.stateVariablesSizes..., TF)
            @test V_sdp2[1,1,1] <= V_sdp3[1,1,1]

            state_ref = zeros(2)
            state_ref[1] = stocks_sdp2[2,1,1]
            state_ref[2] = stocks_sdp2[2,1,2]
            w = [4]

            @test (get_control(modelSDP,paramsSDP,V_sdp3, 1, x, w)[1] >= CONTROL_MIN) == true
            @test (get_control(modelSDP,paramsSDP,V_sdp3, 1, x, w)[1] <= CONTROL_MAX) == true

            paramsSDP.infoStructure = "DH"
            @test (get_control(modelSDP,paramsSDP,V_sdp3, 1, x)[1] >= CONTROL_MIN)
            @test (get_control(modelSDP,paramsSDP,V_sdp3, 1, x)[1] <= CONTROL_MAX)

            @test size(stocks_sdp2) == (3,1,2)
            @test size(controls_sdp2) == (2,1,2)

        end

    end
