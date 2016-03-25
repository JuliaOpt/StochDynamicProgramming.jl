srand(2713)
push!(LOAD_PATH, "../src")
# include("../src/objects.jl")
# include("../src/SDPoptimize.jl")

using StochDynamicProgramming, JuMP, Clp, Distributions

const SOLVER = ClpSolver()
# const SOLVER = CplexSolver(CPX_PARAM_SIMDISPLAY=0)

const EPSILON = .05
const MAX_ITER = 20

# Define number of stages and scenarios:
const N_STAGES = 3
const N_SCENARIOS = 10

# Define time horizon:
const TF = N_STAGES-1

# Randomnly generate a cost scenario fixed for the whole problem:
const COST = [-12, -200, -67]

# Define bounds for states and controls:
const VOLUME_MAX = 50
const VOLUME_MIN = 0

const CONTROL_MAX = 50
const CONTROL_MIN = 0

# Define realistic bounds for aleas:
const W_MAX = 40
const W_MIN = 0

# Randomly generate two deterministic scenarios for rain
alea_year1 =round(Int, (W_MAX-W_MIN)*rand(TF)-W_MIN)

# Define initial states of both dams:
const X0 = [50]


# Define dynamic of the dams:
function dynamic(t, x, u, w)
    return [x[1] - u[1] - u[2] + w[1]]
end

function dynamic_HD(t, x, u, w)
    return [x[1] - u[1] - u[2] + w[1]]
end

function dynamic_DH(t, x, u, w)
    x1=x[1] - u[1] + w[1]
    return [min(VOLUME_MAX,max(0,x1))]
end

# Define cost corresponding to each timestep:
function cost_t(t, x, u, w)
    return COST[t] * (u[1])
end

function cost_t_HD(t, x, u, w)
    return COST[t] * (u[1])
end

function cost_t_DH(t, x, u, w)
    x1=x[1] - u[1] + w[1]
    return COST[t] * (x[1]+w[1]-min(VOLUME_MAX,max(0,x1)))
end

function finalCostFunction(x)
    return 0.
end

function constraints(t, x1, u, w)

    Bu = (x1[1]<=VOLUME_MAX)
    Bl = (x1[1]>=VOLUME_MIN)

    return Bu&Bl

end


"""Solve the problem with a solver, supposing the aleas are known
in advance."""
function solve_determinist_problem()
    m = Model(solver=SOLVER)


    @defVar(m,  VOLUME_MIN  <= x1[1:(TF+1)]  <= VOLUME_MAX)
    @defVar(m,  CONTROL_MIN <= u1[1:TF]  <= CONTROL_MAX)

    @setObjective(m, Min, sum{COST[i]*(u1[i]), i = 1:TF})

    for i in 1:TF
        @addConstraint(m, x1[i+1] - x1[i] + u1[i] - alea_year1[i] == 0)
    end

    @addConstraint(m, x1[1] == X0[1])

    status = solve(m)
    println(status)
    println(getObjectiveValue(m))
    return getValue(u1), getValue(x1)
end


"""Build admissible scenarios for water inflow over the time horizon."""
function build_scenarios(n_scenarios::Int64)
    scenarios = zeros(n_scenarios, N_STAGES)

    for scen in 1:n_scenarios
        scenarios[scen, :] = round(Int, (W_MAX-W_MIN)*rand(N_STAGES)+W_MIN)
    end
    return scenarios
end


"""Build probability distribution at each timestep based on N scenarios.

Return a Vector{NoiseLaw}"""
function generate_probability_laws()
    aleas = build_scenarios(N_SCENARIOS)

    laws = Vector{NoiseLaw}(N_STAGES)

    # uniform probabilities:
    proba = 1/N_SCENARIOS*ones(N_SCENARIOS)

    for t=1:(N_STAGES)
        laws[t] = NoiseLaw(aleas[:, t], proba)
    end

    return laws
end


"""Instantiate the problem."""

function init_problem_sdp_HD()

    x0 = X0
    aleas = generate_probability_laws()

    x_bounds = [(VOLUME_MIN, VOLUME_MAX)]
    u_bounds = [(CONTROL_MIN, CONTROL_MAX), (0, VOLUME_MAX)]

    N_CONTROLS = 2
    N_STATES = 1
    N_NOISES = 1
    infoStruct = "HD"

    stateSteps = [1]
    controlSteps = [1, 1]
    stateVariablesSizes = [(VOLUME_MAX-VOLUME_MIN)+1]
    controlVariablesSizes = [(CONTROL_MAX-CONTROL_MIN)+1, (VOLUME_MAX)+1]
    totalStateSpaceSize = stateVariablesSizes[1]
    totalControlSpaceSize = controlVariablesSizes[1]*controlVariablesSizes[2]
    monteCarloSize = 10

    model = StochDynProgModel(N_STAGES-1,
                    N_CONTROLS,
                    N_STATES,
                    N_NOISES,
                    x_bounds,
                    u_bounds,
                    x0,
                    cost_t_HD,
                    finalCostFunction,
                    dynamic_HD,
                    constraints,
                    aleas)

    params = SDPparameters(stateSteps, controlSteps, totalStateSpaceSize,
                            totalControlSpaceSize, stateVariablesSizes,
                            controlVariablesSizes, monteCarloSize, infoStruct)

    return model, params
end


function init_problem_sdp_DH()

    x0 = X0
    aleas = generate_probability_laws()

    x_bounds = [(VOLUME_MIN, VOLUME_MAX)]
    u_bounds = [(CONTROL_MIN, CONTROL_MAX)]

    N_CONTROLS = 1
    N_STATES = 1
    N_NOISES = 1
    infoStruct = "DH"

    stateSteps = [1]
    controlSteps = [1]
    stateVariablesSizes = [(VOLUME_MAX-VOLUME_MIN)+1]
    controlVariablesSizes = [(CONTROL_MAX-CONTROL_MIN)+1]
    totalStateSpaceSize = stateVariablesSizes[1]
    totalControlSpaceSize = controlVariablesSizes[1]
    monteCarloSize = 10

    model = StochDynProgModel(N_STAGES-1,
                    N_CONTROLS,
                    N_STATES,
                    N_NOISES,
                    x_bounds,
                    u_bounds,
                    x0,
                    cost_t_DH,
                    finalCostFunction,
                    dynamic_DH,
                    constraints,
                    aleas)

    params = SDPparameters(stateSteps, controlSteps, totalStateSpaceSize,
                            totalControlSpaceSize, stateVariablesSizes,
                            controlVariablesSizes, monteCarloSize, infoStruct)

    return model, params
end


"""Solve the problem."""
function solve_dams_sdp_DH(display=false)
    model, params = init_problem_sdp_DH()

    law = model.noises

    V = sdp_optimize(model, params, display)

    scenar = Array(Array, N_STAGES-1)

    for t in 1:(N_STAGES-1)
        scenar[t] = sampling(law, t)
    end

    costs, stocks, controls = sdp_forward_simulation(model, params,
                                                        scenar, X0, V, true)

    println("SDP DH cost: ", costs)
    return costs, stocks, controls, scenar, V
end

function solve_dams_sdp_HD(display=false)
    model, params = init_problem_sdp_HD()

    law = model.noises

    V = sdp_optimize(model, params, display)

    scenar = Array(Array, (N_STAGES-1))

    for t in 1:(N_STAGES-1)
        scenar[t] = sampling(law, t)
    end

    costs, stocks, controls = sdp_forward_simulation(model, params,
                                                        scenar, X0, V, true)

    println("SDP HD cost: ", costs)
    return costs, stocks, controls, scenar, V
end


function compare_sdp_DH_HD(display=false)
    modelHD, paramsHD = init_problem_sdp_HD()
    modelDH, paramsDH = init_problem_sdp_DH()

    law = modelHD.noises

    scenar = Array(Array, (N_STAGES-1))

    for t in 1:(N_STAGES-1)
        scenar[t] = sampling(law, t)
    end

    VHD = sdp_optimize(modelHD, paramsHD, display)

    costHD, stocksHD, controlsHD = sdp_forward_simulation(modelHD, paramsHD,
                                                            scenar, X0, VHD, true)

    VDH = sdp_optimize(modelDH, paramsDH, display)

    costDH, stocksDH, controlsDH = sdp_forward_simulation(modelDH, paramsDH,
                                                            scenar, X0, VDH, true)

    return costHD, stocksHD, controlsHD, costDH, stocksDH, controlsDH, scenar
end

function init_problem()

    x0 = X0
    aleas = generate_probability_laws()

    x_bounds = [(VOLUME_MIN, VOLUME_MAX)]
    u_bounds = [(CONTROL_MIN, CONTROL_MAX), (0, VOLUME_MAX)]

    N_CONTROLS = 2
    N_STATES = 1
    N_ALEAS = 1

    model = LinearDynamicLinearCostSPmodel(N_STAGES,
                                                u_bounds,
                                                x0,
                                                cost_t,
                                                dynamic,
                                                aleas)

    set_state_bounds(model, x_bounds);

    solver = SOLVER

    params = SDDPparameters(solver, N_SCENARIOS, EPSILON, MAX_ITER)

    return model, params
end

 function solve_dams(display=false)
     model, params = init_problem()

     V, pbs = solve_SDDP(model, params, display)

     params.forwardPassNumber = 1

     aleas = simulate_scenarios(model.noises,
                               (model.stageNumber,
                                params.forwardPassNumber,
                                model.dimNoises))

     costs, stocks, controls = forward_simulations(model, params, V, pbs, aleas)

     println("SDDP cost: ", costs)
     return stocks, V, controls, aleas
 end


 function compare_SDDP_SDP(display=false)

    modelHD, paramsHD = init_problem_sdp_HD()
    modelDH, paramsDH = init_problem_sdp_DH()
    model, params = init_problem()

    params.forwardPassNumber = 1

    aleas = simulate_scenarios(model.noises,
                               (model.stageNumber-1,
                                params.forwardPassNumber,
                                model.dimNoises))

    V, pbs = solve_SDDP(model, params, display)

    costs, stocks, controls = forward_simulations(model, params, V, pbs, aleas)

    VHD = sdp_optimize(modelHD, paramsHD, display)

    costHD, stocksHD, controlsHD = sdp_forward_simulation(modelHD, paramsHD,
                                                            aleas, X0, VHD, true)

    VDH = sdp_optimize(modelDH, paramsDH, display)

    costDH, stocksDH, controlsDH = sdp_forward_simulation(modelDH, paramsDH,
                                                            aleas, X0, VDH, true)

    return costs, stocks, controls, aleas, costHD, stocksHD, controlsHD, costDH, stocksDH, controlsDH
end
