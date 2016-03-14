#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Test SDDP with dam example
# Source: Adrien Cassegrain
#############################################################################

srand(2713)
push!(LOAD_PATH, "../src")

using StochDynamicProgramming, JuMP, Clp


const SOLVER = ClpSolver()
# const SOLVER = CplexSolver(CPX_PARAM_SIMDISPLAY=0)

const EPSILON = .05
const MAX_ITER = 20

const N_STAGES = 2
const N_SCENARIOS = 10

# COST:
const COST = -66*2.7*(1 + .5*(rand(N_STAGES) - .5))

# Constants:
const VOLUME_MAX = 20
const VOLUME_MIN = 0

const CONTROL_MAX = 15
const CONTROL_MIN = 0

const W_MAX = 10
const W_MIN = 0
const DW = 1

const T0 = 1
const HORIZON = (N_STAGES-1)

# Define aleas' space:
const N_ALEAS = Int(round(Int, (W_MAX - W_MIN) / DW + 1))
const ALEAS = linspace(W_MIN, W_MAX, N_ALEAS)

N_CONTROLS = 4
N_STATES = 2
N_NOISES = 1


const X0 = [15, 15]

# Define dynamic of the dam:
function dynamic(t, x, u, w)
    #return [x[1] - u[1] + w[1], x[2] - u[2] + u[1]]
    return [x[1] - u[1] - u[3] + w[1], x[2] - u[2] - u[4] + u[1] + u[3]]
end

# Define cost corresponding to each timestep:
function cost_t(t, x, u, w)
    return COST[t] * (u[1] + u[2])
end

function final_cost(x)
    return 0.
end

function constraints(t, x, u, w)

    Bu = (x[1]<=VOLUME_MAX)&(x[2]<=VOLUME_MAX)
    Bl = (x[1]>=VOLUME_MIN)&(x[2]>=VOLUME_MIN)

    return Bu&Bl

end

"""Solve the deterministic problem using a solver, assuming the aleas are known
in advance."""
function solve_determinist_problem()
    m = Model(solver=SOLVER)


    @defVar(m,  VOLUME_MIN  <= x1[1:N_STAGES]  <= VOLUME_MAX)
    @defVar(m,  VOLUME_MIN  <= x2[1:N_STAGES]  <= VOLUME_MAX)
    @defVar(m,  CONTROL_MIN <= u1[1:(N_STAGES-1)]  <= CONTROL_MAX)
    @defVar(m,  CONTROL_MIN <= u2[1:(N_STAGES-1)]  <= CONTROL_MAX)

    @setObjective(m, Min, sum{COST[i]*(u1[i] + u2[i]), i = 1:(N_STAGES-1)})

    for i in 1:(N_STAGES-1)
        @addConstraint(m, x1[i+1] - x1[i] + u1[i] - alea_year[i] == 0)
        @addConstraint(m, x2[i+1] - x2[i] + u2[i] - u1[i] == 0)
    end

    @addConstraint(m, x1[1] == X0[1])
    @addConstraint(m, x2[1] == X0[2])

    status = solve(m)
    println(status)
    println(getObjectiveValue(m))
    return getValue(u1), getValue(x1), getValue(x2)
end


"""Build aleas probabilities for each month."""
function build_aleas()
    aleas = zeros(N_ALEAS, N_STAGES)

    # take into account seasonality effects:
    unorm_prob = linspace(1, N_ALEAS, N_ALEAS)
    proba1 = unorm_prob / sum(unorm_prob)
    proba2 = proba1[N_ALEAS:-1:1]

    for t in 1:(N_STAGES)
        aleas[:, t] = (1 - sin(pi*t/(N_STAGES-1))) * proba1 + sin(pi*t/(N_STAGES-1)) * proba2
    end
    return aleas
end


"""Build an admissible scenario for water inflow."""
function build_scenarios(n_scenarios::Int64, probabilities)
    scenarios = zeros(n_scenarios, (N_STAGES))

    for scen in 1:n_scenarios
        for t in 1:(N_STAGES)
            Pcum = cumsum(probabilities[:, t])

            n_random = rand()
            prob = findfirst(x -> x > n_random, Pcum)
            scenarios[scen, t] = prob
        end
    end
    return scenarios
end


"""Build probability distribution at each timestep.

Return a Vector{NoiseLaw}"""
function generate_probability_laws()
    aleas = build_scenarios(N_SCENARIOS, build_aleas())

    laws = Vector{NoiseLaw}(N_STAGES)

    # uniform probabilities:
    proba = 1/N_SCENARIOS*ones(N_SCENARIOS)

    for t=1:(N_STAGES)
        laws[t] = NoiseLaw(aleas[:, t], proba)
    end

    return laws
end


"""Instantiate the problem."""
function init_SDDP_model()

    x0 = X0
    aleas = generate_probability_laws()

    x_bounds = [(VOLUME_MIN, VOLUME_MAX), (VOLUME_MIN, VOLUME_MAX)]
    u_bounds = [(CONTROL_MIN, CONTROL_MAX), (CONTROL_MIN, CONTROL_MAX), (0, VOLUME_MAX), (0, VOLUME_MAX)]

    model = LinearDynamicLinearCostSPmodel(N_STAGES,
                                                u_bounds,
                                                x0,
                                                cost_t,
                                                dynamic,
                                                aleas)

    set_state_bounds(model, x_bounds)

    solver = SOLVER
    params = SDDPparameters(solver, N_SCENARIOS, EPSILON, MAX_ITER)

    return model, params
end


function init_SDP_model()

    x0 = X0
    aleas = generate_probability_laws()

    x_bounds = [(VOLUME_MIN, VOLUME_MAX), (VOLUME_MIN, VOLUME_MAX)]
    u_bounds = [(CONTROL_MIN, CONTROL_MAX), (CONTROL_MIN, CONTROL_MAX), (0, VOLUME_MAX), (0, VOLUME_MAX)]

    infoStruct = "HD"

    stateSteps = [1, 1]
    controlSteps = [1, 1, 1, 1]
    stateVariablesSizes = [(VOLUME_MAX-VOLUME_MIN)+1, (VOLUME_MAX-VOLUME_MIN)+1]
    controlVariablesSizes = [(CONTROL_MAX-CONTROL_MIN)+1, (CONTROL_MAX-CONTROL_MIN)+1, (VOLUME_MAX)+1, (VOLUME_MAX)+1]
    totalStateSpaceSize = stateVariablesSizes[1] * stateVariablesSizes[2]
    totalControlSpaceSize = controlVariablesSizes[1] * controlVariablesSizes[2] * controlVariablesSizes[3] * controlVariablesSizes[4]
    monteCarloSize = 10

    model = DPSPmodel(N_STAGES-1,
                    N_CONTROLS,
                    N_STATES,
                    N_NOISES,
                    x_bounds,
                    u_bounds,
                    x0,
                    cost_t,
                    final_cost,
                    dynamic,
                    constraints,
                    aleas)

    #params = SDPparameters(stateSteps, controlSteps, totalStateSpaceSize,
    #                        totalControlSpaceSize, stateVariablesSizes,
    #                        controlVariablesSizes, monteCarloSize, infoStruct)


    params1 = SDPparameters(model, stateSteps, controlSteps, monteCarloSize, infoStruct)

    return model, params1
end


"""Solve the problem."""
function solve_dams(display=false)

    model_sddp, params_sddp = init_SDDP_model()
    model_sdp, params_sdp = init_SDP_model()

    V_sddp, pbs = solve_SDDP(model_sddp, params_sddp, display)

    V_sdp = sdp_optimize(model_sdp, params_sdp, display)

    aleas = simulate_scenarios(model_sddp.noises,
                              (model_sddp.stageNumber-1,
                               params_sddp.forwardPassNumber,
                               model_sddp.dimNoises))

    params_sddp.forwardPassNumber = 1

    costs_sddp, stocks_sddp, controls_sddp = forward_simulations(model_sddp, params_sddp, V_sddp, pbs, aleas)

    costs_sdp, stocks_sdp, controls_sdp = sdp_forward_simulation(model_sdp, params_sdp, aleas, X0, V_sdp, true )

    return costs_sddp, stocks_sddp, controls_sddp, costs_sdp, stocks_sdp, controls_sdp
end