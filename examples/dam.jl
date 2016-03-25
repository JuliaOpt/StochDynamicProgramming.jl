#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Test SDDP with dam example
# Source: Adrien Cassegrain
#############################################################################

push!(LOAD_PATH, "../src")


using StochDynamicProgramming, JuMP, Clp

const SOLVER = ClpSolver()

const EPSILON = .05
const MAX_ITER = 20

# N_STAGES is the number of stage for states, including terminal state.
const N_STAGES = 53
const N_SCENARIOS = 10

alea_year = Array([7.0 7.0 8.0 3.0 1.0 1.0 3.0 4.0 3.0 2.0 6.0 5.0 2.0 6.0 4.0 7.0 3.0 4.0 1.0 1.0 6.0 2.0 2.0 8.0 3.0 7.0 3.0 1.0 4.0 2.0 4.0 1.0 3.0 2.0 8.0 1.0 5.0 5.0 2.0 1.0 6.0 7.0 5.0 1.0 7.0 7.0 7.0 4.0 3.0 2.0 8.0 7.0])


# COST:
const COST = -66*2.7*(1 + .5*(rand(N_STAGES) - .5))

# Constants:
const VOLUME_MAX = 100
const VOLUME_MIN = 0

const CONTROL_MAX = round(Int, .4/7. * VOLUME_MAX) + 1
const CONTROL_MIN = 0

const W_MAX = round(Int, .5/7. * VOLUME_MAX)
const W_MIN = 0
const DW = 1

const T0 = 1
const HORIZON = 52

const X0 = [90]

# Define aleas' space:
const N_ALEAS = Int(round(Int, (W_MAX - W_MIN) / DW + 1))
const ALEAS = linspace(W_MIN, W_MAX, N_ALEAS)


# Define dynamic of the dam:
function dynamic(t, x, u, w)
    return [x[1] - u[1] - u[2] + w[1]]
end

# Define cost corresponding to each timestep:
function cost_t(t, x, u, w)
    return COST[t] * u[1]
end


"""Solve the problem with a solver, supposing the aleas are known
in advance."""
function solve_determinist_problem()
    println(alea_year)
    m = Model(solver=SOLVER)

    @defVar(m,  0           <= x[1:N_STAGES]  <= 100)
    @defVar(m,  0.          <= u[1:N_STAGES-1]  <= 7)
    @defVar(m,  0.          <= s[1:N_STAGES-1]  <= 7)

    @setObjective(m, Min, sum{COST[i]*u[i], i = 1:N_STAGES-1})

    for i in 1:(N_STAGES-1)
        @addConstraint(m, x[i+1] - x[i] + u[i] + s[i] - alea_year[i] == 0)
    end

    @addConstraint(m, x[1] .==X0)

    status = solve(m)
    println(status)
    println(getObjectiveValue(m))
    return getValue(u), getValue(x)
end


"""Build aleas probabilities for each month."""
function build_aleas()
    aleas = zeros(N_ALEAS, N_STAGES)

    # take into account seasonality effects:
    unorm_prob = linspace(1, N_ALEAS, N_ALEAS)
    proba1 = unorm_prob / sum(unorm_prob)
    proba2 = proba1[N_ALEAS:-1:1]

    for t in 1:(N_STAGES-1)
        aleas[:, t] = (1 - sin(pi*t/N_STAGES)) * proba1 + sin(pi*t/N_STAGES) * proba2
    end
    return aleas
end


"""Build an admissible scenario for water inflow."""
function build_scenarios(n_scenarios::Int64, probabilities)
    scenarios = zeros(n_scenarios, N_STAGES)

    for scen in 1:n_scenarios
        for t in 1:(N_STAGES-1)
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

    for t=1:(N_STAGES-1)
        laws[t] = NoiseLaw(aleas[:, t], proba)
    end

    return laws
end


"""Instantiate the problem."""
function init_problem()
    # Instantiate model:
    x0 = X0
    aleas = generate_probability_laws()

    x_bounds = [(0, 100)]
    u_bounds = [(0, 7), (0, 7)]
    model = StochDynamicProgramming.LinearDynamicLinearCostSPmodel(N_STAGES,
                                                u_bounds,
                                                x0,
                                                cost_t,
                                                dynamic, aleas)

    set_state_bounds(model, x_bounds)

    solver = SOLVER
    params = StochDynamicProgramming.SDDPparameters(solver, N_SCENARIOS, EPSILON, MAX_ITER)

    return model, params
end


"""Solve the problem."""
function solve_dams(display=false)
    model, params = init_problem()

    V, pbs = solve_SDDP(model, params, display)
    aleas = simulate_scenarios(model.noises ,(model.stageNumber-1,
                               params.forwardPassNumber , model.dimNoises))
    params.forwardPassNumber = 1

    costs, stocks = forward_simulations(model, params, V, pbs, aleas)

    println("SDDP cost: ", costs)
    return stocks
end
