#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Test SDDP with dam example
# Source: Adrien Cassegrain
#############################################################################

#srand(2713)
push!(LOAD_PATH, "../src")

using StochDynamicProgramming, JuMP, Clp

#Constant that the user have to define himself
const SOLVER = ClpSolver()
# const SOLVER = CplexSolver(CPX_PARAM_SIMDISPLAY=0)

const N_STAGES = 5
const N_SCENARIOS = 3

# FINAL TIME:
const TF = N_STAGES

const T0 = 1
const HORIZON = TF

const DIM_STATES = 2
const DIM_CONTROLS = 4
const DIM_ALEAS = 1



#Constants that the user does not have to define
# COST:
const COST = -66*2.7*(1 + .5*(rand(TF) - .5))

# Constants:
const VOLUME_MAX = 1000
const VOLUME_MIN = -1000

const CONTROL_MAX = round(Int, .4/7. * VOLUME_MAX) + 1
const CONTROL_MIN = 0

const W_MAX = round(Int, .5/7. * VOLUME_MAX)
const W_MIN = 0
const DW = 1

# Define aleas' space:
const N_ALEAS = Int(round(Int, (W_MAX - W_MIN) / DW + 1))
const ALEAS = linspace(W_MIN, W_MAX, N_ALEAS)

const EPSILON = .05
const MAX_ITER = 20

alea_year = Array([7.0 7.0 8.0 3.0 1.0 1.0 3.0 4.0 3.0 2.0 6.0 5.0 2.0 6.0 4.0 7.0 3.0 4.0 1.0 1.0 6.0 2.0 2.0 8.0 3.0 7.0 3.0 1.0 4.0 2.0 4.0 1.0 3.0 2.0 8.0 1.0 5.0 5.0 2.0 1.0 6.0 7.0 5.0 1.0 7.0 7.0 7.0 4.0 3.0 2.0 8.0 7.0])


Ax=[]
Au=[]
Aw=[]

Cx=[]
Cu=[]
Cw=[]

const X0 = [50, 50]



for i=1:TF
        push!(Ax, rand(DIM_STATES,DIM_STATES))
        push!(Au, rand(DIM_STATES,DIM_CONTROLS))
        push!(Aw, rand(DIM_STATES,DIM_ALEAS))

        push!(Cx, rand(1,DIM_STATES))
        push!(Cu, rand(1,DIM_CONTROLS))
        push!(Cw, rand(1,DIM_ALEAS))
end


#TODO faire dépendre de t

# Define dynamic of the dam:
function dynamic(t, x, u, w)
    #return [x[1] - u[1] + w[1], x[2] - u[2] + u[1]]
    #return [x[1] - u[1] - u[3] + w[1], x[2] - u[2] - u[4] + u[1] + u[3]]
    return  Ax[t]*x+Au[t]*u+Aw[t]*w
end

# Define cost corresponding to each timestep:
function cost_t(t, x, u, w)
    #return COST[t] * (u[1] + u[2])
    return Cx[t]*x+Cu[t]*u+Cw[t]*w
end


"""Build aleas probabilities for each month."""
function build_aleas()
    aleas = zeros(N_ALEAS, TF)

    # take into account seasonality effects:
    unorm_prob = linspace(1, N_ALEAS, N_ALEAS)
    proba1 = unorm_prob / sum(unorm_prob)
    proba2 = proba1[N_ALEAS:-1:1]

    for t in 1:TF
        aleas[:, t] = (1 - sin(pi*t/TF)) * proba1 + sin(pi*t/TF) * proba2
    end
    return aleas
end


"""Build an admissible scenario for water inflow."""
function build_scenarios(n_scenarios::Int64, probabilities)
    scenarios = zeros(n_scenarios, TF)

    for scen in 1:n_scenarios
        for t in 1:TF
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

    for t=1:N_STAGES
        laws[t] = NoiseLaw(aleas[:, t], proba)
    end

    return laws
end


"""Instantiate the problem."""
function init_problem()

    x0 = X0
    aleas = generate_probability_laws()

    x_bounds = [(VOLUME_MIN, VOLUME_MAX), (VOLUME_MIN, VOLUME_MAX)]
    u_bounds = [(CONTROL_MIN, CONTROL_MAX), (CONTROL_MIN, CONTROL_MAX), (0, Inf), (0, Inf)]

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

model, params = init_problem()

"""Solve the problem."""
function solve_dams(model,params,display=false)

    #model, params = init_problem()

    V, pbs = solve_SDDP(model, params, display)

    aleas = simulate_scenarios(model.noises,
                              (model.stageNumber,
                               params.forwardPassNumber,
                               model.dimNoises))

    params.forwardPassNumber = 1

    costs, stocks = forward_simulations(model, params, V, pbs, aleas)
    println("SDDP cost: ", costs)
    return stocks, V
end



unsolve = true
sol = 0
i = 0
nb_iter = 10

while i<nb_iter
    sol, status = extensive_formulation(model,params)
    if (status != :Infeasible)
        i = nb_iter;
        unsolve = false;
    end
    Ax = rand(DIM_STATES,DIM_STATES)
    Au = rand(DIM_STATES,DIM_CONTROLS)
    Aw = rand(DIM_STATES,DIM_ALEAS)

    Cx = rand(1,DIM_STATES)
    Cu = rand(1,DIM_CONTROLS)
    Cw = rand(1,DIM_ALEAS)
    println("i =", i)
    i = i+1
end

if unsolve
    println("Change your parameters")
else
    println("solution =",sol) 
end

