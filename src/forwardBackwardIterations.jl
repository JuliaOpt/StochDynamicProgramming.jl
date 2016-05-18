#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Define the Forward / Backward iterations of the SDDP algorithm
#############################################################################


"""
Make a forward pass of the algorithm

Simulate a scenario of noise and compute an optimal trajectory on this
scenario according to the current value functions.

Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize

- param (SDDPparameters)
    the parameters of the SDDP algorithm

- V (PolyhedralFunction)
    the current estimation of Bellman's functions

- solverProblems (Array{JuMP.Model})
    Linear model used to approximate each value function

- xi (Array{float})
    the noise scenarios on which we simulate, each column being one scenario :
    xi[t,k,:] is the alea at time t of scenario k.

- returnCosts (Bool)
    return the cost of each simulated scenario if true

- init (Bool)
    Specify if the problem must be initialized
    (ie cuts are empty)

- display (Bool)
    If specified, display results in shell


Returns (according to the last parameters):
- costs (Array{float,1})
    an array of the simulated costs
    If returnCosts=false, return nothing

- stocks (Array{float})
    the simulated stock trajectories. stocks(t,k,:) is the stock for
    scenario k at time t.

- controls (Array{Float64, 3})
    the simulated controls trajectories. controls(t,k,:) is the control for
    scenario k at time t.
"""
function forward_simulations(model::SPModel,
                            param::SDDPparameters,
                            V::Vector{PolyhedralFunction},
                            solverProblems::Vector{JuMP.Model},
                            xi::Array{Float64},
                            returnCosts=true::Bool,
                            init=false::Bool,
                            display=false::Bool)

    T = model.stageNumber
    nb_forward = size(xi)[2]

    if ndims(xi)!=3
        if ndims(xi)==2
            warn("noise scenario are not given in the right shape. Assumed to be real valued noise.")
            xi = reshape(xi,(T,nb_forward,1))
        else
            error("wrong dimension of noise scenarios")
        end
     end

    stocks = zeros(T, nb_forward, model.dimStates)
    # We got T - 1 control, as terminal state is included into the total number
    # of stages.
    controls = zeros(T - 1, nb_forward, model.dimControls)

    # Set first value of stocks equal to x0:
    for k in 1:nb_forward
        stocks[1, k, :] = model.initialState
    end

    costs = nothing
    if returnCosts
        costs = zeros(nb_forward)
    end

    for t=1:T-1
        for k = 1:nb_forward

            state_t = extract_vector_from_3Dmatrix(stocks, t, k)
            alea_t = extract_vector_from_3Dmatrix(xi, t, k)

            status, nextstep = solve_one_step_one_alea(
                                            model,
                                            param,
                                            solverProblems[t],
                                            t,
                                            state_t,
                                            alea_t,
                                            init)

            stocks[t+1, k, :] = nextstep.next_state
            opt_control = nextstep.optimal_control
            controls[t, k, :] = opt_control

            if returnCosts
                costs[k] += nextstep.cost - nextstep.cost_to_go
                if t==T-1
                    costs[k] += nextstep.cost_to_go
                end
            end
        end
    end
    return costs, stocks, controls
end



"""
Add to polyhedral function a cut with shape Vt >= beta + <lambda,.>

Parameters:
- model (SPModel)
Store the problem definition

- t (Int64)
Current time

- Vt (PolyhedralFunction)
Current lower approximation of the Bellman function at time t

- beta (Float)
affine part of the cut to add

- lambda (Array{float,1})
subgradient of the cut to add

"""
function add_cut!(model::SPModel,
    t::Int64, Vt::PolyhedralFunction,
    beta::Float64, lambda::Vector{Float64})
    Vt.lambdas = vcat(Vt.lambdas, reshape(lambda, 1, model.dimStates))
    Vt.betas = vcat(Vt.betas, beta)
    Vt.numCuts += 1
end


"""
Add a cut to the JuMP linear problem.

Parameters:
- model (SPModel)
Store the problem definition

- problem (JuMP.Model)
Linear problem used to approximate the value functions

- t (Int)
Time index

- beta (Float)
affine part of the cut to add

- lambda (Array{float,1})
subgradient of the cut to add

"""
function add_cut_to_model!(model::SPModel, problem::JuMP.Model,
    t::Int64, beta::Float64, lambda::Vector{Float64})
    alpha = getvariable(problem, :alpha)
    x = getvariable(problem, :x)
    u = getvariable(problem, :u)
    w = getvariable(problem, :w)
    @constraint(problem, beta + dot(lambda, model.dynamics(t, x, u, w)) <= alpha)
end


"""
Make a backward pass of the algorithm

For t:T-1 -> 0, compute a valid cut of the Bellman function
Vt at the state given by stockTrajectories and add them to
the current estimation of Vt.


Parameters:
- model (SPmodel)
the stochastic problem we want to optimize

- param (SDDPparameters)
the parameters of the SDDP algorithm

- V (Array{PolyhedralFunction})
the current estimation of Bellman's functions

- solverProblems (Array{JuMP.Model})
Linear model used to approximate each value function

- stockTrajectories (Array{Float64,3})
stockTrajectories[t,k,:] is the vector of stock where the cut is computed
for scenario k and time t.

- law (Array{NoiseLaw})
Conditionnal distributions of perturbation, for each timestep

- init (Bool)
If specified, then init PolyhedralFunction

- updateV (Bool)
Store new cuts in given Polyhedral functions if specified


Return:
- V0 (Float64)
Approximation of initial cost

"""
function backward_pass!(model::SPModel,
    param::SDDPparameters,
    V::Vector{PolyhedralFunction},
    solverProblems::Vector{JuMP.Model},
    stockTrajectories::Array{Float64, 3},
    law,
    init=false::Bool)

    T = model.stageNumber
    nb_forward = size(stockTrajectories)[2]

    costs::Vector{Float64} = zeros(1)
    costs_npass = zeros(Float64, nb_forward)
    state_t = zeros(Float64, model.dimStates)

    for t = T-1:-1:1
        costs = zeros(law[t].supportSize)

        for k = 1:nb_forward

            subgradient_array = zeros(Float64, model.dimStates, law[t].supportSize)
            state_t = extract_vector_from_3Dmatrix(stockTrajectories, t, k)

            for w in 1:law[t].supportSize

                alea_t  = collect(law[t].support[:, w])

                nextstep = solve_one_step_one_alea(model, param, solverProblems[t], t, state_t, alea_t)[2]
                subgradient_array[:, w] = nextstep.sub_gradient
                costs[w] = nextstep.cost
            end

            # Compute expectation of subgradient:
            subgradient = vec(sum(law[t].proba' .* subgradient_array, 2))
            # ... and esperancy of cost:
            costs_npass[k] = dot(law[t].proba, costs)
            beta = costs_npass[k] - dot(subgradient, state_t)

            # Add cut to polyhedral function and JuMP model:
            if init
                V[t] = PolyhedralFunction([beta], reshape(subgradient, 1, model.dimStates), 1)
                if t > 1
                    add_cut_to_model!(model, solverProblems[t-1], t, beta, subgradient)
                end
            else
                add_cut!(model, t, V[t], beta, subgradient)
                if t > 1
                    add_cut_to_model!(model, solverProblems[t-1], t, beta, subgradient)
                end
            end

        end
    end
end
