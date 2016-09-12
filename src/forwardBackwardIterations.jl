#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Define the Forward / Backward iterations of the SDDP algorithm
#############################################################################


"""
Make a forward pass of the algorithm

# Description
Simulate a scenario of noise and compute an optimal trajectory on this
scenario according to the current value functions.

# Arguments
* `model::SPmodel`: the stochastic problem we want to optimize
* `param::SDDPparameters`: the parameters of the SDDP algorithm
* `solverProblems::Array{JuMP.Model}`:
    Linear model used to approximate each value function
* `xi::Array{float}`:
    the noise scenarios on which we simulate, each column being one scenario :
    xi[t,k,:] is the alea at time t of scenario k.

# Returns
* `costs::Array{float,1}`:
    an array of the simulated costs
    If returnCosts=false, return nothing
* `stocks::Array{float}`:
    the simulated stock trajectories. stocks(t,k,:) is the stock for
    scenario k at time t.
* `controls::Array{Float64, 3}`:
    the simulated controls trajectories. controls(t,k,:) is the control for
    scenario k at time t.
* `callsolver::Int64`:
    the number of solver's call'

"""
function forward_simulations(model::SPModel,
                            param::SDDPparameters,
                            solverProblems::Vector{JuMP.Model},
                            xi::Array{Float64})

    callsolver::Int = 0

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

    # Store costs of different scenarios in an array:
    costs = zeros(nb_forward)

    for t=1:T-1
        for k = 1:nb_forward
            # Collect current state and noise:
            state_t = collect(stocks[t, k, :])
            alea_t = collect(xi[t, k, :])

            callsolver += 1

            # Solve optimization problem corresponding to current position:
            if param.IS_ACCELERATED &&  ~isa(model.refTrajectories, Void)
                xp = collect(model.refTrajectories[t+1, k, :])
                status, nextstep = solve_one_step_one_alea(model, param,
                                                           solverProblems[t], t, state_t, alea_t, xp)
            else
                status, nextstep = solve_one_step_one_alea(model, param,
                                                           solverProblems[t], t, state_t, alea_t)
            end

            # Check if the problem is effectively solved:
            if status
                # Get the next position:
                stocks[t+1, k, :] = nextstep.next_state
                # the optimal control just computed:
                opt_control = nextstep.optimal_control
                controls[t, k, :] = opt_control
                # and the current cost:
                costs[k] += nextstep.cost - nextstep.cost_to_go
                if t==T-1
                    costs[k] += nextstep.cost_to_go
                end
            else
                # if problem is not properly solved, next position if equal
                # to current one:
                stocks[t+1, k, :] = state_t
            end
        end
    end
    return costs, stocks, controls, callsolver
end



"""
Add to polyhedral function a cut with shape Vt >= beta + <lambda,.>

# Arguments
* `model::SPModel`: Store the problem definition
* `t::Int64`: Current time
* `Vt::PolyhedralFunction`:
  Current lower approximation of the Bellman function at time t
* `beta::Float`:
  affine part of the cut to add
* `lambda::Array{float,1}`:
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

# Arguments
* `model::SPModel`:
  Store the problem definition
* `problem::JuMP.Model`:
  Linear problem used to approximate the value functions
* `t::Int`:
  Time index
* `beta::Float`:
  affine part of the cut to add
* `lambda::Array{float,1}`:
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

# Description
For t:T-1 -> 0, compute a valid cut of the Bellman function
Vt at the state given by stockTrajectories and add them to
the current estimation of Vt.

# Arguments
* `model::SPmodel`:
    the stochastic problem we want to optimize
* `param::SDDPparameters`:
    the parameters of the SDDP algorithm
* `V::Array{PolyhedralFunction}`:
    the current estimation of Bellman's functions
* `solverProblems::Array{JuMP.Model}`:
    Linear model used to approximate each value function
* `stockTrajectories::Array{Float64,3}`:
    stockTrajectories[t,k,:] is the vector of stock where the cut is computed
    for scenario k and time t.
* `law::Array{NoiseLaw}`:
    Conditionnal distributions of perturbation, for each timestep
"""
function backward_pass!(model::SPModel,
                        param::SDDPparameters,
                        V::Vector{PolyhedralFunction},
                        solverProblems::Vector{JuMP.Model},
                        stockTrajectories::Array{Float64, 3},
                        law)

    callsolver::Int = 0

    T = model.stageNumber
    nb_forward = size(stockTrajectories)[2]

    costs::Vector{Float64} = zeros(1)
    state_t = zeros(Float64, model.dimStates)

    for t = T-1:-1:1
        costs = zeros(Float64, law[t].supportSize)

        for k = 1:nb_forward

            subgradient_array = zeros(Float64, model.dimStates, law[t].supportSize)
            # We collect current state:
            state_t = collect(stockTrajectories[t, k, :])
            # We will store probabilities in a temporary array.
            # It is initialized at 0. If all problem are infeasible for
            # current timestep, then proba remains equal to 0 and not cut is added.
            proba = zeros(law[t].supportSize)

            # We iterate other the possible realization of noise:
            for w in 1:law[t].supportSize

                # We get current noise:
                alea_t  = collect(law[t].support[:, w])

                callsolver += 1

                # We solve LP problem with current noise and position:
                solved, nextstep = solve_one_step_one_alea(model, param,
                                                           solverProblems[t],
                                                           t, state_t, alea_t,
                                                           relaxation=model.IS_SMIP)

                if solved
                    # We catch the subgradient λ:
                    subgradient_array[:, w] = nextstep.sub_gradient
                    # and the current cost:
                    costs[w] = nextstep.cost
                    # and as problem is solved we store current proba in array:
                    proba[w] = law[t].proba[w]
                end
            end

            # We add cuts only if one solution was being found:
            if sum(proba) > 0
                # Scale probability (useful when some problems where infeasible):
                proba /= sum(proba)

                # Compute expectation of subgradient λ:
                subgradient = vec(sum(proba' .* subgradient_array, 2))
                # ... expectation of cost:
                costs_npass = dot(proba, costs)
                # ... and expectation of slope β:
                beta = costs_npass - dot(subgradient, state_t)

                # Add cut to polyhedral function and JuMP model:
                add_cut!(model, t, V[t], beta, subgradient)
                if t > 1
                    add_cut_to_model!(model, solverProblems[t-1], t, beta, subgradient)
                end
            end

        end
    end
    return callsolver
end
