#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Implement the SDDP solver and initializers:
#  - functions to initialize value functions
#  - functions to build terminal cost
#############################################################################


"""
Solve SDDP algorithm and return estimation of bellman functions.

Alternate forward and backward phase till the stopping criterion is
fulfilled.


Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize

- param (SDDPparameters)
    the parameters of the SDDP algorithm

- display (Int64) - Default is 0
    If non null, display progression in terminal every
    n iterations, where n is number specified by display.


Returns :
- V (Array{PolyhedralFunction})
    the collection of approximation of the bellman functions

- problems (Array{JuMP.Model})
    the collection of linear problems used to approximate
    each value function

"""
function solve_SDDP(model::SPModel,
                    param::SDDPparameters,
                    display=0::Int64,
                    V=nothing)

    # First step: process value functions if hotstart is called
    if isa(V, Vector{PolyhedralFunction})
        # If V is already specified, then call hotstart:
        problems = hotstart_SDDP(model, param, V)
    else
        # Otherwise, initialize value functions:
        V, problems = initialize_value_functions(model, param)
    end

    # Run SDDP upon example:
    run_SDDP!(model, param, V, problems, display)
    return V, problems
end


"""Run SDDP iterations."""
function run_SDDP!(model::SPModel,
                    param::SDDPparameters,
                    V::Vector{PolyhedralFunction},
                    problems::Vector{JuMP.Model},
                    display=0::Int64)

    # Evaluation of initial cost:
    V0::Float64 = 0

    if display > 0
      println("Initialize cuts")
    end


    stopping_test::Bool = false
    iteration_count::Int64 = 0

    while (iteration_count < param.maxItNumber) & (~stopping_test)
        # Time execution of current pass:
        if display > 0
            tic()
        end

        # Build given number of scenarios according to distribution
        # law specified in model.noises:
        noise_scenarios = simulate_scenarios(model.noises, param.forwardPassNumber)

        # Forward pass
        costs, stockTrajectories, _ = forward_simulations(model,
                            param,
                            V,
                            problems,
                            noise_scenarios)

        # Backward pass
        backward_pass!(model,
                      param,
                      V,
                      problems,
                      stockTrajectories,
                      model.noises,
                      false)

        iteration_count += 1


        if (display > 0) && (iteration_count%display==0)
            println("Pass number ", iteration_count,
                    "\tLower-bound: ", round(get_bellman_value(model, param, 1, V[1], model.initialState),4),
                    "\tTime: ", round(toq(),2),"s")
        end

    end

    # Estimate upper bound with a great number of simulations:
    if (display>0)
        upb = upper_bound(costs)
        V0 = get_bellman_value(model, param, 1, V[1], model.initialState)

        println("Estimate upper-bound with Monte-Carlo ...")
        upb, costs = estimate_upper_bound(model, param, V, problems)
        println("Estimation of upper-bound: ", round(upb,4),
                "\tExact lower bound: ", round(V0,4),
                "\t Gap <  ", round(100*(upb-V0)/V0) , "\%  with prob. > 97.5 \%")
        println("Estimation of cost of the solution (fiability 95\%):",
                 round(mean(costs),4), " +/- ", round(1.96*std(costs)/sqrt(length(costs)),4))
    end
end



"""
Estimate upper bound with Monte Carlo.

Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize

- param (SDDPparameters)
    the parameters of the SDDP algorithm

- V (Array{PolyhedralFunction})
    the current estimation of Bellman's functions

- problems (Array{JuMP.Model})
    Linear model used to approximate each value function

- n_simulation (Float64)
    Number of scenarios to use to compute Monte-Carlo estimation


Return:
- upb (Float64)
    estimation of upper bound

- costs (Vector{Float64})
    Costs along different trajectories

"""
function estimate_upper_bound(model::SPModel, param::SDDPparameters, V::Vector{PolyhedralFunction}, problem::Vector{JuMP.Model}, n_simulation=1000::Int)

    aleas = simulate_scenarios(model.noises, n_simulation)

    costs, stockTrajectories, _ = forward_simulations(model,
                                                        param,
                                                        V,
                                                        problem,
                                                        aleas)

    return upper_bound(costs), costs
end



"""Build a collection of cuts initialized at 0"""
function get_null_value_functions_array(model::SPModel)

    V = Vector{PolyhedralFunction}(model.stageNumber)
    for t = 1:model.stageNumber
        V[t] = PolyhedralFunction(zeros(1), zeros(1, model.dimStates), 1)
    end

    return V
end



"""
Build a cut approximating terminal cost with null function


Parameter:
- problem (JuMP.Model)
    Cut approximating the terminal cost

- shape
    If PolyhedralFunction is given, build terminal cost with it
    Else, terminal cost is null

"""
function build_terminal_cost!(model::SPModel, problem::JuMP.Model, Vt::PolyhedralFunction)
    # if shape is PolyhedralFunction, build terminal cost with it:
    alpha = getvariable(problem, :alpha)
    x = getvariable(problem, :x)
    u = getvariable(problem, :u)
    w = getvariable(problem, :w)
    t = model.stageNumber -1
    if isa(Vt, PolyhedralFunction)
        for i in 1:Vt.numCuts
            lambda = vec(Vt.lambdas[i, :])
            @constraint(problem, Vt.betas[i] + dot(lambda, model.dynamics(t, x, u, w)) <= alpha)
        end
    else
        @constraint(problem, alpha >= 0)
    end
end



"""
Initialize each linear problem used to approximate value  functions

This function define the variables and the constraints of each
linear problem.


Parameter:
- model (SPModel)
    Parametrization of the problem

- param (SDDPparameters)
    Parameters of SDDP


Return:
- Array{JuMP.Model}

"""
function build_models(model::SPModel, param::SDDPparameters)

    models = Vector{JuMP.Model}(model.stageNumber-1)

    for t = 1:model.stageNumber-1
        m = Model(solver=param.solver)

        nx = model.dimStates
        nu = model.dimControls
        nw = model.dimNoises

        @variable(m,  model.xlim[i][1] <= x[i=1:nx] <= model.xlim[i][2])
        @variable(m,  model.ulim[i][1] <= u[i=1:nu] <=  model.ulim[i][2])
        @variable(m,  model.xlim[i][1] <= xf[i=1:nx]<= model.xlim[i][2])
        @variable(m, alpha)

        @variable(m, w[1:nw] == 0)
        m.ext[:cons] = @constraint(m, state_constraint, x .== 0)

        @constraint(m, xf .== model.dynamics(t, x, u, w))

        if typeof(model) == LinearDynamicLinearCostSPmodel
            @objective(m, Min, model.costFunctions(t, x, u, w) + alpha)

        elseif typeof(model) == PiecewiseLinearCostSPmodel
            @variable(m, cost)

            for i in 1:length(model.costFunctions)
                @constraint(m, cost >= model.costFunctions[i](t, x, u, w))
            end
            @objective(m, Min, cost + alpha)
        end

        models[t] = m

    end
    return models
end



"""
Initialize value functions along a given trajectory

This function add the fist cut to each PolyhedralFunction stored in a Array


Parameters:
- model (SPModel)

- param (SDDPparameters)

Return:
- V (Array{PolyhedralFunction})
    Return T PolyhedralFunction, where T is the number of stages
    specified in model.

- problems (Array{JuMP.Model})
    the initialization of linear problems used to approximate
    each value function

"""
function initialize_value_functions( model::SPModel,
                                     param::SDDPparameters)

    solverProblems = build_models(model, param)
    solverProblems_null = build_models(model, param)

    V_null = get_null_value_functions_array(model)
    V = Array{PolyhedralFunction}(model.stageNumber)

    # Build scenarios according to distribution laws:
    aleas = simulate_scenarios(model.noises, param.forwardPassNumber)

    # Add final costs to solverProblems:
    if isa(model.finalCost, PolyhedralFunction)
        V[end] = model.finalCost
        build_terminal_cost!(model, solverProblems[end], V[end])
    elseif isa(model.finalCost, Function)
        model.finalCost(model, solverProblems[end])
    end

    stockTrajectories = forward_simulations(model,
                        param,
                        V_null,
                        solverProblems_null,
                        aleas,
                        false, true, false)[2]

    backward_pass!(model,
                  param,
                  V,
                  solverProblems,
                  stockTrajectories,
                  model.noises,
                  true)

    return V, solverProblems
end


"""
Initialize JuMP.Model vector with a already computed PolyhedralFunction
vector.

Parameters:
- model (SPModel)
    Parametrization of the problem

- param (SDDPparameters)
    Parameters of SDDP

- V (Vector{PolyhedralFunction})
    Estimation of bellman functions as Polyhedral functions

"""
function hotstart_SDDP(model::SPModel, param::SDDPparameters, V::Vector{PolyhedralFunction})

    solverProblems = build_models(model, param)

    for t in 1:model.stageNumber-2
        add_cuts_to_model!(model, t, solverProblems[t], V[t+1])
    end

    # Take care of final cost:
    if isa(model.finalCost, PolyhedralFunction)
        add_cuts_to_model!(model, model.stageNumber-1, solverProblems[end], V[end])
    else
        model.finalCost(model, solverProblems[end])
    end
    return solverProblems
end


"""
Compute value of Bellman function at point xt. Return V_t(xt)

Parameters:
- model (SPModel)
    Parametrization of the problem

- param (SDDPparameters)
    Parameters of SDDP

- t (Int64)
    Time t where to solve bellman value

- Vt (Polyhedral function)
    Estimation of bellman function as Polyhedral function

- xt (Vector{Float64})
    Point where to compute Bellman value.


Return:
Bellman value (Float64)

"""
function get_bellman_value(model::SPModel, param::SDDPparameters,
                           t::Int64, Vt::PolyhedralFunction, xt::Vector{Float64})

    m = Model(solver=param.solver)
    @variable(m, alpha)

    for i in 1:Vt.numCuts
        lambda = vec(Vt.lambdas[i, :])
        @constraint(m, Vt.betas[i] + dot(lambda, xt) <= alpha)
    end

    @objective(m, Min, alpha)
    solve(m)
    return getvalue(alpha)
end


"""
Compute value of Bellman function at point xt. Return V_t(xt)

Parameters:
- model (SPModel)
    Parametrization of the problem

- param (SDDPparameters)
    Parameters of SDDP

- V (Vector{Polyhedral function})
    Estimation of bellman function as Polyhedral function

Return:
current lower bound of the problem (Float64)

"""
function get_lower_bound(model::SPModel, param::SDDPparameters,
                            V::Vector{PolyhedralFunction})
    return get_bellman_value(model, param, 1, V[1], model.initialState)
end


"""
Compute optimal control at point xt and time t.

Parameters:
- model (SPModel)
    Parametrization of the problem

- param (SDDPparameters)
    Parameters of SDDP

- lpproblem (Vector{JuMP.Model})
    Linear problems used to approximate the value functions

- t (Int64)
    Time

- xt (Vector{Float64})
    Position where to compute optimal control

- xi (Vector{Float64})
    Alea at time t

Return:
    Vector{Float64}: optimal control at time t

"""
function get_control(model::SPModel, param::SDDPparameters, lpproblem::Vector{JuMP.Model}, t::Int, xt::Vector{Float64}, xi::Vector{Float64})
    return solve_one_step_one_alea(model, param, lpproblem[t], t, xt, xi)[2].optimal_control
end


"""
Add several cuts to JuMP.Model from a PolyhedralFunction

Parameters:
- model (SPModel)
    Store the problem definition

- t (Int)
    Time index

- problem (JuMP.Model)
    Linear problem used to approximate the value functions


- V (PolyhedralFunction)
    Cuts are stored in V

"""
function add_cuts_to_model!(model::SPModel, t::Int64, problem::JuMP.Model, V::PolyhedralFunction)
    alpha = getvariable(problem, :alpha)
    x = getvariable(problem, :x)
    u = getvariable(problem, :u)
    w = getvariable(problem, :w)

    for i in 1:V.numCuts
        lambda = vec(V.lambdas[i, :])
        @constraint(problem, V.betas[i] + dot(lambda, model.dynamics(t, x, u, w)) <= alpha)
    end
end


"""
Exact pruning of all polyhedral functions in input array.

Parameters:
- model (SPModel)
- params (SDDPparameters)
- V Vector{PolyhedralFunction}
    Polyhedral functions where cuts will be removed

"""
function prune_cuts!(model::SPModel, params::SDDPparameters, V::Vector{PolyhedralFunction})
    for i in 1:length(V)
        V[i] = exact_prune_cuts(model, params, V[i])
    end
end


"""
Remove useless cuts in PolyhedralFunction.

Parameters:
- model (SPModel)
- params (SDDPparameters)
- V (PolyhedralFunction)
    Polyhedral function where cuts will be removed

Return:
- PolyhedralFunction: pruned polyhedral function

"""
function exact_prune_cuts(model::SPModel, params::SDDPparameters, V::PolyhedralFunction)
    ncuts = V.numCuts
    # Find all active cuts:
    if ncuts > 1
        active_cuts = Bool[is_cut_relevant(model, i, V, params.solver) for i=1:ncuts]
        return PolyhedralFunction(V.betas[active_cuts], V.lambdas[active_cuts, :], sum(active_cuts))
    else
        return V
    end
end


"""
Test whether the cut number k is relevant to define polyhedral function Vt.

Parameters:
- model (SPModel)
- k (Int)
    Position of cut to test in PolyhedralFunction object
- Vt (PolyhedralFunction)
    Object storing all cuts
- solver
    Solver to use to solve linear problem

Return:
- Bool: true if the cut is useful in the definition, false otherwise

"""
function is_cut_relevant(model::SPModel, k::Int, Vt::PolyhedralFunction, solver)

    m = Model(solver=solver)
    @variable(m, alpha)
    @variable(m, model.xlim[i][1] <= x[i=1:model.dimStates] <= model.xlim[i][2])

    for i in 1:Vt.numCuts
        if i!=k
            lambda = vec(Vt.lambdas[i, :])
            @constraint(m, Vt.betas[i] + dot(lambda, x) <= alpha)
        end
    end

    λ_k = vec(Vt.lambdas[k, :])
    @objective(m, Min, alpha - dot(λ_k, x) - Vt.betas[k])
    solve(m)
    return getobjectivevalue(m) < 0.
end

