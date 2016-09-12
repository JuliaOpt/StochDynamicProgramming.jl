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

# Description
Alternate forward and backward phase till the stopping criterion is
fulfilled.

# Arguments
* `model::SPmodel`:
    the stochastic problem we want to optimize
* `param::SDDPparameters`:
    the parameters of the SDDP algorithm
* `verbose::Int64`:
    Default is `0`
    If non null, display progression in terminal every
    `n` iterations, where `n` is the number specified by display.

# Returns
* `V::Array{PolyhedralFunction}`:
    the collection of approximation of the bellman functions
* `problems::Array{JuMP.Model}`:
    the collection of linear problems used to approximate
    each value function
* `count_callsolver::Int64`:
    number of times the solver has been called

"""
function solve_SDDP(model::SPModel, param::SDDPparameters, verbose=0::Int64)
    if model.IS_SMIP && isa(param.MIPSOLVER, Void)
        error("MIP solver is not defined. Please set `param.MIPSOLVER`")
    end
    (model.IS_SMIP && param.IS_ACCELERATED) && error("Acceleration of SMIP not supported")
    (verbose > 0) && (model.IS_SMIP) && println("SMIP SDDP")
    (verbose > 0) && (param.IS_ACCELERATED) && println("Acceleration: ON")

    # initialize value functions:
    V, problems = initialize_value_functions(model, param)
    (verbose > 0) && println("Initial value function loaded into memory.")
    # Run SDDP:
    sddp_stats = run_SDDP!(model, param, V, problems, verbose)
    return V, problems, sddp_stats
end


function solve_SDDP(model::SPModel, param::SDDPparameters, V::Vector{PolyhedralFunction}, verbose=0::Int64)
    if model.IS_SMIP && isa(param.MIPSOLVER, Void)
        error("MIP solver is not defined. Please set `param.MIPSOLVER`")
    end
    (model.IS_SMIP && param.IS_ACCELERATED) && error("Acceleration of SMIP not supported")

    # First step: process value functions if hotstart is called
    problems = hotstart_SDDP(model, param, V)
    sddp_stats = run_SDDP!(model, param, V, problems, verbose)
    return V, problems, sddp_stats
end


"""Run SDDP iterations."""
function run_SDDP!(model::SPModel,
                    param::SDDPparameters,
                    V::Vector{PolyhedralFunction},
                    problems::Vector{JuMP.Model},
                    verbose=0::Int64)

    #Initialization of the counter
    stats = SDDPStat(0, [], [], [], 0)

    (verbose > 0) && println("Initialize cuts")

    # If computation of upper-bound is needed, a set of scenarios is built
    # to keep always the same realization for upper bound estimation:
    if param.compute_upper_bound > 0
        upperbound_scenarios = simulate_scenarios(model.noises, param.monteCarloSize)
    end

    upb = Inf
    costs = nothing
    stopping_test::Bool = false
    iteration_count::Int64 = 0

    # Launch execution of forward and backward passes:
    while (iteration_count < param.maxItNumber) & (~stopping_test)
        # Time execution of current pass:
        tic()

        # Draw a set of scenarios according to the probability
        # law specified in model.noises:
        noise_scenarios = simulate_scenarios(model.noises, param.forwardPassNumber)

        ####################
        # Forward pass
        # If acceleration is ON, need to build a new array of problem to
        # avoid side effect:
        problems_fp = (param.IS_ACCELERATED)? hotstart_SDDP(model, param, V):problems
        _, stockTrajectories,_,callsolver_forward = forward_simulations(model,
                            param,
                            problems_fp,
                            noise_scenarios)

        # we store these trajectories in model:
        model.refTrajectories = stockTrajectories

        ####################
        # Backward pass
        callsolver_backward = backward_pass!(model,
                      param,
                      V,
                      problems,
                      stockTrajectories,
                      model.noises)

        # Update the number of solver call
        stats.ncallsolver += callsolver_forward + callsolver_backward
        iteration_count += 1
        stats.niterations += 1

        ####################
        # If specified, prune cuts:
        if (param.compute_cuts_pruning > 0) && (iteration_count%param.compute_cuts_pruning==0)
            (verbose > 0) && println("Prune cuts ...")
            remove_redundant_cuts!(V)
            prune_cuts!(model, param, V)
            problems = hotstart_SDDP(model, param, V)
        end

        # Update estimation of lower bound:
        lwb = get_bellman_value(model, param, 1, V[1], model.initialState)
        push!(stats.lower_bounds, lwb)

        ####################
        # If specified, compute upper-bound:
        if (param.compute_upper_bound > 0) && (iteration_count%param.compute_upper_bound==0)
            (verbose > 0) && println("Compute upper-bound with ",
                                      param.monteCarloSize, " scenarios...")
            # estimate upper-bound with Monte-Carlo estimation:
            upb, costs = estimate_upper_bound(model, param, upperbound_scenarios, problems)
            if param.gap > 0.
                stopping_test = test_stopping_criterion(lwb, upb, param.gap)
            end
        end

        push!(stats.exectime, toq())
        push!(stats.upper_bounds, upb)

        if param.IS_ACCELERATED
            # If accelerated, need to update penalization coefficient:
            param.acceleration[:rho] *= param.acceleration[:alpha]
        end

        if (verbose > 0) && (iteration_count%verbose==0)
            print("Pass number ", iteration_count)
            (upb < Inf) && print("\tUpper-bound: ", upb)
            println("\tLower-bound: ", round(stats.lower_bounds[end], 4),
                "\tTime: ", round(stats.exectime[end], 2),"s")
        end

    end

    ##########
    # Estimate final upper bound with param.monteCarloSize simulations:
    if (verbose>0) && (param.compute_upper_bound >= 0)
        V0 = get_bellman_value(model, param, 1, V[1], model.initialState)

        if param.compute_upper_bound == 0
            println("Estimate upper-bound with Monte-Carlo ...")
            upb, costs = estimate_upper_bound(model, param, V, problems, param.monteCarloSize)
        end

        println("Estimation of upper-bound: ", round(upb,4),
                "\tExact lower bound: ", round(V0,4),
                "\t Gap <  ", round(100*(upb-V0)/V0, 2) , "\%  with prob. > 97.5 \%")
        println("Estimation of cost of the solution (fiability 95\%):",
                 round(mean(costs),4), " +/- ", round(1.96*std(costs)/sqrt(length(costs)),4))
    end

    return stats
end


"""
Estimate upper bound with Monte Carlo.

# Arguments
* `model::SPmodel`:
    the stochastic problem we want to optimize
* `param::SDDPparameters`:
    the parameters of the SDDP algorithm
* `V::Array{PolyhedralFunction}`:
    the current estimation of Bellman's functions
* `problems::Array{JuMP.Model}`:
    Linear model used to approximate each value function
* `n_simulation::Float64`:
    Number of scenarios to use to compute Monte-Carlo estimation

# Return
* `upb::Float64`:
    estimation of upper bound
* `costs::Vector{Float64}`:
    Costs along different trajectories
"""
function estimate_upper_bound(model::SPModel, param::SDDPparameters,
                                V::Vector{PolyhedralFunction},
                                problem::Vector{JuMP.Model},
                                n_simulation=1000::Int)
    aleas = simulate_scenarios(model.noises, n_simulation)
    return estimate_upper_bound(model, param, aleas, problem)
end
function estimate_upper_bound(model::SPModel, param::SDDPparameters,
                                aleas::Array{Float64, 3},
                                problem::Vector{JuMP.Model})
    costs = forward_simulations(model, param, problem, aleas)[1]
    return upper_bound(costs), costs
end


"""
Build a cut approximating terminal cost with null function

# Arguments
* `problem::JuMP.Model`:
    Cut approximating the terminal cost
* `shape`:
    If PolyhedralFunction is given, build terminal cost with it
    Else, terminal cost is null
"""
function build_terminal_cost!(model::SPModel, problem::JuMP.Model, Vt::PolyhedralFunction)
    # if shape is PolyhedralFunction, build terminal cost with it:
    alpha = getvariable(problem, :alpha)
    xf = getvariable(problem, :xf)
    t = model.stageNumber -1
    if isa(Vt, PolyhedralFunction)
        for i in 1:Vt.numCuts
            lambda = vec(Vt.lambdas[i, :])
            @constraint(problem, Vt.betas[i] + dot(lambda, xf) <= alpha)
        end
    else
        @constraint(problem, alpha >= 0)
    end
end


"""
Initialize each linear problem used to approximate value  functions

# Description
This function define the variables and the constraints of each
linear problem.

# Arguments
* `model::SPModel`:
    Parametrization of the problem
* `param::SDDPparameters`:
    Parameters of SDDP

# Return
* `Array::JuMP.Model`:
"""
function build_models(model::SPModel, param::SDDPparameters)
    return JuMP.Model[build_model(model, param, t) for t=1:model.stageNumber-1]
end

function build_model(model, param, t)
    m = Model(solver=param.SOLVER)

    nx = model.dimStates
    nu = model.dimControls
    nw = model.dimNoises

    # define variables in JuMP:
    @variable(m,  model.xlim[i][1] <= x[i=1:nx] <= model.xlim[i][2])
    @variable(m,  model.xlim[i][1] <= xf[i=1:nx]<= model.xlim[i][2])
    @variable(m,  model.ulim[i][1] <= u[i=1:nu] <=  model.ulim[i][2])
    @variable(m, alpha)

    @variable(m, w[1:nw] == 0)
    m.ext[:cons] = @constraint(m, state_constraint, x .== 0)

    @constraint(m, xf .== model.dynamics(t, x, u, w))

    # Add equality and inequality constraints:
    if model.equalityConstraints != nothing
        @constraint(m, model.equalityConstraints(t, x, u, w) .== 0)
    end
    if model.inequalityConstraints != nothing
        @constraint(m, model.inequalityConstraints(t, x, u, w) .<= 0)
    end

    # Define objective function (could be linear or piecewise linear)
    if isa(model.costFunctions, Function)
        @objective(m, Min, model.costFunctions(t, x, u, w) + alpha)
    elseif isa(model.costFunctions, Vector{Function})
        @variable(m, cost)

        for i in 1:length(model.costFunctions)
            @constraint(m, cost >= model.costFunctions[i](t, x, u, w))
        end
        @objective(m, Min, cost + alpha)
    end

    # Add binary variable if problem is a SMIP:
    if model.IS_SMIP
        m.colCat[2*nx+1:2*nx+nu] = model.controlCat
    end

    return m
end


"""
Initialize value functions along a given trajectory

# Description
This function add the fist cut to each PolyhedralFunction stored in a Array

# Arguments
* `model::SPModel`:
* `param::SDDPparameters`:

# Return
* `V::Array{PolyhedralFunction}`:
    Return T PolyhedralFunction, where T is the number of stages
    specified in model.
* `problems::Array{JuMP.Model}`:
    the initialization of linear problems used to approximate
    each value function
"""
function initialize_value_functions(model::SPModel,
                                    param::SDDPparameters)

    solverProblems = build_models(model, param)
    V = PolyhedralFunction[
                PolyhedralFunction(model.dimStates) for i in 1:model.stageNumber]

    # Build scenarios according to distribution laws:
    aleas = simulate_scenarios(model.noises, param.forwardPassNumber)

    # Add final costs to solverProblems:
    if isa(model.finalCost, PolyhedralFunction)
        V[end] = model.finalCost
        build_terminal_cost!(model, solverProblems[end], V[end])
    elseif isa(model.finalCost, Function)
        model.finalCost(model, solverProblems[end])
    end

    stockTrajectories = zeros(model.stageNumber, param.forwardPassNumber, model.dimStates)
    for i in 1:model.stageNumber, j in 1:param.forwardPassNumber
        stockTrajectories[i, j, :] = get_random_state(model)
    end

    callsolver = backward_pass!(model, param, V, solverProblems,
                                stockTrajectories, model.noises)

    return V, solverProblems
end


"""
Initialize JuMP.Model vector with a previously computed PolyhedralFunction
vector.

# Arguments
* `model::SPModel`:
    Parametrization of the problem
* `param::SDDPparameters`:
    Parameters of SDDP
* `V::Vector{PolyhedralFunction}`:
    Estimation of bellman functions as Polyhedral functions

# Return
* `Vector{JuMP.Model}`
"""
function hotstart_SDDP(model::SPModel, param::SDDPparameters, V::Vector{PolyhedralFunction})

    solverProblems = build_models(model, param)

    # Add corresponding cuts to each problem:
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

# Arguments
* `model::SPModel`:
    Parametrization of the problem
* `param::SDDPparameters`:
    Parameters of SDDP
* `t::Int64`:
    Time t where to solve bellman value
* `Vt::Polyhedral function`:
    Estimation of bellman function as Polyhedral function
* `xt::Vector{Float64}`:
    Point where to compute Bellman value.

# Return
Bellman value (Float64)
"""
function get_bellman_value(model::SPModel, param::SDDPparameters,
                           t::Int64, Vt::PolyhedralFunction, xt::Vector{Float64})

    m = Model(solver=param.SOLVER)
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
Compute lower-bound of the problem at initial time.

# Arguments
* `model::SPModel`:
    Parametrization of the problem
* `param::SDDPparameters`:
    Parameters of SDDP
* `V::Vector{Polyhedral function}`:
    Estimation of bellman function as Polyhedral function

# Return
current lower bound of the problem (Float64)
"""
function get_lower_bound(model::SPModel, param::SDDPparameters,
                            V::Vector{PolyhedralFunction})
    return get_bellman_value(model, param, 1, V[1], model.initialState)
end


"""
Compute optimal control at point xt and time t.

# Arguments
* `model::SPModel`:
    Parametrization of the problem
* `param::SDDPparameters`:
    Parameters of SDDP
* `lpproblem::Vector{JuMP.Model}`:
    Linear problems used to approximate the value functions
* `t::Int64`:
    Time
* `xt::Vector{Float64}`:
    Position where to compute optimal control
* `xi::Vector{Float64}`:
    Alea at time t

# Return
    `Vector{Float64}`: optimal control at time t
"""
function get_control(model::SPModel, param::SDDPparameters, lpproblem::Vector{JuMP.Model},
                     t::Int, xt::Vector{Float64}, xi::Vector{Float64})
    return solve_one_step_one_alea(model, param, lpproblem[t], t, xt, xi)[2].optimal_control
end


"""
Add several cuts to JuMP.Model from a PolyhedralFunction

# Arguments
* `model::SPModel`:
    Store the problem definition
* `t::Int`:
    Time index
* `problem::JuMP.Model`:
    Linear problem used to approximate the value functions
* `V::PolyhedralFunction`:
    Cuts are stored in V
"""
function add_cuts_to_model!(model::SPModel, t::Int64, problem::JuMP.Model, V::PolyhedralFunction)
    alpha = getvariable(problem, :alpha)
    xf = getvariable(problem, :xf)

    for i in 1:V.numCuts
        lambda = vec(V.lambdas[i, :])
        @constraint(problem, V.betas[i] + dot(lambda, xf) <= alpha)
    end
end


"""
Exact pruning of all polyhedral functions in input array.

# Arguments
* `model::SPModel`:
* `params::SDDPparameters`:
* `Vector{PolyhedralFunction}`:
    Polyhedral functions where cuts will be removed
"""
function prune_cuts!(model::SPModel, params::SDDPparameters, V::Vector{PolyhedralFunction})
    for i in 1:length(V)-1
        V[i] = exact_prune_cuts(model, params, V[i])
    end
end


"""
Remove useless cuts in PolyhedralFunction.

# Arguments
* `model::SPModel`:
* `params::SDDPparameters`:
* `V::PolyhedralFunction`:
    Polyhedral function where cuts will be removed

# Return
* `PolyhedralFunction`: pruned polyhedral function
"""
function exact_prune_cuts(model::SPModel, params::SDDPparameters, V::PolyhedralFunction)
    ncuts = V.numCuts
    # Find all active cuts:
    if ncuts > 1
        active_cuts = Bool[is_cut_relevant(model, i, V, params.SOLVER) for i=1:ncuts]
        return PolyhedralFunction(V.betas[active_cuts], V.lambdas[active_cuts, :], sum(active_cuts))
    else
        return V
    end
end


"""
Test whether the cut number k is relevant to define polyhedral function Vt.

# Arguments
* `model::SPModel`:
* `k::Int`:
    Position of cut to test in PolyhedralFunction object
* `Vt::PolyhedralFunction`:
    Object storing all cuts
* `solver`:
    Solver to use to solve linear problem
* `epsilon::Float64`: default is `1e-5`
    Acceptable tolerance to test cuts relevantness

# Return
* `Bool`: true if the cut is useful in the definition, false otherwise
"""
function is_cut_relevant(model::SPModel, k::Int, Vt::PolyhedralFunction, solver; epsilon=1e-5)
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
    sol = getobjectivevalue(m)
    return sol < epsilon
end

