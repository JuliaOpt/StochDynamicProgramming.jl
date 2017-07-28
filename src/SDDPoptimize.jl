#  Copyright 2017, V.Leclere, H.Gerard, F.Pacaud, T.Rigaut
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Implement the SDDP solver and initializers:
#  - functions to initialize value functions
#  - functions to build terminal cost
#############################################################################


export solve_SDDP, solve!

"""
Solve spmodel using SDDP algorithm and return `SDDPInterface` instance.

$(SIGNATURES)

# Description
Alternate forward and backward phases untill the stopping criterion is
fulfilled.

# Arguments
* `model::SPmodel`:
    the stochastic problem we want to optimize
* `param::SDDPparameters`:
    the parameters of the SDDP algorithm
* `verbosity::Int64`:
    Default is `0`, higher gives more printed information
* `verbose_it::Int64`:
    Default is `1`
    If verbosity >1 , display progression in terminal every
    `verbose_it` iterations.

# Returns
`SDDPInterface`

"""
function solve_SDDP(model::SPModel, param::SDDPparameters, verbosity=0::Int64, verbose_it=1::Int64;
                    stopcrit::AbstractStoppingCriterion=IterLimit(param.max_iterations),
                    prunalgo::AbstractCutPruningAlgo=CutPruners.AvgCutPruningAlgo(-1),
                    regularization=nothing)
    sddp = SDDPInterface(model, param,
                         stopcrit,
                         prunalgo,
                         verbosity=verbosity,
                         verbose_it=verbose_it,
                         regularization=regularization)
    # Run SDDP:
    solve!(sddp)
    sddp
end

"""
Solve spmodel using SDDP algorithm and return `SDDPInterface` instance.
Use hotstart.

$(SIGNATURES)

# Description
Alternate forward and backward phases untill the stopping criterion is
fulfilled.

# Arguments
* `model::SPmodel`:
    the stochastic problem we want to optimize
* `param::SDDPparameters`:
    the parameters of the SDDP algorithm
* `V::Vector{PolyhedralFunction}`:
    current lower approximation of Bellman functions
* `verbosity::Int64`:
    Default is `0`, higher gives more printed information
* `verbose_it::Int64`:
    Default is `1`
    If verbosity >1 , display progression in terminal every
    `verbose_it` iterations.

# Returns
* `SDDPInterface`
"""
function solve_SDDP(model::SPModel, param::SDDPparameters,
                    V::Vector{PolyhedralFunction}, verbosity=0::Int64, verbose_it=1::Int64;
                    stopcrit::AbstractStoppingCriterion=IterLimit(param.max_iterations),
                    prunalgo::AbstractCutPruningAlgo=CutPruners.AvgCutPruningAlgo(-1))

    sddp = SDDPInterface(model, param,
                         stopcrit,
                         prunalgo, V,
                         verbosity=verbosity,
                         verbose_it=verbose_it)
    solve!(sddp)
    sddp
end


"""Run SDDP iterations on `sddp::SDDPInterface` instance.

$(SIGNATURES)

# Description
This function modifies `sddp`:
* if `sddp.init` is false, init `sddp`
* run SDDP iterations and update `sddp` till stopping test is fulfilled

At each iteration, the algorithm runs:
* a forward pass on `sddp` to compute `trajectories`
* a backward pass to update value functions of `sddp`
* a cut pruning to remove outdated cuts in `sddp`
* an estimation of the upper-bound of `sddp`
* an update of the different attributes of `sddp`
* test the stopping criterion

"""
function solve!(sddp::SDDPInterface)

    if ~sddp.init
        init!(sddp)
        (sddp.verbosity > 0) && println("Initialize cuts")
    end
    model = sddp.spmodel
    param = sddp.params
    stats = sddp.stats

    # If computation of upper-bound is needed, a set of scenarios is built
    # to keep always the same realization for upper bound estimation:
    upperbound_scenarios = simulate_scenarios(sddp.spmodel.noises, sddp.params.in_iter_mc)

    upb = [Inf, Inf, Inf]
    stopping_test::Bool = false

    # Launch execution of forward and backward passes:
    (sddp.verbosity > 0) && println("Starting SDDP iterations")
    while !stop(sddp.stopcrit, stats, stats)
        # Time execution of current pass:
        tic()

        ####################
        # Forward pass : compute stockTrajectories
        (sddp.verbosity > 2) && checkit(sddp.verbose_it, sddp.stats.niterations) && println("start forward pass")
        costs, trajectories = forward_pass!(sddp)

        ####################
        # Backward pass : update polyhedral approximation of Bellman functions
        (sddp.verbosity > 2) && checkit(sddp.verbose_it, sddp.stats.niterations) && println("start backward pass")
        backward_pass!(sddp, trajectories)

        ####################
        # Time execution of current pass
        time_pass = toq()

        ####################
        # cut pruning
        (param.prune) && prune!(sddp, trajectories)

        ####################
        # In iteration lower bound estimation
        lwb = lowerbound(sddp)

        ####################
        # In iteration upper bound estimation
        (sddp.verbosity > 2) && checkit(sddp.verbose_it, sddp.stats.niterations) && println("start in-iteration upperbound estimation")
        upb = in_iteration_upb_estimation(model, param, stats.niterations+1, sddp.verbosity,
                                          upperbound_scenarios, upb,
                                          sddp.solverinterface)

        updateSDDP!(sddp, lwb, upb, time_pass, trajectories)

        (sddp.verbosity > 1) && checkit(sddp.verbose_it, sddp.stats.niterations) && println(sddp.stats)
    end

    ##########
    # Estimate final upper bound with param.monteCarloSize simulations:
    finalpass!(sddp)
end


"""Init `sddp::SDDPInterface` object."""
function init!(sddp::SDDPInterface)
    random_pass!(sddp)
    sddp.init = true
end


"""Display final results once SDDP iterations are finished."""
function finalpass!(sddp::SDDPInterface)
    model = sddp.spmodel
    param = sddp.params
    V = sddp.bellmanfunctions
    problems = sddp.solverinterface
    stats = sddp.stats

    if (sddp.verbosity>0) && (param.compute_ub >= 0)
        lwb = lowerbound(sddp)

        if (param.compute_ub == 0) || (param.monteCarloSize > 0)
            println("Compute final upper-bound with ",
                                    param.monteCarloSize, " scenarios...")
            upb, σ, tol = estimate_upper_bound(model, param, V, problems, param.monteCarloSize)
        else
            upb = stats.upper_bounds[end]
            tol = stats.upper_bounds_tol[end]
            σ = stats.upper_bounds_std[end]
        end

        println("\n", "#"^60)
        println("SDDP CONVERGENCE")
        @printf("- Exact lower bound:          %.4e [Gap < %.2f%s]\n",
                lwb, 100*(upb+tol-lwb)/lwb, '%')
        @printf("- Estimation of upper-bound:  %.4e\n", upb)
        @printf("- Upper-bound's s.t.d:        %.4e\n", σ)
        @printf("- Confidence interval (%d%s):  [%.4e , %.4e]",
                100*(1- 2*(1-param.confidence_level)), '\%',upb-tol, upb+tol)
        println("\n", "#"^60)
    end
end


function updateSDDP!(sddp::SDDPInterface, lwb, upb, time_pass, trajectories)
    # Update SDDP stats
    updateSDDPStat!(sddp.stats, lwb, upb, time_pass)

    # If specified, reload JuMP model
    # this step can be useful if MathProgBase interface takes too much
    # room in memory, rendering necessary a call to GC
    if checkit(sddp.params.reload, sddp.stats.niterations)
        (sddp.params.prune) && sync!(sddp)
        (sddp.verbosity >2 )&& println("Reloading JuMP model")
        sddp.solverinterface = hotstart_SDDP(sddp.spmodel,
                                             sddp.params,
                                             sddp.bellmanfunctions)
    end

    # Update regularization
    if !isnull(sddp.regularizer)
        (sddp.verbosity >3) && println("Updating regularization ")
        update_penalization!(get(sddp.regularizer))
        get(sddp.regularizer).incumbents = trajectories
    end
end


"""
Build final cost with PolyhedralFunction function `Vt`.

$(SIGNATURES)

# Arguments
* `model::SPModel`:
    Model description
* `problem::JuMP.Model`:
    Cut approximating the terminal cost
* `Vt::PolyhedralFunction`:
    Final cost given as a PolyhedralFunction
* `verbosity::Int64`:
    Default is `0`, higher gives more printed information
"""
function build_terminal_cost!(model::SPModel,
                              problem::JuMP.Model,
                              Vt::PolyhedralFunction,
                              verbosity::Int64=0)
    # if shape is PolyhedralFunction, build terminal cost with it:
    alpha = problem[:alpha]
    xf = problem[:xf]
    t = model.stageNumber -1
    if isa(Vt, PolyhedralFunction)
        (verbosity >3) && println("Building final cost")
        for i in 1:Vt.numCuts
            lambda = vec(Vt.lambdas[i, :])
            if model.info == :HD
                @constraint(problem, Vt.betas[i] + dot(lambda, xf) <= alpha)
            elseif model.info == :DH
                for ww=1:length(model.noises[t].proba)
                    @constraint(problem, Vt.betas[i] + dot(lambda, xf[:, ww]) <= alpha[ww])
                end
            end
        end
    else
        # else, by default terminal cost is equal to 0
        @constraint(problem, alpha .>= 0)
    end
end


"""
Initialize each linear problem used to approximate value functions

$(SIGNATURES)

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
    if model.info == :HD
        return JuMP.Model[build_model(model, param, t) for t=1:model.stageNumber-1]
    else
        return JuMP.Model[build_model_dh(model, param, t) for t=1:model.stageNumber-1]
    end
end


function build_model(model, param, t,verbosity::Int64=0)
    m = Model(solver=param.SOLVER)

    nx = model.dimStates
    nu = model.dimControls
    nw = model.dimNoises

    # define variables in JuMP:
    @variable(m,  model.xlim[i,t][1] <= x[i=1:nx] <= model.xlim[i,t][2])
    @variable(m,  model.xlim[i,t][1] <= xf[i=1:nx]<= model.xlim[i,t][2])
    @variable(m,  model.ulim[i,t][1] <= u[i=1:nu] <=  model.ulim[i,t][2])
    @variable(m, alpha)

    @variable(m, w[1:nw] == 0)
    m.ext[:cons] = @constraint(m, state_constraint, x .== 0)

    @constraint(m, xf .== model.dynamics(t, x, u, w))

    # Add equality and inequality constraints:
    if ~isnull(model.equalityConstraints)
        @constraint(m, get(model.equalityConstraints)(t, x, u, w) .== 0)
    end
    if ~isnull(model.inequalityConstraints)
        @constraint(m, get(model.inequalityConstraints)(t, x, u, w) .<= 0)
    end

    # Define objective function (could be linear or piecewise linear)
    if isa(model.costFunctions, Function)
        try
            @objective(m, Min, model.costFunctions(t, x, u, w) + alpha)
        catch
            #FIXME: hacky redefinition of costs as JuMP Model
            @objective(m, Min, model.costFunctions(m, t, x, u, w) + alpha)
        end

    elseif isa(model.costFunctions, Vector{Function})
        @variable(m, cost)

        for i in 1:length(model.costFunctions)
            @constraint(m, cost >= model.costFunctions[i](t, x, u, w))
        end
        @objective(m, Min, cost + alpha)
    end

    # store number of cuts
    m.ext[:ncuts] = 0

    # Add binary variable if problem is a SMIP:
    if model.IS_SMIP
        m.colCat[2*nx+1:2*nx+nu] = model.controlCat
    end

    (verbosity >5) && print(m)
    return m
end

"""Build model in Decision-Hazard."""
function build_model_dh(model, param, t, verbosity::Int64=0)
    m = Model(solver=param.SOLVER)
    law = model.noises

    nx = model.dimStates
    nu = model.dimControls
    nw = model.dimNoises

    ns = law[t].supportSize
    ξ = collect(law[t].support[:, :])
    πp = law[t].proba

    @variable(m, model.xlim[i,t][1] <= x[i=1:nx] <= model.xlim[i,t][2])
    @variable(m, model.ulim[i,t][1] <= u[i=1:nu] <=  model.ulim[i,t][2])
    @variable(m, model.xlim[i,t][1] <= xf[i=1:nx, j=1:ns]<= model.xlim[i,t][2])
    @variable(m, alpha[1:ns])

    m.ext[:cons] = @constraint(m, state_constraint, x .== 0)

    for j=1:ns
        @constraint(m, xf[:, j] .== model.dynamics(t, x, u, ξ[:, j]))
    end

    # add objective as minimization of expectancy:
    try
        @objective(m, Min,
                        sum(πp[j]*(model.costFunctions(t, x, u, ξ[:, j]) +
                                    alpha[j]) for j in 1:ns))
    catch
        @objective(m, Min,
                        sum(πp[j]*(model.costFunctions(m, t, x, u, ξ[:, j]) +
                        alpha[j]) for j in 1:ns))
    end

    # store number of cuts
    m.ext[:ncuts] = 0

    (verbosity >5) && print(m)
    return m
end



"""
Initialize value functions along a given trajectory

$(SIGNATURES)

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
    V = getemptyvaluefunctions(model)

    # Build scenarios according to distribution laws:
    aleas = simulate_scenarios(model.noises, param.forwardPassNumber)

    # Add final costs to solverProblems:
    if isa(model.finalCost, PolyhedralFunction)
        V[end] = model.finalCost
        build_terminal_cost!(model, solverProblems[end], V[end])
    elseif isa(model.finalCost, Function)
        # In this case, define a trivial value functions for final cost to avoid problem:
        V[end] = PolyhedralFunction(zeros(1), zeros(1, model.dimStates), 1, UInt64[], 0)
        model.finalCost(model, solverProblems[end])
    end
    return V, solverProblems
end

getemptyvaluefunctions(model) = PolyhedralFunction[PolyhedralFunction(model.dimStates) for i in 1:model.stageNumber]


"""
Run SDDP iteration with random forward pass.

$(SIGNATURES)

# Parameters
* `sddp:SDDPInterface`
    SDDP instance
"""
function random_pass!(sddp::SDDPInterface)
    model = sddp.spmodel
    param = sddp.params

    stockTrajectories = zeros(model.stageNumber, param.forwardPassNumber, model.dimStates)
    for i in 1:model.stageNumber, j in 1:param.forwardPassNumber
        stockTrajectories[i, j, :] = get_random_state(model)
    end

    backward_pass!(sddp, stockTrajectories)
end


"""
Initialize JuMP.Model vector with a previously computed PolyhedralFunction
vector.

$(SIGNATURES)

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
Compute value of Bellman function at point `xt`. Return `V_t(xt)`.

$(SIGNATURES)

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
Get lower bound of SDDP instance `sddp`.

$(SIGNATURES)

"""
function lowerbound(sddp::SDDPInterface)
    return get_bellman_value(sddp.spmodel, sddp.params, 1,
                             sddp.bellmanfunctions[1],
                             sddp.spmodel.initialState)
end


"""
Compute lower-bound of the problem at initial time.

$(SIGNATURES)

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
function get_lower_bound(sddp::SDDPInterface)
    return lowerbound(sddp::SDDPInterface)
end

"""
Compute optimal control at point xt and time t.

$(SIGNATURES)

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
    return solve_one_step_one_alea(model, param, lpproblem[t], t, xt, xi)[1].uopt
end


"""
Add several cuts to JuMP.Model from a PolyhedralFunction

$(SIGNATURES)

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
    alpha = problem[:alpha]
    xf = problem[:xf]

    for i in 1:V.numCuts
        lambda = vec(V.lambdas[i, :])
        if model.info == :HD
            @constraint(problem, V.betas[i] + dot(lambda, xf) <= alpha)
        elseif model.info == :DH
            for j in 1:model.noises[t].supportSize
                @constraint(problem, V.betas[i] + dot(lambda, xf[:, j]) <= alpha[j])
            end
        end
    end
    problem.ext[:ncuts] = V.numCuts
end

"""
Compute subgradient of the problem at time t and state x.

# Arguments
* `V::Vector{Polyhedral function}`:
    Estimation of bellman function as Polyhedral function
* `t::Int64`:
    Timestep used to compute the derivative
* `x::Vector{Float64}`:
    State for which the derivative is computed

# Return
subgradient of the problem at time t and state x (Float64)
"""
function get_subgradient(V::Vector{PolyhedralFunction}, t::Int64, x::Vector{Int64})
    return get_subgradient(V[t],x)
end

"""
Compute subgradient of the problem at time t and state x.

# Arguments
* `Vt::PolyhedralFunction`:
    Estimation of bellman function as Polyhedral function
* `t::Int64`:
    Timestep used to compute the subgradient
* `x::Vector{Float64}`:
    State for which the subgradient is computed

# Return
subgradient of the problem at time t and state x (Float64)
"""
function get_subgradient(Vt::PolyhedralFunction, x::Vector{Int64})
    maxvalue = -Inf
    index = 0
    for i in 1:Vt.numCuts
        lambda = vec(Vt.lambdas[i, :])
        if Vt.betas[i] + dot(lambda, x) >= maxvalue
            maxvalue = Vt.betas[i] + dot(lambda, x)
            index = i
        end
    end
return Vt.lambdas[index, :]
end