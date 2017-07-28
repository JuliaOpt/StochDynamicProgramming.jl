#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Define the Forward / Backward iterations of the SDDP algorithm
#############################################################################

"""
Run a forward pass of the algorithm with `sddp` object

$(SIGNATURES)

# Description
Simulate scenarios of noise and compute optimal trajectories on those
scenarios, with associated costs.

# Arguments
* `sddp::SDDPInterface`:
    SDDP interface object

# Returns
* `costs::Array{float,1}`:
    an array of the simulated costs
* `stockTrajectories::Array{float}`:
    the simulated stock trajectories. stocks(t,k,:) is the stock for
    scenario k at time t.

"""
function forward_pass!(sddp::SDDPInterface)
    model = sddp.spmodel
    param = sddp.params
    solverProblems = sddp.solverinterface
    V = sddp.bellmanfunctions
    problems = sddp.solverinterface
    # Draw a set of scenarios according to the probability
    # law specified in model.noises:
    noise_scenarios = simulate_scenarios(model.noises, param.forwardPassNumber)

    # If regularization is ON, need to build a new array of problem to
    # avoid side effect:
    problems_fp = isregularized(sddp) ? hotstart_SDDP(model, param, V) : problems

    # run forward pass
    costs, stockTrajectories,_,callsolver_forward, tocfw = forward_simulations(model,
                        param,
                        problems_fp,
                        noise_scenarios,
                        pruner=sddp.pruner,
                        regularizer=sddp.regularizer,
                        verbosity = sddp.verbosity)

    # update SDDP's stats
    sddp.stats.nsolved += callsolver_forward
    sddp.stats.solverexectime_fw = vcat(sddp.stats.solverexectime_fw, tocfw)
    return costs, stockTrajectories
end


"""
Simulate a forward pass of the algorithm

$(SIGNATURES)

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
* `pruner::AbstractCutPruner`
    Cut pruner
* `regularizer::SDDPRegularization`
    SDDP regularization to use in forward pass
* `verbosity::Int`
    Log-level

# Returns
* `costs::Array{float,1}`:
    an array of the simulated costs
    If returnCosts=false, return nothing
* `stockTrajectories::Array{float}`:
    the simulated stock trajectories. stocks(t,k,:) is the stock for
    scenario k at time t.
* `controls::Array{Float64, 3}`:
    the simulated controls trajectories. controls(t,k,:) is the control for
    scenario k at time t.
* `callsolver::Int64`:
    the number of solver's call'
* `solvertime::Vector{Float64}`
    Solver's call execution time

"""
function forward_simulations(model::SPModel,
                            param::SDDPparameters,
                            solverProblems::Vector{JuMP.Model},
                            xi::Array{Float64};
                            pruner=Nullable{AbstractCutPruner}(),
                            regularizer=Nullable{SDDPRegularization}(),
                            verbosity::Int64=0)

    callsolver::Int = 0
    solvertime = Float64[]

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

    stockTrajectories = zeros(T, nb_forward, model.dimStates)
    # We got T - 1 control, as terminal state is included into the total number of stages.
    controls = zeros(T - 1, nb_forward, model.dimControls)

    # Set first value of stocks equal to x0:
    for k in 1:nb_forward
        stockTrajectories[1, k, :] = model.initialState
    end

    # Store costs of different scenarios in an array:
    costs = zeros(nb_forward)

    for t=1:T-1
        for k = 1:nb_forward
            # Collect current state and noise:
            state_t = collect(stockTrajectories[t, k, :])
            alea_t = collect(xi[t, k, :])

            callsolver += 1

            # Solve optimization problem corresponding to current position:
            if !isnull(regularizer) && !isa(get(regularizer).incumbents, Void)
                reg = get(regularizer)
                xp = getincumbent(reg, t, k)
                sol, ts = regularize(model, param, reg,
                                                  solverProblems[t], t, state_t, alea_t, xp,verbosity = verbosity)
            else
                # switch between HD and DH info structure
                if model.info == :HD
                    sol, ts = solve_one_step_one_alea(model, param,
                                                           solverProblems[t], t, state_t, alea_t,verbosity=verbosity)
                else
                    sol, ts = solve_dh(model, param, t, state_t,
                                                solverProblems[t],verbosity=verbosity)
                end
            end

            # update solvertime with ts
            push!(solvertime, ts)

            # update cutpruners status with new point
            if param.prune && ~isnull(pruner) && t < T-1
                update!(pruner[t+1], sol.xf, sol.πc)
            end

            # Check if the problem is effectively solved:
            if sol.status
                # Get the next position:
                stockTrajectories[t+1, k, :] = sol.xf
                # the optimal control just computed:
                controls[t, k, :] = sol.uopt
                # and the current cost:
                costs[k] += sol.objval - sol.θ
                if t==T-1
                    costs[k] += sol.θ
                end
            else
                # if problem is not properly solved, next position if equal
                # to current one:
                stockTrajectories[t+1, k, :] = state_t
                # this trajectory is unvalid, the cost is set to Inf to discard it:
                costs[k] += Inf
            end
        end
    end
    return costs, stockTrajectories, controls, callsolver, solvertime
end



"""
Add to polyhedral function a cut with shape Vt >= beta + <lambda,.>

$(SIGNATURES)

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
                  beta::Float64, lambda::Vector{Float64},verbosity=verbosity)
    (verbosity > 4) && println("adding cut to polyhedral function at time t=",t)
    Vt.lambdas = vcat(Vt.lambdas, lambda')
    Vt.betas = vcat(Vt.betas, beta)
    Vt.hashcuts = vcat(Vt.hashcuts, hash(lambda))
    Vt.numCuts += 1
    Vt.newcuts += 1
end

isinside(Vt::PolyhedralFunction, lambda::Vector{Float64})=hash(lambda) in Vt.hashcuts

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
* `lambda::Vector{Float64}`:
  subgradient of the cut to add
"""
function add_cut_to_model!(model::SPModel, problem::JuMP.Model,
                            t::Int64, beta::Float64, lambda::Vector{Float64},verbosity=verbosity)
    (verbosity > 4) && println("adding cut to model at time t=",t)
    alpha = problem[:alpha]
    xf = problem[:xf]
    @constraint(problem, beta + dot(lambda, xf) <= alpha)
    problem.ext[:ncuts] += 1
end

function add_cut_dh!(model::SPModel, problem::JuMP.Model,
                     t::Int64, beta::Float64, lambda::Vector{Float64}, verbosity=verbosity)
    (verbosity > 4) && println("adding cut to dh model at time t=",t)
    alpha = problem[:alpha]
    xf = problem[:xf]

    for j=1:length(model.noises[t].proba)
        @constraint(problem, beta + dot(lambda, xf[:, j]) <= alpha[j])
    end
end

"""
Run a SDDP backward pass on `sddp`.

$(SIGNATURES)

# Description
For t:T-1 -> 0, compute a valid cut of the Bellman function
Vt at the state given by stockTrajectories and add them to
the current estimation of Vt.

# Arguments
* `sddp::SDDPInterface`:
    SDDP instance
* `stockTrajectories::Array{Float64,3}`:
    stockTrajectories[t,k,:] is the vector of stock where the cut is computed
    for scenario k and time t.
"""
function backward_pass!(sddp::SDDPInterface,
                        stockTrajectories::Array{Float64, 3})

    model = sddp.spmodel
    law = model.noises
    param = sddp.params
    solverProblems = sddp.solverinterface
    V = sddp.bellmanfunctions

    solvertime = Float64[]

    T = model.stageNumber
    nb_forward = size(stockTrajectories)[2]


    for t = T-1:-1:1
        for k = 1:nb_forward

            # We collect current state:
            state_t = stockTrajectories[t, k, :]
            if model.info == :HD
                compute_cuts_hd!(model, param, V, solverProblems, t, state_t, solvertime,sddp.verbosity)
            elseif model.info == :DH
                compute_cuts_dh!(model, param, V, solverProblems, t, state_t, solvertime,sddp.verbosity)
            end

        end
    end
    # update stats
    sddp.stats.nsolved += length(solvertime)
    sddp.stats.solverexectime_bw = vcat(sddp.stats.solverexectime_bw, solvertime)
end


"""Compute cuts in Hazard-Decision (classical SDDP)."""
function compute_cuts_hd!(model::SPModel, param::SDDPparameters,
                          V::Vector{PolyhedralFunction},
                          solverProblems::Vector{JuMP.Model}, t::Int,
                          state_t::Vector{Float64}, solvertime::Vector{Float64},verbosity::Int64)
    law = model.noises
    costs = zeros(Float64, model.noises[t].supportSize)
    subgradient_array = zeros(Float64, model.dimStates, model.noises[t].supportSize)

    # We will store probabilities in a temporary array.
    # It is initialized at 0. If all problem are infeasible for
    # current timestep, then proba remains equal to 0 and not cut is added.
    proba = zeros(model.noises[t].supportSize)

    # We iterate other the possible realization of noise:
    for w in 1:model.noises[t].supportSize

        # We get current noise:
        alea_t  = collect(model.noises[t].support[:, w])
        # We solve LP problem with current noise and position:
        sol, ts = solve_one_step_one_alea(model, param,
                                            solverProblems[t],
                                            t, state_t, alea_t,
                                            relaxation=model.IS_SMIP,
                                            verbosity=verbosity)
        push!(solvertime, ts)

        if sol.status
            # We catch the subgradient λ:
            subgradient_array[:, w] = sol.ρe
            # and the current cost:
            costs[w] = sol.objval
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
        if ~isinside(V[t], subgradient)
            add_cut!(model, t, V[t], beta, subgradient, verbosity)
            if t > 1
                add_cut_to_model!(model, solverProblems[t-1], t, beta, subgradient, verbosity)
            end
        end
    end
end


"""Compute cuts in Decision-Hazard (variant of SDDP)."""
function compute_cuts_dh!(model::SPModel, param::SDDPparameters,
                          V::Vector{PolyhedralFunction},
                          solverProblems::Vector{JuMP.Model}, t::Int,
                          state_t::Vector{Float64}, solvertime::Vector{Float64},verbosity::Int64)
    # We solve LP problem in decision-hazard, considering all possible
    # outcomes of randomness:
    sol, ts = solve_dh(model, param, t, state_t, solverProblems[t])

    push!(solvertime, ts)

    # We add cuts only if one solution was being found:
    # Scale probability (useful when some problems where infeasible):
    if sol.status
        # Compute expectation of subgradient λ:
        subgradient = sol.ρe
        # ... expectation of cost:
        costs_npass = sol.objval
        # ... and expectation of slope β:
        beta = costs_npass - dot(subgradient, state_t)

        # Add cut to polyhedral function and JuMP model:
        add_cut!(model, t, V[t], beta, subgradient, verbosity)
        if t > 1
            add_cut_dh!(model, solverProblems[t-1], t, beta, subgradient,verbosity)
        end
    end
end
