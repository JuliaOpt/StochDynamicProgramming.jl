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
Test if the stopping criteria is fulfilled.

Return true if |upper_bound - lower_bound|/lower_bound < epsilon
or iteration_count > maxItNumber

# Arguments
*`param::SDDPparameters`:
    stopping test type defined in SDDPparameters
* `stats::SDDPStat`:
    statistics of the current algorithm

# Return
`Bool`
"""
function test_stopping_criterion(param::SDDPparameters, stats::SDDPStat)
    lb = stats.lower_bounds[end]
    ub = stats.upper_bounds[end] + stats.upper_bounds_tol[end]
    check_gap = (abs((ub-lb)/lb) < param.gap)
    check_iter = stats.niterations >= param.maxItNumber
    return check_gap || check_iter
end


"""
Estimate upperbound during SDDP iterations.

# Arguments
* `model::SPModel`:
* `params::SDDPparameters`:
* `Vector{PolyhedralFunction}`:
    Polyhedral functions where cuts will be removed
* `iteration_count::Int64`:
    current iteration number
* `upperbound_scenarios`
* `verbose::Int64`

# Return
* `upb::Float64`:
    estimation of upper bound
"""
function in_iteration_upb_estimation(model::SPModel,
                                     param::SDDPparameters,
                                     iteration_count::Int64,
                                     verbose::Int64,
                                     upperbound_scenarios,
                                     current_upb,
                                     problems)
    upb, σ, tol = current_upb
    # If specified, compute upper-bound:
    if (param.compute_ub > 0) && (iteration_count%param.compute_ub==0)
        (verbose > 0) && println("Compute upper-bound with ",
                                    param.in_iter_mc, " scenarios...")
        # estimate upper-bound with Monte-Carlo estimation:
        upb, σ, tol = estimate_upper_bound(model, param, upperbound_scenarios, problems)
    end
    return [upb, σ, tol]
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
    # discard unvalid values:
    costs = costs[isfinite.(costs)]
    μ = mean(costs)
    σ = std(costs)
    tol = upper_bound_confidence(costs, param.confidence_level)
    return μ, σ, tol
end


"""
Estimate the upper bound with a distribution of costs

# Description
Given a probability p, we have a confidence interval:
[mu - alpha sigma/sqrt(n), mu + alpha sigma/sqrt(n)]
where alpha depends upon p.

Upper bound is the max of this interval.

# Arguments
* `cost::Vector{Float64}`:
    Costs values
* `probability::Float`:
    Probability to be inside the confidence interval

# Return
estimated-upper bound as `Float`
"""
function upper_bound_confidence(cost::Vector{Float64}, probability=.975)
    tol = sqrt(2) * erfinv(2*probability - 1)
    return tol*std(cost)/sqrt(length(cost))
end

