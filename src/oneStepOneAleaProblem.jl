#  Copyright 2014, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Model and solve the One-Step One Alea problem in different settings
# - used to compute the optimal control (in forward phase / simulation)
# - used to compute the cuts in the Backward phase
#############################################################################

"""
Solve the Bellman equation at time t starting at state x under alea xi
with the current evaluation of Vt+1

# Description
The function solve
min_u current_cost(t,x,u,xi) + current_Bellman_Value_{t+1}(dynamic(t,x,u,xi))
and can return the optimal control and a subgradient of the value of the
problem with respect to the initial state x

# Arguments
* `model::SPmodel`:
    the stochastic problem we want to optimize
* `param::SDDPparameters`:
    the parameters of the SDDP algorithm
* `m::JuMP.Model`:
    The linear problem to solve, in order to approximate the
    current value functions
* `t::int`:
    time step at which the problem is solved
* `xt::Array{Float}`:
    current starting state
* `xi::Array{float}`:
    current noise value
* `relaxation::Bool`: default is false
    If problem is MILP, specify if it is needed to relax integer constraints.
* `init::Bool`:
    If specified, approximate future cost as 0

# Returns
* `solved::Bool`:
    True if the solution is feasible, false otherwise
* `NextStep`:
    Store solution of the problem
"""
function solve_one_step_one_alea(model,
                                 param,
                                 m::JuMP.Model,
                                 t::Int64,
                                 xt::Vector{Float64},
                                 xi::Vector{Float64};
                                 relaxation=false::Bool,
                                 init=false::Bool)
    # Get var defined in JuMP.model:
    u = getvariable(m, :u)
    w = getvariable(m, :w)
    alpha = getvariable(m, :alpha)

    # Update value of w:
    setvalue(w, xi)

    # Update constraint x == xt
    for i in 1:model.dimStates
        JuMP.setRHS(m.ext[:cons][i], xt[i])
    end

    if model.IS_SMIP
        solved = relaxation ? solve_relaxed!(m, param): solve_mip!(m, param)
    else
        status = solve(m)
        solved = (status == :Optimal)
    end

    if solved
        optimalControl = getvalue(u)
        # Return object storing results:
        λ = (~model.IS_SMIP || relaxation)? Float64[getdual(m.ext[:cons][i]) for i in 1:model.dimStates]:nothing

        result = NextStep(
                          model.dynamics(t, xt, optimalControl, xi),
                          optimalControl,
                          λ,
                          getobjectivevalue(m),
                          getvalue(alpha))
    else
        # If no solution is found, then return nothing
        result = nothing
    end

    return solved, result
end

# Solve local problem with a quadratic penalization:
function solve_one_step_one_alea(model, param, m::JuMP.Model, t::Int64,
                                 xt::Vector{Float64}, xi::Vector{Float64}, xp::Vector{Float64})
    # store current objective:
    pobj = m.obj
    xf = getvariable(m, :xf)
    # copy JuMP model to avoid side effect:
    mm = m
    rho = param.acceleration[:rho]
    # build quadratic penalty term:
    qexp = QuadExpr(rho*dot(xf - xp, xf - xp))
    # and update model objective:
    @objective(mm, :Min, m.obj + qexp)
    res = solve_one_step_one_alea(model, param, mm, t, xt, xi)
    m.obj = pobj

    return res
end

"""Solve relaxed MILP problem."""
function solve_relaxed!(m, param)
    setsolver(m, param.SOLVER)
    status = solve(m, relaxation=true)
    return status == :Optimal
end

"""Solve original MILP problem."""
function solve_mip!(m, param)
    setsolver(m, param.MIPSOLVER)
    status = solve(m, relaxation=false)
    return status == :Optimal
end

