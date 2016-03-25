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

The function solve
min_u current_cost(t,x,u,xi) + current_Bellman_Value_{t+1}(dynamic(t,x,u,xi))
and can return the optimal control and a subgradient of the value of the
problem with respect to the initial state x


Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize

- param (SDDPparameters)
    the parameters of the SDDP algorithm

- m (JuMP.Model)
    The linear problem to solve, in order to approximate the
    current value functions

- t (int)
    time step at which the problem is solved

- xt (Array{Float})
    current starting state

- xi (Array{float})
    current noise value

- init (Bool)
    If specified, approximate future cost as 0


Returns:
- Bool
    True if the solution is feasible, false otherwise

- NextStep
    Store solution of the problem solved

"""
function solve_one_step_one_alea(model,
                                 param,
                                 m::JuMP.Model,
                                 t::Int64,
                                 xt::Vector{Float64},
                                 xi::Vector{Float64},
                                 init=false::Bool)
    # Get var defined in JuMP.model:
    u = getVar(m, :u)
    w = getVar(m, :w)
    alpha = getVar(m, :alpha)

    # Update value of w:
    setValue(w, xi)

    # If this is the first call to the solver, value-to-go are approximated
    # with null function:
    if init
        @addConstraint(m, alpha >= 0)
    end
    # Update constraint x == xt
    for i in 1:model.dimStates
        chgConstrRHS(m.ext[:cons][i], xt[i])
    end


    status = solve(m)

    solved = (status == :Optimal)

    if solved
        optimalControl = getValue(u)
        # Return object storing results:
        result = NextStep(
                          model.dynamics(t, xt, optimalControl, xi),
                          optimalControl,
                          [getDual(m.ext[:cons][i]) for i in 1:model.dimStates],
                          getObjectiveValue(m),
                          getValue(alpha))
    else
        # If no solution is found, then return nothing
        result = nothing
    end

    return solved, result
end
