#  Copyright 2014, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Define all types used in this module.
#############################################################################

include("noises.jl")

abstract SPModel

type LinearDynamicLinearCostSPmodel <: SPModel
    # problem dimension
    stageNumber::Int64
    dimControls::Int64
    dimStates::Int64
    dimNoises::Int64

    # Bounds of states and controls:
    xlim::Array{Tuple{Float64,Float64},1}
    ulim::Array{Tuple{Float64,Float64},1}

    initialState::Array{Float64, 1}

    costFunctions::Function
    dynamics::Function
    noises::Vector{NoiseLaw}

    function LinearDynamicLinearCostSPmodel(nstage, ubounds, x0, cost, dynamic, aleas)

        dimStates = length(x0)
        dimControls = length(ubounds)
        dimNoises = length(aleas[1].support[:, 1])

        xbounds = []
        for i = 1:dimStates
            push!(xbounds, (-Inf, Inf))
        end
        return new(nstage, dimControls, dimStates, dimNoises, xbounds, ubounds, x0, cost, dynamic, aleas)
    end
end


type PiecewiseLinearCostSPmodel <: SPModel
    # problem dimension
    stageNumber::Int64
    dimControls::Int64
    dimStates::Int64
    dimNoises::Int64

    # Bounds of states and controls:
    xlim::Array{Tuple{Float64,Float64},1}
    ulim::Array{Tuple{Float64,Float64},1}

    initialState::Array{Float64, 1}

    costFunctions::Vector{Function}
    dynamics::Function
    noises::Vector{NoiseLaw}

    function PiecewiseLinearCostSPmodel(nstage, ubounds, x0, costs, dynamic, aleas)
        dimStates = length(x0)
        dimControls = length(ubounds)
        dimNoises = length(aleas[1].support[:, 1])

        xbounds = []
        for i = 1:dimStates
            push!(xbounds, (-Inf, Inf))
        end
        return new(nstage, dimControls, dimStates, dimNoises, xbounds, ubounds, x0, costs, dynamic, aleas)
    end
end

"""Set bounds on state."""
function set_state_bounds(model::SPModel, xbounds)
    if length(xbounds) != model.dimStates
        error("Bounds dimension, must be ", model.dimStates)
    else
        model.xlim = xbounds
    end
end


type StochDynProgModel <: SPModel
    # problem dimension
    stageNumber::Int64
    dimControls::Int64
    dimStates::Int64
    dimNoises::Int64

    # Bounds of states and controls:
    xlim::Array{Tuple{Float64,Float64},1}
    ulim::Array{Tuple{Float64,Float64},1}

    initialState::Array{Float64, 1}

    costFunctions::Function
    finalCostFunction::Function
    dynamics::Function
    constraints::Function
    noises::Vector{NoiseLaw}

    function StochDynProgModel(model::LinearDynamicLinearCostSPmodel, final, cons)
        return new(model.stageNumber-1, model.dimControls, model.dimStates,
                 model.dimNoises, model.xlim, model.ulim, model.initialState,
                 model.costFunctions, final, model.dynamics, cons,
                 model.noises)
    end

    function StochDynProgModel(model::PiecewiseLinearCostSPmodel, final, cons)

        function cost(t,x,u,w)
            saved_cost = -Inf
            current_cost = 0
            for i in model.costFunctions
                current_cost = i(t,x,u,w)
                if (current_cost>saved_cost)
                    saved_cost = current_cost
                end
            end
            return saved_cost
        end

        return new(model.stageNumber-1, model.dimControls, model.dimStates,
                 model.dimNoises, model.xlim, model.ulim, model.initialState,
                 cost, final, model.dynamics, cons,
                 model.noises)
    end

    function StochDynProgModel(TF, N_CONTROLS, N_STATES, N_NOISES,
                    x_bounds, u_bounds, x0, cost_t, finalCostFunction, dynamic,
                    constraints, aleas)
        return new(TF, N_CONTROLS, N_STATES, N_NOISES,
                    x_bounds, u_bounds, x0, cost_t, finalCostFunction, dynamic,
                    constraints, aleas)
    end

    # TODO: add this attributes to model
    # lowerbounds#::Tuple{Vector{Float64}}
    # upperbounds#::Tuple{Vector{Float64}}
end

type SDDPparameters
    # Solver to solve
    solver
    # number of simulated scenario in the forward pass
    forwardPassNumber::Int64
    # Admissible gap between the estimation of the upper-bound
    sensibility::Float64
    # Maximum iterations of the SDDP algorithms:
    maxItNumber::Int64

    function SDDPparameters(solver, passnumber, sensibility=0.01, max_iterations=20)
        return new(solver, passnumber, sensibility, max_iterations)
    end
end


type SDPparameters
    stateSteps
    controlSteps
    totalStateSpaceSize
    totalControlSpaceSize
    stateVariablesSizes
    controlVariablesSizes
    monteCarloSize
    infoStructure

    function SDPparameters(model, stateSteps, controlSteps, monteCarloSize, infoStruct)

        stateVariablesSizes = zeros(Int64, length(stateSteps))
        controlVariablesSizes = zeros(Int64, length(controlSteps))
        totalStateSpaceSize = 1
        totalControlSpaceSize = 1
        for i=1:length(stateSteps)
            stateVariablesSizes[i] = round(Int64,1 + (model.xlim[i][2]-model.xlim[i][1])/stateSteps[i])
            totalStateSpaceSize *= stateVariablesSizes[i]
        end

        for i=1:length(controlSteps)
            controlVariablesSizes[i] = round(Int64, 1 + (model.ulim[i][2]-model.ulim[i][1])/controlSteps[i])
            totalControlSpaceSize *= controlVariablesSizes[i]
        end

        return new(stateSteps, controlSteps, totalStateSpaceSize,
                    totalControlSpaceSize, stateVariablesSizes,
                    controlVariablesSizes, monteCarloSize, infoStruct)
    end
end

function set_max_iterations(param::SDDPparameters, n_iter::Int)
    param.maxItNumber = n_iter
end


type PolyhedralFunction
    #function defined by max_k betas[k] + lambdas[k,:]*x
    betas::Vector{Float64}
    lambdas::Array{Float64,2} #lambdas[k,:] is the subgradient
    # number of cuts:
    numCuts::Int64
end



type NextStep
    next_state::Array{Float64, 1}
    optimal_control::Array{Float64, 1}
    sub_gradient
    cost::Float64
    cost_to_go::Float64
end
