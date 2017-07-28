#  Copyright 2017, V.Leclere, H.Gerard, F.Pacaud, T.Rigaut
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Stochastic dynamic programming algorithm
#
#############################################################################

using ProgressMeter, Interpolations


"""
Compute interpolation of the value function at time t

# Arguments
* `model::SPmodel`:
* `dim_states::Int`:
    the number of state variables
* `v::Array`:
    the value function to interpolate
* `time::Int`:
    time at which we have to interpolate V

# Return
* Interpolation
    the interpolated value function (working as an array with float indexes)

"""
function value_function_interpolation( dim_states::Int, V::Union{SharedArray, Array}, time::Int)
    return interpolate(V[[Colon() for i in 1:dim_states]...,time], BSpline(Linear()), OnGrid())
end


"""
Compute the cartesian products of discretized state spaces

# Arguments
* `model::SPmodel`:
    the model of the problem
* `param::SDPparameters`:
    the parameters of the problem

# Return
* Iterators: product_states
    the cartesian product iterators for states

"""
function generate_state_grid(model::SPModel, param::SDPparameters, w::Nullable{Array} = Nullable{Array}() )
    product_states = Base.product([model.xlim[i][1]:param.stateSteps[i]:model.xlim[i][2] for i in 1:model.dimStates]...)

    return collect(product_states)
end

"""
Compute the cartesian products of discretized control spaces or more complex space if provided

# Arguments
* `model::SPmodel`:
    the model of the problem
* `param::SDPparameters`:
    the parameters of the problem
* `t::Int`:
    time step of the value function computation
* `x::Array{Float64}`:
    the  current state explored

# Return
* Iterators: product_states and product_controls
    the cartesian product iterators for both states and controls

"""
function generate_control_grid(model::SPModel, param::SDPparameters,
                                t::Nullable{Int} = Nullable{Int}(),
                                x::Nullable{Array} = Nullable{Array}(),
                                w::Nullable{Array} = Nullable{Array}())

    if (isnull(model.build_search_space))||(isnull(t))||(isnull(x))
        product_controls = Base.product([model.ulim[i][1]:param.controlSteps[i]:model.ulim[i][2] for i in 1:model.dimControls]...)
    else
        product_controls = model.build_search_space(t, x, w)

    end
    return collect(product_controls)
end


"""
Transform a general SPmodel into a StochDynProgModel

# Arguments
* `model::SPmodel`:
    the model of the problem
* `param::SDPparameters`:
    the parameters of the problem

# Return
* `sdpmodel::StochDynProgModel:
    the corresponding StochDynProgModel

"""
function build_sdpmodel_from_spmodel(model::SPModel)

    function zero_fun(x)
        return 0
    end

    if isa(model,LinearSPModel)
        
        cons_fun(t,x,u,w) = true 
        
        if in(:finalCostFunction,fieldnames(model))
            SDPmodel = StochDynProgModel(model, model.finalCostFunction, cons_fun)
        else
            SDPmodel = StochDynProgModel(model, zero_fun, cons_fun)
        end
    elseif isa(model,StochDynProgModel)
        SDPmodel = model
    else
        error("cannot build StochDynProgModel from current SPmodel. You need to
        implement a new StochDynProgModel constructor.")
    end

    return SDPmodel
end


"""
Dynamic programming algorithm to compute optimal value functions
by backward induction using bellman equation in the finite horizon case.
The information structure can be Decision Hazard (DH) or Hazard Decision (HD)

# Arguments
* `model::SPmodel`:
    the DPSPmodel of our problem
* `param::SDPparameters`:
    the parameters for the SDP algorithm
* `display::Int`:
    the output display or verbosity parameter

# Return
* `value_functions::Array`:
    the vector representing the value functions as functions of the state
    of the system at each time step

"""
function solve_dp(model::SPModel, param::SDPparameters, display=0::Int64)

    SDPmodel = build_sdpmodel_from_spmodel(model)

    # Start of the algorithm
    V = compute_value_functions_grid(SDPmodel, param, display)
    return V
end


"""
Dynamic Programming algorithm to compute optimal value functions

# Parameters
* `model::StochDynProgModel`:
    the StochDynProgModel of the problem
* `param::SDPparameters`:
    the parameters for the algorithm
* `display::Int`:
    the output display or verbosity parameter

# Returns
* `value_functions::Array`:
    the vector representing the value functions as functions of the state
    of the system at each time step

"""
function compute_value_functions_grid(model::StochDynProgModel,
                                        param::SDPparameters,
                                        display=0::Int64)

    TF = model.stageNumber
    next_state = zeros(Float64, model.dimStates)

    u_bounds = model.ulim
    x_bounds = model.xlim
    x_steps = param.stateSteps
    x_dim = model.dimStates

    dynamics = model.dynamics
    constraints = model.constraints
    cost = model.costFunctions

    law = model.noises

    build_Ux = model.build_search_space

    #Compute cartesian product spaces
    product_states = generate_state_grid(model, param)

    product_controls = generate_control_grid(model, param)

    V = SharedArray{Float64}(zeros(Float64, param.stateVariablesSizes..., TF))

    #Compute final value functions
    for x in product_states
        ind_x = SdpLoops.index_from_variable(x, x_bounds, x_steps)
        V[ind_x..., TF] = model.finalCostFunction(x)
    end

    if param.expectation_computation!="MonteCarlo" && param.expectation_computation!="Exact"
        warn("param.expectation_computation should be 'MonteCarlo' or 'Exact'.
                Defaulted to 'exact'")
        param.expectation_computation="Exact"
    end

    if param.infoStructure == "DH"
        get_V_t_x = SdpLoops.sdp_u_w_loop
    elseif param.infoStructure == "HD"
        get_V_t_x = SdpLoops.sdp_w_u_loop
    else
        warn("Information structure should be DH or HD. Defaulted to DH")
        param.infoStructure = "DH"
        get_V_t_x = SdpLoops.sdp_u_w_loop
    end

    #Construct a progress meter
    p = 0
    if display > 0
        p = Progress((TF-1), 1)
        println("[SDP] Starting value functions computation:")
    end

    # Loop over time:
    for t = (TF-1):-1:1

        if display > 0
            next!(p)
        end

        if (param.expectation_computation=="MonteCarlo")
            sampling_size = param.monteCarloSize
            samples = [sampling(law,t) for i in 1:sampling_size]
            probas = (1/sampling_size)
        else
            sampling_size = law[t].supportSize
            samples = law[t].support
            probas = law[t].proba
        end

        Vitp = value_function_interpolation(x_dim, V, t+1)

        @sync @parallel for indx in 1:length(product_states)
            x = product_states[indx]
            ind_x = SdpLoops.index_from_variable(x, x_bounds, x_steps)
            V[ind_x..., t] = get_V_t_x(sampling_size, samples, probas,
                                            u_bounds, x_bounds, x_steps, x_dim,
                                            product_controls, dynamics,
                                            constraints, cost, Vitp, t,
                                            x, build_Ux)[1]
        end

    end
    return V
end

"""
Get the optimal value of the problem from the optimal Bellman Function

# Arguments
* `model::SPmodel`:
    the DPSPmodel of our problem
* `param::SDPparameters`:
    the parameters for the SDP algorithm
* `V::Array{Float64}`:
    the Bellman Functions

# Return
* `V_x0::Float64`:

"""
function get_bellman_value(model::SPModel, param::SDPparameters,
                            V::Union{SharedArray, Array})
    ind_x0 = SdpLoops.real_index_from_variable(model.initialState, model.xlim, param.stateSteps)
    Vi = value_function_interpolation(model.dimStates, V, 1)
    return Vi[ind_x0...,1]
end


"""
Get the optimal control at time t knowing the state of the system in the decision
hazard case

# Arguments
* `model::SPmodel`:
    the DPSPmodel of our problem
* `param::SDPparameters`:
    the parameters for the SDP algorithm
* `V::Array{Float64}`:
    the Bellman Functions
* `t::int`:
    the time step
* `x::Array`:
    the state variable
* `w::Array`:
    the alea realization

# Return
* `V_x0::Float64`:

"""
function get_control(model::SPModel,param::SDPparameters,
                     V, t::Int64, x::Array, w::Union{Void, Array} = nothing)

    sdp_model = build_sdpmodel_from_spmodel(model)

    args = []
    optional_args = []

    if w==nothing
        law = sdp_model.noises
        get_u = SdpLoops.sdp_dh_get_u
        if (param.expectation_computation=="MonteCarlo")
            sampling_size = param.monteCarloSize
            push!(args,sampling_size,
                    [sampling(law,t) for i in 1:sampling_size],
                    (1./sampling_size)*ones(sampling_size))
        else
            push!(args,law[t].supportSize, law[t].support, law[t].proba)
        end
        push!(optional_args, sdp_model.build_search_space)
    else
        get_u = SdpLoops.sdp_hd_get_u
        push!(optional_args, w, sdp_model.build_search_space)
    end

    push!(args, sdp_model.ulim, sdp_model.xlim, param.stateSteps,
            sdp_model.dimStates, generate_control_grid(sdp_model, param),
            sdp_model.dynamics, sdp_model.constraints, sdp_model.costFunctions,
            value_function_interpolation(sdp_model.dimStates, V, t+1), t, x)

    return get_u(args..., optional_args...)[1]
end


"""
Simulation of optimal control given an initial state and multiple scenarios

# Arguments
* `model::SPmodel`:
    the DPSPmodel of our problem
* `param::SDPparameters`:
    the parameters for the SDP algorithm
* `scenarios::Array`:
    the scenarios of uncertainties realizations we want to simulate on
* `X0::SDPparameters`:
    the initial state of the system
* `V::Array`:
    the vector representing the value functions as functions of the state
    of the system at each time step
* `display::Bool`:
    the output display or verbosity parameter

# Return
* `costs::Array{Float}`:
    the cost of the optimal control over the scenario provided
* `states::Array`:
    the state of the controlled system at each time step for each scenario
* `controls::Array`:
    the controls applied to the system at each time step for each scenario
"""
function forward_simulations(model::SPModel,
                            param::SDPparameters,
                            V::Union{SharedArray, Array},
                            scenarios::Array,
                            display=true::Bool)

    SDPmodel = build_sdpmodel_from_spmodel(model)

    nb_scenarios = size(scenarios,2)

    TF = SDPmodel.stageNumber
    law = SDPmodel.noises
    x_dim = SDPmodel.dimStates
    product_states = generate_state_grid(SDPmodel, param)
    costs = SharedArray{Float64}(zeros(nb_scenarios))
    states = SharedArray{Float64}(zeros(TF,nb_scenarios,x_dim))
    controls = SharedArray{Float64}(zeros(TF-1,nb_scenarios,SDPmodel.dimControls))

    dynamics = SDPmodel.dynamics
    cost = SDPmodel.costFunctions

    args = [SDPmodel.ulim, SDPmodel.xlim, param.stateSteps, x_dim,
    generate_control_grid(SDPmodel, param), dynamics, SDPmodel.constraints,
    cost]

    X0 = SDPmodel.initialState
    for s in 1:nb_scenarios
        states[1, s, :] = X0
    end

    best_control = tuple()

    info = param.infoStructure

    if  info == "DH"
        get_u = SdpLoops.sdp_dh_get_u
    elseif info == "HD"
        get_u = SdpLoops.sdp_hd_get_u
    else
        warn("Information structure should be DH or HD. Defaulted to DH")
        get_u = SdpLoops.sdp_dh_get_u
    end

    build_Ux = Nullable{Function}(SDPmodel.build_search_space)


    @sync @parallel for s in 1:nb_scenarios

        current_scen = scenarios[:,s,:]

        for t = 1:(TF-1)

            args_w = []

            x = states[t,s,:]
            w = current_scen[t,:]
            args_t = [value_function_interpolation(x_dim, V, t+1), t, x]

            if info == "DH"
                if (param.expectation_computation=="MonteCarlo")
                    sampling_size = param.monteCarloSize
                    push!(args_w,sampling_size,
                            [sampling(law,t) for i in 1:sampling_size],
                            (1./sampling_size)*ones(sampling_size))
                else
                    push!(args_w,law[t].supportSize, law[t].support, law[t].proba)
                end
            else
                push!(args_t, w)
            end

            best_control = get_u(args_w..., args..., args_t..., build_Ux)[1]

            if best_control == tuple()
                error("No u admissible")
            else
                controls[t,s,:] = [best_control...]
                states[t+1,s,:] = dynamics(t, x, best_control, w)
                costs[s] = costs[s] + cost(t, x, best_control, w)
            end
        end

    end

    for s in 1:nb_scenarios
        costs[s] = costs[s] + SDPmodel.finalCostFunction(states[TF,s,:])
    end

    return costs, states, controls
end


