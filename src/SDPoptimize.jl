#  Copyright 2015, Vincent Leclere, Francois Pacaud, Henri Gerard and
#  Tristan Rigaut
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Stochastic dynamic programming algorithm
#
#############################################################################

using ProgressMeter
using Iterators
using Interpolations

"""
Convert the state and control float tuples (stored as arrays or tuples) of the
problem into integer tuples that can be used as indexes for the value function

Parameters:
- variable (Array)
    the vector variable we want to convert to an index (integer)

- bounds (Array)
    the lower bounds for each component of the variable

- variable_steps (Array)
    discretization step for each component


Returns :
- index (tuple of integers)
    the indexes of the variable

"""
function index_from_variable( variable,
                    bounds::Array,
                    variable_steps::Array)

    return tuple([ 1 + floor(Int64,(1e-10+( variable[i] - bounds[i][1] )/ variable_steps[i] )) for i in 1:length(variable)]...)
end

"""
Convert the state and control float tuples (stored as arrays or tuples) of the
problem into float tuples that can be used as indexes for the interpolated
value function

Parameters:
- variable (Array)
    the vector variable we want to convert to an index (integer)

- bounds (Array)
    the lower bounds for each component of the variable

- variable_steps (Array)
    discretization step for each component


Returns :
- index (tuple of integers)
    the indexes of the variable

"""
function real_index_from_variable( variable,
                    bounds::Array,
                    variable_steps::Array)

    return tuple([1 + ( variable[i] - bounds[i][1] )/variable_steps[i] for i in 1:length(variable)]...)
end

"""
Compute interpolation of the value function at time t

Parameters:
- model (SPmodel)

- v (Array)
    the value function to interpolate

- time (Int)
    time at which we have to interpolate V


Returns :
- Interpolation
    the interpolated value function (working as an array with float indexes)
"""
function value_function_interpolation( model::SPModel,
                                    V::Array,
                                    time::Int)

    return interpolate(V[[Colon() for i in 1:model.dimStates]...,time], BSpline(Linear()), OnGrid())
end

"""
Compute interpolation of the value function at time t

Parameters:
- model (SPmodel)
    the model of the problem

- param (SDPparameters)
    the parameters of the problem


Returns :
- Iterators : product_states and product_controls
    the cartesian product iterators for both states and controls
"""
function generate_grid(model::SPModel, param::SDPparameters)

    product_states = product([model.xlim[i][1]:param.stateSteps[i]:model.xlim[i][2] for i in 1:model.dimStates]...)

    product_controls = product([model.ulim[i][1]:param.controlSteps[i]:model.ulim[i][2] for i in 1:model.dimControls]...)

    return product_states, product_controls

end

"""
Transform a general SPmondel into a StochDynProgModel

Parameters:
- model (SPmodel)
    the model of the problem

- param (SDPparameters)
    the parameters of the problem


Returns :
- sdpmodel : (StochDynProgModel)
    the corresponding StochDynProgModel
"""
function SPmodel_to_SDPmodel(model::SPModel, param::SDPparameters)

    function zero_fun(x)
        return 0
    end

    if isa(model,PiecewiseLinearCostSPmodel)||isa(model,LinearDynamicLinearCostSPmodel)
        function cons_fun(t,x,u,w)
            test = true
            for i in 1:model.dimStates
                test &= (x[i]>=model.xlim[i][1])&(x[i]<=model.xlim[i][2])
            end
            return test
        end
        if in(:finalCostFunction,fieldnames(model))
            SDPmodel = StochDynProgModel(model, model.finalCostFunction, cons_fun)
        else
            SDPmodel = StochDynProgModel(model, zero_fun, cons_fun)
        end
    elseif isa(model,StochDynProgModel)
        SDPmodel = model
    else
        error("cannot build StochDynProgModel from current SPmodel. You need to implement
        a new StochDynProgModel constructor.")
    end
    return(SDPmodel)
end

"""
Value iteration algorithm to compute optimal value functions in
the Decision Hazard (DH) as well as the Hazard Decision (HD) case

Parameters:
- model (SPmodel)
    the DPSPmodel of our problem

- param (SDPparameters)
    the parameters for the SDP algorithm

- display (Bool)
    the output display or verbosity parameter


Returns :
- value_functions (Array)
    the vector representing the value functions as functions of the state
    of the system at each time step

"""
function sdp_optimize(model::SPModel,
                  param::SDPparameters,
                  display=true::Bool)

    SDPmodel = SPmodel_to_SDPmodel(model, param)

    #Display start of the algorithm in DH and HD cases
    if (param.infoStructure == "DH")
        V = sdp_solve_DH(SDPmodel, param, display)
    elseif (param.infoStructure == "HD")
        V = sdp_solve_HD(SDPmodel, param, display)
    else
        error("param.infoStructure is neither 'DH' nor 'HD'")
    end

    return V
end


"""
Value iteration algorithm to compute optimal value functions in
the Decision Hazard (DH) case

Parameters:
- model (StochDynProgModel)
    the DPSPmodel of our problem

- param (SDPparameters)
    the parameters for the SDP algorithm

- display (Bool)
    the output display or verbosity parameter


Returns :
- value_functions (Array)
    the vector representing the value functions as functions of the state
    of the system at each time step

"""
function sdp_solve_DH(model::StochDynProgModel,
                  param::SDPparameters,
                  display=true::Bool)

    TF = model.stageNumber
    next_state = zeros(Float64, model.dimStates)
    law = model.noises

    u_bounds = model.ulim
    x_bounds = model.xlim
    x_steps = param.stateSteps

    #Compute cartesian product spaces
    product_states, product_controls = generate_grid(model, param)

    V = zeros(Float64, param.stateVariablesSizes..., TF)

    #Compute final value functions
    for x in product_states
        ind_x = index_from_variable(x, x_bounds, x_steps)
        V[ind_x..., TF] = model.finalCostFunction(x)
    end

    #Construct a progress meter
    if display
        p = Progress((TF-1)*param.totalStateSpaceSize, 1)
    end

    #Display start of the algorithm in DH and HD cases
    if display
        println("Starting stochastic dynamic programming decision hazard computation")
    end

        #Loop over time
    for t = (TF-1):-1:1
        Vitp = value_function_interpolation(model, V, t+1)

            #Loop over states
        for x in product_states

            if display
                next!(p)
            end

            expected_V = Inf
            optimal_u = tuple()
            current_cost = 0

            #Loop over controls
            for u = product_controls

                expected_V_u = 0.
                count_admissible_w = 0

                if (param.expectation_computation=="MonteCarlo")
                    sampling_size = param.monteCarloSize
                    samples = [sampling(law,t) for i in 1:sampling_size]
                    probas = (1/sampling_size)
                else
                    sampling_size = law[t].supportSize
                    samples = law[t].support
                    probas = law[t].proba
                end

                for w = 1:sampling_size
                    w_sample = samples[:, w]
                    proba = probas[w]
                    next_state = model.dynamics(t, x, u, w_sample)


                    if model.constraints(t, next_state, u, w_sample)

                        count_admissible_w = count_admissible_w + proba
                        ind_next_state = real_index_from_variable(next_state, x_bounds, x_steps)
                        next_V = Vitp[ind_next_state...]
                        current_cost = model.costFunctions(t, x, u, w_sample)
                        expected_V_u += proba*(current_cost + next_V)

                    end
                end

                if (count_admissible_w>0)

                    next_V = next_V / count_admissible_w

                    if (expected_V_u < expected_V)

                        expected_V = expected_V_u
                        optimal_u = u

                    end
                 end
            end
            ind_x = index_from_variable(x, x_bounds, x_steps)

            V[ind_x..., t] = expected_V
        end
    end
    return V
end

"""
Value iteration algorithm to compute optimal value functions in
the Hazard Decision (HD) case

Parameters:
- model (StochDynProgModel)
    the DPSPmodel of our problem

- param (SDPparameters)
    the parameters for the SDP algorithm

- display (Bool)
    the output display or verbosity parameter


Returns :
- value_functions (Array)
    the vector representing the value functions as functions of the state
    of the system at each time step

"""
function sdp_solve_HD(model::StochDynProgModel,
                  param::SDPparameters,
                  display=true::Bool)

    TF = model.stageNumber
    next_state = zeros(Float64, model.dimStates)
    law = model.noises

    u_bounds = model.ulim
    x_bounds = model.xlim
    x_steps = param.stateSteps

    #Compute cartesian product spaces
    product_states, product_controls = generate_grid(model, param)

    V = zeros(Float64, param.stateVariablesSizes..., TF)

    #Compute final value functions
    for x in product_states
        ind_x = index_from_variable(x, x_bounds, x_steps)
        V[ind_x..., TF] = model.finalCostFunction(x)
    end

    #Construct a progress meter
    if display
        p = Progress((TF-1)*param.totalStateSpaceSize, 1)
    end

    if display
        println("Starting stochastic dynamic programming hazard decision computation")
    end

    #Loop over time
    for t = (TF-1):-1:1
        Vitp = value_function_interpolation(model, V, t+1)

        #Loop over states
        for x in product_states

            if display
                next!(p)
            end

            expected_V = 0.
            current_cost = 0.
            count_admissible_w = 0.

                #Tuning expectation computation parameters
            if param.expectation_computation!="MonteCarlo" && param.expectation_computation!="Exact"
                warn("param.expectation_computation should be 'MonteCarlo' or 'Exact'. Defaulted to 'exact'")
                param.expectation_computation="Exact"
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

            #Compute expectation
            for w in 1:sampling_size
                admissible_u_w_count = 0
                best_V_x_w = Inf
                next_V_x_w = Inf
                w_sample = samples[:, w]
                proba = probas[w]

                #Loop over controls to find best next value function
                for u in product_controls

                    next_state = model.dynamics(t, x, u, w_sample)

                    if model.constraints(t, next_state, u, w_sample)
                        admissible_u_w_count += 1
                        current_cost = model.costFunctions(t, x, u, w_sample)
                        ind_next_state = real_index_from_variable(next_state, x_bounds, x_steps)
                        next_V_x_w_u = Vitp[ind_next_state...]
                        next_V_x_w = current_cost + next_V_x_w_u

                        if (next_V_x_w < best_V_x_w)
                            best_V_x_w = next_V_x_w
                        end

                    end
                end

                expected_V += proba*best_V_x_w
                count_admissible_w += (admissible_u_w_count>0)*proba
            end
            if (count_admissible_w>0.)
                expected_V = expected_V / count_admissible_w
            end
            ind_x = index_from_variable(x, x_bounds, x_steps)
            V[ind_x..., t] = expected_V
        end
    end
    return V
end

"""
Get the optimal value of the problem from the optimal Bellman Function

Parameters:
- model (SPmodel)
    the DPSPmodel of our problem

- param (SDPparameters)
    the parameters for the SDP algorithm

- V (Array{Float64})
    the Bellman Functions

Returns :
- V(x0) (Float64)
"""
function get_value(model::SPModel,param::SDPparameters,V::Array{Float64})
    ind_x0 = real_index_from_variable(model.initialState, model.xlim, param.stateSteps)
    Vi = value_function_interpolation(model, V, 1)
    return Vi[ind_x0...,1]
end

"""
Get the optimal control at time t knowing the state of the system in the decision hazard case

Parameters:
- model (SPmodel)
    the DPSPmodel of our problem

- param (SDPparameters)
    the parameters for the SDP algorithm

- V (Array{Float64})
    the Bellman Functions

- t (int)
    the time step

- x (Array)
    the state variable

Returns :
- V(x0) (Float64)
"""
function get_control(model::SPModel,param::SDPparameters,V::Array{Float64}, t::Int64, x::Array)

    SDPmodel = SPmodel_to_SDPmodel(model, param)

    product_controls = product([SDPmodel.ulim[i][1]:param.controlSteps[i]:SDPmodel.ulim[i][2] for i in 1:SDPmodel.dimControls]...)

    law = SDPmodel.noises
    best_control = tuple()
    Vitp = value_function_interpolation(SDPmodel, V, t+1)

    u_bounds = SDPmodel.ulim
    x_bounds = SDPmodel.xlim
    x_steps = param.stateSteps

    best_V = Inf

    for u in product_controls

        count_admissible_w = 0.
        current_V = 0.

        if (param.expectation_computation=="MonteCarlo")
            sampling_size = param.monteCarloSize
            samples = [sampling(law,t) for i in 1:sampling_size]
            probas = (1/sampling_size)
        else
            sampling_size = law[t].supportSize
            samples = law[t].support[:]
            probas = law[t].proba
        end

        for w = 1:sampling_size

            w_sample = samples[w]
            proba = probas[w]

            next_state = SDPmodel.dynamics(t, x, u, w_sample)

            if SDPmodel.constraints(t, next_state, u, w_sample)
                ind_next_state = real_index_from_variable(next_state, x_bounds, x_steps)
                next_V = Vitp[ind_next_state...]
                current_V += proba *(SDPmodel.costFunctions(t, x, u, w_sample) + next_V)
                count_admissible_w = count_admissible_w + proba
            end
        end
        current_V = current_V/count_admissible_w
        if (current_V < best_V)&(count_admissible_w>0)
            best_control = u
            best_V = current_V
        end
    end

    return best_control
end

"""
Get the optimal control at time t knowing the state of the system in the hazard decision case

Parameters:
- model (SPmodel)
    the DPSPmodel of our problem

- param (SDPparameters)
    the parameters for the SDP algorithm

- V (Array{Float64})
    the Bellman Functions

- t (int)
    the time step

- x (Array)
    the state variable

- w (Array)
the alea realization

Returns :
- V(x0) (Float64)
"""
function get_control(model::SPModel,param::SDPparameters,V::Array{Float64}, t::Int64, x::Array, w::Array)

    SDPmodel = SPmodel_to_SDPmodel(model, param)

    product_controls = product([SDPmodel.ulim[i][1]:param.controlSteps[i]:SDPmodel.ulim[i][2] for i in 1:SDPmodel.dimControls]...)

    law = SDPmodel.noises
    best_control = tuple()
    Vitp = value_function_interpolation(SDPmodel, V, t+1)

    u_bounds = SDPmodel.ulim
    x_bounds = SDPmodel.xlim
    x_steps = param.stateSteps

    best_V = Inf

    for u = product_controls

        next_state = SDPmodel.dynamics(t, x, u, w)

        if SDPmodel.constraints(t, next_state, u, w)
            ind_next_state = real_index_from_variable(next_state, x_bounds, x_steps)
            next_V = Vitp[ind_next_state...]
            current_V = SDPmodel.costFunctions(t, x, u, w) + next_V
            if (current_V < best_V)
                best_control = u
                best_state = SDPmodel.dynamics(t, x, u, w)
                best_V = current_V
            end
        end

    end

    return best_control
end

"""
Simulation of optimal control given an initial state and an alea scenario

Parameters:
- model (SPmodel)
    the DPSPmodel of our problem

- param (SDPparameters)
    the parameters for the SDP algorithm

- scenario (Array)
    the scenario of uncertainties realizations we want to simulate on

- X0 (SDPparameters)
    the initial state of the system

- value_functions (Array)
    the vector representing the value functions as functions of the state
    of the system at each time step

- display (Bool)
    the output display or verbosity parameter

Returns :

- J (Float)
    the cost of the optimal control over the scenario provided

- stocks (Array)
    the state of the controlled system at each time step

- controls (Array)
    the controls applied to the system at each time step

"""
function sdp_forward_simulation(model::SPModel,
                  param::SDPparameters,
                  scenario::Array,
                  X0::Array,
                  value::Array,
                  display=true::Bool)

    TF = model.stageNumber
    law = model.noises
    u_bounds = model.ulim
    x_bounds = model.xlim
    x_steps = param.stateSteps

    #Compute cartesian product spaces
    product_states, product_controls = generate_grid(model, param)

    controls = Inf*ones(TF-1, 1, model.dimControls)
    states = Inf*ones(TF, 1, model.dimStates)

    state_num = 0
    for xj in X0
        state_num += 1
        states[1, 1, state_num] = xj
    end

    J = 0
    best_state = X0

    best_control = tuple()

    if (param.infoStructure == "DH")
        #Decision hazard forward simulation
        for t = 1:(TF-1)

            x = states[t,1,:]

            best_V = Inf
            Vitp = value_function_interpolation(model, value, t+1)

            for u in product_controls

                count_admissible_w = 0.
                current_V = 0.

                if (param.expectation_computation=="MonteCarlo")
                    sampling_size = param.monteCarloSize
                    samples = [sampling(law,t) for i in 1:sampling_size]
                    probas = (1/sampling_size)
                else
                    sampling_size = law[t].supportSize
                    samples = law[t].support[:]
                    probas = law[t].proba
                end

                for w = 1:sampling_size

                    w_sample = samples[w]
                    proba = probas[w]

                    next_state = model.dynamics(t, x, u, w_sample)

                    if model.constraints(t, next_state, u, w_sample)
                        ind_next_state = real_index_from_variable(next_state, x_bounds, x_steps)
                        next_V = Vitp[ind_next_state...]
                        current_V += proba *(model.costFunctions(t, x, u, w_sample) + next_V)
                        count_admissible_w = count_admissible_w + proba
                    end
                end
                current_V = current_V/count_admissible_w
                if (current_V < best_V)&(count_admissible_w>0)
                    best_control = u
                    best_state = model.dynamics(t, x, u, scenario[t,1,:])
                    best_V = current_V
                end
            end

            index_control = 0
            for uj in best_control
                index_control += 1
                controls[t,1,index_control] = uj
            end

            index_state = 0
            for xj in best_state
                index_state = index_state +1
                states[t+1,1,index_state] = xj
            end
            J += model.costFunctions(t, x, best_control, scenario[t,1,:])
        end

    else
        #Hazard desision forward simulation
        for t = 1:TF-1

            x = states[t,1,:]

            Vitp = value_function_interpolation(model, value, t+1)

            best_V = Inf

            for u = product_controls

                next_state = model.dynamics(t, x, u, scenario[t,1,:])

                if model.constraints(t, next_state, u, scenario[t])
                    ind_next_state = real_index_from_variable(next_state, x_bounds, x_steps)
                    next_V = Vitp[ind_next_state...]
                    current_V = model.costFunctions(t, x, u, scenario[t,1,:]) + next_V
                    if (current_V < best_V)
                        best_control = u
                        best_state = model.dynamics(t, x, u, scenario[t,1,:])
                        best_V = current_V
                    end
                end

            end
            index_control = 0
            for uj in best_control
                index_control += 1
                controls[t,1,index_control] = uj
            end

            index_state = 0
            for xj in best_state
                index_state = index_state +1
                states[t+1,1,index_state] = xj
            end
            J += model.costFunctions(t, x, best_control, scenario[t,1,:])
        end
    end

    x = states[TF,1,:]
    J = J + model.finalCostFunction(x)

    return J, states, controls
end

