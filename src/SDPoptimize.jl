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

"""
Convert the state and control tuples (stored as arrays) of the problem into integers

Parameters:
- variable (Array)
    the vector variable we want to convert to an index (integer)

- lower_bounds (Array)
    the lower bounds for each component of the variable

- variable_sizes (Array)
    the number of possibilities at each component knowing the upper and lower bounds
    and the discretizations

- variable_steps (Array)
    discretization step for each component


Returns :
- index (integer)
    the index of the variable for the problem knowing the upper and lower bounds
    as well as the discretization

"""
function index_from_variable( variable::Array,
                    lower_bounds::Array,
                    variable_sizes::Array,
                    variable_steps::Array)
    index = 1;
    j = 1;

    for i = 1:length(variable)
        index += j * (floor( Int, 1e-10 + ( variable[i] - lower_bounds[i] ) / variable_steps[i] ))
        j *= variable_sizes[i]
    end

    return index
end

"""
Convert the index (integer) of a variable into the real variable (Array)

Parameters:
- index (integer)
    the index of the variable for the problem knowing the upper and lower bounds
    as well as the discretization

- lower_bounds (Array)
    the lower bounds for each component of the variable

- variable_sizes (Array)
    the number of possibilities at each component knowing the upper and lower bounds
    and the discretizations

- variable_steps (Array)
    discretization step for each component


Returns :
- variable (Array)
    the vector variable we want to convert to an index (integer)

"""
function variable_from_index( variable_ind::Int,
                    lower_bounds::Array,
                    variable_sizes::Array,
                    variable_steps::Array)

    dim = length(lower_bounds)
    index = variable_ind-1
    variable = zeros(dim)

    if (variable_ind>-1)
        for i=1:dim
            variable[i] = lower_bounds[i] + variable_steps[i] * ((index % variable_sizes[i]))
            index += -(index % variable_sizes[i])
            index /= variable_sizes[i]
        end
    end

    return variable
end

"""
Compute nearest neighbor of a continuous variable in the discrete variable space

Parameters:
- variable (Array)
    the vector variable we want to compute the nearest neighbor index (integer)

- lower_bounds (Array)
    the lower bounds for each component of the variable

- variable_sizes (Array)
    the number of possibilities at each component knowing the upper and lower bounds
    and the discretizations

- variable_steps (Array)
    discretization step for each component


Returns :
- index_of_nearest_neighbor (integer)
    the index of the nearest neighbor in the grid of the variable
    for the problem knowing the upper and lower bounds
    as well as the discretization
"""
function nearest_neighbor( variable::Array,
                    lower_bounds::Array,
                    variable_sizes::Array,
                    variable_steps::Array)

    index = index_from_variable(variable, lower_bounds, variable_sizes, variable_steps)-1

    neighbors = [index]
    if ((index % variable_sizes[1]) < (variable_sizes[1]-1))
        push!(neighbors, index+1)
    end

    if length(variable)>1
        K=1
        for i = 2:length(variable)
            K=K*variable_sizes[i-1]
            neighbors0 = copy(neighbors)
            for j in neighbors0
                if (((j-j%K)/K)%variable_sizes[i] <variable_sizes[i]-1)
                    push!(neighbors, j + K)
                end
            end
        end
    end

    ref = -1
    ref_dist = Inf

    for inn0 in neighbors
        nn0 = variable_from_index(inn0 + 1, lower_bounds, variable_sizes,
                                    variable_steps)

        dist = norm(variable-nn0)

        if (dist < ref_dist)
            ref =  inn0 + 1
            ref_dist = dist
        end
    end

    return ref
end


"""
Compute barycentre of value function with state neighbors in a discrete
state space
"""
function value_function_barycentre( model::SPModel,
                                    param::SDPparameters,
                                    V::Array,
                                    time::Int,
                                    variable::Array)

    TF = model.stageNumber
    value_function = 0.
    neighbors_sum = 0.
    lower_bounds = [ i for (i , j) in model.xlim]
    variable_sizes = param.stateVariablesSizes
    variable_steps = param.stateSteps

    index = index_from_variable(variable, lower_bounds, variable_sizes, variable_steps);

    neighbors = [index]
    if ((index % variable_sizes[1]) < (variable_sizes[1]-1))
        push!(neighbors, index+1)
    end

    if length(variable)>1
        K=1
        for i = 2:length(variable)
            K=K*variable_sizes[i-1]
            neighbors0 = copy(neighbors)
            for j in neighbors0
                if (((j-j%K)/K)%variable_sizes[i] <variable_sizes[i]-1)
                    push!(neighbors, j + K)
                end
            end
        end
    end

    sum_dist = 0.

    for inn0 in neighbors
        nn0 = variable_from_index(inn0 + 1, lower_bounds, variable_sizes,
                                    variable_steps)
        dist = norm(variable-nn0)
        value_function += dist*V[inn0 + 1, time]
        neighbors_sum += V[inn0 + 1, time]
        sum_dist += dist
    end

    return neighbors_sum-(value_function/sum_dist)
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

    TF = model.stageNumber
    V = zeros(Float64, param.totalStateSpaceSize, TF+1)
    x = zeros(Float64, model.dimStates)
    x1 = zeros(Float64, model.dimStates)
    law = model.noises

    u_lower_bounds = [ i for (i , j) in model.ulim]
    x_lower_bounds = [ i for (i , j) in model.xlim]

    count_iteration = 1

    #Compute final value functions
    for indx = 1:(param.totalStateSpaceSize)
        x = variable_from_index(indx, x_lower_bounds, param.stateVariablesSizes,
                                param.stateSteps)
        V[indx, TF+1] = model.finalCostFunction(x)
    end

    #Construct a progress meter
    p = Progress(TF*param.totalStateSpaceSize, 1)

    #Display start of the algorithm in DH and HD cases
    #Define the loops order in sdp
    if (param.infoStructure == "DH")
        if display
            println("Starting stochastic dynamic programming
                    decision hazard computation")
        end

        #Loop over time
        for t = TF:-1:1
            count_iteration = count_iteration + 1

            #Loop over states
            for indx = 1:(param.totalStateSpaceSize)

                if display
                    next!(p)
                end

                v = Inf
                v1 = 0
                indu1 = -1
                Lv = 0
                x = variable_from_index(indx, x_lower_bounds,
                                        param.stateVariablesSizes,
                                        param.stateSteps)

                #Loop over controls
                for indu = 1:(param.totalControlSpaceSize)

                    v1 = 0
                    count = 0
                    u = variable_from_index(indu, u_lower_bounds,
                                            param.controlVariablesSizes,
                                            param.controlSteps)

                    #Loop over uncertainty samples
                    for w = 1:param.monteCarloSize

                        wsample = sampling( law, t)
                        x1 = model.dynamics(t, x, u, wsample)

                        if model.constraints(t, x1, u, wsample)

                            count = count + 1
                            indx1 = nearest_neighbor(x1,
                                                    x_lower_bounds,
                                                    param.stateVariablesSizes,
                                                    param.stateSteps)
                            Lv = model.costFunctions(t, x, u, wsample)
                            v1 += Lv + V[indx1, t + 1]

                        end
                    end

                    if (count>0)

                        v1 = v1 / count

                        if (v1 < v)

                            v = v1
                            indu1 = indu

                        end
                    end
                end

                V[indx, t] = v
            end
        end

    else
        if display
            println("Starting stochastic dynamic programming
                    hazard decision computation")
        end

        #Loop over time
        for t = TF:-1:1
            count_iteration = count_iteration + 1

            #Loop over states
            for indx = 1:(param.totalStateSpaceSize)

                if display
                    next!(p)
                end

                v     = 0
                indu1 = -1
                Lv    = 0
                x     = variable_from_index(indx, x_lower_bounds,
                                            param.stateVariablesSizes,
                                            param.stateSteps)

                count = 0
                #Loop over uncertainty samples
                for w in 1:param.monteCarloSize

                    admissible_u_w_count = 0
                    v_x_w = 0
                    v_x_w1 = Inf
                    wsample = sampling( law, t)

                    #Loop over controls
                    for indu in 1:(param.totalControlSpaceSize)

                        u = variable_from_index(indu, u_lower_bounds,
                                                param.controlVariablesSizes,
                                                param.controlSteps)

                        x1 = model.dynamics(t, x, u, wsample)

                        if model.constraints(t, x1, u, wsample)

                            if (admissible_u_w_count == 0)
                                admissible_u_w_count = 1
                            end

                            indx1 = nearest_neighbor(x1, x_lower_bounds,
                                                    param.stateVariablesSizes,
                                                    param.stateSteps)
                            Lv = model.costFunctions(t, x, u, wsample)
                            v_x_w1 = Lv + V[indx1, t+1]

                            if (v_x_w1 < v_x_w)
                                v_x_w = v_x_w1
                            end

                        end
                    end

                    v = v + v_x_w
                    count += 1
                end

                if (count>0)
                    v = v / count
                end

                V[indx, t] = v
            end
        end
    end

    return V
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
    u_lower_bounds = [ i for (i , j) in model.ulim]
    x_lower_bounds = [ i for (i , j) in model.xlim]

    controls = Inf*ones(TF, 1, model.dimControls)
    states = Inf*ones(TF + 1, 1, model.dimStates)

    inc = 0
    for xj in X0
        inc = inc + 1
        states[1, 1, inc] = xj
    end

    J = 0
    xRef = X0

    if (param.infoStructure == "DH")
        #Decision hazard forward simulation
        for t = 1:TF

            x = states[t,1,:]

            induRef = 0

            LvRef = Inf

            for indu = 1:(param.totalControlSpaceSize)

                u = variable_from_index(indu, u_lower_bounds,
                                        param.controlVariablesSizes,
                                        param.controlSteps)

                countW = 0.
                Lv = 0.
                for w = 1:param.monteCarloSize

                    wsample = sampling( law, t)

                    x1 = model.dynamics(t, x, u, wsample)

                    indx1 = nearest_neighbor(x1, x_lower_bounds,
                                            param.stateVariablesSizes,
                                            param.stateSteps)

                    if model.constraints(t, x1, u, scenario[t])
                        Lv = Lv + model.costFunctions(t, x, u, scenario[t]) + value[indx1, t+1]
                        countW = countW +1.
                    end
                end
                Lv = Lv/countW
                if (Lv < LvRef)&(countW>0)
                    induRef = indu
                    xRef = model.dynamics(t, x, u, scenario[t,1,:])
                    LvRef = Lv
                end
            end

            inc = 0
            uRef = variable_from_index(induRef, u_lower_bounds,
                                        param.controlVariablesSizes,
                                        param.controlSteps)
            for uj in uRef
                inc = inc +1
                controls[t,1,inc] = uj
            end

            inc = 0
            for xj in xRef
                inc = inc +1
                states[t+1,1,inc] = xj
            end

            J = J + model.costFunctions(t, x, uRef, scenario[t,1,:])

        end


    else
        #Hazard desision forward simulation
        for t = 1:TF

            x = states[t,1,:]

            induRef = 0

            indxRef = 0
            LvRef = Inf

            for indu = 1:(param.totalControlSpaceSize)

                u = variable_from_index(indu, u_lower_bounds,
                                        param.controlVariablesSizes,
                                        param.controlSteps)

                x1 = model.dynamics(t, x, u, scenario[t,1,:])

                indx1 = nearest_neighbor(x1, x_lower_bounds,
                                            param.stateVariablesSizes,
                                            param.stateSteps)

                if model.constraints(t, x1, u, scenario[t])
                    Lv = model.costFunctions(t, x, u, scenario[t,1,:]) + value[indx1, t+1]
                    if (Lv < LvRef)
                        induRef = indu
                        xRef = model.dynamics(t, x, u, scenario[t,1,:])
                        LvRef = Lv
                    end
                end

            end

            inc = 0
            uRef = variable_from_index(induRef, u_lower_bounds,
                                                param.controlVariablesSizes,
                                                param.controlSteps)
            for uj in uRef
                inc = inc +1
                controls[t,1,inc] = uj
            end

            inc = 0
            for xj in xRef
                inc = inc +1
                states[t+1,1,inc] = xj
            end

            J = J + model.costFunctions(t, x, uRef, scenario[t])

        end
    end

    x = states[TF+1,1,:]
    J = J + model.finalCostFunction(x)

    return J, states, controls
end

