#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Define the extensive formulation to check the results of small problems.
#  The problem is instantiate on a tree.
#############################################################################

"""
Contruct the scenario tree and solve the problem with measurability constraints
"""
function extensive_formulation(model,
                               param)

    #Recover all the constant in the model or in param
    laws = model.noises

    DIM_STATE = model.dimStates
    DIM_CONTROL = model.dimControls

    X_init = model.initialState

    T = model.stageNumber-1

    mod = Model(solver=param.solver)


    #Calculate the number of nodes n at each step on the scenario tree
    N = Array{Int64,2}(zeros(T+1,1))
    N[1] = 1
    for t = 1 : (T)
        N[t+1] = N[t]*laws[t].supportSize
    end



    #Define the variables for the extensive formulation
    #At each node, we have as many variables as nodes
    @variable(mod,  u[t=1:T,n=1:DIM_CONTROL*N[t+1]])
    @variable(mod,  x[t=1:T+1,n=1:DIM_STATE*N[t]])
    @variable(mod,  c[t=1:T,n=1:laws[t].supportSize*N[t]])


    #Computes the total probability of each node from the conditional probabilities
    proba    = Vector{typeof(laws[1].proba)}(T)
    proba[1] = laws[1].proba
    for t = 2 : T
        proba[t] = zeros(N[t+1])
        for j = 1 : N[t]
            for k = 1 : laws[t].supportSize
                proba[t][laws[t].supportSize*(j-1)+k] = laws[t].proba[k]*proba[t-1][j]
            end
        end
    end

    #Instantiate the problem creating dynamic constraint at each node
    for t = 1 : (T)
        for n = 1 : N[t]
            @constraint(mod,[x[t,DIM_STATE*(n-1)+k] for k = 1:DIM_STATE] .>= [model.xlim[k][1] for k = 1:DIM_STATE])
            @constraint(mod,[x[t,DIM_STATE*(n-1)+k] for k = 1:DIM_STATE] .<= [model.xlim[k][2] for k = 1:DIM_STATE])
            for xi = 1 : laws[t].supportSize
                m = (n-1)*laws[t].supportSize+xi

                #Add bounds constraint on the control
                @constraint(mod,[u[t,DIM_CONTROL*(m-1)+k] for k = 1:DIM_CONTROL] .>= [model.ulim[k][1] for k = 1:DIM_CONTROL])
                @constraint(mod,[u[t,DIM_CONTROL*(m-1)+k] for k = 1:DIM_CONTROL] .<= [model.ulim[k][2] for k = 1:DIM_CONTROL])

                #Add dynamic constraints
                @constraint(mod,
                [x[t+1,DIM_STATE*(m-1)+k] for k = 1:DIM_STATE] .== model.dynamics(t,
                                                                                    [x[t,DIM_STATE*(n-1)+k] for k = 1:DIM_STATE],
                                                                                    [u[t,DIM_CONTROL*(m-1)+k] for k = 1:DIM_CONTROL],
                                                                                    laws[t].support[xi]))

                #Add constraints to define the cost at each node
                @constraint(mod,
                c[t,m] == model.costFunctions(t,
                                                [x[t,DIM_STATE*(n-1)+k] for k = 1:DIM_STATE],
                                                [u[t,DIM_CONTROL*(m-1)+k] for k = 1:DIM_CONTROL],
                                                laws[t].support[xi]))
            end
        end
    end

    #Initial state
    @constraint(mod, [x[1,k] for k = 1:DIM_STATE] .== X_init)


    #Define the objective of the function
    @objective(mod, Min, sum{ sum{proba[t][laws[t].supportSize*(n-1)+k]*c[t,laws[t].supportSize*(n-1)+k],k = 1:laws[t].supportSize} , t = 1:T, n=1:div(N[t+1],laws[t].supportSize)})

    status = solve(mod)

    solved = (status == :Optimal)

    if solved
        return getobjectivevalue(mod), status
    else
        return -1., status
    end

end
