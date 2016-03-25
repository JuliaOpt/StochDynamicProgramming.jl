#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Define the extensive formulation to check the results of small problems
#############################################################################

"""
Contruct the scenario tree and solve the problem with measurability constraints
"""
function extensive_formulation(model,
                               params)


    #Recover all the constant in the model or in param
    laws = model.noises
    N_NOISES = laws[1].supportSize

    DIM_STATE = model.dimStates
    DIM_CONTROL = model.dimControls

    X_init = model.initialState


    T = model.stageNumber-1

    mod = Model(solver=params.solver)


    #Calculate the number of nodes n at each step on the scenario tree
    N = Array{Int64,2}(zeros(T+1,1))
    N[1] = 1
    for t = 1 : (T)
        N[t+1] = N[t]*laws[t].supportSize
    end



    #Define the variables for the extensive formulation
    #At each node, we have as many variables as nodes
    @defVar(mod,  u[t=1:T,n=1:DIM_CONTROL*N[t+1]])
    @defVar(mod,  x[t=1:T+1,n=1:DIM_STATE*N[t]])
    @defVar(mod,  c[t=1:T,n=1:N_NOISES*N[t]])


    #Computes the total probability of each node from the conditional probabilities
    proba = []
    push!(proba, laws[1].proba)
    for t = 2 : T
        push!(proba, zeros(N[t+1]))
        for j = 1 : N[t]
            for k = 1 : N_NOISES
                proba[t][N_NOISES*(j-1)+k] = laws[t].proba[k]*proba[t-1][j]
            end
        end
    end


    #Instantiate the problem creating dynamic constraint at each node
    for t = 1 : (T)
        for n = 1 : N[t]
            @addConstraint(mod,[x[t,DIM_STATE*(n-1)+k] for k = 1:DIM_STATE] .>= [model.xlim[k][1] for k = 1:DIM_STATE])
            @addConstraint(mod,[x[t,DIM_STATE*(n-1)+k] for k = 1:DIM_STATE] .<= [model.xlim[k][2] for k = 1:DIM_STATE])
            for xi = 1 : laws[t].supportSize
                m = (n-1)*laws[t].supportSize+xi

                #Add bounds constraint on the control
                @addConstraint(mod,[u[t,DIM_CONTROL*(m-1)+k] for k = 1:DIM_CONTROL] .>= [model.ulim[k][1] for k = 1:DIM_CONTROL])
                @addConstraint(mod,[u[t,DIM_CONTROL*(m-1)+k] for k = 1:DIM_CONTROL] .<= [model.ulim[k][2] for k = 1:DIM_CONTROL])
    
                #Add dynamic constraints
                @addConstraint(mod,
                [x[t+1,DIM_STATE*(m-1)+k] for k = 1:DIM_STATE] .== model.dynamics(t,
                                                                                    [x[t,DIM_STATE*(n-1)+k] for k = 1:DIM_STATE],
                                                                                    [u[t,DIM_CONTROL*(m-1)+k] for k = 1:DIM_CONTROL],
                                                                                    laws[t].support[xi]))
                                                                                   
                #Add constraints to define the cost at each node
                @addConstraint(mod,
                c[t,m] == model.costFunctions(t,
                                                [x[t,DIM_STATE*(n-1)+k] for k = 1:DIM_STATE],
                                                [u[t,DIM_CONTROL*(m-1)+k] for k = 1:DIM_CONTROL],
                                                laws[t].support[xi]))
            end
        end
    end

    #Initial state
    @addConstraint(mod, [x[1,k] for k = 1:DIM_STATE] .== X_init)


    #Define the objective of the function
    @setObjective(mod, Min, sum{ sum{proba[t][N_NOISES*(n-1)+k]*c[t,N_NOISES*(n-1)+k],k = 1:N_NOISES} , t = 1:T, n=1:div(N[t+1],N_NOISES)})

    status = solve(mod)

    solved = (status == :Optimal)

    if solved
        return getObjectiveValue(mod), status
    else
        return -1., status
    end

end
