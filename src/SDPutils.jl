module SDPutils
using Interpolations

export index_from_variable, real_index_from_variable,
        compute_V_given_t_DH, compute_V_given_t_HD

"""
Convert the state and control float tuples (stored as arrays or tuples) of the
problem into int tuples that can be used as indexes for the discretized
value functions

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
- index (tuple of floats)
    the indexes of the variable
"""
function real_index_from_variable( variable,
                    bounds::Array,
                    variable_steps::Array)

    return tuple([1 + ( variable[i] - bounds[i][1] )/variable_steps[i] for i in 1:length(variable)]...)
end


"""
Computes the value function at time t evaluated at state x in a decision
hazard setting

Parameters:
- sampling_size (int)
    the size of the uncertainty space

- samples (Array)
    the uncertainties realizations

- probas (Array)
    the probabilities of all the uncertainties realizations

- u_bounds (Array)
    the control variables bounds

- x_bounds (Array)
    the state variables bounds

- x_steps (Array)
    the state variables steps

- x_dim (int)
    the number of state variables

- product_controls (Array)
    the discretized control space

- dynamics (Function)
    the dynamic of the problem

- constraints (Function)
    the constraints of the problem

- cost (Function)
    the cost function of the problem

- V (Array)
    the value functions ofthe problem

- Vitp (Interpolations)
    the interpolated value function at time t+1

- t (float)
    the time step at which the value function is computed

- x (Array or Tuple)
    the state at which the value function needs to be evaluated

"""
function compute_V_given_x_t_DH(sampling_size, samples, probas, u_bounds,
                                x_bounds, x_steps, x_dim, product_controls,
                                dynamics, constraints, cost, V, Vitp, t, x)
    expected_V = Inf
    optimal_u = tuple()
    current_cost = 0
    #Loop over controls
    for u = product_controls

        expected_V_u = 0.
        count_admissible_w = 0

        for w = 1:sampling_size
            w_sample = samples[:, w]
            proba = probas[w]
            next_state = dynamics(t, x, u, w_sample)

            if constraints(t, next_state, u, w_sample)

                count_admissible_w = count_admissible_w + proba
                ind_next_state = real_index_from_variable(next_state, x_bounds,
                                                            x_steps)
                next_V = Vitp[ind_next_state...]
                current_cost = cost(t, x, u, w_sample)
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

"""
Computes the value function at time t evaluated at state x in a hazard
decision setting

Parameters:
- sampling_size (int)
    the size of the uncertainty space

- samples (Array)
    the uncertainties realizations

- probas (Array)
    the probabilities of all the uncertainties realizations

- u_bounds (Array)
    the control variables bounds

- x_bounds (Array)
    the state variables bounds

- x_steps (Array)
    the state variables steps

- x_dim (int)
    the number of state variables

- product_controls (Array)
    the discretized control space

- dynamics (Function)
    the dynamic of the problem

- constraints (Function)
    the constraints of the problem

- cost (Function)
    the cost function of the problem

- V (Array)
    the value functions ofthe problem

- Vitp (Interpolations)
    the interpolated value function at time t+1

- t (float)
    the time step at which the value function is computed

- x (Array or Tuple)
    the state at which the value function needs to be evaluated

"""
function compute_V_given_x_t_HD(sampling_size, samples, probas, u_bounds,
                                x_bounds, x_steps, x_dim, product_controls,
                                dynamics, constraints, cost, V, Vitp, t, x)

    expected_V = 0.
    current_cost = 0.
    count_admissible_w = 0.
    admissible_u_w_count = 0
    best_V_x_w = Inf
    next_V_x_w = Inf

    #Compute expectation
    for w in 1:sampling_size
        admissible_u_w_count = 0
        best_V_x_w = Inf
        next_V_x_w = Inf
        w_sample = samples[:, w]
        proba = probas[w]

        #Loop over controls to find best next value function
        for u in product_controls

            next_state = dynamics(t, x, u, w_sample)

            if constraints(t, next_state, u, w_sample)
                admissible_u_w_count += 1
                current_cost = cost(t, x, u, w_sample)
                ind_next_state = real_index_from_variable(next_state, x_bounds,
                                                            x_steps)
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