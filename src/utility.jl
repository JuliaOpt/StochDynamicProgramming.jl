#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  Some useful methods
#
#############################################################################

"""
Instantiate a Polyhedral function corresponding to f -> 0

"""
function get_null_value_functions()
    return PolyhedralFunction(zeros(1), zeros(1, 1), 1)
end


"""
Extract a vector stored in a 3D Array


Parameters:
- input_array (Array{Float64, 3})
    array storing the values of vectors

- nx (Int64)
    Position of vector in first dimension

- ny (Int64)
    Position of vector in second dimension

Return:
- Vector{Float64}

"""
function extract_vector_from_3Dmatrix(input_array::Array{Float64, 3},
                                      nx::Int64,
                                      ny::Int64)

    state_dimension = size(input_array)[3]
    return reshape(input_array[ny, nx, :], state_dimension)
end


"""
Estimate the upper bound with the Monte-Carlo error


Parameters:
- model (SPmodel)
    the stochastic problem we want to optimize

- param (SDDPparameters)
    the parameters of the SDDP algorithm

- V (bellmanFunctions)
    the current estimation of Bellman's functions

- forwardPassNumber (int)
    number of Monte-Carlo simulation

- returnMCerror (Bool)
    return or not the estimation of the MC error


Returns :
- estimated-upper bound

- Monte-Carlo error on the upper bound (if returnMCerror)

"""
function upper_bound(cost::Vector{Float64})
    return mean(cost) + 1.96*std(cost)/sqrt(length(cost))
end


"""
Test if the stopping criteria is fulfilled.

Return true if |V0 - upb|/V0 < epsilon


Parameters:
- V0 (Float)
    Approximation of initial cost

- upb (Float)
    Approximation of the upper bound given by Monte-Carlo estimation

- epsilon (Float)
    Sensibility


Return:
Bool

"""
function test_stopping_criterion(V0::Float64, upb::Float64, epsilon::Float64)
    return abs((V0-upb)/V0) < epsilon
end
