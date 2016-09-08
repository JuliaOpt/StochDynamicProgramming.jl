#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  utility functions
#
#############################################################################


"""
Dump Polyhedral functions in a text file.

# Arguments
* `dump::String`:
    Name of output filt
* `Vts::Vector{PolyhedralFunction}`:
    Vector of polyhedral functions to save
"""
function dump_polyhedral_functions(dump::AbstractString, Vts::Vector{PolyhedralFunction})
    outfile = open(dump, "w")

    time = 1
    for V in Vts
        ncuts = V.numCuts

        for i in 1:ncuts
            write(outfile, join((time, V.betas[i], tuple(V.lambdas[i, :]...)...), ","), "\n")
        end

        time += 1
    end

    close(outfile)
end


"""
Import cuts from a dump text file.

# Argument
* `dump::String`:
    Name of file to import
"""
function read_polyhedral_functions(dump::AbstractString)
    # Store all values in a two dimensional array:
    process = readdlm(dump, ',')

    ntime = round(Int, maximum(process[:, 1]))
    V = Vector{PolyhedralFunction}(ntime)
    total_cuts = size(process)[1]
    dim_state = size(process)[2] - 2

    for it in 1:total_cuts
        # Read time corresponding to cuts:
        t = round(Int, process[it, 1])
        beta = process[it, 2]
        lambda = vec(process[it, 3:end])

        try
            V[t].lambdas = vcat(V[t].lambdas, lambda')
            V[t].betas = vcat(V[t].betas, beta)
            V[t].numCuts += 1
        catch
            V[t] = PolyhedralFunction([beta],
                                       reshape(lambda, 1, dim_state), 1)
        end
    end
    return V
end


""" Remove redundant cuts in Polyhedral Value function `V`"""
function remove_cuts(V::PolyhedralFunction)
    Vf = hcat(V.lambdas, V.betas)
    Vf = unique(Vf, 1)
    return PolyhedralFunction(Vf[:, end], Vf[:, 1:end-1], size(Vf)[1])
end


""" Remove redundant cuts in a vector of Polyhedral Functions `Vts`."""
function remove_redundant_cuts!(Vts::Vector{PolyhedralFunction})
    n_functions = length(Vts)-1
    for i in 1:n_functions
        Vts[i] = remove_cuts(Vts[i])
    end
end

"""Concatenate collection of arrays of PolyhedralFunction."""
function catcutsarray(polyfunarray::Vector{StochDynamicProgramming.PolyhedralFunction}...)
    assert(length(polyfunarray) > 0)
    ntimes = length(polyfunarray[1])
    # Concatenate cuts in polyfunarray, and discard final time as we do not add cuts at final time:
    concatcuts = StochDynamicProgramming.PolyhedralFunction[catcuts([V[t] for V in polyfunarray]...) for t in 1:ntimes-1]
    return vcat(concatcuts, polyfunarray[1][end])
end


"""Concatenate collection of PolyhedralFunction."""
function catcuts(Vts::StochDynamicProgramming.PolyhedralFunction...)
    betas = vcat([V.betas for V in Vts]...)
    lambdas = vcat([V.lambdas for V in Vts]...)
    numcuts = sum([V.numCuts for V in Vts])
    return StochDynamicProgramming.PolyhedralFunction(betas, lambdas, numcuts)
end

"""
Extract a vector stored in a 3D Array

# Arguments
* `input_array::Array{Float64, 3}`:
    array storing the values of vectors
* `nx::Int64`:
    Position of vector in first dimension
* `ny::Int64`:
    Position of vector in second dimension

# Return
`Vector{Float64}`
"""
function extract_vector_from_3Dmatrix(input_array::Array{Float64, 3},
                                      nx::Int64,
                                      ny::Int64)
    info("extract_vector_from_3Dmatrix is now deprecated. Use collect instead.")
    state_dimension = size(input_array)[3]
    return reshape(input_array[nx, ny, :], state_dimension)
end


"""
Generate a random state.

# Arguments
* `model::SPModel`:

# Return
`Vector{Float64}`, shape=`(model.dimStates,)`

"""
function get_random_state(model)
    return [model.xlim[i][1] + rand()*(model.xlim[i][2] - model.xlim[i][1]) for i in 1:model.dimStates]
end


"""
Estimate the upper bound with a distribution of costs

# Description
Given a probability p, we have a confidence interval:
[mu - alpha sigma/sqrt(n), mu + alpha sigma/sqrt(n)]
where alpha depends upon p.

Upper bound is the max of this interval.

# Arguments
* `cost::Vector{Float64}`:
    Costs values
* `probability::Float`:
    Probability to be inside the confidence interval

# Return
estimated-upper bound as `Float`
"""
function upper_bound(cost::Vector{Float64}, probability=.975)
    tol = sqrt(2) * erfinv(2*probability - 1)
    return mean(cost) + tol*std(cost)/sqrt(length(cost))
end


"""
Test if the stopping criteria is fulfilled.

Return true if |V0 - upb|/V0 < epsilon

# Arguments
* `V0::Float`:
    Approximation of initial cost
* `upb::Float`:
    Approximation of the upper bound given by Monte-Carlo estimation
*  `epsilon::Float`:
    Sensibility

# Return
`Bool`
"""
function test_stopping_criterion(V0::Float64, upb::Float64, epsilon::Float64)
    return abs((V0-upb)/V0) < epsilon
end

