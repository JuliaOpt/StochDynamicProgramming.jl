#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Probability utilities
#############################################################################

type NoiseLaw
    # Number of points in distribution:
    supportSize::Int64
    # Position of points:
    support::Array{Float64,2}
    # Probabilities of points:
    proba::Vector{Float64}
end


"""
Instantiate an element of NoiseLaw


Parameters:
- supportSize (Int64)
    Number of points in discrete distribution

- support
    Position of each point

- proba
    Probabilities of each point


Return:
- NoiseLaw

"""
function NoiseLaw_const(supportSize, support, proba)
    supportSize = convert(Int64,supportSize)
    if ndims(support)==1
        support = reshape(support,1,length(support))
    end

    if ndims(proba) == 2
        proba = vec(proba)
    elseif  ndims(proba) >= 2
        proba = squeeze(proba,1)
    end

    return NoiseLaw(supportSize,support,proba)
end



"""
Generic constructor to instantiate NoiseLaw


Parameters:
- support
    Position of each point

- proba
    Probabilities of each point


Return:
- NoiseLaw

"""
function NoiseLaw(support, proba)
    return NoiseLaw_const(length(proba), support, proba)
end




"""
Simulate n scenario according to a given NoiseLaw

Parameters:
- law::Vector{NoiseLaw}
    Vector of discrete independent random variables

- n::Int
    number of simulations computed


Returns :
- scenarios Array(Float64,n,T)
    an Array of scenario, scenarios[i,:] being the ith noise scenario
"""
function simulate(law::Vector{NoiseLaw}, n::Int64)
    if n <= 0
        error("negative number of simulations")
    end
    Tf = length(law)
    scenarios = Array{Vector{Float64}}(n,Tf)
    for i = 1:n#TODO can be parallelized
        scenario = []
        for t=1:Tf
            new_val = law[t].support[:,rand(Categorical(law[t].proba))]
            push!(scenario, new_val)
        end
        scenarios[i,:]=scenario
    end

    return scenarios
end


"""
Simulate n scenario and return a 3D array


Parameters:
- laws (Vector{NoiseLaw})
    Distribution laws corresponding to each timestep

- dims (3-tuple)
    Dimension of array to return. Its shape is:
        (time, numberScenario, dimAlea)

Return:
- Array{Float64, 3}

"""
function simulate_scenarios(laws, dims::Tuple)

    if typeof(laws) == Distributions.Normal
        scenarios = rand(laws, dims)
    else
        scenarios = zeros(dims)

    for k=1:dims[2]
        for t=1:dims[1]
            gen = Categorical(laws[t].proba)
            scenarios[t, k, :] = laws[t].support[:, rand(gen)]
        end

    end
    end
    return scenarios

end
