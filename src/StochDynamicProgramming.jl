#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# SDDP is an implementation of the Stochastic Dual Dynamic Programming
# algorithm for multi-stage stochastic convex optimization problem
# see TODO
#############################################################################

module StochDynamicProgramming

using JuMP, Distributions

export solve_SDDP, NoiseLaw, simulate_scenarios,
        SDDPparameters, LinearDynamicLinearCostSPmodel, set_state_bounds,
        PiecewiseLinearCostSPmodel,
        PolyhedralFunction, NextStep, forward_simulations,
        StochDynProgModel, SDPparameters, sdp_optimize,
        sdp_forward_simulation, sampling, extensive_formulation

include("objects.jl")
include("utils.jl")
include("oneStepOneAleaProblem.jl")
include("forwardBackwardIterations.jl")
include("noises.jl")
include("SDDPoptimize.jl")
include("extensiveFormulation.jl")
include("SDPoptimize.jl")
end
