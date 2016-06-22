#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# SDDP is an implementation of the Stochastic Dual Dynamic Programming
# algorithm for multi-stage stochastic convex optimization problem
# see TODO
#############################################################################
include("SDPutils.jl")

module StochDynamicProgramming

using JuMP, Distributions
import SDPutils

export solve_SDDP,
            NoiseLaw, simulate_scenarios,
        SDDPparameters, LinearDynamicLinearCostSPmodel, set_state_bounds,
        PiecewiseLinearCostSPmodel, extensive_formulation,
        PolyhedralFunction, NextStep, forward_simulations,
        StochDynProgModel, SDPparameters, solve_DP,
        sdp_forward_simulation, sampling, get_control, get_bellman_value

include("objects.jl")
include("utils.jl")
include("oneStepOneAleaProblem.jl")
include("forwardBackwardIterations.jl")
include("noises.jl")
include("SDDPoptimize.jl")
include("extensiveFormulation.jl")
include("SDPoptimize.jl")
end
