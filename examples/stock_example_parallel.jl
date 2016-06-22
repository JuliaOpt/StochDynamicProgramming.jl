#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Solving an decision hazard stock problem using multiprocessed dynamic programming :
# We decide which quantity to store before the randomness happens
# If too much energy comes, the excess is wasted
# If not enough energy comes, we lower accordingly what was decided to discharge
# Min   E [\sum_{t=1}^TF c_t C_t(u_t)]
# s.t.    s_{t+1} = s_t - u_t + xi_t, if 0 <= s_t - u_t + xi_t <= s_{max}
#         s_{t+1} = s_{max}, if s_t - u_t + xi_t > s_{max}
#         s_{t+1} = 0, if s_t - u_t + xi_t < 0
#         C_t(u_t) = u_t, if 0 <= s_t - u_t + xi_t
#         C_t(u_t) = -(0 - s_{t} - xi_t), otherwise
#         s_0 given
#         0 <= s_t <= 1
#         u_min <= u_t <= u_max
#         u_t choosen knowing xi_1 .. xi_t
#############################################################################

import StochDynamicProgramming, Distributions
println("library loaded")

@everywhere begin

    run_sdp = true

    ######## Stochastic Model  Parameters  ########
    const N_STAGES = 50
    const COSTS = rand(N_STAGES)

    const CONTROL_MAX = 0.5
    const CONTROL_MIN = 0

    const STATE_MAX = 1
    const STATE_MIN = 0

    const XI_MAX = 0.3
    const XI_MIN = 0
    const N_XI = 10
    # initial stock
    const S0 = 0.5

    # create law of noises
    proba = 1/N_XI*ones(N_XI) # uniform probabilities
    xi_support = collect(linspace(XI_MIN,XI_MAX,N_XI))
    xi_law = StochDynamicProgramming.NoiseLaw(xi_support, proba)
    xi_laws = StochDynamicProgramming.NoiseLaw[xi_law for t in 1:N_STAGES-1]

    # Define dynamic of the stock:
    function dynamic(t, x, u, xi)
        return [min(STATE_MAX, max(STATE_MIN,x[1] - u[1] + xi[1]))]
    end

    # Define cost corresponding to each timestep:
    function cost_t(t, x, u, xi)
        a = u[1]
        b = dynamic(t, x, u, xi)[1]
        if b==0
            a = x[1] + xi[1]
        end
        return -COSTS[t] * a
    end

    ######## Setting up the SPmodel
    s_bounds = [(0, 1)]
    u_bounds = [(CONTROL_MIN, CONTROL_MAX)]
    spmodel = StochDynamicProgramming.LinearDynamicLinearCostSPmodel(N_STAGES,
                                                                    u_bounds,
                                                                    [S0],
                                                                    cost_t,
                                                                    dynamic,
                                                                    xi_laws)
    StochDynamicProgramming.set_state_bounds(spmodel, s_bounds)

    scenarios = StochDynamicProgramming.simulate_scenarios(xi_laws,1000)

    stateSteps = [0.01]
    controlSteps = [0.0001]
    infoStruct = "DH" # noise at time t is not known before taking the decision at time t

    paramSDP = StochDynamicProgramming.SDPparameters(spmodel, stateSteps,
                                                    controlSteps, infoStruct)
end

Vs = []

@time for i in 1:1
Vs = StochDynamicProgramming.solve_DP(spmodel,paramSDP, 1)
end

lb_sdp = StochDynamicProgramming.get_bellman_value(spmodel,paramSDP,Vs)
println("Value obtained by SDP: "*string(lb_sdp))
costsdp, states, stocks = StochDynamicProgramming.sdp_forward_simulation(spmodel,paramSDP,scenarios,Vs)
println(mean(costsdp))

