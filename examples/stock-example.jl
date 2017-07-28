#  Copyright 2015, Vincent Leclere, Francois Pacaud and Henri Gerard
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Compare different ways of solving a stock problem :
# Min   E [\sum_{t=1}^TF c_t u_t]
# s.t.    s_{t+1} = s_t + u_t - xi_t, s_0 given    # Stock dynamic equation
#         0 <= s_t <= 1                            # Constraint on stock
#         u_min <= u_t <= u_max                    # Constraint on control
#         u_t choosen knowing xi_1 .. xi_t         # information constraint
#############################################################################

using StochDynamicProgramming, Clp # or CPLEX
println("library loaded")

run_sddp = true # false if you don't want to run sddp
run_sdp  = true # false if you don't want to run sdp
run_ef   = true # false if you don't want to run extensive formulation
test_simulation = true # false if you don't want to test your strategies

######## Optimization parameters  ########
# choose the LP solver used.
SOLVER = ClpSolver() 			   # require "using Clp"
#const SOLVER = CplexSolver(CPX_PARAM_SIMDISPLAY=0, CPX_PARAM_SCRIND=0) # require "using CPLEX"

# convergence test
MAX_ITER = 10 # number of iterations of SDDP
step = 0.01   # discretization step for SDP

######## Stochastic Model  Parameters  ########
N_STAGES = 6              # number of stages of the SP problem
COSTS = [sin(3*t)-1 for t in 1:N_STAGES-1]
#const COSTS = rand(N_STAGES)    # randomly generating deterministic costs

CONTROL_MAX = 0.5         # bounds on the control
CONTROL_MIN = 0

XI_MAX = 0.3              # bounds on the noise
XI_MIN = 0
N_XI = 10                 # discretization of the noise

S0 = 0.5                  # initial stock

# create law of noises
proba = 1/N_XI*ones(N_XI) # uniform probabilities
xi_support = collect(linspace(XI_MIN,XI_MAX,N_XI))
xi_law = NoiseLaw(xi_support, proba)
xi_laws = NoiseLaw[xi_law for t in 1:N_STAGES-1]

# Define dynamic of the stock:
function dynamic(t, x, u, xi)
    return [x[1] + u[1] - xi[1]]
end

# Define cost corresponding to each timestep:
function cost_t(t, x, u, w)
    return COSTS[t]* u[1] #
end

######## Setting up the SPmodel
s_bounds = [(0, 1)] 			# bounds on the state
u_bounds = [(CONTROL_MIN, CONTROL_MAX)] # bounds on controls
spmodel = LinearSPModel(N_STAGES,u_bounds,[S0],cost_t,dynamic,xi_laws)
set_state_bounds(spmodel, s_bounds) 	# adding the bounds to the model
println("Model set up")

######### Solving the problem via SDDP
if run_sddp
    tic()
    println("Starting resolution by SDDP")
    # 10 forward pass, stop at MAX_ITER
    paramSDDP = SDDPparameters(SOLVER,
                               passnumber=10,
                               max_iterations=MAX_ITER)
    sddp = solve_SDDP(spmodel, paramSDDP, 2) # display information every 2 iterations
    lb_sddp = StochDynamicProgramming.get_lower_bound(spmodel, paramSDDP, sddp.bellmanfunctions)
    println("Lower bound obtained by SDDP: "*string(round(lb_sddp,4)))
    toc(); println();
end

######### Solving the problem via Dynamic Programming
if run_sdp
    tic()
    println("Starting resolution by SDP")
    stateSteps = [step] # discretization step of the state
    controlSteps = [step] # discretization step of the control
    infoStruct = "HD" # noise at time t is known before taking the decision at time t
    paramSDP = SDPparameters(spmodel, stateSteps, controlSteps, infoStruct)
    Vs = solve_dp(spmodel,paramSDP, 1)
    value_sdp = StochDynamicProgramming.get_bellman_value(spmodel,paramSDP,Vs)
    println("Value obtained by SDP: "*string(round(value_sdp,4)))
    toc(); println();
end

######### Solving the problem via Extensive Formulation
if run_ef
    tic()
    println("Starting resolution by Extensive Formulation")
    value_ef = extensive_formulation(spmodel, paramSDDP)[1]
    println("Value obtained by Extensive Formulation: "*string(round(value_ef,4)))
    println("Relative error of SDP value: "*string(100*round(abs(value_sdp/value_ef)-1,4))*"%")
    println("Relative error of SDDP lower bound: "*string(100*round(abs(lb_sddp/value_ef)-1,4))*"%")
    toc(); println();
end

######### Comparing the solutions on simulated scenarios.

#srand(1234) # to fix the random seed accross runs
if run_sddp && run_sdp && test_simulation
    scenarios = StochDynamicProgramming.simulate_scenarios(xi_laws,1000)
    costsddp, stocks = forward_simulations(spmodel, paramSDDP, sddp.solverinterface, scenarios)
    costsdp, states, controls = forward_simulations(spmodel,paramSDP, Vs, scenarios)
    println("Simulated relative gain of sddp over sdp: "
            *string(round(200*mean(costsdp-costsddp)/abs(mean(costsddp+costsdp)),3))*"%")
end
