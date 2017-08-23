using StochDynamicProgramming,  JuMP, Clp, Cbc#, ECOS#,Gurobi, CPLEX#

    solverLP = ClpSolver()
    solverQP = ClpSolver() #ECOSSolver()# GurobiSolver(OutputFlag=0)#  #CplexSolver(CPX_PARAM_SIMDISPLAY=0)
    solverMILP = CbcSolver(logLevel=0) #GurobiSolver(OutputFlag=0)# CbcSolver(OutputFlag=0)

    # SDDP's tolerance:
    epsilon = .05
    # maximum number of iterations:
    max_iterations = 2
    # number of scenarios in forward and backward pass:
    n_scenarios = 10
    # number of aleas:
    n_aleas = 5
    # number of stages:
    n_stages = 3

    # define dynamic:
    function dynamic(t, x, u, w)
        return [x[1] - u[1] - u[2] + w[1]]
    end
    # define cost:
    function cost(t, x, u, w)
        return -u[1]*w[2]
    end

    # Generate probability laws:
    laws = Vector{NoiseLaw}(n_stages)
    proba = 1/n_aleas*ones(n_aleas)
    for t=1:n_stages
        laws[t] = NoiseLaw([0 1; 1 2; 3 1; 4 2; 6 1]', proba)#
    end

    # set initial position:
    x0 = [10.]
    # set bounds on state:
    x_bounds = [(0., 100.)]
    # set bounds on control:
    u_bounds = [(0., 7.), (0., Inf)]

    # Instantiate parameters of SDDP:
    param = StochDynamicProgramming.SDDPparameters(solverLP,
                                                   passnumber=n_scenarios,
                                                   gap=epsilon,
                                                   max_iterations=max_iterations,
                                                   prune_cuts=0)

    V = nothing
    model = StochDynamicProgramming.LinearSPModel(n_stages, u_bounds,
                                                  x0, cost, dynamic, laws)

    set_state_bounds(model, x_bounds)

    noise_scenarios = simulate_scenarios(model.noises,param.forwardPassNumber)
