#  Copyright 2017, V.Leclere, H.Gerard, F.Pacaud, T.Rigaut
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Define all types used in this module.
#############################################################################


@compat abstract type SPModel end


type PolyhedralFunction
    #function defined by max_k betas[k] + lambdas[k,:]*x
    betas::Vector{Float64}
    lambdas::Array{Float64,2} #lambdas[k,:] is the subgradient
    # number of cuts:
    numCuts::Int64
    hashcuts::Vector{UInt64}
    newcuts::Int
end

PolyhedralFunction(n_dim::Int) = PolyhedralFunction(Float64[], Array{Float64}(0, n_dim), 0, UInt64[], 0)
PolyhedralFunction(beta, lambda) = PolyhedralFunction(beta, lambda, length(beta), UInt64[], 0)

function fetchnewcuts!(V::PolyhedralFunction)
    β = V.betas[end-V.newcuts+1:end]
    λ = V.lambdas[end-V.newcuts+1:end, :]
    V.newcuts = 0
    return β, λ
end

type LinearSPModel <: SPModel
    # problem dimension
    stageNumber::Int64 #number of information step + 1
    dimControls::Int64
    dimStates::Int64
    dimNoises::Int64

    # Bounds of states and controls (state and control are in column)
    xlim::Array{Tuple{Float64,Float64},2}
    ulim::Array{Tuple{Float64,Float64},2}


    initialState::Array{Float64, 1}

    #FIXME: add a correct typage for costFunctions that dont break in 0.5
    costFunctions
    dynamics::Function
    noises::Vector{NoiseLaw}

    finalCost::Union{Function, PolyhedralFunction}

    controlCat::Vector{Symbol}
    equalityConstraints::Nullable{Function}
    inequalityConstraints::Nullable{Function}
    info::Symbol

    IS_SMIP::Bool

    function LinearSPModel(n_stage,            # number of stages
                           u_bounds,           # bounds of control
                           x0,                 # initial state
                           cost,               # cost function
                           dynamic,            # dynamic
                           aleas;              # modelling of noises
                           x_bounds =nothing,  # state bounds
                           Vfinal=nothing,     # final cost
                           eqconstr=nothing,   # equality constraints
                           ineqconstr=nothing, # inequality constraints
                           info=:HD,           # information structure
                           control_cat=nothing) # category of controls

        dimStates   = length(x0)
        dimControls = size(u_bounds,1)
        dimNoises   = aleas[1].dimNoises

        # First step: process terminal costs.
        # If not specified, default value is null function
        if isa(Vfinal, Function) || isa(Vfinal, PolyhedralFunction)
            Vf = Vfinal
        else
            Vf = PolyhedralFunction(zeros(1), zeros(1, dimStates), 1, UInt64[], 0)
        end

        isbu = isa(control_cat, Vector{Symbol})? control_cat: [:Cont for i in 1:dimControls]
        is_smip = (:Int in isbu)||(:Bin in isbu)

        if (x_bounds == nothing)
            x_bounds = repmat([(-Inf,Inf)],dimStates,n_stage)
        end
        u_bounds = test_and_reshape_bounds(u_bounds, dimControls,n_stage, "Controls")

        return new(n_stage, dimControls, dimStates, dimNoises, x_bounds, u_bounds,
                   x0, cost, dynamic, aleas, Vf, isbu, eqconstr, ineqconstr, info, is_smip)
    end
end


"""Set bounds on state."""
function set_state_bounds(model::SPModel, x_bounds)
   nx = model.dimStates
   ns = model.stageNumber
   model.xlim = test_and_reshape_bounds(x_bounds, nx,ns, "State")
end

""" Checking state and control bounds and duplicating if needed

If bounds is a vector of length nx (or nx*1 array) duplicate to a matrix nx*ns,
if already a matrix keep it this way, else return an error"""
function test_and_reshape_bounds(bounds, nx,ns, variable)
    if (ndims(bounds) == 1 && length(bounds) == nx)||(ndims(bounds) == 2 && size(bounds) == (nx,1))
        return repmat(bounds,1,ns)
    elseif ndims(bounds) == 2 && size(bounds) == (nx,ns)
        return bounds
     else
        error(variable, " bounds dimension should be ", nx," or (",nx,",",ns,")")
     end
 end


function iswithinbounds(x, bounds::Array)
    test = true
    for (i, xi) in enumerate(x)
        test = test&(xi <= bounds[i][2])&(xi >= bounds[i][1])
    end
    test
end

function max_bounds(bounds::Array)
    warn("Varying bounds badly not in sdp, define in constraint function")
    m_bounds = ones(size(bounds)[1],2)
    for j in 1:size(bounds)[2]
        for i in 1:size(bounds)[1]
            m_bounds[i,1] = min(m_bounds[i,1], bounds[i,j][1])
            m_bounds[i,2] = max(m_bounds[i,2], bounds[i,j][2])
        end
    end
    [(m_bounds[i,1], m_bounds[i,2]) for i in 1:size(bounds)[1]]
end

type StochDynProgModel <: SPModel
    # problem dimension
    stageNumber::Int64
    dimControls::Int64
    dimStates::Int64
    dimNoises::Int64

    # Bounds of states and controls:
    xlim::Array
    ulim::Array


    initialState::Array

    costFunctions::Function
    finalCostFunction::Function
    dynamics::Function
    constraints::Function
    noises::Vector{NoiseLaw}

    build_search_space::Nullable{Function}

    function StochDynProgModel(model::LinearSPModel, final, cons)
        if isa(model.costFunctions, Function)
            cost = model.costFunctions
        #FIXME: broken test since 0.5 release
        else
            function cost(t,x,u,w)
                current_cost = -Inf
                for aff_func in model.costFunctions
                    current_cost = aff_func(t,x,u,w)
                end
            return current_cost
            end
        end
        return StochDynProgModel(model.stageNumber, model.xlim, model.ulim, model.initialState,
                 cost, final, model.dynamics, cons,
                 model.noises)
    end

    function StochDynProgModel(TF::Int, x_bounds, u_bounds, x0, costFunctions,
                                finalCostFunction, dynamic, constraints, aleas, search_space_builder = Nullable{Function}())
        dimState = length(x0)
        dimControls = size(u_bounds)[1]
        u_bounds1 = ndims(u_bounds) == 1? u_bounds : max_bounds(u_bounds)
        x_bounds1 = ndims(x_bounds) == 1? x_bounds : max_bounds(x_bounds)

        # cons_fun_x(t,x,u,w) = true
        # cons_fun_u(t,x,u,w) = true

        # if ndims(x_bounds) > 1
        #     cons_fun_x(t,x,u,w) = iswithinbounds(x, x_bounds[:,t])
        # end

        # println(cons_fun_x(2,[1. 1.],[1.,1.],[1.,1.]))
        # if ndims(u_bounds) > 1
        #     cons_fun_u(t,x,u,w) = iswithinbounds(u, u_bounds[:,t])
        # end

        # cons_fun(t,x,u,w) = return constraints(t,x,u,w)&&cons_fun_x(t,x,u,w)&&cons_fun_u(t,x,u,w)

        return new(TF, dimControls, dimState, length(aleas[1].support[:, 1]),
                    x_bounds1, u_bounds1, x0, costFunctions, finalCostFunction, dynamic,
                    constraints, aleas, search_space_builder)
    end


end


type SDPparameters
    stateSteps
    controlSteps
    totalStateSpaceSize
    totalControlSpaceSize
    stateVariablesSizes
    controlVariablesSizes
    monteCarloSize
    infoStructure
    expectation_computation

    function SDPparameters(model, stateSteps, controlSteps, infoStruct,
                            expectation_computation="Exact" ,monteCarloSize=1000)

        stateVariablesSizes = zeros(Int64, length(stateSteps))
        controlVariablesSizes = zeros(Int64, length(controlSteps))
        totalStateSpaceSize = 1
        totalControlSpaceSize = 1
        for i=1:length(stateSteps)
            stateVariablesSizes[i] = round(Int64,1 +
                                    (model.xlim[i][2]-model.xlim[i][1])/stateSteps[i])
            totalStateSpaceSize *= stateVariablesSizes[i]
        end

        for i=1:length(controlSteps)
            controlVariablesSizes[i] = round(Int64, 1 +
                                    (model.ulim[i][2]-model.ulim[i][1])/controlSteps[i])
            totalControlSpaceSize *= controlVariablesSizes[i]
        end

        return new(stateSteps, controlSteps, totalStateSpaceSize,
                    totalControlSpaceSize, stateVariablesSizes,
                    controlVariablesSizes, monteCarloSize, infoStruct,
                    expectation_computation)
    end

end


@compat abstract type AbstractSDDPStats end

# Define an object to store evolution of solution
# along iterations:
type SDDPStat <: AbstractSDDPStats
    # Number of iterations:
    niterations::Int
    # evolution of lower bound:
    lower_bounds::Vector{Float64}
    # evolution of upper bound:
    upper_bounds::Vector{Float64}
    # standard deviation of upper-bound's estimation
    upper_bounds_std::Vector{Float64}
    # tolerance of upper-bounds estimation:
    upper_bounds_tol::Vector{Float64}
    # evolution of execution time:
    exectime::Vector{Float64}
    # time used to solve each LP:
    solverexectime_fw::Vector{Float64}
    solverexectime_bw::Vector{Float64}
    # number of calls to solver:
    nsolved::Int
    # number of optimality cuts
    nocuts::Int
    npaths::Int
    # current lower bound
    lowerbound::Float64
    # current lower bound
    upperbound::Float64
    # upper-bound std:
    σ_UB::Float64
    # total time
    time::Float64
end


SDDPStat() = SDDPStat(0, [], [], [], [], [], [], [], 0, 0, 0, 0., 0., 0., 0.)

"""
Update the SDDPStat object with the results of current iterations.

# Arguments
* `stats::SDDPStat`:
    statistics of the current algorithm
* `call_solver_at_it::Int64`:
    number of time a solver was called during the current iteration
* `lwb::Float64`:
    lowerbound obtained
* `upb::Float64`:
    upperbound estimated
* `time`
"""
function updateSDDPStat!(stats::SDDPStat,
                         lwb::Float64,
                         upb::Vector{Float64},
                         time)
    stats.niterations += 1
    push!(stats.lower_bounds, lwb)
    push!(stats.upper_bounds, upb[1])
    push!(stats.upper_bounds_tol, upb[3])
    push!(stats.upper_bounds_std, upb[2])
    push!(stats.exectime, time)
    stats.lowerbound = lwb
    stats.upperbound = upb[1]
    stats.σ_UB = upb[2]
    stats.time += time
end


type NLDSSolution
    # solver status:
    status::Bool
    # cost:
    objval::Float64
    # next position:
    xf::Array{Float64, 1}
    # optimal control:
    uopt::Array{Float64, 1}
    # Subgradient:
    ρe::Array{Float64, 1}
    # cost-to-go:
    θ::Float64
    # cuts' multipliers
    πc::Vector{Float64}
end

NLDSSolution() = NLDSSolution(false, Inf, Array{Float64, 1}(), Array{Float64, 1}(), Array{Float64, 1}(), Inf, Array{Float64, 1}())
