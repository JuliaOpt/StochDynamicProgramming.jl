#  Copyright 2017, V.Leclere, H.Gerard, F.Pacaud, T.Rigaut
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
#  SDDP interface
#############################################################################

type SDDPInterface
    init::Bool
    # Stochastic model to solve
    spmodel::SPModel
    # SDDP parameters
    params::SDDPparameters
    # statistics
    stats::SDDPStat
    # stopping criterion
    stopcrit::AbstractStoppingCriterion
    # cut pruner:
    pruner::Vector{CutPruners.AbstractCutPruner}
    # regularization scheme:
    regularizer::Nullable{AbstractRegularization}

    # solution
    bellmanfunctions::Vector{PolyhedralFunction}
    solverinterface::Vector{JuMP.Model}

    verbosity::Int #0: no output, 1: big phases, 2: every verbose_it iterations, 3: inside iterations, 4: detailed inside iterations, 6: showing LP problems
    verbose_it::Int

    # Init SDDP interface
    function SDDPInterface(model::SPModel, # SP Model
                           param::SDDPparameters,# parameters
                           stopcrit::AbstractStoppingCriterion,
                           prunalgo::AbstractCutPruningAlgo;
                           regularization=nothing,
                           verbosity::Int=2,
                           verbose_it::Int=1)

        check_SDDPparameters(model, param, verbosity)
        # initialize value functions:
        V, problems = initialize_value_functions(model, param)
        (verbosity > 0) && println("SDDP Interface initialized")

        pruner = initpruner(prunalgo, model.stageNumber, model.dimStates)
        #Initialization of stats
        stats = SDDPStat()
        return new(false, model, param, stats, stopcrit, pruner, regularization, V,
                   problems, verbosity,verbose_it)
    end

    function SDDPInterface(model::SPModel,
                        params::SDDPparameters,
                        stopcrit::AbstractStoppingCriterion,
                        prunalgo::AbstractCutPruningAlgo,
                        V::Vector{PolyhedralFunction};
                        regularization=nothing,
                        verbosity::Int=2,
                        verbose_it::Int=1)

        check_SDDPparameters(model, params, verbosity)
        # First step: process value functions if hotstart is called
        problems = hotstart_SDDP(model, params, V)
        pruner = initpruner(prunalgo, model.stageNumber, model.dimStates)

        stats = SDDPStat()
        return new(false, model, params, stats, stopcrit, pruner, regularization,
                   V, problems, verbosity,verbose_it)
    end

    function SDDPInterface(model::SPModel,
                        params::SDDPparameters;
                        stopcrit::AbstractStoppingCriterion=IterLimit(params.max_iterations),
                        prunalgo::AbstractCutPruningAlgo=CutPruners.AvgCutPruningAlgo(-1),
                        regularization=nothing,
                        verbosity::Int=2,
                        verbose_it::Int=1)
        return SDDPInterface(model,params,stopcrit,prunalgo,regularization=regularization,verbosity = verbosity,verbose_it=verbose_it)
    end
end


function initpruner(algo, n_stages, n_dim)
    # Initialize cuts container for cuts pruning:
    return [CutPruners.CutPruner{n_dim, Float64}(algo, :Max) for i in 1:n_stages-1]
end

isregularized(sddp::SDDPInterface) = !isnull(sddp.regularizer)
