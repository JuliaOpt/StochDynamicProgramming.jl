#############################################################################
#  Copyright 2017, V.Leclere, H.Gerard, F.Pacaud, T.Rigaut
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# Wrapper of cut's pruning algorithm from CutPruners
#############################################################################


"""
Exact pruning of all polyhedral functions in input array.

# Arguments
* `model::SPModel`:
* `trajectories::Array{Float64, 3}`
    Previous trajectories
"""
function prune!(sddp::SDDPInterface,
                trajectories::Array{Float64, 3},
                )
    # Basic pruning: remove redundant cuts
    for t in 1:sddp.spmodel.stageNumber-1
        b, A = fetchnewcuts!(sddp.bellmanfunctions[t])
        nnew = length(b)
        if nnew > 0
            mycut = Bool[true for _ in 1:length(b)]
            CutPruners.addcuts!(sddp.pruner[t], A, b, mycut)
        end
    end
end


"""Synchronise cuts between `sddp.pruner` and `sddp.bellmanfunctions`."""
function sync!(sddp::SDDPInterface)
    for t in 1:sddp.spmodel.stageNumber-1
        sddp.bellmanfunctions[t] = PolyhedralFunction(sddp.pruner[t].b, sddp.pruner[t].A)
    end
end


"""Prune cuts with exact pruning."""
function cleancuts!(sddp::SDDPInterface)
    for t in 1:sddp.spmodel.stageNumber-1
        ub = [x[2] for x in sddp.spmodel.xlim[:, t]]
        lb = [x[1] for x in sddp.spmodel.xlim[:, t]]
        exactpruning!(sddp.pruner[t], sddp.params.SOLVER, ub=ub, lb=lb)
    end
end


# Return total number of cuts in PolyhedralFunction array:
ncuts(V::PolyhedralFunction) = length(V.betas)
ncuts(V::Array{PolyhedralFunction}) = sum([ncuts(v) for v in V])

# Update cut pruner
update!{T}(pruner::CutPruners.DeMatosCutPruner, x::Vector{T}, λ::Vector{T})=addposition!(pruner, x)
update!{T}(pruner::CutPruners.AvgCutPruner, x::Vector{T}, λ::Vector{T})=addusage!(pruner, λ)
update!{T}(pruner::CutPruners.DecayCutPruner, x::Vector{T}, λ::Vector{T})=addusage!(pruner, λ)

