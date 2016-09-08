
import StochDynamicProgramming


"""
Solve SDDP in parallel.

# Arguments
* `model::SPmodel`:
    the stochastic problem we want to optimize
* `param::SDDPparameters`:
    the parameters of the SDDP algorithm
* `V::Array{PolyhedralFunction}`:
    the current estimation of Bellman's functions
* `n_parallel_pass::Int`: default is 4
    Number of parallel pass to compute
* `synchronize::Int`: default is 5
    Set when to synchronize the cuts between the different processes.
* `display::Int`: default is 0
    Says whether to display results or not

# Return
* `V::Array{PolyhedralFunction}`:
    the collection of approximation of the bellman functions
"""
function psolve_sddp(model, params, V; n_parallel_pass=4,
                     synchronize=5, display=0)
    # Redefine seeds in every processes to maximize randomness:
    @everywhere srand()

    mitn = params.maxItNumber
    params.maxItNumber = synchronize

    # Count number of available CPU:
    ncpu = nprocs() - 1
    (display > 0) && println("\nLaunch simulation on ", ncpu, " processes")
    workers = procs()[2:end]

    fpn = params.forwardPassNumber
    # As we distribute computation in n process, we perform forward pass in parallel:
    params.forwardPassNumber = max(1, round(Int, params.forwardPassNumber/ncpu))

    # Start parallel computation:
    for i in 1:n_parallel_pass
        # Distribute computation of SDDP in each process:
        refs = [@spawnat w StochDynamicProgramming.solve_SDDP(model, params, V, display)[1] for w in workers]
        # Catch the result in the main process:
        V = StochDynamicProgramming.catcutsarray([fetch(r) for r in refs]...)
        # We clean the resultant cuts:
        StochDynamicProgramming.remove_redundant_cuts!(V)
        (display > 0) && println("Lower bound at pass ", i, ": ", StochDynamicProgramming.get_lower_bound(model, params, V))
    end

    params.forwardPassNumber = fpn
    params.maxItNumber = mitn
    return V
end
