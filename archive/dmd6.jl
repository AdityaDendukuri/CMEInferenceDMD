# inference_cme_fixed.jl

using Catalyst, JumpProcesses, DifferentialEquations
using LinearAlgebra, SparseArrays, StatsBase, ProgressMeter

# ────────────────────────────────────────────────────────────────────────────────
# 1) Simulate trajectories via SSA
# ────────────────────────────────────────────────────────────────────────────────
function simulate_trajectories(rn::ReactionSystem, u0, tspan, params;
                               n_traj::Int=500, Δt::Float64=0.1)
    jprob = JumpProblem(JumpInputs(rn, u0, tspan, params))
    trajs = Vector{ODESolution}(undef, n_traj)
    @showprogress for i in 1:n_traj
        trajs[i] = solve(jprob, SSAStepper(), saveat=Δt)
    end
    return trajs
end

# ────────────────────────────────────────────────────────────────────────────────
# 2) Build per‐time‐point histograms with local state mappings
# ────────────────────────────────────────────────────────────────────────────────
function build_time_histograms(trajs, Δt; boundary_condition=nothing)
    T_max      = maximum(maximum(sol.t) for sol in trajs)
    time_pts   = 0:Δt:T_max
    hist       = Dict{Float64, Vector{Float64}}()
    st2i_map   = Dict{Float64, Dict{String,Int}}()
    i2st_map   = Dict{Float64, Dict{Int,Vector{Int}}}()

    @showprogress "Building histograms..." for t in time_pts
        buf = Vector{Vector{Int}}()
        for sol in trajs
            idx = argmin(abs.(sol.t .- t))
            if abs(sol.t[idx] - t) ≤ Δt/2
                x = convert(Vector{Int}, sol.u[idx])
                if isnothing(boundary_condition) || boundary_condition(x)
                    push!(buf, x)
                end
            end
        end

        # unique states
        uniq = Dict{String,Vector{Int}}()
        for x in buf
            uniq[join(x, ",")] = x
        end

        keys_sorted = sort(collect(keys(uniq)))
        st2i, i2st = Dict{String,Int}(), Dict{Int,Vector{Int}}()
        for (i,k) in enumerate(keys_sorted)
            st2i[k] = i
            i2st[i] = uniq[k]
        end

        # normalized histogram
        h = zeros(Float64, length(i2st))
        for x in buf
            h[st2i[join(x, ",")]] += 1
        end
        total = sum(h)
        if total > 0
            h ./= total
        end

        hist[t]       = h
        st2i_map[t]   = st2i
        i2st_map[t]   = i2st
    end

    return hist, st2i_map, i2st_map
end

# ────────────────────────────────────────────────────────────────────────────────
# 3) Compute generator matrices for each chunk
# ────────────────────────────────────────────────────────────────────────────────
function compute_generator_matrices(hist, i2st_map, Δt; threshold=1e-6)
    times = sort(collect(keys(hist)))
    gens  = Dict{Float64, Tuple{SparseMatrixCSC{Float64,Int},Dict{Int,Vector{Int}}}}()

    @showprogress "Computing generators..." for k in 1:length(times)-1
        t, tnext = times[k], times[k+1]
        if abs((tnext - t) - Δt) > 1e-8
            continue
        end
        p, p_next = hist[t], hist[tnext]
        if length(p) != length(p_next)
            continue
        end

        dp = (p_next .- p) ./ Δt
        n  = length(p)
        A  = zeros(Float64, n, n)

        for j in 1:n
            if p[j] < threshold
                continue
            end
            for i in 1:n
                if i != j && dp[i] > threshold
                    A[i,j] = dp[i] / p[j]
                end
            end
            A[j,j] = -sum(A[:,j])
        end

        gens[t] = (sparse(A), i2st_map[t])
    end

    return gens
end

# ────────────────────────────────────────────────────────────────────────────────
# 4) Extract stoichiometric vectors using each chunk’s mapping
# ────────────────────────────────────────────────────────────────────────────────
function extract_stoichiometry_from_generators(gens; threshold=1e-8)
    reactions = []

    # iterate in time order by sorting keys, not pairs
    for t in sort(collect(keys(gens)))
        A, i2st = gens[t]
        rows, cols, vals = findnz(A)
        for (i,j,v) in zip(rows, cols, vals)
            if i == j || abs(v) < threshold
                continue
            end
            Xi = i2st[i]
            Xj = i2st[j]
            ν  = Xi .- Xj

            reactants = [(k, -d) for (k,d) in enumerate(ν) if d < 0]
            products  = [(k,  d) for (k,d) in enumerate(ν) if d > 0]

            push!(reactions, (
                time       = t,
                reactants  = reactants,
                products   = products,
                ν          = ν,
                rate       = v
            ))
        end
    end

    return reactions
end
using StatsBase  # for mean

"""
    aggregate_reactions(rxns)

Given a Vector of NamedTuples
  (time, reactants, products, ν, rate),
group by ν and return a summary vector of
  (ν, count, mean_rate, times).
"""
function aggregate_reactions(rxns)
    # bucket each reaction by its ν‐vector
    buckets = Dict{Vector{Int}, Vector{NamedTuple}}()
    for r in rxns
        push!(get!(buckets, r.ν, Vector{NamedTuple}()), r)
    end

    # build summaries
    summaries = []
    for (ν, recs) in buckets
        rates = [r.rate for r in recs]
        times = [r.time for r in recs]
        cnt   = length(recs)
        μ     = mean(rates)
        push!(summaries, (ν=ν, count=cnt, mean_rate=μ, times=times))
    end

    return summaries
end

# ────────────────────────────────────────────────────────────────────────────────
# Update run_mm_example to call aggregate_reactions
# ────────────────────────────────────────────────────────────────────────────────
"""
    run_mm_example_agg(; n_traj=500, Δt=0.1)

Run the CME‐inference on Michaelis–Menten, then aggregate and summarize all recovered
reactions over time.
"""
function run_mm_example_agg(; n_traj::Int=500, Δt::Float64=0.1)
    # ─── 1) Define your ground‐truth MM system ────────────────────────────────
    rn = @reaction_network begin
        kB, S + E --> SE
        kD, SE    --> S + E
        kP, SE    --> P + E
    end

    u0     = [:S=>50, :E=>10, :SE=>1, :P=>1]
    tspan  = (0.0, 100.0)
    params = [:kB=>0.01, :kD=>0.1, :kP=>0.1]

    println("Running CME inference on Michaelis–Menten…")
    # ─── 2) Infer all per‐chunk reactions ──────────────────────────────────
    rxns, gens = infer_crm_cme(rn, u0, tspan, params;
                               n_traj=n_traj, Δt=Δt, threshold=1e-8)

    # ─── 3) Aggregate across time bins ─────────────────────────────────────
    summary = aggregate_reactions(rxns)

    # ─── 4) Print a human‐readable summary ────────────────────────────────
    species = Catalyst.get_species(rn)           # [:S, :E, :SE, :P]
    syms    = Symbol.(string.(species))

    println("\nAggregated reactions:")
    for s in summary
        # build reaction strings from ν
        ν      = s.ν
        react  = [(k, -d) for (k,d) in enumerate(ν) if d < 0]
        prod   = [(k,  d) for (k,d) in enumerate(ν) if d > 0]
        rstr = isempty(react) ? "∅" :
               join(map(t-> t[2]>1 ?
                        "$(t[2])$(syms[t[1]])" :
                        string(syms[t[1]]), react), " + ")
        pstr = isempty(prod)  ? "∅" :
               join(map(t-> t[2]>1 ?
                        "$(t[2])$(syms[t[1]])" :
                        string(syms[t[1]]), prod), " + ")

        println("ν=$(ν):  $rstr --> $pstr")
        println("  count      = $(s.count)")
        println("  mean rate  = $(round(s.mean_rate, sigdigits=4))")
        println("  time bins  = $(round.(s.times, digits=2))\n")
    end

    return summary
end


# ────────────────────────────────────────────────────────────────────────────────
# 5) High-level pipeline
# ────────────────────────────────────────────────────────────────────────────────
function infer_crm_cme(rn::ReactionSystem, u0, tspan, params;
                       n_traj::Int=500, Δt::Float64=0.1, threshold=1e-6,
                       boundary_condition=nothing)
    trajs = simulate_trajectories(rn, u0, tspan, params,
                                  n_traj=n_traj, Δt=Δt)
    hist, st2i_map, i2st_map = build_time_histograms(trajs, Δt;
                                                    boundary_condition=boundary_condition)
    gens = compute_generator_matrices(hist, i2st_map, Δt;
                                      threshold=threshold)
    rxns = extract_stoichiometry_from_generators(gens;
                                                 threshold=threshold)
    return rxns, gens
end

# ────────────────────────────────────────────────────────────────────────────────
# 6) Example: Michaelis–Menten
# ────────────────────────────────────────────────────────────────────────────────
function run_mm_example(; n_traj=500, Δt::Float64=0.1)
    rn     = @reaction_network begin
        kB, S + E --> SE
        kD, SE    --> S + E
        kP, SE    --> P + E
    end
    u0     = [:S=>50, :E=>10, :SE=>1, :P=>1]
    tspan  = (0.0, 100.0)
    params = [:kB=>0.01, :kD=>0.1, :kP=>0.1]

    println("Running CME inference on Michaelis–Menten…")
    rxns, gens = infer_crm_cme(rn, u0, tspan, params;
                               n_traj=n_traj, Δt=Δt, threshold=1e-8)

    species = Catalyst.get_species(rn)   # [:S, :E, :SE, :P]
    syms    = Symbol.(string.(species))

    println("\nRecovered reactions:")
    for r in rxns
        rstr = isempty(r.reactants) ? "∅" :
               join(map(t->t[2]>1 ?
                        "$(t[2])$(syms[t[1]])" :
                        string(syms[t[1]]),
                        r.reactants), " + ")
        pstr = isempty(r.products)  ? "∅" :
               join(map(t->t[2]>1 ?
                        "$(t[2])$(syms[t[1]])" :
                        string(syms[t[1]]),
                        r.products), " + ")
        println("t=$(round(r.time,digits=3)): $rstr --> $pstr, rate=$(round(r.rate,digits=5)), ν=$(r.ν)")
    end

    return rxns, gens
end

# Run when script is called directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_mm_example(n_traj=1000, Δt=0.05)
end

