using Catalyst, ModelingToolkit, JumpProcesses, Plots, StatsBase, ProgressMeter

# ─── Ground Truth Reaction Network ──────────────────────────────────────────────
rn_true = @reaction_network begin
    kB, S + E --> SE
    kD, SE    --> S + E
    kP, SE    --> P + E
end

# ─── Inferred Reaction Network ─────────────────────────────────────────────────
rn_inferred = @reaction_network begin
    k1, SE     --> S + E    # ≈ 0.08571
    k2, SE     --> E + P    # ≈ 0.075
    k3, S      --> P        # ≈ 0.0381
    k4, P      --> S        # ≈ 0.02051
    k5, S + E  --> SE       # ≈ 0.01
    k6, E + P  --> SE       # ≈ 0.004
end

# ─── Initial conditions and time span ───────────────────────────────────────────
u0    = [:S => 50, :E => 10, :SE => 1, :P => 1]
tspan = (0.0, 200.0)

# ─── Parameters ─────────────────────────────────────────────────────────────────
ps_true     = [:kB => 0.01,  :kD => 0.1,   :kP => 0.1]
ps_inferred = [
    :k1 => 0.08571, :k2 => 0.075,
    :k3 => 0.0381,  :k4 => 0.02051,
    :k5 => 0.01,    :k6 => 0.004
]

# ─── Build JumpProblems ─────────────────────────────────────────────────────────
jprob_true     = JumpProblem(JumpInputs(rn_true,     u0, tspan, ps_true))
jprob_inferred = JumpProblem(JumpInputs(rn_inferred, u0, tspan, ps_inferred))

# ─── Simulate many trajectories ─────────────────────────────────────────────────
n_trajectories = 100
save_times     = 0.0:1.0:200.0

println("Generating trajectories for comparison…")
true_trajs, inferred_trajs = [], []

@showprogress for i in 1:n_trajectories
    seed = rand(UInt)
    try
        push!(true_trajs,     solve(jprob_true,     SSAStepper(), saveat=save_times, seed=seed))
        push!(inferred_trajs, solve(jprob_inferred, SSAStepper(), saveat=save_times, seed=seed))
    catch e
        @warn "Skipping trajectory $i due to error: $e"
    end
end

println("Successfully generated $(length(true_trajs)) trajectories")

# ─── FIX #2: Align species columns by symbol lookup ──────────────────────────────

# 1) canonical list of species (symbols)
species_names = [:S, :E, :SE, :P]

# 2) pull the ModelingToolkit state‐variable list for each network
order_true = states(rn_true)
order_inf  = states(rn_inferred)

# 3) build Dicts: symbol → column index in the u‐vector
idx_true = Dict(sym => findfirst(x->name(x)==sym, order_true) for sym in species_names)
idx_inf  = Dict(sym => findfirst(x->name(x)==sym, order_inf)  for sym in species_names)

# 4) compute means & variances using those indices
function compute_trajectory_statistics(true_trajs, inferred_trajs,
                                       species_names, idx_true, idx_inf)
    time_pts       = true_trajs[1].t
    n_time, n_spec = length(time_pts), length(species_names)

    tm = zeros(n_time, n_spec)   # true means
    tv = zeros(n_time, n_spec)   # true vars
    im = zeros(n_time, n_spec)   # inferred means
    iv = zeros(n_time, n_spec)   # inferred vars

    for (j, sym) in enumerate(species_names)
        ti = idx_true[sym]
        ii = idx_inf[sym]

        for t in 1:n_time
            vals_true = [traj.u[t][ti] for traj in true_trajs     if length(traj.u) ≥ t]
            vals_inf  = [traj.u[t][ii] for traj in inferred_trajs if length(traj.u) ≥ t]

            if !isempty(vals_true)
                tm[t,j] = mean(vals_true)
                tv[t,j] = var(vals_true)
            end
            if !isempty(vals_inf)
                im[t,j] = mean(vals_inf)
                iv[t,j] = var(vals_inf)
            end
        end
    end

    return (time_pts = time_pts, true_means = tm, true_vars = tv,
            inf_means = im,  inf_vars  = iv)
end

# run statistics
stats = compute_trajectory_statistics(true_trajs, inferred_trajs,
                                      species_names, idx_true, idx_inf)

# ─── Plot comparison ────────────────────────────────────────────────────────────
function create_comparison_plots(stats, species_names)
    tp = stats.time_pts
    plots = []

    for (i, sym) in enumerate(species_names)
        μ_true = stats.true_means[:, i]
        σ_true = sqrt.(stats.true_vars[:, i])
        μ_inf  = stats.inf_means[:, i]
        σ_inf  = sqrt.(stats.inf_vars[:, i])

        p = plot(tp, μ_true, ribbon=σ_true, fillalpha=0.2, label="Ground Truth",
                 title="Species: $sym", xlabel="Time", ylabel="Population")
        plot!(p, tp, μ_inf,  ribbon=σ_inf,  fillalpha=0.2, label="Inferred",
              linestyle=:dash)
        push!(plots, p)
    end

    return plot(plots..., layout=(2,2), size=(1200,800))
end

comparison_plot = create_comparison_plots(stats, species_names)
display(comparison_plot)
savefig(comparison_plot, "mm_trajectory_comparison_fixed.png")

# ─── Single‐trajectory example ──────────────────────────────────────────────────
println("Generating single trajectory example…")
seed_ex  = rand(UInt)
true_ex  = solve(jprob_true,     SSAStepper(), saveat=save_times, seed=seed_ex)
inf_ex   = solve(jprob_inferred, SSAStepper(), saveat=save_times, seed=seed_ex)

function plot_single_trajectory(t_true, t_inf, species_names, idx_true, idx_inf)
    tmax = min(length(t_true.t), length(t_inf.t))
    tp   = t_true.t[1:tmax]
    plots = []

    for sym in species_names
        ti = idx_true[sym]
        ii = idx_inf[sym]
        vals_t = [t_true.u[t][ti] for t in 1:tmax]
        vals_i = [t_inf.u[t][ii]  for t in 1:tmax]

        p = plot(tp, vals_t, label="Ground Truth", title="Species: $sym (Single Trajectory)",
                 xlabel="Time", ylabel="Population")
        plot!(p, tp, vals_i, label="Inferred", linestyle=:dash)
        push!(plots, p)
    end

    return plot(plots..., layout=(2,2), size=(1200,800))
end

single_plot = plot_single_trajectory(true_ex, inf_ex,
                                     species_names, idx_true, idx_inf)
display(single_plot)
savefig(single_plot, "mm_single_trajectory_fixed.png")


