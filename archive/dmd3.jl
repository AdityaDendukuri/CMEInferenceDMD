using JumpProcesses
using Catalyst
using DifferentialEquations # For DiscreteProblem, solve
using ModelingToolkit # For states, nameof
using LinearAlgebra
using SparseArrays
using StatsBase
using ProgressMeter
using Plots

# --- Ground Truth Model Definition (from your script) ---
rn = @reaction_network begin
    kB, S + E --> SE
    kD, SE --> S + E
    kP, SE --> P + E
end
u0_map = Dict(:S => 50, :E => 10, :SE => 1, :P => 1)
# Ensure species are always in the same order for state vectors: S, E, SE, P
species_order = [:S, :E, :SE, :P]
# At the global scope
u0_map = Dict(:S => 50, :E => 10, :SE => 1, :P => 1) # Values are Int
params_map = Dict(:kB => 0.01, :kD => 0.1, :kP => 0.1) # Values are Float64

tspan = (0.0, 20.0) # Shorter tspan for quicker tests, adjust as needed
true_rates = params_map # For later comparison

# --- Helper Functions ---

"""
Generate trajectories using SSA.
"""

function generate_trajectories(rn::ReactionSystem,
                               u0_input::Union{Dict{Symbol,Int}, Vector{Pair{Symbol,Int}}},
                               tspan::Tuple{Float64,Float64},
                               params_input::Union{Dict{Symbol,Float64}, Vector{Pair{Symbol,Float64}}},
                               num_trajectories::Int,
                               saveat_dt::Float64,
                               output_species_order::Vector{Symbol}) # Desired order for species in output

    # Construct JumpInputs and JumpProblem
    jinput = JumpInputs(rn, u0_input, tspan, params_input)
    jprob = JumpProblem(jinput) # Following your working MWE

    trajectories = []
    times = tspan[1]:saveat_dt:tspan[2]

    # Get the internal symbolic state variables from the ReactionSystem.
    # The order of states in this vector is the order used by the solver.
    internal_symbolic_states = ModelingToolkit.get_states(rn)

    # For efficient lookup: create a map from the `nameof` each internal symbolic state
    # to its index in the `internal_symbolic_states` vector (i.e., its index in the solver's state vector).
    internal_state_name_to_solver_idx = Dict{Symbol, Int}()
    for (idx, state_var) in enumerate(internal_symbolic_states)
        internal_state_name_to_solver_idx[ModelingToolkit.nameof(state_var)] = idx
    end

    @showprogress "Generating $num_trajectories trajectories..." for _ in 1:num_trajectories
        sol = solve(jprob, SSAStepper(), saveat=times)
        
        traj_data_for_one_realization = Vector{Vector{Int}}(undef, length(sol.t))
        for i_t in 1:length(sol.t)
            current_solver_output_state = sol.u[i_t] # State vector from solver, ordered as per `internal_symbolic_states`
            ordered_state_for_output = Vector{Int}(undef, length(output_species_order))
            
            for (j_out, out_spec_name) in enumerate(output_species_order) # Iterate through your desired output order (e.g., :S, then :E)
                # Find the index that `out_spec_name` (e.g., :S) corresponds to in the solver's state vector
                solver_idx = get(internal_state_name_to_solver_idx, out_spec_name, 0) # Returns 0 if not found
                
                if solver_idx > 0 # Check if out_spec_name was found in the system's states
                    ordered_state_for_output[j_out] = current_solver_output_state[solver_idx]
                else
                    # This means a species name from your `output_species_order` was not found
                    # among the states defined in the `ReactionSystem rn`.
                    available_internal_names = collect(keys(internal_state_name_to_solver_idx))
                    error("Species symbol '$out_spec_name' from `output_species_order` was not found in the reaction network's internal states. Available internal state names from `rn`: $(available_internal_names). Your `output_species_order`: $(output_species_order)")
                end
            end
            traj_data_for_one_realization[i_t] = ordered_state_for_output
        end
        push!(trajectories, (t=sol.t, u=traj_data_for_one_realization))
    end
    return trajectories, times
end
"""
Construct empirical probability distributions (histograms) from trajectories.
Optionally restrict to a defined state_space_map (for FSP experiment).
"""
function get_empirical_distributions(trajectories, time_points, state_space_map=nothing)
    num_times = length(time_points)
    num_trajectories = length(trajectories)
    
    # Build a consistent mapping from state vector to index for all time points
    # If state_space_map is provided, it pre-defines the space.
    # Otherwise, discover states from data.
    current_state_to_idx = Dict{Vector{Int}, Int}()
    current_idx_to_state = Vector{Vector{Int}}()

    if isnothing(state_space_map)
        # Discover states from all trajectories at all time points
        all_observed_states = Set{Vector{Int}}()
        for traj in trajectories
            for state_vec in traj.u
                push!(all_observed_states, state_vec)
            end
        end
        for state_vec in sort(collect(all_observed_states)) # Sort for consistency
            push!(current_idx_to_state, state_vec)
            current_state_to_idx[state_vec] = length(current_idx_to_state)
        end
    else
        current_state_to_idx = state_space_map["state_to_idx"]
        current_idx_to_state = state_space_map["idx_to_state"]
    end
    
    num_states = length(current_idx_to_state)
    histograms = [spzeros(num_states) for _ in 1:num_times] # Use sparse vectors

    for (k, t_idx) in enumerate(1:num_times) # Iterate through time_points indices
        counts = zeros(Int, num_states)
        for traj in trajectories
            # Find the trajectory data point closest to time_points[k] if saveat doesn't match perfectly
            # Assuming trajectories are saved at time_points
            state_vec = traj.u[t_idx]
            
            if haskey(current_state_to_idx, state_vec)
                state_idx = current_state_to_idx[state_vec]
                counts[state_idx] += 1
            # Else: state is outside the predefined FSP state_space_map, so ignore it.
            end
        end
        if sum(counts) > 0
            histograms[k] = sparse(counts ./ sum(counts)) # Normalize to get probability
        else # Handle case where no states in the current FSP fall into this time point
             # This can happen if FSP is too restrictive
            histograms[k] = spzeros(num_states)
        end
    end
    return histograms, current_state_to_idx, current_idx_to_state
end


"""
Compute DMD operator and optionally the reconstructed generator A_hat.
X = [p(t0), p(t1), ..., p(tm-1)]
X_prime = [p(t1), p(t2), ..., p(tm)]
A_dmd = X_prime * pinv(X, rank_svd)
"""
function compute_dmd_and_generator(histograms, dt, rank_svd::Union{Nothing,Int}=nothing)
    # Ensure histograms are column vectors in a matrix
    if isempty(histograms) || length(histograms) < 2
        error("Need at least 2 histogram snapshots for DMD.")
    end
    
    # Convert sparse histograms to dense for SVD, ensure they are columns
    # Transpose because Julia's hcat makes them columns, but we need states as rows
    # No, hcat([h1, h2, ...]) where h are vectors, makes states rows, snapshots cols.
    
    # Convert sparse histograms to dense for SVD
    # X_matrix = Matrix(hcat(histograms[1:end-1]...))
    # X_prime_matrix = Matrix(hcat(histograms[2:end]...))
    
    # Assuming histograms are sparse vectors, hcat them:
    X_matrix_sparse = hcat(histograms[1:end-1]...)
    X_prime_matrix_sparse = hcat(histograms[2:end]...)

    X_matrix = Matrix(X_matrix_sparse) # Convert to dense for SVD
    
    # Compute SVD and pseudoinverse
    U, S_vals, V = svd(X_matrix)
    
    local pinv_X
    if isnothing(rank_svd) || rank_svd > length(S_vals)
        # Full rank or rank exceeds available singular values
        pinv_S = diagm(1 ./ S_vals)
        pinv_X = V * pinv_S * U'
    else
        # Truncated SVD
        U_r = U[:, 1:rank_svd]
        S_r_inv = diagm(1 ./ S_vals[1:rank_svd])
        V_r = V[:, 1:rank_svd]
        pinv_X = V_r * S_r_inv * U_r'
    end
    
    A_dmd_matrix = Matrix(X_prime_matrix_sparse) * pinv_X # A_dmd approximates e^(A*dt)
    
    # Reconstruct generator A_hat = log(A_dmd) / dt
    # Handle complex eigenvalues from log carefully if A_dmd has negative real eigenvalues
    # For now, assume we can take real part if small imag part, or handle carefully
    # eigen_decomp = eigen(A_dmd_matrix)
    # log_Lambda = diagm(log.(complex(eigen_decomp.values))) # Ensure complex log
    # Phi = eigen_decomp.vectors
    # A_hat_matrix = real.(Phi * log_Lambda * inv(Phi)) ./ dt # Take real part
    # A_hat_matrix = (log(A_dmd_matrix) ./ dt) # Matrix logarithm
    # Using simple log for eigenvalues for now, more robust matrix log is needed for general case.
    # This is a simplification; a robust matrix logarithm (e.g., via Schur decomposition) is better.
    evals_dmd, evecs_dmd = eigen(A_dmd_matrix)
    log_evals = log.(complex(evals_dmd)) # Use complex log
    
    # Check for suitability of matrix logarithm:
    # All eigenvalues of A_dmd should ideally be positive if it represents e^(A*dt)
    # where A is a CME generator (real eigenvalues <= 0).
    # However, noise can make A_dmd eigenvalues complex or negative.
    
    A_hat_matrix = real.(evecs_dmd * diagm(log_evals) * inv(evecs_dmd) ./ dt)

    return A_dmd_matrix, A_hat_matrix
end


"""
Define a state space based on max counts for each species.
Returns a dictionary with state_to_idx and idx_to_state.
"""
function define_fsp_space(max_counts::Vector{Int})
    # max_counts should be in species_order: [max_S, max_E, max_SE, max_P]
    @assert length(max_counts) == length(species_order) "Max counts length must match number of species"
    
    fsp_state_to_idx = Dict{Vector{Int}, Int}()
    fsp_idx_to_state = Vector{Vector{Int}}()
    
    # Iterate over all combinations of species counts up to max_counts
    # This generates states in a lexicographical order if iterators are nested correctly
    # For 4 species:
    current_idx = 0
    for s_count in 0:max_counts[1]
        for e_count in 0:max_counts[2]
            for se_count in 0:max_counts[3]
                for p_count in 0:max_counts[4]
                    state_vec = [s_count, e_count, se_count, p_count]
                    current_idx += 1
                    fsp_state_to_idx[state_vec] = current_idx
                    push!(fsp_idx_to_state, state_vec)
                end
            end
        end
    end
    return Dict("state_to_idx" => fsp_state_to_idx, "idx_to_state" => fsp_idx_to_state)
end


# --- "True" values for comparison ---
# Generate a large number of trajectories for a "true" reference
SAVEAT_DT = 0.5 # Sampling interval for trajectories
TRUE_N_TRAJECTORIES = 10000 # Number of trajectories for ground truth
println("Generating ground truth data (this may take a while)...")
true_trajectories, true_time_points = generate_trajectories(rn, u0_map, tspan, params_map, TRUE_N_TRAJECTORIES, SAVEAT_DT);

# For a very large state space to approximate true distributions well
# This defines the maximal possible states we might see within these bounds
# Adjust based on typical population sizes observed in true_trajectories
# For example, find max observed for each species in true_trajectories
max_obs_counts = [maximum(traj.u[t_idx][spec_idx] for traj in true_trajectories for t_idx in 1:length(traj.t)) for spec_idx in 1:length(species_order)]
println("Max observed counts for S, E, SE, P: $max_obs_counts")
# Let's cap it a bit for performance if max_obs_counts are too high
# For MM example, they should be reasonable.
# max_s, max_e, max_se, max_p = max_obs_counts[1]+2, max_obs_counts[2]+2, max_obs_counts[3]+2, max_obs_counts[4]+2
# Manually set for MM system for consistency, e.g., [60, 20, 20, 60] if P can grow
# Based on u0: S=50, E=10, SE=1, P=1. Max S+SE can be 51. Max E+SE can be 11.
# Total molecules involving S and E are conserved in some way for E. S can deplete. P can grow.
# Let's use a reasonable fixed large space for "truth".
true_fsp_max_counts = [60, 15, 15, 60] # S, E, SE, P
true_state_space_map = define_fsp_space(true_fsp_max_counts);

true_histograms, true_s2i, true_i2s = get_empirical_distributions(true_trajectories, true_time_points, true_state_space_map);

# True Koopman operator (approximated by DMD on true_histograms with high rank)
# The rank here should be high, ideally min(num_states, num_snapshots-1)
num_true_states = length(true_i2s)
num_true_snapshots = length(true_histograms)
max_possible_rank = min(num_true_states, num_true_snapshots -1)
println("Number of true states: $num_true_states, Max possible rank for true DMD: $max_possible_rank")

# Ensure rank is not too large if num_true_states is small
true_dmd_rank = max_possible_rank > 0 ? max_possible_rank : nothing 

if isnothing(true_dmd_rank)
    println("Warning: Not enough snapshots or states for true DMD computation.")
    K_true = missing # Or handle error appropriately
    A_true = missing
else
    println("Computing true Koopman operator (K_true) and generator (A_true)...")
    # Need to ensure histograms used for K_true are from the *same defined space*
    # as those used in experiments, if we plan to compare generators directly.
    # Or, K_true can be on its own "true_state_space_map".
    # For eigenvalue comparison, this is fine. For matrix norm, spaces must match.
    K_true, A_true = compute_dmd_and_generator(true_histograms, SAVEAT_DT, true_dmd_rank);
    true_eigenvalues = eigen(K_true).values
    println("Computed K_true and A_true.")
end


# --- Experiment 1: Histogram Error vs. N ---
println("\n--- Experiment 1: Histogram Error vs. N ---")
N_values = [10, 50, 100, 500, 1000, 2000] #, 5000] # Reduced for speed
histogram_errors_l1 = []

# Use the same true_state_space_map for consistent histogram comparison
# This means we are measuring error within this pre-defined large space.
target_time_idx = div(length(true_time_points), 2) # A midpoint in time

for N_val in N_values
    println("Running for N = $N_val")
    # Take a subset of trajectories
    exp1_trajectories = true_trajectories[1:N_val]
    exp1_histograms, _, _ = get_empirical_distributions(exp1_trajectories, true_time_points, true_state_space_map)
    
    # Compare histogram at target_time_idx
    # Ensure vectors are dense for norm calculation
    p_hat_N = Vector(exp1_histograms[target_time_idx])
    p_true = Vector(true_histograms[target_time_idx])
    
    # Pad if dimensions don't match (should not happen if using same state_space_map)
    # This padding is a simple way if states discovered are different.
    # Better: use the same state_to_idx map for both. (Done by passing true_state_space_map)

    err_l1 = norm(p_hat_N - p_true, 1)
    push!(histogram_errors_l1, err_l1)
    println("N = $N_val, L1 Error = $err_l1")
end

plot_hist_err = plot(N_values, histogram_errors_l1, xlabel="Number of Trajectories (N)", ylabel="L1 Histogram Error", title="Histogram Error vs. N", legend=false, yscale=:log10, xscale=:log10, marker=:circle)
plot!(N_values, N_values.^(-0.5) .* (histogram_errors_l1[1] / (N_values[1]^-0.5)), label="O(N^-0.5) guide", linestyle=:dash)
display(plot_hist_err)
savefig(plot_hist_err, "histogram_error_vs_N.png")

# --- Experiment 2: FSP Error vs. State Space Size ---
println("\n--- Experiment 2: FSP Error vs. State Space Size ---")
# Define a sequence of increasing state space sizes by varying max_counts
# Example: S, E, SE, P
fsp_max_counts_series = [
    [10, 5, 5, 10],   # Smallest
    [20, 8, 8, 20],
    [30, 10, 10, 30],
    [40, 12, 12, 40],
    [50, 14, 14, 50], # Max S is 50 initially
    true_fsp_max_counts # Largest, corresponds to A_true space
]
fsp_errors = [] # Store error in leading eigenvalue of K or norm of A_hat
fsp_state_space_sizes = []

# Use a fixed, moderate N for this experiment (e.g., N=1000 from true_trajectories)
N_for_fsp_exp = 1000
exp2_trajectories = true_trajectories[1:N_for_fsp_exp];

for max_c in fsp_max_counts_series
    println("Running for FSP max_counts = $max_c")
    current_fsp_map = define_fsp_space(max_c)
    push!(fsp_state_space_sizes, length(current_fsp_map["idx_to_state"]))
    
    exp2_histograms, s2i_exp2, i2s_exp2 = get_empirical_distributions(exp2_trajectories, true_time_points, current_fsp_map)
    
    if length(i2s_exp2) < 2 || length(exp2_histograms) < 2 || size(hcat(exp2_histograms...), 2) < 2
         println("Skipping FSP size $(length(i2s_exp2)) due to insufficient states/snapshots for DMD.")
         push!(fsp_errors, NaN) # Or some large error value
         continue
    end
    
    # Rank for DMD in FSP experiment: use full available rank for the current space
    fsp_dmd_rank = min(length(i2s_exp2), length(exp2_histograms)-1)
    if fsp_dmd_rank <=0
        println("Skipping FSP size $(length(i2s_exp2)) due to rank issue.")
        push!(fsp_errors, NaN)
        continue
    end

    K_fsp, A_hat_fsp = compute_dmd_and_generator(exp2_histograms, SAVEAT_DT, fsp_dmd_rank)
    
    # Error metric: Compare leading eigenvalue of K_fsp with K_true's leading eigenvalue
    # Or, project A_true onto the current FSP space and compare A_hat_fsp
    
    # Simpler: compare a few dominant eigenvalues of K_fsp to K_true
    # This requires K_true's modes to be projectable or comparable.
    # Let's use the error in the reconstructed generator A_hat_fsp compared to A_true
    # projected onto the current FSP space.
    
    # Project A_true onto the current FSP space (A_true_proj)
    # This is non-trivial as A_true is on true_state_space_map
    # Need to map states from current_fsp_map to true_state_space_map indices
    if !ismissing(A_true)
        num_current_fsp_states = length(i2s_exp2)
        A_true_proj = spzeros(num_current_fsp_states, num_current_fsp_states)
        
        for r_idx in 1:num_current_fsp_states # row in A_hat_fsp
            state_r_fsp = i2s_exp2[r_idx]
            if haskey(true_s2i, state_r_fsp)
                true_r_idx = true_s2i[state_r_fsp]
                for c_idx in 1:num_current_fsp_states # col in A_hat_fsp
                    state_c_fsp = i2s_exp2[c_idx]
                    if haskey(true_s2i, state_c_fsp)
                        true_c_idx = true_s2i[state_c_fsp]
                        A_true_proj[r_idx, c_idx] = A_true[true_r_idx, true_c_idx]
                    end
                end
            end
        end
        err_fsp_gen = norm(A_hat_fsp - A_true_proj, 2) # Frobenius norm or 2-norm
        push!(fsp_errors, err_fsp_gen)
        println("FSP Size = $(fsp_state_space_sizes[end]), Generator Error = $err_fsp_gen")
    else
        # Fallback: compare dominant eigenvalue of K_fsp
        evals_fsp = eigen(K_fsp).values
        # Sort by magnitude, take largest real one (should be close to 1)
        lead_eval_fsp = maximum(real(evals_fsp[abs.(imag(evals_fsp)) .< 1e-6])) # Heuristic
        lead_eval_true = maximum(real(true_eigenvalues[abs.(imag(true_eigenvalues)) .< 1e-6]))

        err_fsp_eval = abs(lead_eval_fsp - lead_eval_true)
        push!(fsp_errors, err_fsp_eval)
        println("FSP Size = $(fsp_state_space_sizes[end]), Leading Eigenvalue Error = $err_fsp_eval")
    end
end

# Filter out NaNs for plotting if any FSP sizes were skipped
valid_fsp_indices = .!isnan.(fsp_errors)
plot_fsp_err = plot(fsp_state_space_sizes[valid_fsp_indices], fsp_errors[valid_fsp_indices], xlabel="State Space Size (|X_J|)", ylabel="Approximation Error", title="FSP Error vs. State Space Size", legend=false, marker=:circle, yscale=:log10, xscale=:log10)
display(plot_fsp_err)
savefig(plot_fsp_err, "fsp_error_vs_size.png")


# --- Experiment 3: DMD Spectral Error vs. Rank r ---
println("\n--- Experiment 3: DMD Spectral Error vs. Rank r ---")
# Use a fixed, large N and large state space (true_state_space_map)
N_for_dmd_exp = 2000 # Should be large enough
exp3_trajectories = true_trajectories[1:N_for_dmd_exp];
exp3_histograms, s2i_exp3, i2s_exp3 = get_empirical_distributions(exp3_trajectories, true_time_points, true_state_space_map);

# Determine max possible rank for this setup
max_rank_exp3 = min(length(i2s_exp3), length(exp3_histograms)-1)
# rank_values = unique(round.(Int, exp.(range(log(1), log(max_rank_exp3), length=20)))) # Log scale for ranks
rank_values = filter(r -> r > 0, unique(round.(Int, range(1, max_rank_exp3, length=20)))) # Linear scale

dmd_spectral_errors = [] # Compare leading eigenvalue or a few eigenvalues

if ismissing(K_true)
    println("Skipping DMD rank experiment as K_true is missing.")
else
    for r_val in rank_values
        println("Running for DMD rank r = $r_val")
        if r_val == 0 continue end # Rank must be positive
        K_dmd_r, _ = compute_dmd_and_generator(exp3_histograms, SAVEAT_DT, r_val)
        
        # Compare dominant eigenvalue
        evals_dmd_r = eigen(K_dmd_r).values
        # Heuristic for leading eigenvalue (stationary mode, should be close to 1)
        lead_eval_dmd_r = maximum(real(evals_dmd_r[abs.(imag(evals_dmd_r)) .< 1e-6]))
        lead_eval_true = maximum(real(true_eigenvalues[abs.(imag(true_eigenvalues)) .< 1e-6])) # From K_true

        err_dmd_eval = abs(lead_eval_dmd_r - lead_eval_true)
        push!(dmd_spectral_errors, err_dmd_eval)
        println("DMD Rank r = $r_val, Leading Eigenvalue Error = $err_dmd_eval")
    end

    plot_dmd_err = plot(rank_values, dmd_spectral_errors, xlabel="DMD Rank (r)", ylabel="Leading Eigenvalue Error", title="DMD Spectral Error vs. Rank r", legend=false, marker=:circle, yscale=:log10)
    display(plot_dmd_err)
    savefig(plot_dmd_err, "dmd_error_vs_rank.png")
end

println("All experiments complete. Plots saved.")
