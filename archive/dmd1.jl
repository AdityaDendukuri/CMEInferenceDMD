using JumpProcesses
using Catalyst
using DifferentialEquations
using LinearAlgebra
using SparseArrays
using StatsBase
using ProgressMeter
using Plots
using LaTeXStrings
using Random
using Statistics
using GLM
using DataFrames
using LsqFit

# Set seed for reproducibility
Random.seed!(123)

# Define the Michaelis-Menten system
function define_mm_system(params=(0.01, 0.1, 0.1))
    kB, kD, kP = params
    
    rn = @reaction_network begin
        kB, S + E --> SE
        kD, SE --> S + E
        kP, SE --> P + E
    end

    u0_integers = [:S => 50, :E => 10, :SE => 1, :P => 1]
    tspan = (0., 100.)
    ps = [:kB => kB, :kD => kD, :kP => kP]

    jinput = JumpInputs(rn, u0_integers, tspan, ps)
    jprob = JumpProblem(jinput)
    
    return jprob
end

# Generate trajectories
function generate_trajectories(jprob, n_trajs)
    println("Generating $n_trajs stochastic trajectories...")
    ssa_trajs = []
    @showprogress for i in 1:n_trajs 
        push!(ssa_trajs, solve(jprob, SSAStepper()))
    end
    return ssa_trajs
end

# Extract state from solution at time t
function get_state_at_time(sol, t)
    t_idx = argmin(abs.(sol.t .- t))
    return Int.(round.(sol[t_idx]))
end

# Construct histograms from trajectories with a fixed state space
function construct_histograms(trajectories, times; fixed_state_to_idx=nothing)
    if fixed_state_to_idx === nothing
        state_to_idx = Dict{Vector{Int}, Int}()
        idx_to_state = Dict{Int, Vector{Int}}()
        
        println("Building state space...")
        # First pass: build state space
        for traj in trajectories
            for t in times
                state = get_state_at_time(traj, t)
                if !haskey(state_to_idx, state)
                    idx = length(state_to_idx) + 1
                    state_to_idx[state] = idx
                    idx_to_state[idx] = state
                end
            end
        end
    else
        state_to_idx = fixed_state_to_idx
        idx_to_state = Dict(v => k for (k, v) in state_to_idx)
    end
    
    # Initialize histogram vectors
    n_states = length(state_to_idx)
    if fixed_state_to_idx === nothing
        println("Found $n_states unique states")
    end
    
    # Second pass: count states at each time point
    println("Computing probability distributions...")
    histograms = Dict{Float64, Vector{Float64}}()
    for t in times
        counts = zeros(Int, n_states)
        for traj in trajectories
            state = get_state_at_time(traj, t)
            if haskey(state_to_idx, state)
                idx = state_to_idx[state]
                counts[idx] += 1
            end
        end
        # Convert to probability distribution
        histograms[t] = counts ./ length(trajectories)
    end
    
    return histograms, state_to_idx, idx_to_state, n_states
end

# Compute forward operator
function compute_forward_operator(histograms, times, delta_t)
    time_array = sort(collect(keys(histograms)))
    operators = Dict{Float64, Matrix{Float64}}()
    
    println("Computing forward operators directly...")
    for i in 1:(length(time_array)-1)
        t = time_array[i]
        t_next = time_array[i+1]
        
        # Only process if time difference matches delta_t
        if abs(t_next - t - delta_t) < 0.1*delta_t
            p_t = histograms[t]
            p_t_next = histograms[t_next]
            
            # Create forward operator (initialized to identity)
            n = length(p_t)
            P = Matrix{Float64}(I, n, n)
            
            # For each state with non-zero probability at t
            active_indices = findall(x -> x > 1e-12, p_t)
            
            for j in active_indices
                if p_t[j] > 1e-12
                    # Estimate transition probabilities
                    for i in 1:n
                        # Lower threshold for considering a transition
                        if p_t_next[i] > 1e-12
                            # More permissive calculation of transition probability
                            P[i, j] = min(1.0, p_t_next[i] / p_t[j])
                        end
                    end
                    
                    # Ensure column sums to 1
                    col_sum = sum(P[:, j])
                    if col_sum > 0
                        P[:, j] ./= col_sum
                    end
                end
            end
            
            operators[t] = P
        end
    end
    
    return operators
end

#############################################################
# APPROACH 1: Fitting the Exponents in Theoretical Bounds
#############################################################

# Function to fit exponents in histogram error bound: C * |X_J|^α / N^β
function fit_histogram_error_exponents(jprob, max_time, delta_t)
    # Generate high-quality reference dataset
    println("Computing reference solution with 50,000 trajectories...")
    ref_trajs = generate_trajectories(jprob, 50000)
    times = 0:delta_t:max_time
    
    # Fix the state space based on reference solution
    ref_histograms, ref_state_to_idx, ref_idx_to_state, ref_n_states = 
        construct_histograms(ref_trajs, times)
    
    println("Reference state space has $ref_n_states states")
    
    # Test with varying trajectory counts
    n_trajectories = [10, 30, 100, 300, 1000, 3000, 10000]
    errors = []
    
    println("Testing histogram error with different trajectory counts...")
    for n in n_trajectories
        println("Testing with $n trajectories...")
        trajs = generate_trajectories(jprob, n)
        
        # Use fixed state space
        histograms, _, _, _ = construct_histograms(trajs, times, fixed_state_to_idx=ref_state_to_idx)
        
        # Compute L1 error at each time point
        time_errors = []
        for t in times
            if haskey(histograms, t) && haskey(ref_histograms, t)
                push!(time_errors, norm(histograms[t] - ref_histograms[t], 1))
            end
        end
        
        # Average error across time points
        avg_error = mean(time_errors)
        push!(errors, avg_error)
    end
    
    # Log-transform for linear fitting
    log_N = log.(n_trajectories)
    log_error = log.(errors)
    
    # Fit exponent β in error ~ N^(-β)
    df = DataFrame(X=log_N, Y=log_error)
    model = lm(@formula(Y ~ X), df)
    
    # Extract parameters
    coefficients = coef(model)
    intercept, slope = coefficients
    beta = -slope
    
    # Compute C using the intercept
    C = exp(intercept) * ref_n_states^0  # initially assume α=0
    
    println("Fitted exponent β for N: $beta (theoretical is 0.5)")
    println("Fitted constant C: $C")
    
    # Fit more complex model with varying state space size
    # For this, we need data with different state space sizes
    println("\nFitting model with varying state space size...")
    
    # Generate trajectories with different state space truncations
    n_fixed = 1000  # Fix number of trajectories
    trajs_fixed = generate_trajectories(jprob, n_fixed)
    
    # Create different state space sizes by percentile truncation
    percentiles = [50, 65, 80, 95, 99]
    state_sizes = []
    errors_vs_states = []
    
    for p in percentiles
        println("Testing with $p-th percentile truncation...")
        
        # Extract states from trajectories
        all_states = Dict()
        for traj in trajs_fixed
            for t in times
                state = get_state_at_time(traj, t)
                state_tuple = Tuple(state)
                if !haskey(all_states, state_tuple)
                    all_states[state_tuple] = 0
                end
                all_states[state_tuple] += 1
            end
        end
        
        # Create truncated state space based on percentile
        species_counts = Dict{Int, Vector{Int}}()
        for state in keys(all_states)
            for (i, count) in enumerate(state)
                if !haskey(species_counts, i)
                    species_counts[i] = Int[]
                end
                push!(species_counts[i], count)
            end
        end
        
        thresholds = Dict()
        for (species, counts) in species_counts
            thresholds[species] = ceil(Int, quantile(counts, p/100))
        end
        
        # Create truncated state maps
        trunc_state_to_idx = Dict{Vector{Int}, Int}()
        idx = 1
        for state in keys(all_states)
            state_vec = collect(state)
            if all(state_vec[j] <= thresholds[j] for j in 1:length(state_vec))
                trunc_state_to_idx[state_vec] = idx
                idx += 1
            end
        end
        
        n_trunc_states = length(trunc_state_to_idx)
        push!(state_sizes, n_trunc_states)
        
        # Compute histograms with truncated state space
        trunc_histograms, _, _, _ = construct_histograms(trajs_fixed, times, fixed_state_to_idx=trunc_state_to_idx)
        
        # Compute error compared to reference
        truncation_errors = []
        for t in times
            if haskey(trunc_histograms, t) && haskey(ref_histograms, t)
                # Need to align the distributions
                trunc_hist = zeros(ref_n_states)
                for (idx, state) in ref_idx_to_state
                    if haskey(trunc_state_to_idx, state)
                        trunc_idx = trunc_state_to_idx[state]
                        if trunc_idx <= length(trunc_histograms[t])
                            trunc_hist[idx] = trunc_histograms[t][trunc_idx]
                        end
                    end
                end
                push!(truncation_errors, norm(trunc_hist - ref_histograms[t], 1))
            end
        end
        
        avg_trunc_error = mean(truncation_errors)
        push!(errors_vs_states, avg_trunc_error)
    end
    
    # Fit exponent α in error ~ |X_J|^α
    log_S = log.(state_sizes)
    log_error_S = log.(errors_vs_states)
    
    df_S = DataFrame(X=log_S, Y=log_error_S)
    model_S = lm(@formula(Y ~ X), df_S)
    
    # Extract parameters
    coefficients_S = coef(model_S)
    intercept_S, slope_S = coefficients_S
    alpha = slope_S
    
    println("Fitted exponent α for |X_J|: $alpha (theoretical is 1.0)")
    
    # Now fit the combined model
    # Create synthetic dataset combining both effects
    n_samples = 20
    sim_Ns = 10 .^ rand(1:4, n_samples)  # 10 to 10000
    sim_states = 50 .^ rand(1:3, n_samples)  # 50 to 125000
    
    sim_errors = []
    for i in 1:n_samples
        # Estimate error: C * |X_J|^α / N^β
        sim_error = C * sim_states[i]^alpha / sim_Ns[i]^beta
        push!(sim_errors, sim_error)
    end
    
    # Fit full model: error ~ C * |X_J|^α / N^β
    # This is log(error) = log(C) + α*log(|X_J|) - β*log(N)
    df_full = DataFrame(X1=log.(sim_states), X2=log.(sim_Ns), Y=log.(sim_errors))
    model_full = lm(@formula(Y ~ X1 + X2), df_full)
    
    # Extract parameters
    coefficients_full = coef(model_full)
    intercept_full, alpha_full, beta_neg_full = coefficients_full
    beta_full = -beta_neg_full
    C_full = exp(intercept_full)
    
    println("Full model fit:")
    println("C = $C_full, α = $alpha_full, β = $beta_full")
    
    # Create theoretical curves with fitted exponents
    fitted_curve_N = C * ref_n_states^alpha ./ n_trajectories.^beta
    fitted_curve_S = C * state_sizes.^alpha / n_fixed^beta
    
    # Create theoretical curves with original exponents (α=1, β=0.5)
    original_curve_N = (errors[1] / (ref_n_states / sqrt(n_trajectories[1]))) * ref_n_states ./ sqrt.(n_trajectories)
    original_curve_S = (errors_vs_states[1] / (state_sizes[1] / sqrt(n_fixed))) * state_sizes ./ sqrt(n_fixed)
    
    # Plot results
    p1 = plot(n_trajectories, errors, 
              xscale=:log10, yscale=:log10,
              marker=:circle, markersize=6, label="Experimental Data",
              xlabel="Number of Trajectories (N)", 
              ylabel="Histogram Error ‖p̂-p‖₁",
              title="Histogram Error vs Number of Trajectories",
              legend=:topright, dpi=300)
    
    plot!(p1, n_trajectories, fitted_curve_N, 
          line=:dash, linewidth=2, label="Fitted: C·|X_J|^$alpha/N^$beta")
    
    plot!(p1, n_trajectories, original_curve_N, 
          line=:dot, linewidth=2, label="Original: C·|X_J|/√N")
    
    p2 = plot(state_sizes, errors_vs_states, 
              xscale=:log10, yscale=:log10,
              marker=:circle, markersize=6, label="Experimental Data",
              xlabel="State Space Size |X_J|", 
              ylabel="Histogram Error ‖p̂-p‖₁",
              title="Histogram Error vs State Space Size",
              legend=:topright, dpi=300)
    
    plot!(p2, state_sizes, fitted_curve_S, 
          line=:dash, linewidth=2, label="Fitted: C·|X_J|^$alpha/N^$beta")
    
    plot!(p2, state_sizes, original_curve_S, 
          line=:dot, linewidth=2, label="Original: C·|X_J|/√N")
    
    savefig(p1, "figures/histogram_fitted_n.png")
    savefig(p2, "figures/histogram_fitted_s.png")
    
    return Dict(
        "N_traj" => n_trajectories,
        "errors_N" => errors,
        "state_sizes" => state_sizes,
        "errors_S" => errors_vs_states,
        "fitted_exponent_beta" => beta,
        "fitted_exponent_alpha" => alpha,
        "fitted_constant_C" => C,
        "fitted_curve_N" => fitted_curve_N,
        "fitted_curve_S" => fitted_curve_S,
        "original_curve_N" => original_curve_N,
        "original_curve_S" => original_curve_S,
        "plots" => (p1, p2)
    )
end

# Function to fit exponents in DMD error bound: C * (|X_J|^α/N^β + σ_{r+1}^γ)
function fit_dmd_error_exponents(jprob, max_time, delta_t)
    # Generate high-quality trajectories
    println("Generating 10,000 trajectories for DMD validation...")
    trajs = generate_trajectories(jprob, 10000)
    times = 0:delta_t:max_time
    
    # Compute histograms
    histograms, _, _, n_states = construct_histograms(trajs, times)
    
    # Set up snapshot matrices
    time_array = sort(collect(keys(histograms)))
    X = hcat([histograms[t] for t in time_array[1:end-1]]...)
    X_prime = hcat([histograms[t] for t in time_array[2:end]]...)
    
    println("Computing full SVD for reference...")
    # Full SVD for reference
    U, Σ, V = svd(X)
    
    # Get the exact Koopman operator via regularized pseudoinverse
    reg_param = 1e-10
    X_pinv = V * Diagonal(1.0 ./ (Σ .+ reg_param)) * U'
    K_exact = X_prime * X_pinv
    
    # Test different ranks for DMD
    println("Testing DMD error with different rank truncations...")
    ranks = [2, 5, 10, 20, 50, 100, 200]
    dmd_errors = []
    sing_vals = []
    
    for r in ranks
        println("Testing with rank $r...")
        # Truncated SVD
        r_actual = min(r, length(Σ), size(U, 2), size(V, 2))
        Ur = U[:, 1:r_actual]
        Σr = Σ[1:r_actual]
        Vr = V[:, 1:r_actual]
        
        # Truncated DMD
        K_r = X_prime * Vr * inv(Diagonal(Σr) + reg_param*I) * Ur'
        
        # Compute error using operator norm
        operator_error = norm(K_r - K_exact, 2) / norm(K_exact, 2)
        push!(dmd_errors, operator_error)
        
        # Store the first neglected singular value
        if r_actual < length(Σ)
            push!(sing_vals, Σ[r_actual+1])
        else
            push!(sing_vals, 0.0)
        end
    end
    
    # Test effect of sampling error by varying N
    println("Testing effect of sampling error on DMD...")
    fixed_rank = 50  # Fixed rank
    n_trajectories = [100, 300, 1000, 3000, 10000]
    sampling_errors = []
    
    for n in n_trajectories
        println("Testing with $n trajectories...")
        trajs_n = generate_trajectories(jprob, n)
        
        # Compute histograms
        histograms_n, _, _, _ = construct_histograms(trajs_n, times)
        
        # Set up snapshot matrices
        X_n = hcat([histograms_n[t] for t in time_array[1:end-1]]...)
        X_prime_n = hcat([histograms_n[t] for t in time_array[2:end]]...)
        
        # SVD
        U_n, Σ_n, V_n = svd(X_n)
        
        # Truncated DMD
        r_actual = min(fixed_rank, length(Σ_n), size(U_n, 2), size(V_n, 2))
        Ur_n = U_n[:, 1:r_actual]
        Σr_n = Σ_n[1:r_actual]
        Vr_n = V_n[:, 1:r_actual]
        
        K_n = X_prime_n * Vr_n * inv(Diagonal(Σr_n) + reg_param*I) * Ur_n'
        
        # Compare to reference K_exact (recompute with same size)
        min_size = min(size(K_n, 1), size(K_exact, 1))
        K_n_trunc = K_n[1:min_size, 1:min_size]
        K_exact_trunc = K_exact[1:min_size, 1:min_size]
        
        operator_error = norm(K_n_trunc - K_exact_trunc, 2) / norm(K_exact_trunc, 2)
        push!(sampling_errors, operator_error)
    end
    
    # Fit exponent for rank effect: error ~ r^(-γ)
    non_zero_indices = findall(x -> x > 0, sing_vals)
    if !isempty(non_zero_indices)
        log_r = log.(ranks[non_zero_indices])
        log_error_r = log.(dmd_errors[non_zero_indices])
        
        df_r = DataFrame(X=log_r, Y=log_error_r)
        model_r = lm(@formula(Y ~ X), df_r)
        
        # Extract parameters
        coefficients_r = coef(model_r)
        if length(coefficients_r) >= 2
            intercept_r, slope_r = coefficients_r
            gamma = -slope_r
            
            println("Fitted exponent γ for rank r: $gamma (theoretical is 1.0)")
            
            # Fit exponent for singular value effect: error ~ σ_{r+1}^ρ
            log_sigma = log.(sing_vals[non_zero_indices])
            
            df_sigma = DataFrame(X=log_sigma, Y=log_error_r)
            model_sigma = lm(@formula(Y ~ X), df_sigma)
            
            # Extract parameters
            coefficients_sigma = coef(model_sigma)
            if length(coefficients_sigma) >= 2
                intercept_sigma, slope_sigma = coefficients_sigma
                rho = slope_sigma
                
                println("Fitted exponent ρ for σ_{r+1}: $rho (theoretical is 1.0)")
                
                # Fit exponent for sampling effect: error ~ N^(-β)
                log_N = log.(n_trajectories)
                log_error_N = log.(sampling_errors)
                
                df_N = DataFrame(X=log_N, Y=log_error_N)
                model_N = lm(@formula(Y ~ X), df_N)
                
                # Extract parameters
                coefficients_N = coef(model_N)
                intercept_N, slope_N = coefficients_N
                beta = -slope_N
                
                println("Fitted exponent β for N: $beta (theoretical is 0.5)")
                
                # Fit combined model
                # error ~ C * (|X_J|^α/N^β + σ_{r+1}^ρ)
                # We'll assume α = 1 for simplicity
                
                # Generate synthetic data
                n_samples = 20
                sim_rs = rand(2:200, n_samples)
                sim_sigmas = [r < length(Σ) ? Σ[r+1] : 0.0 for r in sim_rs]
                sim_Ns = 10 .^ rand(2:4, n_samples)
                
                # Calculate expected errors
                sim_errors = []
                C1 = exp(intercept_N - log(n_states))  # adjust for n_states
                C2 = exp(intercept_sigma)
                
                for i in 1:n_samples
                    term1 = C1 * n_states / sim_Ns[i]^beta
                    term2 = C2 * sim_sigmas[i]^rho
                    push!(sim_errors, term1 + term2)
                end
                
                # Create fitted curves
                C_rank = exp(intercept_r)
                fitted_curve_rank = C_rank ./ ranks.^gamma
                
                C_sv = exp(intercept_sigma)
                non_zero_sv = sing_vals[non_zero_indices]
                fitted_curve_sv = C_sv .* non_zero_sv.^rho
                
                C_N = exp(intercept_N)
                fitted_curve_N = C_N ./ n_trajectories.^beta
                
                # Original theoretical curves
                original_curve_rank = (dmd_errors[1] / (1.0/ranks[1])) ./ ranks
                original_curve_sv = (dmd_errors[non_zero_indices][1] / sing_vals[non_zero_indices][1]) .* non_zero_sv
                original_curve_N = (sampling_errors[1] / (n_states/sqrt(n_trajectories[1]))) * n_states ./ sqrt.(n_trajectories)
                
                # Plot results
                p1 = plot(ranks, dmd_errors, 
                          xscale=:log10, yscale=:log10,
                          marker=:circle, markersize=6, label="Experimental Data",
                          xlabel="DMD Rank (r)", 
                          ylabel="DMD Error ‖K_r - K‖",
                          title="DMD Error vs Rank",
                          legend=:topright, dpi=300)
                
                plot!(p1, ranks, fitted_curve_rank, 
                      line=:dash, linewidth=2, label="Fitted: C/r^$gamma")
                
                plot!(p1, ranks, original_curve_rank, 
                      line=:dot, linewidth=2, label="Original: C/r")
                
                p2 = plot(sing_vals[non_zero_indices], dmd_errors[non_zero_indices], 
                          xscale=:log10, yscale=:log10,
                          marker=:circle, markersize=6, label="Experimental Data",
                          xlabel="First Neglected Singular Value (σ_{r+1})", 
                          ylabel="DMD Error ‖K_r - K‖",
                          title="DMD Error vs Singular Value",
                          legend=:topright, dpi=300)
                
                plot!(p2, sing_vals[non_zero_indices], fitted_curve_sv, 
                      line=:dash, linewidth=2, label="Fitted: C·σ_{r+1}^$rho")
                
                plot!(p2, sing_vals[non_zero_indices], original_curve_sv, 
                      line=:dot, linewidth=2, label="Original: C·σ_{r+1}")
                
                p3 = plot(n_trajectories, sampling_errors, 
                          xscale=:log10, yscale=:log10,
                          marker=:circle, markersize=6, label="Experimental Data",
                          xlabel="Number of Trajectories (N)", 
                          ylabel="DMD Error ‖K_N - K‖",
                          title="DMD Error vs Number of Trajectories",
                          legend=:topright, dpi=300)
                
                plot!(p3, n_trajectories, fitted_curve_N, 
                      line=:dash, linewidth=2, label="Fitted: C·|X_J|/N^$beta")
                
                plot!(p3, n_trajectories, original_curve_N, 
                      line=:dot, linewidth=2, label="Original: C·|X_J|/√N")
                
                savefig(p1, "figures/dmd_fitted_rank.png")
                savefig(p2, "figures/dmd_fitted_sv.png")
                savefig(p3, "figures/dmd_fitted_n.png")
                
                return Dict(
                    "ranks" => ranks,
                    "dmd_errors" => dmd_errors,
                    "sing_vals" => sing_vals,
                    "n_trajectories" => n_trajectories,
                    "sampling_errors" => sampling_errors,
                    "fitted_exponent_gamma" => gamma,
                    "fitted_exponent_rho" => rho,
                    "fitted_exponent_beta" => beta,
                    "fitted_curve_rank" => fitted_curve_rank,
                    "fitted_curve_sv" => fitted_curve_sv,
                    "fitted_curve_N" => fitted_curve_N,
                    "original_curve_rank" => original_curve_rank,
                    "original_curve_sv" => original_curve_sv,
                    "original_curve_N" => original_curve_N,
                    "plots" => (p1, p2, p3)
                )
            end
        end
    end
    
    # If fitting failed, return only the raw data
    return Dict(
        "ranks" => ranks,
        "dmd_errors" => dmd_errors,
        "sing_vals" => sing_vals,
        "n_trajectories" => n_trajectories,
        "sampling_errors" => sampling_errors
    )
end

# Function to fit exponents in FSP error bound: C * |X_J|^(-α)
function fit_fsp_error_exponents(jprob, max_time, delta_t)
    # Generate a very large reference dataset
    println("Generating 10,000 trajectories for FSP validation...")
    ref_trajs = generate_trajectories(jprob, 10000)
    
    # Focus on steady-state distribution
    steady_state_time = max_time * 0.9
    println("Analyzing steady-state at t = $steady_state_time")
    
    # Extract steady-state distribution
    full_state_probs = Dict()
    species_counts = Dict{Int, Vector{Int}}()
    
    for traj in ref_trajs
        t_idx = argmin(abs.(traj.t .- steady_state_time))
        state = Int.(round.(traj[t_idx]))
        
        # Store state in dictionary
        state_tuple = Tuple(state)
        if !haskey(full_state_probs, state_tuple)
            full_state_probs[state_tuple] = 0
        end
        full_state_probs[state_tuple] += 1
        
        # Store individual species counts
        for (i, count) in enumerate(state)
            if !haskey(species_counts, i)
                species_counts[i] = Int[]
            end
            push!(species_counts[i], count)
        end
    end
    
    # Normalize to get probabilities
    for state in keys(full_state_probs)
        full_state_probs[state] /= length(ref_trajs)
    end
    
    # Test different state space truncations
    println("Testing FSP error with various state space truncations...")
    # Use more varied truncation levels
    percentiles = [25, 40, 55, 70, 85, 95, 99]
    state_space_sizes = []
    probability_leakages = []
    
    for p in percentiles
        println("Testing with $p-th percentile truncation...")
        # Compute truncation thresholds
        thresholds = Dict()
        for (species, counts) in species_counts
            thresholds[species] = ceil(Int, quantile(counts, p/100))
        end
        
        # Count states and probability in truncated space
        n_states_in_truncation = 0
        prob_in_truncation = 0.0
        
        for (state, prob) in full_state_probs
            state_vec = collect(state)
            if all(state_vec[i] <= thresholds[i] for i in 1:length(state_vec))
                n_states_in_truncation += 1
                prob_in_truncation += prob
            end
        end
        
        push!(state_space_sizes, n_states_in_truncation)
        leakage = 1.0 - prob_in_truncation
        push!(probability_leakages, leakage)
    end
    
    # Fit exponent α in error ~ |X_J|^(-α)
    log_S = log.(state_space_sizes)
    log_error = log.(probability_leakages)
    
    df = DataFrame(X=log_S, Y=log_error)
    model = lm(@formula(Y ~ X), df)
    
    # Extract parameters
    coefficients = coef(model)
    intercept, slope = coefficients
    alpha = -slope  # Note: the negative sign because we expect inverse scaling
    
    println("Fitted exponent α for FSP: $alpha (theoretical is 1.0)")
    
    # Calculate constant C
    C = exp(intercept + alpha * log_S[1]) * state_space_sizes[1]^alpha
    println("Fitted constant C: $C")
    
    # Create fitted curve
    fitted_curve = C ./ state_space_sizes.^alpha
    
    # Create theoretical curve with α=1
    original_curve = (probability_leakages[1] / (1.0/state_space_sizes[1])) ./ state_space_sizes
    
    # Plot results
    p = plot(state_space_sizes, probability_leakages, 
             xscale=:log10, yscale=:log10,
             marker=:circle, markersize=6, label="Experimental Data",
             xlabel="State Space Size |X_J|", 
             ylabel="FSP Error (Probability Leakage)",
             title="FSP Error vs State Space Size",
             legend=:topright, dpi=300)
    
    plot!(p, state_space_sizes, fitted_curve, 
          line=:dash, linewidth=2, label="Fitted: C/|X_J|^$alpha")
    
    plot!(p, state_space_sizes, original_curve, 
          line=:dot, linewidth=2, label="Original: C/|X_J|")
    
    savefig(p, "figures/fsp_fitted.png")
    
    return Dict(
        "percentiles" => percentiles,
        "state_space_sizes" => state_space_sizes,
        "probability_leakages" => probability_leakages,
        "fitted_exponent_alpha" => alpha,
        "fitted_constant_C" => C,
        "fitted_curve" => fitted_curve,
        "original_curve" => original_curve,
        "plot" => p
    )
end

#############################################################
# APPROACH 2: Non-Asymptotic Bounds
#############################################################

# Function to derive tighter non-asymptotic bounds for histogram error
function derive_non_asymptotic_bounds(jprob, max_time, delta_t)
    println("Deriving non-asymptotic bounds for histogram error...")
    
    # Generate reference solution
    ref_trajs = generate_trajectories(jprob, 50000)
    times = 0:delta_t:max_time
    
    # Compute reference histograms
    ref_histograms, ref_state_to_idx, ref_idx_to_state, ref_n_states = 
        construct_histograms(ref_trajs, times)
    
    println("Reference state space has $ref_n_states states")
    
    # Calculate variance and range for each state
    variances = Dict()
    ranges = Dict()
    
    # Extract middle time point for analysis
    mid_time = times[div(length(times), 2)]
    
    # Compute bootstrap replicates to estimate variance
    n_bootstrap = 1000
    bootstrap_size = 5000
    bootstrap_histograms = []
    
    for i in 1:n_bootstrap
        # Sample trajectories with replacement
        indices = rand(1:length(ref_trajs), bootstrap_size)
        bootstrap_trajs = ref_trajs[indices]
        
        # Compute histogram for this bootstrap sample
        counts = zeros(Int, ref_n_states)
        for traj in bootstrap_trajs
            state = get_state_at_time(traj, mid_time)
            if haskey(ref_state_to_idx, state)
                idx = ref_state_to_idx[state]
                counts[idx] += 1
            end
        end
        
        # Convert to probability distribution
        hist = counts ./ bootstrap_size
        push!(bootstrap_histograms, hist)
    end
    
    # Compute variance for each state
    state_variances = zeros(ref_n_states)
    for i in 1:ref_n_states
        values = [h[i] for h in bootstrap_histograms]
        state_variances[i] = var(values)
    end
    
    # Compute maximum variance
    max_var = maximum(state_variances)
    println("Maximum variance across states: $max_var")
    
    # Compute average variance
    avg_var = mean(state_variances)
    println("Average variance across states: $avg_var")
    
    # Compute total L1 norm variance
    var_L1 = sum(sqrt.(state_variances))
    println("Total L1 norm variance: $var_L1")
    
    # Implement Bernstein's inequality for L1 error
    # P(‖p̂ - p‖₁ > ε) ≤ 2 exp(-Nε²/(2∑var + 2ε/3))
    # Solving for ε at confidence level 0.95
    function bernstein_bound(N, confidence=0.95)
        # Calculate epsilon by iterative method
        target = log(2/(1-confidence))
        f(eps) = N*eps^2/(2*var_L1 + 2*eps/3) - target
        
        # Find root by bisection
        eps_min, eps_max = 0.001, 10.0
        while eps_max - eps_min > 1e-6
            eps_mid = (eps_min + eps_max) / 2
            if f(eps_mid) < 0
                eps_min = eps_mid
            else
                eps_max = eps_mid
            end
        end
        
        return (eps_min + eps_max) / 2
    end
    
    # Implement Hoeffding's inequality for L1 error
    # P(‖p̂ - p‖₁ > ε) ≤ 2|X_J|exp(-Nε²/(2|X_J|²))
    function hoeffding_bound(N, n_states, confidence=0.95)
        # Calculate epsilon
        target = log(2*n_states/(1-confidence))
        return sqrt(2*n_states^2*target/N)
    end
    
    # Test with different sample sizes
    n_trajectories = [10, 30, 100, 300, 1000, 3000, 10000]
    bernstein_bounds = []
    hoeffding_bounds = []
    
    for n in n_trajectories
        push!(bernstein_bounds, bernstein_bound(n))
        push!(hoeffding_bounds, hoeffding_bound(n, ref_n_states))
    end
    
    # Compute actual errors
    println("Computing actual errors for various trajectory counts...")
    actual_errors = []
    
    for n in n_trajectories
        println("Testing with $n trajectories...")
        trajs = generate_trajectories(jprob, n)
        
        # Compute histograms with reference state space
        histograms, _, _, _ = construct_histograms(trajs, times, fixed_state_to_idx=ref_state_to_idx)
        
        # Compute L1 error
        time_errors = []
        for t in times
            if haskey(histograms, t) && haskey(ref_histograms, t)
                push!(time_errors, norm(histograms[t] - ref_histograms[t], 1))
            end
        end
        
        avg_error = mean(time_errors)
        push!(actual_errors, avg_error)
    end
    
    # Plot results
    p = plot(n_trajectories, actual_errors, 
             xscale=:log10, yscale=:log10,
             marker=:circle, markersize=6, label="Actual Error",
             xlabel="Number of Trajectories (N)", 
             ylabel="Histogram Error ‖p̂-p‖₁",
             title="Non-Asymptotic Bounds for Histogram Error",
             legend=:topright, dpi=300)
    
    plot!(p, n_trajectories, bernstein_bounds, 
          line=:dash, linewidth=2, label="Bernstein Bound")
    
    plot!(p, n_trajectories, hoeffding_bounds, 
          line=:dot, linewidth=2, label="Hoeffding Bound")
    
    # Add theoretical asymptotic bound
    asymptotic_scale = actual_errors[1] / (ref_n_states / sqrt(n_trajectories[1]))
    asymptotic_bounds = asymptotic_scale * ref_n_states ./ sqrt.(n_trajectories)
    
    plot!(p, n_trajectories, asymptotic_bounds, 
          line=(:dashdot, :black), linewidth=2, label="Original O(|X_J|/√N)")
    
    savefig(p, "figures/non_asymptotic_bounds.png")
    
    return Dict(
        "n_trajectories" => n_trajectories,
        "actual_errors" => actual_errors,
        "bernstein_bounds" => bernstein_bounds,
        "hoeffding_bounds" => hoeffding_bounds,
        "asymptotic_bounds" => asymptotic_bounds,
        "state_variances" => state_variances,
        "max_variance" => max_var,
        "avg_variance" => avg_var,
        "L1_variance" => var_L1,
        "plot" => p
    )
end

#############################################################
# APPROACH 3: System-Specific Constants
#############################################################

# Function to explore how constants depend on system parameters
function explore_system_specific_constants(max_time=50.0, delta_t=0.5)
    println("Exploring system-specific constants for different parameter sets...")
    
    # Define different parameter sets for Michaelis-Menten system
    # [kB, kD, kP]
    parameter_sets = [
        (0.01, 0.1, 0.1),   # Original
        (0.005, 0.1, 0.1),  # Lower binding rate
        (0.02, 0.1, 0.1),   # Higher binding rate
        (0.01, 0.05, 0.1),  # Lower unbinding rate
        (0.01, 0.2, 0.1),   # Higher unbinding rate
        (0.01, 0.1, 0.05),  # Lower product formation rate
        (0.01, 0.1, 0.2)    # Higher product formation rate
    ]
    
    # Compute equilibrium constants for each parameter set
    K_eq = []
    for (kB, kD, kP) in parameter_sets
        push!(K_eq, kB/kD)  # Equilibrium constant for binding/unbinding
    end
    
    # Define fixed trajectory count and rank for experiments
    fixed_n = 1000
    fixed_rank = 20
    
    # Store results
    histogram_errors = []
    dmd_errors = []
    fsp_errors = []
    state_sizes = []
    
    for (i, params) in enumerate(parameter_sets)
        kB, kD, kP = params
        println("\nTesting parameter set $i: kB=$kB, kD=$kD, kP=$kP")
        
        # Define system with these parameters
        jprob = define_mm_system(params)
        
        # Generate reference solution
        ref_trajs = generate_trajectories(jprob, 10000)
        times = 0:delta_t:max_time
        
        # Compute reference histograms
        ref_histograms, ref_state_to_idx, ref_idx_to_state, ref_n_states = 
            construct_histograms(ref_trajs, times)
        
        push!(state_sizes, ref_n_states)
        println("State space size: $ref_n_states")
        
        # Histogram error
        println("Computing histogram error...")
        trajs = generate_trajectories(jprob, fixed_n)
        
        # Compute histograms with reference state space
        histograms, _, _, _ = construct_histograms(trajs, times, fixed_state_to_idx=ref_state_to_idx)
        
        # Compute L1 error
        time_errors = []
        for t in times
            if haskey(histograms, t) && haskey(ref_histograms, t)
                push!(time_errors, norm(histograms[t] - ref_histograms[t], 1))
            end
        end
        
        avg_error = mean(time_errors)
        push!(histogram_errors, avg_error)
        
        # Scale by theoretical factor to get constant
        hist_constant = avg_error / (ref_n_states / sqrt(fixed_n))
        println("Histogram error constant: $hist_constant")
        
        # DMD error
        println("Computing DMD error...")
        # Set up snapshot matrices
        time_array = sort(collect(keys(histograms)))
        X = hcat([histograms[t] for t in time_array[1:end-1]]...)
        X_prime = hcat([histograms[t] for t in time_array[2:end]]...)
        
        # SVD
        U, Σ, V = svd(X)
        
        # Reference operator
        reg_param = 1e-10
        X_pinv = V * Diagonal(1.0 ./ (Σ .+ reg_param)) * U'
        K_exact = X_prime * X_pinv
        
        # Truncated DMD
        r_actual = min(fixed_rank, length(Σ), size(U, 2), size(V, 2))
        Ur = U[:, 1:r_actual]
        Σr = Σ[1:r_actual]
        Vr = V[:, 1:r_actual]
        
        K_r = X_prime * Vr * inv(Diagonal(Σr) + reg_param*I) * Ur'
        
        # Compute error
        operator_error = norm(K_r - K_exact, 2) / norm(K_exact, 2)
        push!(dmd_errors, operator_error)
        
        # Scale by theoretical factor to get constant
        dmd_constant = operator_error * fixed_rank / ref_n_states * sqrt(fixed_n)
        println("DMD error constant: $dmd_constant")
        
        # FSP error
        println("Computing FSP error...")
        # Use 95th percentile truncation
        species_counts = Dict{Int, Vector{Int}}()
        for traj in ref_trajs
            for t in times
                state = get_state_at_time(traj, t)
                for (i, count) in enumerate(state)
                    if !haskey(species_counts, i)
                        species_counts[i] = Int[]
                    end
                    push!(species_counts[i], count)
                end
            end
        end
        
        thresholds = Dict()
        for (species, counts) in species_counts
            thresholds[species] = ceil(Int, quantile(counts, 0.95))
        end
        
        # Count states and probability in truncated space
        full_state_probs = Dict()
        for traj in ref_trajs
            t_idx = argmin(abs.(traj.t .- max_time*0.9))
            state = Int.(round.(traj[t_idx]))
            
            state_tuple = Tuple(state)
            if !haskey(full_state_probs, state_tuple)
                full_state_probs[state_tuple] = 0
            end
            full_state_probs[state_tuple] += 1
        end
        
        # Normalize
        for state in keys(full_state_probs)
            full_state_probs[state] /= length(ref_trajs)
        end
        
        # Count states and probability in truncated space
        n_states_in_truncation = 0
        prob_in_truncation = 0.0
        
        for (state, prob) in full_state_probs
            state_vec = collect(state)
            if all(state_vec[i] <= thresholds[i] for i in 1:length(state_vec))
                n_states_in_truncation += 1
                prob_in_truncation += prob
            end
        end
        
        leakage = 1.0 - prob_in_truncation
        push!(fsp_errors, leakage)
        
        # Scale by theoretical factor to get constant
        fsp_constant = leakage * n_states_in_truncation
        println("FSP error constant: $fsp_constant")
    end
    
    # Fit models relating constants to parameters
    # For histogram error
    df_hist = DataFrame(
        K_eq = K_eq,
        kB = [p[1] for p in parameter_sets],
        kD = [p[2] for p in parameter_sets],
        kP = [p[3] for p in parameter_sets],
        state_size = state_sizes,
        error = histogram_errors,
        constant = [histogram_errors[i] / (state_sizes[i] / sqrt(fixed_n)) for i in 1:length(histogram_errors)]
    )
    
    # For DMD error
    df_dmd = DataFrame(
        K_eq = K_eq,
        kB = [p[1] for p in parameter_sets],
        kD = [p[2] for p in parameter_sets],
        kP = [p[3] for p in parameter_sets],
        state_size = state_sizes,
        error = dmd_errors,
        constant = [dmd_errors[i] * fixed_rank / state_sizes[i] * sqrt(fixed_n) for i in 1:length(dmd_errors)]
    )
    
    # For FSP error
    df_fsp = DataFrame(
        K_eq = K_eq,
        kB = [p[1] for p in parameter_sets],
        kD = [p[2] for p in parameter_sets],
        kP = [p[3] for p in parameter_sets],
        state_size = state_sizes,
        error = fsp_errors,
        constant = [fsp_errors[i] * state_sizes[i] for i in 1:length(fsp_errors)]
    )
    
    # Try to fit relationships
    println("\nFitting models for system-specific constants:")
    
    # Histogram error
    println("\nHistogram Error Constants:")
    model_hist_K = lm(@formula(constant ~ K_eq), df_hist)
    println("Model with K_eq: p-value = $(round(coeftable(model_hist_K).cols[4][2], digits=5))")
    
    model_hist_kB = lm(@formula(constant ~ kB), df_hist)
    println("Model with kB: p-value = $(round(coeftable(model_hist_kB).cols[4][2], digits=5))")
    
    model_hist_kD = lm(@formula(constant ~ kD), df_hist)
    println("Model with kD: p-value = $(round(coeftable(model_hist_kD).cols[4][2], digits=5))")
    
    model_hist_kP = lm(@formula(constant ~ kP), df_hist)
    println("Model with kP: p-value = $(round(coeftable(model_hist_kP).cols[4][2], digits=5))")
    
    # DMD error
    println("\nDMD Error Constants:")
    model_dmd_K = lm(@formula(constant ~ K_eq), df_dmd)
    println("Model with K_eq: p-value = $(round(coeftable(model_dmd_K).cols[4][2], digits=5))")
    
    model_dmd_kB = lm(@formula(constant ~ kB), df_dmd)
    println("Model with kB: p-value = $(round(coeftable(model_dmd_kB).cols[4][2], digits=5))")
    
    model_dmd_kD = lm(@formula(constant ~ kD), df_dmd)
    println("Model with kD: p-value = $(round(coeftable(model_dmd_kD).cols[4][2], digits=5))")
    
    model_dmd_kP = lm(@formula(constant ~ kP), df_dmd)
    println("Model with kP: p-value = $(round(coeftable(model_dmd_kP).cols[4][2], digits=5))")
    
    # FSP error
    println("\nFSP Error Constants:")
    model_fsp_K = lm(@formula(constant ~ K_eq), df_fsp)
    println("Model with K_eq: p-value = $(round(coeftable(model_fsp_K).cols[4][2], digits=5))")
    
    model_fsp_kB = lm(@formula(constant ~ kB), df_fsp)
    println("Model with kB: p-value = $(round(coeftable(model_fsp_kB).cols[4][2], digits=5))")
    
    model_fsp_kD = lm(@formula(constant ~ kD), df_fsp)
    println("Model with kD: p-value = $(round(coeftable(model_fsp_kD).cols[4][2], digits=5))")
    
    model_fsp_kP = lm(@formula(constant ~ kP), df_fsp)
    println("Model with kP: p-value = $(round(coeftable(model_fsp_kP).cols[4][2], digits=5))")
    
    # Plot relationships
    p1 = plot(df_hist.K_eq, df_hist.constant,
              marker=:circle, markersize=6, label="Data",
              xlabel="Equilibrium Constant (K_eq)", 
              ylabel="Constant C",
              title="Histogram Error Constant vs K_eq",
              legend=:topright, dpi=300)
    
    # Add fitted line if significant
    if coeftable(model_hist_K).cols[4][2] < 0.05
        intercept, slope = coef(model_hist_K)
        fitted_line = intercept .+ slope .* df_hist.K_eq
        plot!(p1, df_hist.K_eq, fitted_line,
              line=:dash, linewidth=2, label="Fitted Line")
    end
    
    p2 = plot(df_dmd.K_eq, df_dmd.constant,
              marker=:circle, markersize=6, label="Data",
              xlabel="Equilibrium Constant (K_eq)", 
              ylabel="Constant C",
              title="DMD Error Constant vs K_eq",
              legend=:topright, dpi=300)
    
    if coeftable(model_dmd_K).cols[4][2] < 0.05
        intercept, slope = coef(model_dmd_K)
        fitted_line = intercept .+ slope .* df_dmd.K_eq
        plot!(p2, df_dmd.K_eq, fitted_line,
              line=:dash, linewidth=2, label="Fitted Line")
    end
    
    p3 = plot(df_fsp.K_eq, df_fsp.constant,
              marker=:circle, markersize=6, label="Data",
              xlabel="Equilibrium Constant (K_eq)", 
              ylabel="Constant C",
              title="FSP Error Constant vs K_eq",
              legend=:topright, dpi=300)
    
    if coeftable(model_fsp_K).cols[4][2] < 0.05
        intercept, slope = coef(model_fsp_K)
        fitted_line = intercept .+ slope .* df_fsp.K_eq
        plot!(p3, df_fsp.K_eq, fitted_line,
              line=:dash, linewidth=2, label="Fitted Line")
    end
    
    savefig(p1, "figures/hist_constant_vs_keq.png")
    savefig(p2, "figures/dmd_constant_vs_keq.png")
    savefig(p3, "figures/fsp_constant_vs_keq.png")
    
    # Combined visualization
    p4 = plot(layout=(1,3), size=(1200, 400), dpi=300)
    
    # Normalize constants to [0,1] for comparison
    function normalize(x)
        return (x .- minimum(x)) ./ (maximum(x) - minimum(x))
    end
    
    norm_hist = normalize(df_hist.constant)
    norm_dmd = normalize(df_dmd.constant)
    norm_fsp = normalize(df_fsp.constant)
    
    plot!(p4[1], df_hist.K_eq, norm_hist,
          marker=:circle, markersize=4, label="",
          xlabel="K_eq", 
          ylabel="Normalized Constant",
          title="(a) Histogram Error")
    
    plot!(p4[2], df_dmd.K_eq, norm_dmd,
          marker=:circle, markersize=4, label="",
          xlabel="K_eq", 
          ylabel="",
          title="(b) DMD Error")
    
    plot!(p4[3], df_fsp.K_eq, norm_fsp,
          marker=:circle, markersize=4, label="",
          xlabel="K_eq", 
          ylabel="",
          title="(c) FSP Error")
    
    savefig(p4, "figures/combined_constants_vs_keq.png")
    
    return Dict(
        "parameter_sets" => parameter_sets,
        "K_eq" => K_eq,
        "state_sizes" => state_sizes,
        "histogram_errors" => histogram_errors,
        "dmd_errors" => dmd_errors,
        "fsp_errors" => fsp_errors,
        "df_hist" => df_hist,
        "df_dmd" => df_dmd,
        "df_fsp" => df_fsp,
        "plots" => (p1, p2, p3, p4)
    )
end

# Main function combining all approaches
function refined_theoretical_bounds()
    # Create directory for results
    mkpath("figures")
    
    # Set plotting defaults
    default(fontfamily="Computer Modern", framestyle=:box, grid=false, size=(600,400))
    
    println("===== APPROACH 1: Fitting the Exponents in Theoretical Bounds =====")
    println("\n-- Histogram Error Bound --")
    hist_results = fit_histogram_error_exponents(define_mm_system(), 50.0, 0.5)
    
    println("\n-- DMD Error Bound --")
    dmd_results = fit_dmd_error_exponents(define_mm_system(), 50.0, 0.5)
    
    println("\n-- FSP Error Bound --")
    fsp_results = fit_fsp_error_exponents(define_mm_system(), 50.0, 0.5)
    
    println("\n===== APPROACH 2: Non-Asymptotic Bounds =====")
    non_asymptotic_results = derive_non_asymptotic_bounds(define_mm_system(), 50.0, 0.5)
    
    println("\n===== APPROACH 3: System-Specific Constants =====")
    system_specific_results = explore_system_specific_constants(50.0, 0.5)
    
    println("\n===== Summary of Findings =====")
    println("\nHistogram Error Bound:")
    println("  Original: O(|X_J|/√N)")
    println("  Fitted: O(|X_J|^$(round(hist_results["fitted_exponent_alpha"], digits=2))/N^$(round(hist_results["fitted_exponent_beta"], digits=2)))")
    
    println("\nDMD Error Bound:")
    if haskey(dmd_results, "fitted_exponent_gamma")
        println("  Original: O(1/r) + O(σ_{r+1})")
        println("  Fitted: O(1/r^$(round(dmd_results["fitted_exponent_gamma"], digits=2))) + O(σ_{r+1}^$(round(dmd_results["fitted_exponent_rho"], digits=2)))")
    end
    
    println("\nFSP Error Bound:")
    println("  Original: O(1/|X_J|)")
    println("  Fitted: O(1/|X_J|^$(round(fsp_results["fitted_exponent_alpha"], digits=2)))")
    
    # Create summary plot
    p_summary = plot(layout=(2,2), size=(900,800), dpi=300)
    
    # Add histogram error plot
    plot!(p_summary[1], hist_results["N_traj"], hist_results["errors_N"], 
          xscale=:log10, yscale=:log10,
          marker=:circle, markersize=4, label="Data",
          xlabel="Number of Trajectories (N)", 
          ylabel="Error ‖p̂-p‖₁",
          title="(a) Histogram Error vs N",
          legend=:topright)
    
    plot!(p_summary[1], hist_results["N_traj"], hist_results["fitted_curve_N"], 
          line=:dash, linewidth=2, label="Fitted")
    
    plot!(p_summary[1], hist_results["N_traj"], hist_results["original_curve_N"], 
          line=:dot, linewidth=2, label="Original")
    
    # Add FSP error plot
    plot!(p_summary[2], fsp_results["state_space_sizes"], fsp_results["probability_leakages"], 
          xscale=:log10, yscale=:log10,
          marker=:circle, markersize=4, label="Data",
          xlabel="State Space Size |X_J|", 
          ylabel="Error ε_J",
          title="(b) FSP Error vs |X_J|",
          legend=:topright)
    
    plot!(p_summary[2], fsp_results["state_space_sizes"], fsp_results["fitted_curve"], 
          line=:dash, linewidth=2, label="Fitted")
    
    plot!(p_summary[2], fsp_results["state_space_sizes"], fsp_results["original_curve"], 
          line=:dot, linewidth=2, label="Original")
    
    # Add DMD error plot if available
    if haskey(dmd_results, "plots")
        plot!(p_summary[3], dmd_results["ranks"], dmd_results["dmd_errors"], 
              xscale=:log10, yscale=:log10,
              marker=:circle, markersize=4, label="Data",
              xlabel="DMD Rank (r)", 
              ylabel="Error ‖K_r-K‖",
              title="(c) DMD Error vs Rank",
              legend=:topright)
        
        plot!(p_summary[3], dmd_results["ranks"], dmd_results["fitted_curve_rank"], 
              line=:dash, linewidth=2, label="Fitted")
        
        plot!(p_summary[3], dmd_results["ranks"], dmd_results["original_curve_rank"], 
              line=:dot, linewidth=2, label="Original")
    end
    
    # Add non-asymptotic bounds plot
    plot!(p_summary[4], non_asymptotic_results["n_trajectories"], non_asymptotic_results["actual_errors"], 
          xscale=:log10, yscale=:log10,
          marker=:circle, markersize=4, label="Actual",
          xlabel="Number of Trajectories (N)", 
          ylabel="Error ‖p̂-p‖₁",
          title="(d) Non-Asymptotic Bounds",
          legend=:topright)
    
    plot!(p_summary[4], non_asymptotic_results["n_trajectories"], non_asymptotic_results["bernstein_bounds"], 
          line=:dash, linewidth=2, label="Bernstein")
    
    plot!(p_summary[4], non_asymptotic_results["n_trajectories"], non_asymptotic_results["hoeffding_bounds"], 
          line=:dot, linewidth=2, label="Hoeffding")
    
    plot!(p_summary[4], non_asymptotic_results["n_trajectories"], non_asymptotic_results["asymptotic_bounds"], 
          line=(:dashdot, :black), linewidth=2, label="Original")
    
    savefig(p_summary, "figures/refined_bounds_summary.png")
    
    return Dict(
        "histogram_results" => hist_results,
        "dmd_results" => dmd_results,
        "fsp_results" => fsp_results,
        "non_asymptotic_results" => non_asymptotic_results,
        "system_specific_results" => system_specific_results,
        "summary_plot" => p_summary
    )
end

# Run the refined analysis
function main()
    println("===== Refined Theoretical Bounds Analysis =====")
    results = refined_theoretical_bounds()
    println("\nAnalysis complete. All figures saved to the 'figures' directory.")
    return results
end

# Execute the main function
main()
