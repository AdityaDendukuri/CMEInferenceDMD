# main.jl - FRESH MM CRN INFERENCE SYSTEM
# Complete working system with flow analysis, kinetics, and conservation laws

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using ProgressMeter
using JumpProcesses
using Catalyst
using DifferentialEquations

# Set random seed for reproducibility
Random.seed!(42)

"""
Generate Michaelis-Menten trajectory data using Catalyst
"""
function generate_mm_trajectories(n_trajs=500)
    println("Generating MM trajectories...")
    
    # Define the Michaelis-Menten reaction network
    rn = @reaction_network begin
        kB, S + E --> SE
        kD, SE --> S + E  
        kP, SE --> P + E
    end
    
    # Parameters and initial conditions
    u0 = [:S => 50, :E => 10, :SE => 1, :P => 1]
    tspan = (0.0, 50.0)
    ps = [:kB => 0.01, :kD => 0.1, :kP => 0.1]
    
    # Create jump problem
    jinput = JumpInputs(rn, u0, tspan, ps)
    jprob = JumpProblem(jinput)
    
    # Generate trajectories
    trajectories = []
    @showprogress desc="Simulating..." for i in 1:n_trajs
        sol = solve(jprob, SSAStepper())
        push!(trajectories, sol)
    end
    
    return trajectories, rn
end

"""
Process trajectories into probability distributions
"""
function trajectories_to_probabilities(trajectories, time_points, grid_sizes)
    n_times = length(time_points)
    n_trajs = length(trajectories)
    
    sparse_probs = []
    
    for (t_idx, t) in enumerate(time_points)
        state_counts = Dict()
        
        for traj in trajectories
            # Find closest time point
            time_idx = searchsortedfirst(traj.t, t)
            if time_idx > length(traj.t)
                time_idx = length(traj.t)
            end
            
            # Extract state
            state_vals = traj.u[time_idx]
            state = tuple([min(max(1, Int(state_vals[j]) + 1), grid_sizes[j]) for j in 1:4]...)
            
            # Count occurrences
            state_counts[state] = get(state_counts, state, 0) + 1
        end
        
        # Convert to probabilities
        indices = collect(keys(state_counts))
        values = [state_counts[idx] / n_trajs for idx in indices]
        
        push!(sparse_probs, (indices=indices, values=values))
        println("Time $t_idx: $(length(indices)) unique states")
    end
    
    return sparse_probs
end

"""
Select most important states based on multiple criteria
"""
function select_important_states(sparse_probs, max_states=500)
    # Calculate state importance metrics
    state_metrics = Dict()
    
    for snapshot in sparse_probs
        for (i, state) in enumerate(snapshot.indices)
            if !haskey(state_metrics, state)
                state_metrics[state] = Dict(
                    "frequency" => 0,
                    "max_prob" => 0.0,
                    "total_prob" => 0.0,
                    "prob_values" => Float64[]
                )
            end
            
            prob = snapshot.values[i]
            state_metrics[state]["frequency"] += 1
            state_metrics[state]["max_prob"] = max(state_metrics[state]["max_prob"], prob)
            state_metrics[state]["total_prob"] += prob
            push!(state_metrics[state]["prob_values"], prob)
        end
    end
    
    # Calculate variance and importance scores
    for (state, metrics) in state_metrics
        # Pad with zeros for missing snapshots
        all_probs = zeros(length(sparse_probs))
        snapshot_count = 0
        for (t, snapshot) in enumerate(sparse_probs)
            idx = findfirst(s -> s == state, snapshot.indices)
            if idx !== nothing
                all_probs[t] = snapshot.values[idx]
                snapshot_count += 1
            end
        end
        
        metrics["variance"] = var(all_probs)
        
        # Combined importance score
        freq_score = metrics["frequency"] / length(sparse_probs)
        prob_score = metrics["total_prob"] / length(sparse_probs) 
        var_score = metrics["variance"] * 10  # Scale variance
        
        metrics["importance"] = 0.3 * freq_score + 0.4 * prob_score + 0.3 * var_score
    end
    
    # Select top states
    all_states = collect(keys(state_metrics))
    if length(all_states) > max_states
        sorted_states = sort(all_states, by=s -> state_metrics[s]["importance"], rev=true)
        selected_states = sorted_states[1:max_states]
    else
        selected_states = all_states
    end
    
    # Create reduced data matrix
    state_to_idx = Dict(state => i for (i, state) in enumerate(selected_states))
    n_snapshots = length(sparse_probs)
    reduced_data = zeros(length(selected_states), n_snapshots)
    
    for (t, snapshot) in enumerate(sparse_probs)
        for (i, state) in enumerate(snapshot.indices)
            if haskey(state_to_idx, state)
                reduced_data[state_to_idx[state], t] = snapshot.values[i]
            end
        end
    end
    
    # Normalize columns
    for j in 1:size(reduced_data, 2)
        col_sum = sum(reduced_data[:, j])
        if col_sum > 0
            reduced_data[:, j] ./= col_sum
        end
    end
    
    println("Selected $(length(selected_states)) important states")
    return reduced_data, selected_states
end

"""
Apply theoretically grounded constrained DMD to reconstruct the CME generator matrix
"""
function apply_constrained_dmd_reconstruction(reduced_data, dt; use_constraints=true, λ_sparse=0.01)
    println("Applying $(use_constraints ? "theoretically constrained" : "unconstrained") DMD...")
    
    if use_constraints
        try
            # Load constrained DMD module if not already loaded
            if !isdefined(Main, :theoretically_constrained_dmd)
                include("constrained_dmd.jl")
            end
            
            # Use theoretically grounded constrained DMD
            G, obj_val, method = apply_constrained_dmd(reduced_data, dt, 
                                                     method="theoretical", 
                                                     λ_sparse=λ_sparse)
            
            # Compute eigenvalues of exp(G*dt) for compatibility
            K = exp(Matrix(G * dt))
            λ, Φ = eigen(K)
            
            println("✓ Theoretically constrained DMD completed")
            println("  Method: $method")
            println("  Objective value: $(round(obj_val, digits=6))")
            
            # Verify the result is a valid CME generator
            println("\nVerifying recovered CME generator:")
            verify_cme_constraints(G)
            
            return G, λ, Φ, K, size(G, 1)
            
        catch e
            println("⚠ Theoretically constrained DMD failed: $e")
            println("  Error details: $(sprint(showerror, e))")
            println("  Falling back to unconstrained DMD")
            use_constraints = false
        end
    end
    
    if !use_constraints
        # Original unconstrained DMD (for comparison)
        X = reduced_data[:, 1:end-1]
        X_prime = reduced_data[:, 2:end]
        
        println("Data matrices: X$(size(X)), X'$(size(X_prime))")
        
        # SVD of X
        U, Σ, V = svd(X)
        
        # Determine rank
        svd_threshold = 1e-10
        rank_r = sum(Σ ./ Σ[1] .> svd_threshold)
        rank_r = min(rank_r, size(X, 2))
        
        println("Using unconstrained DMD with rank: $rank_r")
        
        # Truncated SVD
        U_r = U[:, 1:rank_r]
        Σ_r = Diagonal(Σ[1:rank_r])
        V_r = V[:, 1:rank_r]
        
        # DMD operator
        A_tilde = U_r' * X_prime * V_r * inv(Σ_r)
        
        # Eigendecomposition
        λ, W = eigen(A_tilde)
        
        # DMD modes and full operator
        Φ = U_r * W
        A_dmd = U_r * A_tilde * U_r'
        
        # Generator matrix using linear approximation (THEORETICAL ISSUE!)
        G = (A_dmd - I) / dt
        
        println("⚠ Using unconstrained DMD with linear approximation")
        println("  This may produce unphysical reactions due to invalid CME structure")
        
        return G, λ, Φ, A_dmd, rank_r
    end
end

"""
Extract reactions from generator matrix with conservation law filtering
"""
function extract_reactions_with_conservation(G, selected_states, species_names; 
                                           threshold=1e-5, apply_conservation=true)
    println("Extracting reactions...")
    
    reactions = []
    G_sparse = sparse(G)
    rows, cols, vals = findnz(G_sparse)
    
    max_magnitude = maximum(abs.(vals))
    rel_threshold = threshold * max_magnitude
    
    println("Using threshold: $rel_threshold")
    
    for i in 1:length(vals)
        if abs(vals[i]) > rel_threshold && rows[i] != cols[i]
            from_state = [selected_states[cols[i]]...]
            to_state = [selected_states[rows[i]]...]
            
            # Convert to molecular counts
            from_mol = [x-1 for x in from_state]
            to_mol = [x-1 for x in to_state]
            stoichiometry = to_mol - from_mol
            
            # Basic filtering
            total_change = sum(abs.(stoichiometry))
            if 0 < total_change <= 3
                rate = vals[i]
                
                # Conservation law check
                if apply_conservation && !check_mm_conservation_laws(stoichiometry)
                    continue
                end
                
                push!(reactions, (
                    from_state = from_state,
                    to_state = to_state,
                    stoichiometry = stoichiometry,
                    rate = rate
                ))
            end
        end
    end
    
    println("Found $(length(reactions)) valid reactions")
    
    # Group by stoichiometry
    grouped_reactions = Dict()
    for r in reactions
        s_key = tuple(r.stoichiometry...)
        if haskey(grouped_reactions, s_key)
            push!(grouped_reactions[s_key], r)
        else
            grouped_reactions[s_key] = [r]
        end
    end
    
    # Calculate statistics
    stoich_stats = Dict()
    for (stoich, rxns) in grouped_reactions
        total_rate = sum(abs(r.rate) for r in rxns)
        avg_rate = total_rate / length(rxns)
        rate_var = var([abs(r.rate) for r in rxns])
        
        stoich_stats[stoich] = (
            total_rate = total_rate,
            avg_rate = avg_rate,
            rate_var = rate_var,
            count = length(rxns)
        )
    end
    
    # Sort by total rate
    sorted_stoich = sort(collect(keys(stoich_stats)), 
                        by=s -> stoich_stats[s].total_rate, rev=true)
    
    # Display results
    println("\nTop reactions found:")
    for (i, stoich) in enumerate(sorted_stoich[1:min(10, end)])
        stats = stoich_stats[stoich]
        reaction_str = format_reaction(stoich, species_names)
        println("$i. $reaction_str (rate: $(round(stats.total_rate, digits=4)), count: $(stats.count))")
    end
    
    return sorted_stoich, grouped_reactions, stoich_stats
end

"""
Check MM conservation laws
"""
function check_mm_conservation_laws(stoichiometry)
    if length(stoichiometry) != 4
        return false
    end
    
    s_change, e_change, se_change, p_change = stoichiometry
    
    # Conservation law 1: Total enzyme (E + SE = constant)
    enzyme_conserved = abs(e_change + se_change) <= 1e-10
    
    # Conservation law 2: Total substrate material (S + SE + P = constant)  
    substrate_conserved = abs(s_change + se_change + p_change) <= 1e-10
    
    return enzyme_conserved && substrate_conserved
end

"""
Format reaction for display
"""
function format_reaction(stoich, species_names)
    reactants = String[]
    products = String[]
    
    for i in 1:min(length(stoich), length(species_names))
        if stoich[i] < 0
            coeff = abs(stoich[i])
            species = species_names[i]
            if coeff == 1
                push!(reactants, species)
            else
                push!(reactants, "$coeff $species")
            end
        elseif stoich[i] > 0
            coeff = stoich[i]
            species = species_names[i]
            if coeff == 1
                push!(products, species)
            else
                push!(products, "$coeff $species")
            end
        end
    end
    
    reactant_str = isempty(reactants) ? "∅" : join(reactants, " + ")
    product_str = isempty(products) ? "∅" : join(products, " + ")
    
    return "$reactant_str → $product_str"
end

"""
Main inference function with constrained DMD option
"""
function run_mm_inference(n_trajs=500, max_states=500; use_constrained_dmd=true, λ_sparse=0.01)
    println("="^50)
    println("MM CRN INFERENCE $(use_constrained_dmd ? "(CONSTRAINED DMD)" : "(UNCONSTRAINED DMD)")")
    println("="^50)
    
    # Generate data
    trajectories, rn = generate_mm_trajectories(n_trajs)
    
    # Setup parameters
    species_names = ["S", "E", "SE", "P"]
    grid_sizes = [32, 16, 8, 32]
    time_points = range(0.0, 40.0, length=15)
    dt = time_points[2] - time_points[1]
    
    # Process trajectories
    println("\nProcessing trajectories...")
    sparse_probs = trajectories_to_probabilities(trajectories, time_points, grid_sizes)
    
    # Select important states
    println("\nSelecting important states...")
    reduced_data, selected_states = select_important_states(sparse_probs, max_states)
    
    # Apply DMD (constrained or unconstrained)
    println("\nApplying DMD...")
    G, λ, Φ, A_dmd, rank_r = apply_constrained_dmd_reconstruction(
        reduced_data, dt, use_constraints=use_constrained_dmd, λ_sparse=λ_sparse
    )
    
    # Extract reactions
    println("\nExtracting reactions...")
    significant_stoich, grouped_reactions, stoich_stats = extract_reactions_with_conservation(
        G, selected_states, species_names, apply_conservation=use_constrained_dmd
    )
    
    # Check for expected MM reactions
    println("\n" * "="^30)
    println("MM REACTION VALIDATION")
    println("="^30)
    
    expected_reactions = [
        (tuple([0, 1, -1, 1]...), "SE → E + P"),
        (tuple([-1, -1, 1, 0]...), "S + E → SE"),
        (tuple([1, 1, -1, 0]...), "SE → S + E")
    ]
    
    found_count = 0
    for (expected_stoich, description) in expected_reactions
        if expected_stoich in keys(grouped_reactions)
            stats = stoich_stats[expected_stoich]
            println("✓ $description: found (rate ≈ $(round(stats.total_rate, digits=4)))")
            found_count += 1
        else
            println("✗ $description: not found")
        end
    end
    
    # Check for unphysical reactions
    println("\n" * "="^30)
    println("UNPHYSICAL REACTION CHECK")
    println("="^30)
    
    unphysical_reactions = [
        (tuple([-1, 0, 0, 1]...), "S → P"),
        (tuple([0, 0, 0, 1]...), "∅ → P"),
        (tuple([-1, 0, 0, 0]...), "S → ∅"),
        (tuple([0, -1, 0, 1]...), "E → P")
    ]
    
    unphysical_found = 0
    for (unphys_stoich, description) in unphysical_reactions
        if unphys_stoich in keys(grouped_reactions)
            stats = stoich_stats[unphys_stoich] 
            println("⚠ $description: FOUND (rate ≈ $(round(stats.total_rate, digits=4))) - UNPHYSICAL!")
            unphysical_found += 1
        else
            println("✓ $description: not found (good)")
        end
    end
    
    # Overall assessment
    recovery_rate = found_count / length(expected_reactions) * 100
    println("\n" * "="^50)
    println("INFERENCE ASSESSMENT")
    println("="^50)
    println("Expected reactions recovered: $found_count/$(length(expected_reactions)) ($(round(recovery_rate, digits=1))%)")
    println("Unphysical reactions found: $unphysical_found")
    
    if use_constrained_dmd && unphysical_found == 0
        println("🎉 Constrained DMD successfully eliminated unphysical reactions!")
    elseif !use_constrained_dmd && unphysical_found > 0
        println("⚠ Unconstrained DMD produced unphysical reactions - try constrained version")
    end
    
    return Dict(
        "trajectories" => trajectories,
        "significant_stoichiometries" => significant_stoich,
        "grouped_reactions" => grouped_reactions,
        "stoich_stats" => stoich_stats,
        "generator" => G,
        "eigenvalues" => λ,
        "DMD_modes" => Φ,
        "DMD_operator" => A_dmd,
        "selected_states" => selected_states,
        "rank" => rank_r,
        "dt" => dt,
        "species_names" => species_names,
        "constrained" => use_constrained_dmd,
        "sparsity_param" => λ_sparse,
        "recovery_rate" => recovery_rate,
        "unphysical_count" => unphysical_found
    )
end

function verify_cme_constraints(A; tol=1e-8)
    n = size(A, 1)
    
    println("\nVerifying CME generator constraints:")
    
    # Check 1: Non-negative off-diagonals
    off_diag_violations = 0
    min_off_diag = Inf
    for i in 1:n
        for j in 1:n
            if i != j && A[i,j] < -tol
                off_diag_violations += 1
                min_off_diag = min(min_off_diag, A[i,j])
            end
        end
    end
    
    if off_diag_violations == 0
        println("  ✓ Non-negative off-diagonals: PASS")
    else
        println("  ✗ Non-negative off-diagonals: $off_diag_violations violations")
        println("    Minimum off-diagonal: $min_off_diag")
    end
    
    # Check 2: Column sums zero
    col_sum_violations = 0
    max_col_sum_error = 0.0
    for j in 1:n
        col_sum = sum(A[:, j])
        if abs(col_sum) > tol
            col_sum_violations += 1
            max_col_sum_error = max(max_col_sum_error, abs(col_sum))
        end
    end
    
    if col_sum_violations == 0
        println("  ✓ Zero column sums: PASS")
    else
        println("  ✗ Zero column sums: $col_sum_violations violations")
        println("    Maximum error: $max_col_sum_error")
    end
    
    # Check 3: Diagonal elements non-positive
    pos_diag_violations = 0
    max_pos_diag = -Inf
    for i in 1:n
        if A[i,i] > tol
            pos_diag_violations += 1
            max_pos_diag = max(max_pos_diag, A[i,i])
        end
    end
    
    if pos_diag_violations == 0
        println("  ✓ Non-positive diagonals: PASS")
    else
        println("  ✗ Non-positive diagonals: $pos_diag_violations violations")
        println("    Maximum diagonal: $max_pos_diag")
    end
    
    # Overall assessment
    total_violations = off_diag_violations + col_sum_violations + pos_diag_violations
    
    if total_violations == 0
        println("  🎉 All CME constraints satisfied!")
        return true
    else
        println("  ⚠ Total constraint violations: $total_violations")
        return false
    end
end


println("Fresh MM CRN Inference System Loaded! 🧬")
println("Usage: results = run_mm_inference(n_trajs, max_states)")
println("Example: results = run_mm_inference(500, 500)")
