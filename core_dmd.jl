# core_dmd.jl - SYSTEM-AGNOSTIC DMD ANALYSIS
# Clean separation of algorithm from application

using LinearAlgebra
using SparseArrays
using Statistics

"""
    create_time_segments(total_time_points, segment_length=8, overlap_fraction=0.3)

Create overlapping time segments for multigrid DMD.
"""
function create_time_segments(total_time_points, segment_length=8, overlap_fraction=0.3)
    overlap_points = round(Int, segment_length * overlap_fraction)
    stride = segment_length - overlap_points
    
    segments = []
    start_idx = 1
    
    while start_idx <= total_time_points - segment_length + 1
        end_idx = min(start_idx + segment_length - 1, total_time_points)
        push!(segments, (start_idx, end_idx))
        
        start_idx += stride
        
        if start_idx > total_time_points - segment_length + 1 && end_idx < total_time_points
            push!(segments, (total_time_points - segment_length + 1, total_time_points))
            break
        end
    end
    
    return segments
end

"""
    apply_local_dmd(segment_data, dt; regularization=0.02)

Apply DMD to a single time segment.
"""
function apply_local_dmd(segment_data, dt; regularization=0.02)
    n_states, n_times = size(segment_data)
    
    if n_times < 3
        return nothing, Inf, []
    end
    
    # Form snapshot matrices
    X = segment_data[:, 1:end-1]
    X_prime = segment_data[:, 2:end]
    
    # SVD with conservative rank selection
    U, Σ, V = svd(X)
    
    max_rank = min(min(size(X)...) - 1, 8)
    sig_threshold = 0.01 * Σ[1]
    r = min(sum(Σ .> sig_threshold), max_rank)
    
    if r < 2
        return nothing, Inf, []
    end
    
    U_r = U[:, 1:r]
    Σ_r = Diagonal(Σ[1:r])
    V_r = V[:, 1:r]
    
    # DMD operator in reduced space
    A_tilde = U_r' * X_prime * V_r * inv(Σ_r)
    
    # Project back to full space
    A_full = U_r * A_tilde * U_r'
    
    # Apply generator constraints
    G_local = apply_generator_constraints((A_full - I) / dt, regularization)
    
    # DEBUG: Check DMD reconstruction at segment level
    if rand() < 0.3  # Debug ~30% of segments to avoid spam
        println("    🔧 SEGMENT DEBUG:")
        println("      Segment data shape: $(size(segment_data))")
        println("      Max |A_full| entry: $(maximum(abs.(A_full)))")
        println("      Max |A_full - I| entry: $(maximum(abs.(A_full - I)))")
        println("      dt used: $dt")
        println("      Max |G_local| entry: $(maximum(abs.(G_local)))")
        println("      Theoretical rate scale: ~dt=$(dt) suggests rates ~$(1/dt)")
    end
    
    # Compute reconstruction error
    K_constrained = I + G_local * dt
    prediction_error = norm(X_prime - K_constrained * X)
    
    return G_local, prediction_error, []
end

"""
    apply_generator_constraints(G_raw, reg_strength)

Apply basic CME generator constraints.
"""
function apply_generator_constraints(G_raw, reg_strength)
    n_states = size(G_raw, 1)
    G_constrained = copy(G_raw)
    
    for iter in 1:20
        G_old = copy(G_constrained)
        
        # Sparsity threshold
        threshold = reg_strength * maximum(abs.(G_constrained))
        for i in 1:n_states, j in 1:n_states
            if i != j && abs(G_constrained[i,j]) < threshold
                G_constrained[i,j] = 0
            end
        end
        
        # Non-negative off-diagonals
        for i in 1:n_states, j in 1:n_states
            if i != j
                G_constrained[i,j] = max(0, G_constrained[i,j])
            end
        end
        
        # Zero column sums (probability conservation)
        for j in 1:n_states
            off_diag_sum = sum(G_constrained[i,j] for i in 1:n_states if i != j)
            G_constrained[j,j] = -off_diag_sum
        end
        
        if norm(G_constrained - G_old) < 1e-8
            break
        end
    end
    
    return G_constrained
end

"""
    compute_reachability_matrix(trajectories, selected_states; 
                               min_observations=2, confidence_threshold=0.8)

Compute elementary reachability matrix from trajectory data.
"""
function compute_reachability_matrix(trajectories, selected_states; 
                                   min_observations=2, confidence_threshold=0.8)
    n_states = length(selected_states)
    state_to_idx = Dict(Tuple(state) => i for (i, state) in enumerate(selected_states))
    
    # Track transition counts
    transition_counts = zeros(Int, n_states, n_states)
    total_observations = zeros(Int, n_states)
    
    elementary_steps = 0
    captured_steps = 0
    
    for traj in trajectories
        n_species = length(traj.u[1])
        
        for i in 1:(length(traj.t)-1)
            state1 = [traj.u[i][j] for j in 1:n_species]
            state2 = [traj.u[i+1][j] for j in 1:n_species]
            
            elementary_steps += 1
            
            state1_tuple = Tuple(state1)
            state2_tuple = Tuple(state2)
            
            if haskey(state_to_idx, state1_tuple) && haskey(state_to_idx, state2_tuple)
                idx1 = state_to_idx[state1_tuple]
                idx2 = state_to_idx[state2_tuple]
                
                transition_counts[idx2, idx1] += 1
                total_observations[idx1] += 1
                captured_steps += 1
            end
        end
    end
    
    # Create confidence-weighted reachability matrix
    R = zeros(n_states, n_states)
    
    for i in 1:n_states, j in 1:n_states
        if i != j
            count = transition_counts[i, j]
            total_from_j = total_observations[j]
            
            if count >= min_observations && total_from_j > 0
                confidence = min(1.0, count / max(1, total_from_j / 10))
                R[i, j] = confidence >= confidence_threshold ? 1.0 : confidence
            elseif count > 0
                R[i, j] = min(0.5, count / max(1, min_observations))
            end
        else
            R[i, j] = 1.0  # Self-transitions always allowed
        end
    end
    
    println("Reachability matrix computed:")
    println("  Elementary steps captured: $captured_steps / $elementary_steps ($(round(captured_steps/elementary_steps*100, digits=2))%)")
    println("  Transition confidence: $(count(R .> 0.8)) high, $(count(0.2 .< R .<= 0.8)) medium, $(count(0 .< R .<= 0.2)) low")
    
    return R, transition_counts, total_observations
end

"""
    apply_reachability_masking(A_DMD, R; masking_strength=1.0)

Apply reachability masking to DMD operator.
"""
function apply_reachability_masking(A_DMD, R; masking_strength=1.0)
    return A_DMD .* (R .^ masking_strength)
end

"""
    run_multigrid_dmd(prob_matrix, dt, selected_states; 
                     segment_length=8, overlap_fraction=0.3,
                     use_reachability=false, reachability_matrix=nothing,
                     masking_strength=1.0)

Run multigrid DMD analysis (system-agnostic).
"""
function run_multigrid_dmd(prob_matrix, dt, selected_states; 
                          segment_length=8, overlap_fraction=0.3,
                          use_reachability=false, reachability_matrix=nothing,
                          masking_strength=1.0)
    
    println("\n" * "="^60)
    println("MULTIGRID DMD ANALYSIS")
    println("="^60)
    
    n_states, total_time_points = size(prob_matrix)
    println("Data: $n_states states × $total_time_points time points")
    
    # Create time segments
    segments = create_time_segments(total_time_points, segment_length, overlap_fraction)
    println("Created $(length(segments)) segments")
    
    # Process each segment
    segment_results = []
    successful_segments = 0
    
    for (seg_idx, (start_t, end_t)) in enumerate(segments)
        segment_data = prob_matrix[:, start_t:end_t]
        
        G_local, error, _ = apply_local_dmd(segment_data, dt)
        
        if G_local !== nothing
            # Apply reachability masking if provided
            if use_reachability && reachability_matrix !== nothing
                G_local = apply_reachability_masking(G_local, reachability_matrix, masking_strength=masking_strength)
                G_local = apply_generator_constraints(G_local, 0.02)  # Re-enforce constraints
            end
            
            successful_segments += 1
            println("  Segment $seg_idx: ✓ (error: $(round(error, digits=4)))")
        else
            println("  Segment $seg_idx: ✗")
        end
        
        push!(segment_results, (G_local, error))
    end
    
    println("Successfully processed $successful_segments/$(length(segments)) segments")
    
    # Combine results
    if successful_segments >= 1
        G_combined = combine_segment_generators(segment_results, n_states)
        λ_combined, Φ_combined = eigen(G_combined)
        
        return G_combined, λ_combined, Φ_combined, successful_segments
    else
        return zeros(n_states, n_states), [], [], 0
    end
end

"""
    combine_segment_generators(segment_results, n_states)

Combine generators by aggregating transition evidence (FIXED - no rate dilution).
"""
function combine_segment_generators(segment_results, n_states)
    # Instead of averaging generator matrices, aggregate transition evidence
    transition_evidence = zeros(n_states, n_states)
    evidence_weights = zeros(n_states, n_states)
    
    for (G_local, error) in segment_results
        if G_local !== nothing
            # Weight based on reconstruction quality
            segment_weight = 1.0 / (1.0 + error)
            
            # Add evidence for each transition (not average)
            for i in 1:n_states, j in 1:n_states
                if abs(G_local[i, j]) > 1e-8
                    transition_evidence[i, j] += segment_weight * abs(G_local[i, j])
                    evidence_weights[i, j] += segment_weight
                end
            end
        end
    end
    
    # Build combined generator from aggregated evidence
    G_combined = zeros(n_states, n_states)
    
    for i in 1:n_states, j in 1:n_states
        if evidence_weights[i, j] > 0 && i != j
            # Use evidence-weighted rate (preserves magnitude)
            G_combined[i, j] = transition_evidence[i, j] / evidence_weights[i, j]
        end
    end
    
    # Enforce generator constraints (probability conservation)
    for j in 1:n_states
        off_diag_sum = sum(G_combined[i, j] for i in 1:n_states if i != j)
        G_combined[j, j] = -off_diag_sum
    end
    
    return G_combined
end

"""
    extract_reactions_from_generator(G, selected_states; threshold=1e-5)

Extract reactions with proper individual transition data.
"""
function extract_reactions_from_generator(G, selected_states; threshold=1e-5)
    # Extract individual transitions first
    individual_transitions = extract_individual_transitions(G, selected_states, threshold=threshold)
    
    # Group by stoichiometry and compute statistics
    sorted_stoich, stoich_stats = group_transitions_by_stoichiometry(individual_transitions)
    
    # Display results
    println("\nTop reactions found:")
    for (i, stoich) in enumerate(sorted_stoich[1:min(10, end)])
        stats = stoich_stats[stoich]
        reaction_str = format_reaction_string(collect(stoich), ["S", "E", "SE", "P"])
        println("$i. $reaction_str (rate: $(round(stats.total_rate, digits=4)), n=$(stats.count))")
    end
    
    return sorted_stoich, stoich_stats
end

"""
    extract_individual_transitions(G, selected_states; threshold=1e-5)

Extract individual transitions from generator matrix with state-rate pairs.
"""
function extract_individual_transitions(G, selected_states; threshold=1e-5)
    println("Extracting individual transitions from generator matrix...")
    
    individual_transitions = []
    n_states = size(G, 1)
    
    for i in 1:n_states, j in 1:n_states
        if i != j && abs(G[i, j]) > threshold
            if i <= length(selected_states) && j <= length(selected_states)
                from_state = selected_states[j]
                to_state = selected_states[i]
                
                # Compute stoichiometry
                stoichiometry = to_state - from_state
                
                # Filter for elementary reactions
                total_change = sum(abs.(stoichiometry))
                if 0 < total_change <= 3
                    push!(individual_transitions, (
                        from_state = from_state,
                        to_state = to_state,
                        stoichiometry = stoichiometry,
                        rate = G[i, j],  # Use actual generator matrix entry (can be negative)
                        from_idx = j,
                        to_idx = i
                    ))
                end
            end
        end
    end
    
    println("Found $(length(individual_transitions)) individual elementary transitions")
    return individual_transitions
end

"""
    group_transitions_by_stoichiometry(individual_transitions)

Group individual transitions by stoichiometry and compute statistics.
"""
function group_transitions_by_stoichiometry(individual_transitions)
    grouped_transitions = Dict()
    stoich_stats = Dict()
    
    for transition in individual_transitions
        s_key = tuple(transition.stoichiometry...)
        
        if haskey(grouped_transitions, s_key)
            push!(grouped_transitions[s_key], transition)
        else
            grouped_transitions[s_key] = [transition]
        end
    end
    
    # Calculate statistics for each stoichiometry
    for (stoich, transitions) in grouped_transitions
        rates = [abs(t.rate) for t in transitions]  # Take absolute value for stats
        
        stoich_stats[stoich] = (
            total_rate = sum(rates),
            avg_rate = mean(rates),
            rate_var = var(rates, corrected=false),
            count = length(transitions),
            transitions = transitions  # Keep individual transitions for kinetics analysis
        )
    end
    
    # Sort by total rate
    sorted_stoich = sort(collect(keys(stoich_stats)), 
                        by=s -> stoich_stats[s].total_rate, rev=true)
    
    return sorted_stoich, stoich_stats
end

"""
    format_reaction_string(stoich, species_names)

Format stoichiometry as readable reaction string.
"""
function format_reaction_string(stoich, species_names)
    reactants = String[]
    products = String[]
    
    for (i, coeff) in enumerate(stoich)
        if i <= length(species_names)
            species = species_names[i]
            if coeff < 0
                abs_coeff = abs(coeff)
                if abs_coeff == 1
                    push!(reactants, species)
                else
                    push!(reactants, "$abs_coeff $species")
                end
            elseif coeff > 0
                if coeff == 1
                    push!(products, species)
                else
                    push!(products, "$coeff $species")
                end
            end
        end
    end
    
    reactant_str = isempty(reactants) ? "∅" : join(reactants, " + ")
    product_str = isempty(products) ? "∅" : join(products, " + ")
    
    return "$reactant_str → $product_str"
end

println("Core DMD Analysis Module Loaded! 🔄")
println("System-agnostic functions:")
println("  run_multigrid_dmd(prob_matrix, dt, states)")
println("  compute_reachability_matrix(trajectories, states)")
println("  extract_reactions_from_generator(G, states)")
println("  apply_reachability_masking(A_DMD, R)")
