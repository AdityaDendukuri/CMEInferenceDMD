# dmd.jl - CLEAN DMD MODULE
# All DMD-related functionality with multigrid as default

using LinearAlgebra
using SparseArrays
using Statistics

"""
    create_dmd_time_segments(total_time_points, segment_length=8, overlap_fraction=0.3)

Create overlapping time segments for multigrid DMD.
"""
function create_dmd_time_segments(total_time_points, segment_length=8, overlap_fraction=0.3)
    overlap_points = round(Int, segment_length * overlap_fraction)
    stride = segment_length - overlap_points
    
    segments = []
    start_idx = 1
    
    while start_idx <= total_time_points - segment_length + 1
        end_idx = min(start_idx + segment_length - 1, total_time_points)
        push!(segments, (start_idx, end_idx))
        
        start_idx += stride
        
        # Ensure we capture the end of the time series
        if start_idx > total_time_points - segment_length + 1 && end_idx < total_time_points
            push!(segments, (total_time_points - segment_length + 1, total_time_points))
            break
        end
    end
    
    return segments
end

"""
    apply_local_dmd(segment_data, dt; regularization_strength=0.02)

Apply DMD to a single time segment with constraint enforcement.
"""
function apply_local_dmd(segment_data, dt; regularization_strength=0.02)
    n_states, n_times = size(segment_data)
    
    if n_times < 3
        return nothing, 0, []
    end
    
    # Form snapshot matrices
    X_local = segment_data[:, 1:end-1]
    X_prime_local = segment_data[:, 2:end]
    
    # SVD with conservative rank selection
    U, Σ, V = svd(X_local)
    
    max_rank = min(min(size(X_local)...) - 1, 8)
    sig_threshold = 0.01 * Σ[1]
    r = min(sum(Σ .> sig_threshold), max_rank)
    
    if r < 2
        return nothing, 0, []
    end
    
    U_r = U[:, 1:r]
    Σ_r = Diagonal(Σ[1:r])
    V_r = V[:, 1:r]
    
    # DMD operator in reduced space
    A_tilde = U_r' * X_prime_local * V_r * inv(Σ_r)
    
    # Project back to full space
    A_full = U_r * A_tilde * U_r'
    
    # Apply light regularization for stability
    G_local = apply_generator_constraints((A_full - I) / dt, regularization_strength)
    
    # Compute reconstruction error
    K_constrained = I + G_local * dt
    prediction_error = norm(X_prime_local - K_constrained * X_local)
    
    # Extract elementary reactions
    local_reactions = extract_local_reactions(G_local, prediction_error)
    
    return G_local, prediction_error, local_reactions
end

"""
    apply_generator_constraints(G_raw, reg_strength)

Apply basic CME generator constraints (non-negative off-diagonals, zero column sums).
"""
function apply_generator_constraints(G_raw, reg_strength)
    n_states = size(G_raw, 1)
    G_constrained = copy(G_raw)
    
    # Iterative constraint enforcement
    for iter in 1:20
        G_old = copy(G_constrained)
        
        # 1. Sparsity: threshold small elements
        threshold = reg_strength * maximum(abs.(G_constrained))
        for i in 1:n_states
            for j in 1:n_states
                if i != j && abs(G_constrained[i,j]) < threshold
                    G_constrained[i,j] = 0
                end
            end
        end
        
        # 2. Non-negative off-diagonals
        for i in 1:n_states
            for j in 1:n_states
                if i != j
                    G_constrained[i,j] = max(0, G_constrained[i,j])
                end
            end
        end
        
        # 3. Zero column sums (probability conservation)
        for j in 1:n_states
            off_diag_sum = sum(G_constrained[i,j] for i in 1:n_states if i != j)
            G_constrained[j,j] = -off_diag_sum
        end
        
        # Check convergence
        if norm(G_constrained - G_old) < 1e-8
            break
        end
    end
    
    return G_constrained
end

"""
    extract_local_reactions(G_local, error; threshold=0.001)

Extract reactions from local generator with confidence scoring.
"""
function extract_local_reactions(G_local, error; threshold=0.001)
    n_states = size(G_local, 1)
    local_reactions = []
    
    # Confidence based on reconstruction error
    confidence = 1.0 / (1.0 + error)
    
    # Find significant transitions
    rate_threshold = threshold * maximum(abs.(G_local))
    
    for i in 1:n_states
        for j in 1:n_states
            if i != j && abs(G_local[i,j]) > rate_threshold
                rate = G_local[i,j]
                push!(local_reactions, (
                    from_state = j,
                    to_state = i,
                    rate = rate,
                    confidence = confidence
                ))
            end
        end
    end
    
    return local_reactions
end

"""
    fuse_segment_reactions(segment_results, selected_states, species_names)

Fuse reactions discovered across multiple time segments.
"""
function fuse_segment_reactions(segment_results, selected_states, species_names)
    println("Fusing reactions across segments...")
    
    # Collect all reactions with confidences
    all_reactions = Dict()
    
    for (segment_idx, (G_local, error, local_reactions)) in enumerate(segment_results)
        if G_local === nothing
            continue
        end
        
        for reaction in local_reactions
            from_idx = reaction.from_state
            to_idx = reaction.to_state
            
            if from_idx <= length(selected_states) && to_idx <= length(selected_states)
                from_state = selected_states[from_idx]
                to_state = selected_states[to_idx]
                
                # Compute stoichiometry
                stoichiometry = to_state - from_state
                
                # Filter elementary reactions
                total_change = sum(abs.(stoichiometry))
                if 0 < total_change <= 3
                    stoich_key = tuple(stoichiometry...)
                    
                    if !haskey(all_reactions, stoich_key)
                        all_reactions[stoich_key] = []
                    end
                    
                    push!(all_reactions[stoich_key], (
                        rate = abs(reaction.rate),
                        confidence = reaction.confidence,
                        segment = segment_idx
                    ))
                end
            end
        end
    end
    
    # Fuse reactions by weighted averaging
    fused_reactions = Dict()
    reaction_stats = Dict()
    
    for (stoich, rate_data) in all_reactions
        if length(rate_data) >= 1  # Accept single-segment reactions
            # Weighted average by confidence
            total_weight = sum(rd.confidence for rd in rate_data)
            weighted_rate = sum(rd.rate * rd.confidence for rd in rate_data) / total_weight
            
            # Consistency score
            rates = [rd.rate for rd in rate_data]
            rate_variance = length(rates) > 1 ? var(rates) : 0.0
            consistency = 1.0 / (1.0 + rate_variance)
            
            # Overall confidence
            overall_confidence = (total_weight / length(rate_data)) * consistency
            
            fused_reactions[stoich] = []
            reaction_stats[stoich] = (
                total_rate = weighted_rate,
                avg_rate = weighted_rate,
                rate_var = rate_variance,
                count = length(rate_data),
                confidence = overall_confidence,
                segments = [rd.segment for rd in rate_data]
            )
        end
    end
    
    # Sort by confidence * rate
    sorted_stoich = sort(collect(keys(reaction_stats)),
                        by=s -> reaction_stats[s].confidence * reaction_stats[s].total_rate,
                        rev=true)
    
    println("Fused $(length(sorted_stoich)) reactions from $(length(all_reactions)) candidates")
    
    return sorted_stoich, fused_reactions, reaction_stats
end

"""
    compute_elementary_reachability_matrix(ssa_trajs, time_points, selected_states; 
                                         min_observations=2, confidence_threshold=0.8)

Compute reachability matrix with safeguards against data limitation.
"""
function compute_elementary_reachability_matrix(ssa_trajs, time_points, selected_states; 
                                              min_observations=2, confidence_threshold=0.8)
    n_states = length(selected_states)
    state_to_idx = Dict(Tuple(state) => i for (i, state) in enumerate(selected_states))
    
    # Track transition counts and confidence
    transition_counts = zeros(Int, n_states, n_states)
    total_observations = zeros(Int, n_states)  # How often each state was observed
    
    elementary_steps = 0
    captured_steps = 0
    
    # Count all elementary transitions
    for traj in ssa_trajs
        for i in 1:(length(traj.t)-1)
            state1 = [traj.u[i][j] for j in 1:4]
            state2 = [traj.u[i+1][j] for j in 1:4]
            
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
    
    for i in 1:n_states
        for j in 1:n_states
            if i != j
                count = transition_counts[i, j]
                total_from_j = total_observations[j]
                
                if count >= min_observations && total_from_j > 0
                    # Confidence based on observation frequency
                    confidence = min(1.0, count / max(1, total_from_j / 10))
                    
                    if confidence >= confidence_threshold
                        R[i, j] = 1.0  # High confidence transition
                    else
                        R[i, j] = confidence  # Partial confidence (soft masking)
                    end
                elseif count > 0
                    # Low confidence but observed - use soft masking
                    R[i, j] = min(0.5, count / max(1, min_observations))
                end
                # If count == 0, R[i,j] remains 0 (hard mask)
            else
                R[i, j] = 1.0  # Always allow self-transitions
            end
        end
    end
    
    println("Elementary reachability computed with confidence weighting:")
    println("  Elementary steps: $captured_steps / $elementary_steps captured")
    
    # Report confidence statistics
    high_conf = count(R .> 0.8)
    med_conf = count(0.2 .< R .<= 0.8)
    low_conf = count(0 .< R .<= 0.2)
    blocked = count(R .== 0)
    
    println("  Transition confidence: $high_conf high, $med_conf medium, $low_conf low, $blocked blocked")
    
    return R, transition_counts, total_observations
end

"""
    apply_confidence_weighted_masking(A_DMD, R, masking_strength=1.0)

Apply confidence-weighted masking instead of hard binary masking.
"""
function apply_confidence_weighted_masking(A_DMD, R, masking_strength=1.0)
    # Soft masking: A_masked[i,j] = A_DMD[i,j] * R[i,j]^masking_strength
    # masking_strength = 1.0: linear confidence weighting
    # masking_strength > 1.0: more aggressive masking of low-confidence transitions
    A_masked = A_DMD .* (R .^ masking_strength)
    
    return A_masked
end

"""
    apply_multigrid_dmd(prob_matrix, dt, selected_states, species_names; 
                       segment_length=8, overlap_fraction=0.3, 
                       use_reachability=true, masking_strength=1.0,
                       min_observations=2, confidence_threshold=0.8,
                       ssa_trajs=nothing, time_points=nothing)

Apply multigrid DMD with optional reachability masking and data-limitation safeguards.
"""
function apply_multigrid_dmd(prob_matrix, dt, selected_states, species_names; 
                            segment_length=8, overlap_fraction=0.3,
                            use_reachability=true, masking_strength=1.0,
                            min_observations=2, confidence_threshold=0.8,
                            ssa_trajs=nothing, time_points=nothing)
    
    println("\n" * "="^60)
    println("MULTIGRID DMD ANALYSIS")
    if use_reachability
        println("WITH CONFIDENCE-WEIGHTED REACHABILITY MASKING")
    end
    println("="^60)
    
    n_states, total_time_points = size(prob_matrix)
    println("Data: $n_states states × $total_time_points time points")
    println("Segment length: $segment_length, overlap: $(overlap_fraction*100)%")
    
    # Compute reachability matrix if requested and trajectory data available
    R = nothing
    if use_reachability && ssa_trajs !== nothing && time_points !== nothing
        println("\n🔍 Computing confidence-weighted reachability matrix...")
        R, transition_counts, obs_counts = compute_elementary_reachability_matrix(
            ssa_trajs, time_points, selected_states,
            min_observations=min_observations, 
            confidence_threshold=confidence_threshold
        )
        
        # Analyze potential data limitations
        analyze_data_limitations(R, transition_counts, obs_counts, species_names)
    elseif use_reachability
        println("⚠️  Reachability masking requested but trajectory data not provided")
        println("   Proceeding without reachability constraints")
        use_reachability = false
    end
    
    # Create time segments
    dmd_segments = create_dmd_time_segments(total_time_points, segment_length, overlap_fraction)
    println("Created $(length(dmd_segments)) DMD segments")
    
    # Process each segment with optional reachability masking
    segment_results = []
    successful_segments = 0
    
    for (seg_idx, (start_t, end_t)) in enumerate(dmd_segments)
        println("\n--- Processing Segment $seg_idx: t[$start_t:$end_t] ---")
        
        segment_data = prob_matrix[:, start_t:end_t]
        
        # Apply local DMD with optional reachability
        G_local, error, local_reactions = apply_local_dmd_with_reachability(
            segment_data, dt, R, use_reachability, masking_strength
        )
        
        if G_local !== nothing
            successful_segments += 1
            println("  ✓ Found $(length(local_reactions)) reactions (error: $(round(error, digits=4)))")
        else
            println("  ✗ Segment processing failed")
        end
        
        push!(segment_results, (G_local, error, local_reactions))
    end
    
    println("\nSuccessfully processed $successful_segments/$(length(dmd_segments)) segments")
    
    # Fuse results across segments
    if successful_segments >= 1
        sorted_stoich, fused_reactions, reaction_stats = fuse_segment_reactions(
            segment_results, selected_states, species_names
        )
        
        # Create combined generator
        G_combined = create_combined_generator(segment_results, n_states)
        
        # Apply reachability to combined generator if used
        if use_reachability && R !== nothing
            println("Applying reachability masking to combined generator...")
            G_combined = apply_confidence_weighted_masking(G_combined, R, masking_strength)
        end
        
        # Compute combined eigenvalues for flow analysis on CLEAN spectrum
        λ_combined, Φ_combined = eigen(G_combined)
        
        # Return additional reachability info for analysis
        reachability_info = use_reachability ? Dict(
            "reachability_matrix" => R,
            "masking_strength" => masking_strength,
            "confidence_threshold" => confidence_threshold
        ) : Dict()
        
        return G_combined, λ_combined, Φ_combined, sorted_stoich, reaction_stats, successful_segments, reachability_info
    else
        println("⚠ No successful segments for analysis")
        return zeros(n_states, n_states), [], [], [], Dict(), 0, Dict()
    end
end

"""
    apply_local_dmd_with_reachability(segment_data, dt, R, use_reachability, masking_strength)

Apply local DMD with optional reachability masking.
"""
function apply_local_dmd_with_reachability(segment_data, dt, R, use_reachability, masking_strength)
    n_states, n_times = size(segment_data)
    
    if n_times < 3
        return nothing, 0, []
    end
    
    # Form snapshot matrices
    X_local = segment_data[:, 1:end-1]
    X_prime_local = segment_data[:, 2:end]
    
    # SVD with conservative rank selection
    U, Σ, V = svd(X_local)
    
    max_rank = min(min(size(X_local)...) - 1, 8)
    sig_threshold = 0.01 * Σ[1]
    r = min(sum(Σ .> sig_threshold), max_rank)
    
    if r < 2
        return nothing, 0, []
    end
    
    U_r = U[:, 1:r]
    Σ_r = Diagonal(Σ[1:r])
    V_r = V[:, 1:r]
    
    # DMD operator in reduced space
    A_tilde = U_r' * X_prime_local * V_r * inv(Σ_r)
    
    # Project back to full space
    A_full = U_r * A_tilde * U_r'
    
    # Apply reachability masking BEFORE constraint enforcement
    if use_reachability && R !== nothing
        A_masked = apply_confidence_weighted_masking(A_full, R, masking_strength)
    else
        A_masked = A_full
    end
    
    # Apply generator constraints to masked operator
    G_local = apply_generator_constraints((A_masked - I) / dt, 0.02)
    
    # Compute reconstruction error
    K_constrained = I + G_local * dt
    prediction_error = norm(X_prime_local - K_constrained * X_local)
    
    # Extract elementary reactions
    local_reactions = extract_local_reactions(G_local, prediction_error)
    
    return G_local, prediction_error, local_reactions
end

"""
    analyze_data_limitations(R, transition_counts, obs_counts, species_names)

Analyze potential data limitations that could mask genuine pathways.
"""
function analyze_data_limitations(R, transition_counts, obs_counts, species_names)
    println("\n📊 Data Limitation Analysis:")
    
    n_states = size(R, 1)
    
    # Find under-sampled states (potential data limitations)
    under_sampled = findall(obs_counts .< 10)  # States observed < 10 times
    
    if !isempty(under_sampled)
        println("  ⚠️  Under-sampled states: $(length(under_sampled))/$(n_states)")
        println("     These states may have missing genuine transitions")
    else
        println("  ✅ All states well-sampled")
    end
    
    # Find completely blocked transitions vs partially masked
    completely_blocked = count(R .== 0) - n_states  # Exclude diagonal
    partially_masked = count(0 .< R .< 1.0)
    
    println("  Transition masking:")
    println("    Completely blocked: $completely_blocked")
    println("    Partially masked: $partially_masked (soft masking)")
    
    # Check for potential MM pathway masking
    check_mm_pathway_masking(R, species_names)
end

"""
    check_mm_pathway_masking(R, species_names)

Check if critical MM pathways might be masked due to data limitations.
"""
function check_mm_pathway_masking(R, species_names)
    if length(species_names) != 4
        return
    end
    
    println("  MM pathway check:")
    
    # Look for any evidence of SE complex dynamics
    se_transitions = 0
    total_transitions = 0
    
    for i in 1:size(R, 1), j in 1:size(R, 2)
        if i != j && R[i, j] > 0
            total_transitions += 1
            # This is a simplified check - in practice would need state mapping
            if R[i, j] > 0.1  # Some confidence in transition
                se_transitions += 1
            end
        end
    end
    
    if total_transitions > 0
        se_fraction = se_transitions / total_transitions
        if se_fraction < 0.1
            println("    ⚠️  Very few confident transitions detected")
            println("       May be masking genuine MM pathways")
        else
            println("    ✅ Reasonable transition coverage for MM analysis")
        end
    end
end

"""
    create_combined_generator(segment_results, n_states)

Create approximate combined generator from segment results.
"""
function create_combined_generator(segment_results, n_states)
    G_combined = zeros(n_states, n_states)
    weight_sum = zeros(n_states, n_states)
    
    for (G_local, error, local_reactions) in segment_results
        if G_local !== nothing
            weight = 1.0 / (1.0 + error)
            G_combined .+= weight * G_local
            weight_sum .+= weight
        end
    end
    
    # Normalize by weights
    for i in 1:n_states, j in 1:n_states
        if weight_sum[i,j] > 0
            G_combined[i,j] /= weight_sum[i,j]
        end
    end
    
    return G_combined
end

# dmd.jl - CLEAN DMD MODULE (FIXED SYNTAX)
# All DMD-related functionality with multigrid as default

using LinearAlgebra
using SparseArrays
using Statistics

"""
    create_dmd_time_segments(total_time_points, segment_length=8, overlap_fraction=0.3)

Create overlapping time segments for multigrid DMD.
"""
function create_dmd_time_segments(total_time_points, segment_length=8, overlap_fraction=0.3)
    overlap_points = round(Int, segment_length * overlap_fraction)
    stride = segment_length - overlap_points
    
    segments = []
    start_idx = 1
    
    while start_idx <= total_time_points - segment_length + 1
        end_idx = min(start_idx + segment_length - 1, total_time_points)
        push!(segments, (start_idx, end_idx))
        
        start_idx += stride
        
        # Ensure we capture the end of the time series
        if start_idx > total_time_points - segment_length + 1 && end_idx < total_time_points
            push!(segments, (total_time_points - segment_length + 1, total_time_points))
            break
        end
    end
    
    return segments
end

"""
    apply_generator_constraints(G_raw, reg_strength)

Apply basic CME generator constraints (non-negative off-diagonals, zero column sums).
"""
function apply_generator_constraints(G_raw, reg_strength)
    n_states = size(G_raw, 1)
    G_constrained = copy(G_raw)
    
    # Iterative constraint enforcement
    for iter in 1:20
        G_old = copy(G_constrained)
        
        # 1. Sparsity: threshold small elements
        threshold = reg_strength * maximum(abs.(G_constrained))
        for i in 1:n_states
            for j in 1:n_states
                if i != j && abs(G_constrained[i,j]) < threshold
                    G_constrained[i,j] = 0
                end
            end
        end
        
        # 2. Non-negative off-diagonals
        for i in 1:n_states
            for j in 1:n_states
                if i != j
                    G_constrained[i,j] = max(0, G_constrained[i,j])
                end
            end
        end
        
        # 3. Zero column sums (probability conservation)
        for j in 1:n_states
            off_diag_sum = sum(G_constrained[i,j] for i in 1:n_states if i != j)
            G_constrained[j,j] = -off_diag_sum
        end
        
        # Check convergence
        if norm(G_constrained - G_old) < 1e-8
            break
        end
    end
    
    return G_constrained
end

"""
    extract_local_reactions(G_local, error; threshold=0.001)

Extract reactions from local generator with confidence scoring.
"""
function extract_local_reactions(G_local, error; threshold=0.001)
    n_states = size(G_local, 1)
    local_reactions = []
    
    # Confidence based on reconstruction error
    confidence = 1.0 / (1.0 + error)
    
    # Find significant transitions
    rate_threshold = threshold * maximum(abs.(G_local))
    
    for i in 1:n_states
        for j in 1:n_states
            if i != j && abs(G_local[i,j]) > rate_threshold
                rate = G_local[i,j]
                push!(local_reactions, (
                    from_state = j,
                    to_state = i,
                    rate = rate,
                    confidence = confidence
                ))
            end
        end
    end
    
    return local_reactions
end

"""
    compute_elementary_reachability_matrix(ssa_trajs, time_points, selected_states; 
                                         min_observations=2, confidence_threshold=0.8)

Compute reachability matrix with safeguards against data limitation.
"""
function compute_elementary_reachability_matrix(ssa_trajs, time_points, selected_states; 
                                              min_observations=2, confidence_threshold=0.8)
    n_states = length(selected_states)
    state_to_idx = Dict(Tuple(state) => i for (i, state) in enumerate(selected_states))
    
    # Track transition counts and confidence
    transition_counts = zeros(Int, n_states, n_states)
    total_observations = zeros(Int, n_states)  # How often each state was observed
    
    elementary_steps = 0
    captured_steps = 0
    
    # Count all elementary transitions
    for traj in ssa_trajs
        for i in 1:(length(traj.t)-1)
            state1 = [traj.u[i][j] for j in 1:4]
            state2 = [traj.u[i+1][j] for j in 1:4]
            
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
    
    for i in 1:n_states
        for j in 1:n_states
            if i != j
                count = transition_counts[i, j]
                total_from_j = total_observations[j]
                
                if count >= min_observations && total_from_j > 0
                    # Confidence based on observation frequency
                    confidence = min(1.0, count / max(1, total_from_j / 10))
                    
                    if confidence >= confidence_threshold
                        R[i, j] = 1.0  # High confidence transition
                    else
                        R[i, j] = confidence  # Partial confidence (soft masking)
                    end
                elseif count > 0
                    # Low confidence but observed - use soft masking
                    R[i, j] = min(0.5, count / max(1, min_observations))
                end
                # If count == 0, R[i,j] remains 0 (hard mask)
            else
                R[i, j] = 1.0  # Always allow self-transitions
            end
        end
    end
    
    println("Elementary reachability computed with confidence weighting:")
    println("  Elementary steps: $captured_steps / $elementary_steps captured")
    
    # Report confidence statistics
    high_conf = count(R .> 0.8)
    med_conf = count(0.2 .< R .<= 0.8)
    low_conf = count(0 .< R .<= 0.2)
    blocked = count(R .== 0)
    
    println("  Transition confidence: $high_conf high, $med_conf medium, $low_conf low, $blocked blocked")
    
    return R, transition_counts, total_observations
end

"""
    apply_confidence_weighted_masking(A_DMD, R, masking_strength=1.0)

Apply confidence-weighted masking instead of hard binary masking.
"""
function apply_confidence_weighted_masking(A_DMD, R, masking_strength=1.0)
    # Soft masking: A_masked[i,j] = A_DMD[i,j] * R[i,j]^masking_strength
    A_masked = A_DMD .* (R .^ masking_strength)
    return A_masked
end

"""
    apply_local_dmd_with_reachability(segment_data, dt, R, use_reachability, masking_strength)

Apply local DMD with optional reachability masking.
"""
function apply_local_dmd_with_reachability(segment_data, dt, R, use_reachability, masking_strength)
    n_states, n_times = size(segment_data)
    
    if n_times < 3
        return nothing, 0, []
    end
    
    # Form snapshot matrices
    X_local = segment_data[:, 1:end-1]
    X_prime_local = segment_data[:, 2:end]
    
    # SVD with conservative rank selection
    U, Σ, V = svd(X_local)
    
    max_rank = min(min(size(X_local)...) - 1, 8)
    sig_threshold = 0.01 * Σ[1]
    r = min(sum(Σ .> sig_threshold), max_rank)
    
    if r < 2
        return nothing, 0, []
    end
    
    U_r = U[:, 1:r]
    Σ_r = Diagonal(Σ[1:r])
    V_r = V[:, 1:r]
    
    # DMD operator in reduced space
    A_tilde = U_r' * X_prime_local * V_r * inv(Σ_r)
    
    # Project back to full space
    A_full = U_r * A_tilde * U_r'
    
    # Apply reachability masking BEFORE constraint enforcement
    if use_reachability && R !== nothing
        A_masked = apply_confidence_weighted_masking(A_full, R, masking_strength)
    else
        A_masked = A_full
    end
    
    # Apply generator constraints to masked operator
    G_local = apply_generator_constraints((A_masked - I) / dt, 0.02)
    
    # Compute reconstruction error
    K_constrained = I + G_local * dt
    prediction_error = norm(X_prime_local - K_constrained * X_local)
    
    # Extract elementary reactions
    local_reactions = extract_local_reactions(G_local, prediction_error)
    
    return G_local, prediction_error, local_reactions
end

"""
    fuse_segment_reactions(segment_results, selected_states, species_names)

Fuse reactions discovered across multiple time segments.
"""
function fuse_segment_reactions(segment_results, selected_states, species_names)
    println("Fusing reactions across segments...")
    
    # Collect all reactions with confidences
    all_reactions = Dict()
    
    for (segment_idx, (G_local, error, local_reactions)) in enumerate(segment_results)
        if G_local === nothing
            continue
        end
        
        for reaction in local_reactions
            from_idx = reaction.from_state
            to_idx = reaction.to_state
            
            if from_idx <= length(selected_states) && to_idx <= length(selected_states)
                from_state = selected_states[from_idx]
                to_state = selected_states[to_idx]
                
                # Compute stoichiometry
                stoichiometry = to_state - from_state
                
                # Filter elementary reactions
                total_change = sum(abs.(stoichiometry))
                if 0 < total_change <= 3
                    stoich_key = tuple(stoichiometry...)
                    
                    if !haskey(all_reactions, stoich_key)
                        all_reactions[stoich_key] = []
                    end
                    
                    push!(all_reactions[stoich_key], (
                        rate = abs(reaction.rate),
                        confidence = reaction.confidence,
                        segment = segment_idx
                    ))
                end
            end
        end
    end
    
    # Fuse reactions by weighted averaging
    fused_reactions = Dict()
    reaction_stats = Dict()
    
    for (stoich, rate_data) in all_reactions
        if length(rate_data) >= 1  # Accept single-segment reactions
            # Weighted average by confidence
            total_weight = sum(rd.confidence for rd in rate_data)
            weighted_rate = sum(rd.rate * rd.confidence for rd in rate_data) / total_weight
            
            # Consistency score
            rates = [rd.rate for rd in rate_data]
            rate_variance = length(rates) > 1 ? var(rates) : 0.0
            consistency = 1.0 / (1.0 + rate_variance)
            
            # Overall confidence
            overall_confidence = (total_weight / length(rate_data)) * consistency
            
            fused_reactions[stoich] = []
            reaction_stats[stoich] = (
                total_rate = weighted_rate,
                avg_rate = weighted_rate,
                rate_var = rate_variance,
                count = length(rate_data),
                confidence = overall_confidence,
                segments = [rd.segment for rd in rate_data]
            )
        end
    end
    
    # Sort by confidence * rate
    sorted_stoich = sort(collect(keys(reaction_stats)),
                        by=s -> reaction_stats[s].confidence * reaction_stats[s].total_rate,
                        rev=true)
    
    println("Fused $(length(sorted_stoich)) reactions from $(length(all_reactions)) candidates")
    
    return sorted_stoich, fused_reactions, reaction_stats
end

"""
    create_combined_generator(segment_results, n_states)

Create approximate combined generator from segment results.
"""
function create_combined_generator(segment_results, n_states)
    G_combined = zeros(n_states, n_states)
    weight_sum = zeros(n_states, n_states)
    
    for (G_local, error, local_reactions) in segment_results
        if G_local !== nothing
            weight = 1.0 / (1.0 + error)
            G_combined .+= weight * G_local
            weight_sum .+= weight
        end
    end
    
    # Normalize by weights
    for i in 1:n_states, j in 1:n_states
        if weight_sum[i,j] > 0
            G_combined[i,j] /= weight_sum[i,j]
        end
    end
    
    return G_combined
end

"""
    apply_multigrid_dmd(prob_matrix, dt, selected_states, species_names; 
                       segment_length=8, overlap_fraction=0.3, 
                       use_reachability=true, masking_strength=1.0,
                       min_observations=2, confidence_threshold=0.8,
                       ssa_trajs=nothing, time_points=nothing)

Apply multigrid DMD with optional reachability masking.
"""
function apply_multigrid_dmd(prob_matrix, dt, selected_states, species_names; 
                            segment_length=8, overlap_fraction=0.3,
                            use_reachability=true, masking_strength=1.0,
                            min_observations=2, confidence_threshold=0.8,
                            ssa_trajs=nothing, time_points=nothing)
    
    println("\n" * "="^60)
    println("MULTIGRID DMD ANALYSIS")
    if use_reachability
        println("WITH CONFIDENCE-WEIGHTED REACHABILITY MASKING")
    end
    println("="^60)
    
    n_states, total_time_points = size(prob_matrix)
    println("Data: $n_states states × $total_time_points time points")
    println("Segment length: $segment_length, overlap: $(overlap_fraction*100)%")
    
    # Compute reachability matrix if requested and trajectory data available
    R = nothing
    if use_reachability && ssa_trajs !== nothing && time_points !== nothing
        println("\n🔍 Computing confidence-weighted reachability matrix...")
        R, transition_counts, obs_counts = compute_elementary_reachability_matrix(
            ssa_trajs, time_points, selected_states,
            min_observations=min_observations, 
            confidence_threshold=confidence_threshold
        )
    elseif use_reachability
        println("⚠️  Reachability masking requested but trajectory data not provided")
        println("   Proceeding without reachability constraints")
        use_reachability = false
    end
    
    # Create time segments
    dmd_segments = create_dmd_time_segments(total_time_points, segment_length, overlap_fraction)
    println("Created $(length(dmd_segments)) DMD segments")
    
    # Process each segment with optional reachability masking
    segment_results = []
    successful_segments = 0
    
    for (seg_idx, (start_t, end_t)) in enumerate(dmd_segments)
        println("\n--- Processing Segment $seg_idx: t[$start_t:$end_t] ---")
        
        segment_data = prob_matrix[:, start_t:end_t]
        
        # Apply local DMD with optional reachability
        G_local, error, local_reactions = apply_local_dmd_with_reachability(
            segment_data, dt, R, use_reachability, masking_strength
        )
        
        if G_local !== nothing
            successful_segments += 1
            println("  ✓ Found $(length(local_reactions)) reactions (error: $(round(error, digits=4)))")
        else
            println("  ✗ Segment processing failed")
        end
        
        push!(segment_results, (G_local, error, local_reactions))
    end
    
    println("\nSuccessfully processed $successful_segments/$(length(dmd_segments)) segments")
    
    # Fuse results across segments
    if successful_segments >= 1
        sorted_stoich, fused_reactions, reaction_stats = fuse_segment_reactions(
            segment_results, selected_states, species_names
        )
        
        # Create combined generator
        G_combined = create_combined_generator(segment_results, n_states)
        
        # Apply reachability to combined generator if used
        if use_reachability && R !== nothing
            println("Applying reachability masking to combined generator...")
            G_combined = apply_confidence_weighted_masking(G_combined, R, masking_strength)
        end
        
        # Compute combined eigenvalues for flow analysis on CLEAN spectrum
        λ_combined, Φ_combined = eigen(G_combined)
        
        # Return additional reachability info for analysis
        reachability_info = use_reachability ? Dict(
            "reachability_matrix" => R,
            "masking_strength" => masking_strength,
            "confidence_threshold" => confidence_threshold
        ) : Dict()
        
        return G_combined, λ_combined, Φ_combined, sorted_stoich, reaction_stats, successful_segments, reachability_info
    else
        println("⚠ No successful segments for analysis")
        return zeros(n_states, n_states), [], [], [], Dict(), 0, Dict()
    end
end

"""
    run_dmd_analysis(data_dict; dmd_method=:multigrid, use_reachability=true, 
                    masking_strength=1.0, min_observations=2, confidence_threshold=0.8)

Run DMD analysis with optional reachability masking and data-limitation safeguards.
"""
function run_dmd_analysis(data_dict; dmd_method=:multigrid, use_reachability=true,
                         masking_strength=1.0, min_observations=2, confidence_threshold=0.8, kwargs...)
    println("\n" * "="^50)
    println("DMD ANALYSIS ($(uppercase(string(dmd_method))))")
    if use_reachability
        println("WITH CONFIDENCE-WEIGHTED REACHABILITY MASKING")
    end
    println("="^50)
    
    # Extract data
    prob_matrix = data_dict["probability_matrix"]
    dt = data_dict["dt"]
    selected_states = data_dict["selected_states"]
    species_names = data_dict["species_names"]
    
    # Get trajectory data for reachability if available
    ssa_trajs = get(data_dict, "trajectories", nothing)
    time_points = get(data_dict, "time_points", nothing)
    
    if dmd_method == :multigrid
        G, λ, Φ, sorted_stoich, reaction_stats, n_segments, reachability_info = apply_multigrid_dmd(
            prob_matrix, dt, selected_states, species_names;
            use_reachability=use_reachability,
            masking_strength=masking_strength,
            min_observations=min_observations,
            confidence_threshold=confidence_threshold,
            ssa_trajs=ssa_trajs,
            time_points=time_points,
            kwargs...
        )
        
        # Add results to data dictionary
        data_dict["generator"] = G
        data_dict["eigenvalues"] = λ
        data_dict["modes"] = Φ
        data_dict["significant_stoichiometries"] = sorted_stoich
        data_dict["reaction_stats"] = reaction_stats
        data_dict["successful_segments"] = n_segments
        data_dict["dmd_method"] = "multigrid"
        data_dict["reachability_info"] = reachability_info
        
        # Report reachability impact
        if use_reachability && !isempty(reachability_info)
            println("\n📊 Reachability Masking Summary:")
            println("  Masking strength: $(masking_strength)")
            println("  Confidence threshold: $(confidence_threshold)")
            println("  Min observations: $(min_observations)")
        end
        
    else
        error("Unknown DMD method: $dmd_method. Available: :multigrid")
    end
    
    println("\nDMD analysis completed!")
    return data_dict
end

println("Clean DMD Module with Reachability Masking Loaded! 🔄🎯")
println("Available methods:")
println("  :multigrid (default) - Robust multiscale DMD with optional reachability")
println()
println("Reachability features:")
println("  • Confidence-weighted masking (soft transitions)")
println("  • Data limitation safeguards")
println("  • Spurious reaction elimination")
println("  • Adjustable masking strength")

println("Clean DMD Module Loaded! 🔄")
println("Available methods:")
println("  :multigrid (default) - Robust multiscale DMD analysis")
println()
println("Functions:")
println("  apply_multigrid_dmd(matrix, dt, states, species)")
println("  run_dmd_analysis(data_dict, dmd_method=:multigrid)")
println()
println("Multigrid DMD provides:")
println("  • Robust reaction discovery across timescales")
println("  • Automatic constraint enforcement")
println("  • Confidence-weighted reaction fusion")
