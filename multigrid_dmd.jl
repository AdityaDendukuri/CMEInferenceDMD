# multigrid_dmd.jl - MULTIGRID DMD FOR SCALABLE CME RECOVERY
# Implements adaptive multiscale approach for computationally tractable CME inference

using LinearAlgebra
using SparseArrays
using Statistics

"""
    create_dmd_time_grid(total_time_points, segment_length, overlap_fraction=0.2)

Create DMD grid segments from the data time grid.
"""
function create_dmd_time_grid(total_time_points, segment_length, overlap_fraction=0.2)
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
    local_constrained_dmd(segment_data, dt; regularization_params=(0.02, 0.001, 0.1))

Apply computationally tractable constrained DMD to a small time segment.
"""
function local_constrained_dmd(segment_data, dt; regularization_params=(0.02, 0.001, 0.1))
    n_states, n_times = size(segment_data)
    
    if n_times < 3
        return nothing, 0, []  # Need at least 3 time points
    end
    
    # Form local snapshot matrices
    X_local = segment_data[:, 1:end-1]
    X_prime_local = segment_data[:, 2:end]
    
    # For small segments, use simplified constrained approach
    # This is computationally tractable unlike the full matrix exponential optimization
    
    # Step 1: Standard DMD
    U, Σ, V = svd(X_local)
    
    # Use aggressive rank reduction for computational tractability
    max_rank = min(min(size(X_local)...) - 1, 10)  # Keep very low rank
    
    # Filter singular values
    sig_threshold = 0.01 * Σ[1]  # More aggressive threshold
    r = min(sum(Σ .> sig_threshold), max_rank)
    
    if r < 2
        return nothing, 0, []  # Need at least rank 2
    end
    
    U_r = U[:, 1:r]
    Σ_r = Diagonal(Σ[1:r])
    V_r = V[:, 1:r]
    
    # DMD operator in reduced space
    A_tilde = U_r' * X_prime_local * V_r * inv(Σ_r)
    
    # Project back to full space
    A_full = U_r * A_tilde * U_r'
    
    # Convert to generator with proper constraints
    K_local = A_full
    
    # Simple constraint enforcement for computational tractability
    G_local = constrain_local_generator((K_local - I) / dt, regularization_params)
    
    # Compute reconstruction error
    K_constrained = I + G_local * dt
    prediction_error = norm(X_prime_local - K_constrained * X_local)
    
    # Extract reactions from local generator
    local_reactions = extract_local_reactions(G_local, prediction_error)
    
    return G_local, prediction_error, local_reactions
end

"""
    constrain_local_generator(G_raw, reg_params)

Apply computationally efficient constraints to local generator.
"""
function constrain_local_generator(G_raw, reg_params)
    λ_sparsity, λ_smooth, λ_structure = reg_params
    n_states = size(G_raw, 1)
    
    G_constrained = copy(G_raw)
    
    # Iterative constraint projection (computationally efficient)
    for iter in 1:20  # Limited iterations for speed
        G_old = copy(G_constrained)
        
        # 1. Sparsity: threshold small elements
        threshold = λ_sparsity * maximum(abs.(G_constrained))
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
    extract_local_reactions(G_local, error)

Extract reactions from local generator with confidence scoring.
"""
function extract_local_reactions(G_local, error)
    n_states = size(G_local, 1)
    local_reactions = []
    
    # Confidence based on reconstruction error (lower error = higher confidence)
    confidence = 1.0 / (1.0 + error)
    
    # Find significant transitions
    threshold = 0.001 * maximum(abs.(G_local))
    
    for i in 1:n_states
        for j in 1:n_states
            if i != j && abs(G_local[i,j]) > threshold
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
    adaptive_reaction_fusion(segment_results, selected_states, species_names)

Adaptively fuse reactions discovered across different time segments.
"""
function adaptive_reaction_fusion(segment_results, selected_states, species_names)
    println("\n=== Adaptive Reaction Fusion ===")
    
    # Collect all reactions with their confidences
    all_reactions = Dict()  # stoichiometry -> list of (rate, confidence, segment)
    
    for (segment_idx, (G_local, error, local_reactions)) in enumerate(segment_results)
        if G_local === nothing
            continue
        end
        
        println("Processing segment $segment_idx: $(length(local_reactions)) local reactions")
        
        # Debug: check what's in local_reactions
        if !isempty(local_reactions)
            println("  Sample reaction: from_state=$(local_reactions[1].from_state), to_state=$(local_reactions[1].to_state)")
        end
        
        candidates_processed = 0
        valid_reactions = 0
        
        for reaction in local_reactions
            candidates_processed += 1
            
            from_idx = reaction.from_state  
            to_idx = reaction.to_state
            
            # Debug bounds check
            if from_idx > length(selected_states) || to_idx > length(selected_states)
                println("    Reaction $candidates_processed: indices out of bounds ($from_idx, $to_idx) vs $(length(selected_states))")
                continue
            end
            
            if from_idx <= length(selected_states) && to_idx <= length(selected_states)
                from_state = selected_states[from_idx]
                to_state = selected_states[to_idx]
                
                # Compute stoichiometry
                from_mol = [max(0, x-1) for x in from_state]
                to_mol = [max(0, x-1) for x in to_state]
                stoichiometry = to_mol - from_mol
                
                # Filter elementary reactions
                total_change = sum(abs.(stoichiometry))
                
                # Debug stoichiometry
                if candidates_processed <= 3  # Only print first few
                    println("    Reaction $candidates_processed: stoich=$stoichiometry, total_change=$total_change")
                end
                
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
                    
                    valid_reactions += 1
                end
            end
        end
        
        println("  Processed $candidates_processed candidates, $valid_reactions valid reactions")
    end
    
    println("Total reaction candidates found: $(length(all_reactions))")
    
    # Fuse reactions by weighted averaging
    fused_reactions = Dict()
    reaction_stats = Dict()
    
    for (stoich, rate_data) in all_reactions
        if length(rate_data) >= 1  # Accept single-segment reactions - CHANGED FROM >= 2
            
            # Weighted average by confidence
            total_weight = sum(rd.confidence for rd in rate_data)
            weighted_rate = sum(rd.rate * rd.confidence for rd in rate_data) / total_weight
            
            # Consistency score (lower variance = higher consistency)
            rates = [rd.rate for rd in rate_data]
            rate_variance = length(rates) > 1 ? var(rates) : 0.0
            consistency = 1.0 / (1.0 + rate_variance)
            
            # Overall confidence
            overall_confidence = (total_weight / length(rate_data)) * consistency
            
            fused_reactions[stoich] = []  # Placeholder for compatibility
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
    
    # Sort by confidence * rate (most reliable and significant)
    sorted_stoich = sort(collect(keys(reaction_stats)), 
                        by=s -> reaction_stats[s].confidence * reaction_stats[s].total_rate, 
                        rev=true)
    
    println("Fused $(length(sorted_stoich)) reliable reactions from $(length(all_reactions)) candidates")
    
    # Display top reactions
    println("\nTop fused reactions:")
    for (i, stoich) in enumerate(sorted_stoich[1:min(10, end)])
        stats = reaction_stats[stoich]
        reaction_str = format_reaction(stoich, species_names)
        println("$i. $reaction_str")
        println("   Rate: $(round(stats.total_rate, digits=4)), Confidence: $(round(stats.confidence, digits=3))")
        println("   Segments: $(stats.segments), Variance: $(round(stats.rate_var, digits=6))")
    end
    
    return sorted_stoich, fused_reactions, reaction_stats
end

"""
    multigrid_constrained_dmd(reduced_data, dt, selected_states, species_names; 
                             segment_length=8, overlap_fraction=0.3)

Main multigrid DMD function for scalable CME recovery.
"""
function multigrid_constrained_dmd(reduced_data, dt, selected_states, species_names; 
                                  segment_length=8, overlap_fraction=0.3)
    
    println("\n" * "="^60)
    println("MULTIGRID CONSTRAINED DMD")
    println("="^60)
    
    n_states, total_time_points = size(reduced_data)
    println("Data: $n_states states × $total_time_points time points")
    println("Segment length: $segment_length, Overlap: $(overlap_fraction*100)%")
    
    # Create DMD time grid
    dmd_segments = create_dmd_time_grid(total_time_points, segment_length, overlap_fraction)
    println("Created $(length(dmd_segments)) DMD segments")
    
    # Process each segment
    segment_results = []
    successful_segments = 0
    
    for (seg_idx, (start_t, end_t)) in enumerate(dmd_segments)
        println("\n--- Processing Segment $seg_idx: t[$start_t:$end_t] ---")
        
        segment_data = reduced_data[:, start_t:end_t]
        segment_dt = dt  # Assuming uniform time spacing
        
        # Apply local constrained DMD
        G_local, error, local_reactions = local_constrained_dmd(
            segment_data, segment_dt, 
            regularization_params=(0.02, 0.001, 0.1)
        )
        
        if G_local !== nothing
            successful_segments += 1
            println("  ✓ Recovered $(length(local_reactions)) local reactions (error: $(round(error, digits=4)))")
            
            # Quick constraint check for this segment
            violations_count = count_constraint_violations(G_local)
            println("  Constraint violations: $violations_count")
        else
            println("  ✗ Segment processing failed")
        end
        
        push!(segment_results, (G_local, error, local_reactions))
    end
    
    println("\nSuccessfully processed $successful_segments/$(length(dmd_segments)) segments")
    
    # Adaptive fusion of results
    if successful_segments >= 2
        sorted_stoich, fused_reactions, reaction_stats = adaptive_reaction_fusion(
            segment_results, selected_states, species_names
        )
        
        # Create dummy generator for compatibility (weighted combination)
        # This is approximate but maintains the interface
        G_combined = create_combined_generator(segment_results, n_states)
        
        return G_combined, sorted_stoich, fused_reactions, reaction_stats, successful_segments
    else
        println("⚠ Insufficient successful segments for reliable fusion")
        return zeros(n_states, n_states), [], Dict(), Dict(), 0
    end
end

"""
    count_constraint_violations(G; tol=1e-6)

Quick constraint violation check for small matrices.
"""
function count_constraint_violations(G; tol=1e-6)
    n = size(G, 1)
    violations = 0
    
    # Off-diagonal non-negative
    for i in 1:n, j in 1:n
        if i != j && G[i,j] < -tol
            violations += 1
        end
    end
    
    # Column sums zero
    for j in 1:n
        if abs(sum(G[:, j])) > tol
            violations += 1
        end
    end
    
    return violations
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
            weight = 1.0 / (1.0 + error)  # Higher weight for lower error
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

# Helper function for reaction formatting (assuming it exists)
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

println("Multigrid DMD Module Loaded! 🔄")
println("Key functions:")
println("  multigrid_constrained_dmd(data, dt, states, species_names)")
println("  create_dmd_time_grid(time_points, segment_length)")
println("  adaptive_reaction_fusion(segment_results, states, species_names)")
println()
println("This multigrid approach enables:")
println("  • Computationally tractable constraint enforcement")
println("  • Local reaction discovery across time scales")
println("  • Adaptive fusion with confidence weighting")
println("  • Scalable processing of large state spaces")
