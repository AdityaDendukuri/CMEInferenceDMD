# multigrid_dmd.jl - COMPLETE MULTIGRID DMD FOR UNIVERSAL SYSTEMS
# Supports MM, Lotka-Volterra, Toggle Switch, and general systems

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
    is_biochemically_valid_reaction(stoich, species_names, system_type)

Check if a reaction satisfies system-appropriate biochemical constraints.
"""
function is_biochemically_valid_reaction(stoich, species_names, system_type)
    n_species = length(stoich)
    
    # Universal constraints (apply to all systems)
    total_change = sum(abs.(stoich))
    
    # 1. No trivial reactions
    if total_change == 0
        return false
    end
    
    # 2. Reasonable stoichiometric bounds (prevent numerical artifacts)
    max_reasonable_change = 10
    if total_change > max_reasonable_change
        return false
    end
    
    # System-specific constraints
    if system_type == "mm"
        return is_mm_valid_reaction(stoich, species_names)
    elseif system_type == "lotka_volterra"
        return is_lv_valid_reaction(stoich, species_names)
    elseif system_type == "toggle_switch"
        return is_toggle_valid_reaction(stoich, species_names)
    else
        # General system - only universal constraints
        return is_general_valid_reaction(stoich, species_names)
    end
end

"""
    is_mm_valid_reaction(stoich, species_names)

MM-specific biochemical constraints using conservation laws.
"""
function is_mm_valid_reaction(stoich, species_names)
    if length(stoich) != 4 || length(species_names) != 4
        return false
    end
    
    s_change, e_change, se_change, p_change = stoich
    
    # MM-specific conservation laws (derived from atomic composition)
    # Conservation 1: Substrate material balance (S + SE + P = constant)
    substrate_balance = s_change + se_change + p_change
    
    # Conservation 2: Enzyme material balance (E + SE = constant)  
    enzyme_balance = e_change + se_change
    
    # Conservation tolerance
    conservation_tolerance = 1
    conserves_substrate = abs(substrate_balance) <= conservation_tolerance
    conserves_enzyme = abs(enzyme_balance) <= conservation_tolerance
    
    # Universal chemical principles
    # No direct S → P without enzyme involvement
    direct_substrate_product = (s_change != 0 && p_change != 0 && e_change == 0 && se_change == 0)
    if direct_substrate_product
        return false
    end
    
    # No spontaneous enzyme creation/destruction
    enzyme_creation = (e_change > 0 && se_change == 0 && s_change == 0 && p_change == 0)
    enzyme_destruction = (e_change < 0 && se_change == 0 && s_change == 0 && p_change == 0)
    if enzyme_creation || enzyme_destruction
        return false
    end
    
    return conserves_substrate && conserves_enzyme
end

"""
    is_lv_valid_reaction(stoich, species_names)

Lotka-Volterra specific constraints (minimal - mostly universal).
"""
function is_lv_valid_reaction(stoich, species_names)
    if length(stoich) != 2 || length(species_names) != 2
        return false
    end
    
    x_change, y_change = stoich  # X = prey, Y = predator
    
    # Universal constraints only for LV (no strict conservation laws)
    # Allow birth, death, and predation patterns
    
    # Prevent absurd simultaneous changes
    if abs(x_change) > 5 || abs(y_change) > 5
        return false
    end
    
    # All other patterns allowed for ecological dynamics
    return true
end

"""
    is_toggle_valid_reaction(stoich, species_names)

Toggle switch specific constraints.
"""
function is_toggle_valid_reaction(stoich, species_names)
    if length(stoich) != 2 || length(species_names) != 2
        return false
    end
    
    a_change, b_change = stoich  # A and B are the toggle switch proteins
    
    # Universal constraints for gene regulatory networks
    # Allow production from nothing (transcription) and degradation to nothing
    
    # Prevent absurd simultaneous changes
    if abs(a_change) > 3 || abs(b_change) > 3
        return false
    end
    
    # All other patterns allowed for gene regulation
    return true
end

"""
    is_general_valid_reaction(stoich, species_names)

General system constraints - only universal chemical principles.
"""
function is_general_valid_reaction(stoich, species_names)
    # Only universal constraints
    total_change = sum(abs.(stoich))
    
    # Prevent very large changes (likely numerical artifacts)
    if total_change > 8
        return false
    end
    
    # Prevent all species changing by large amounts simultaneously
    if all(abs.(stoich) .> 2)
        return false
    end
    
    return true
end

"""
    local_constrained_dmd(segment_data, dt, selected_states, species_names, system_type; 
                          regularization_params=(0.02, 0.001, 0.1))

Apply system-appropriate constrained DMD to a small time segment.
"""
function local_constrained_dmd(segment_data, dt, selected_states, species_names, system_type; 
                              regularization_params=(0.02, 0.001, 0.1))
    n_states, n_times = size(segment_data)
    
    if n_times < 3
        return nothing, 0, []
    end
    
    # Form local snapshot matrices
    X_local = segment_data[:, 1:end-1]
    X_prime_local = segment_data[:, 2:end]
    
    # Conservative rank selection
    U, Σ, V = svd(X_local)
    
    max_rank = min(min(size(X_local)...) - 1, 10)
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
    
    # Apply system-appropriate constraints to generator
    G_local = constrain_local_generator((A_full - I) / dt, regularization_params, 
                                       selected_states, species_names, system_type)
    
    # Compute reconstruction error
    K_constrained = I + G_local * dt
    prediction_error = norm(X_prime_local - K_constrained * X_local)
    
    # Extract system-appropriate reactions
    local_reactions = extract_system_valid_reactions(G_local, selected_states, species_names, 
                                                    system_type, prediction_error)
    
    return G_local, prediction_error, local_reactions
end

"""
    constrain_local_generator(G_raw, reg_params, selected_states, species_names, system_type)

Apply system-appropriate constraints to local generator.
"""
function constrain_local_generator(G_raw, reg_params, selected_states, species_names, system_type)
    λ_sparsity, λ_smooth, λ_structure = reg_params
    n_states = size(G_raw, 1)
    
    G_constrained = copy(G_raw)
    
    # Pre-filter: Remove transitions that violate system-specific constraints
    valid_transitions = 0
    invalid_transitions = 0
    
    for i in 1:n_states
        for j in 1:n_states
            if i != j && i <= length(selected_states) && j <= length(selected_states)
                from_state = selected_states[j]
                to_state = selected_states[i]
                
                # Compute stoichiometry
                from_mol = [max(0, x-1) for x in from_state]
                to_mol = [max(0, x-1) for x in to_state]
                stoichiometry = to_mol - from_mol
                
                # Apply system-specific validation
                if is_biochemically_valid_reaction(stoichiometry, species_names, system_type)
                    valid_transitions += 1
                else
                    G_constrained[i,j] = 0
                    invalid_transitions += 1
                end
            end
        end
    end
    
    # Iterative constraint projection
    for iter in 1:30
        G_old = copy(G_constrained)
        
        # 1. Sparsity with system-awareness
        threshold = λ_sparsity * maximum(abs.(G_constrained))
        for i in 1:n_states
            for j in 1:n_states
                if i != j && abs(G_constrained[i,j]) < threshold
                    G_constrained[i,j] = 0
                end
            end
        end
        
        # 2. Non-negative off-diagonals (universal generator constraint)
        for i in 1:n_states
            for j in 1:n_states
                if i != j
                    G_constrained[i,j] = max(0, G_constrained[i,j])
                end
            end
        end
        
        # 3. Zero column sums (probability conservation - universal)
        for j in 1:n_states
            off_diag_sum = sum(G_constrained[i,j] for i in 1:n_states if i != j)
            G_constrained[j,j] = -off_diag_sum
        end
        
        # Check convergence
        if norm(G_constrained - G_old) < 1e-10
            break
        end
    end
    
    return G_constrained
end

"""
    extract_system_valid_reactions(G_local, selected_states, species_names, system_type, error)

Extract system-appropriate valid reactions.
"""
function extract_system_valid_reactions(G_local, selected_states, species_names, system_type, error)
    n_states = size(G_local, 1)
    local_reactions = []
    
    # Confidence based on reconstruction error
    confidence = 1.0 / (1.0 + error)
    
    # Adaptive threshold
    threshold = 0.001 * maximum(abs.(G_local))
    
    valid_found = 0
    invalid_skipped = 0
    
    for i in 1:n_states
        for j in 1:n_states
            if i != j && abs(G_local[i,j]) > threshold && 
               i <= length(selected_states) && j <= length(selected_states)
                
                # Compute stoichiometry
                from_state = selected_states[j]
                to_state = selected_states[i]
                
                from_mol = [max(0, x-1) for x in from_state]
                to_mol = [max(0, x-1) for x in to_state]
                stoichiometry = to_mol - from_mol
                
                # Check system-specific validity
                if is_biochemically_valid_reaction(stoichiometry, species_names, system_type)
                    rate = G_local[i,j]
                    total_change = sum(abs.(stoichiometry))
                    
                    push!(local_reactions, (
                        from_state = j,
                        to_state = i,
                        stoichiometry = stoichiometry,
                        rate = rate,
                        confidence = confidence,
                        total_change = total_change
                    ))
                    valid_found += 1
                else
                    invalid_skipped += 1
                end
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
    
    # Collect all valid reactions with their confidences
    all_reactions = Dict()
    
    for (segment_idx, (G_local, error, local_reactions)) in enumerate(segment_results)
        if G_local === nothing || isempty(local_reactions)
            continue
        end
        
        println("Processing segment $segment_idx: $(length(local_reactions)) valid reactions")
        
        for reaction in local_reactions
            stoich = reaction.stoichiometry
            stoich_key = tuple(stoich...)
            
            if !haskey(all_reactions, stoich_key)
                all_reactions[stoich_key] = []
            end
            
            push!(all_reactions[stoich_key], (
                rate = abs(reaction.rate),
                confidence = reaction.confidence,
                segment = segment_idx,
                total_change = reaction.total_change
            ))
        end
    end
    
    println("Found $(length(all_reactions)) unique valid reaction types")
    
    # Fuse reactions with enhanced statistics
    fused_reactions = Dict()
    reaction_stats = Dict()
    
    for (stoich, rate_data) in all_reactions
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
            segments = [rd.segment for rd in rate_data],
            total_change = rate_data[1].total_change
        )
    end
    
    # Sort by confidence * rate
    sorted_stoich = sort(collect(keys(reaction_stats)), 
                        by=s -> reaction_stats[s].confidence * reaction_stats[s].total_rate, 
                        rev=true)
    
    println("\nTop valid reactions:")
    for (i, stoich) in enumerate(sorted_stoich[1:min(10, end)])
        stats = reaction_stats[stoich]
        reaction_str = format_reaction(stoich, species_names)
        println("$i. $reaction_str")
        println("   Rate: $(round(stats.total_rate, digits=4)), Confidence: $(round(stats.confidence, digits=3))")
        println("   Change: $(stats.total_change) molecules, Segments: $(stats.segments)")
    end
    
    return sorted_stoich, fused_reactions, reaction_stats
end

"""
    multigrid_constrained_dmd(reduced_data, dt, selected_states, species_names; 
                             segment_length=8, overlap_fraction=0.3, system_type="general")

Main multigrid DMD function for universal system support.
"""
function multigrid_constrained_dmd(reduced_data, dt, selected_states, species_names; 
                                  segment_length=8, overlap_fraction=0.3, system_type="general")
    
    println("\n" * "="^60)
    println("UNIVERSAL MULTIGRID CONSTRAINED DMD")
    println("="^60)
    
    n_states, total_time_points = size(reduced_data)
    println("System type: $system_type")
    println("Data: $n_states states × $total_time_points time points")
    println("Segment length: $segment_length, Overlap: $(overlap_fraction*100)%")
    
    # Create DMD time grid
    dmd_segments = create_dmd_time_grid(total_time_points, segment_length, overlap_fraction)
    println("Created $(length(dmd_segments)) DMD segments")
    
    # Process each segment with system-specific constraints
    segment_results = []
    successful_segments = 0
    total_valid_reactions = 0
    
    for (seg_idx, (start_t, end_t)) in enumerate(dmd_segments)
        println("\n--- Processing Segment $seg_idx: t[$start_t:$end_t] ---")
        
        segment_data = reduced_data[:, start_t:end_t]
        
        # Apply system-specific constrained DMD
        G_local, error, local_reactions = local_constrained_dmd(
            segment_data, dt, selected_states, species_names, system_type,
            regularization_params=(0.02, 0.001, 0.1)
        )
        
        if G_local !== nothing
            successful_segments += 1
            valid_count = length(local_reactions)
            total_valid_reactions += valid_count
            
            println("  ✓ Recovered $valid_count system-valid reactions (error: $(round(error, digits=4)))")
            
            # Show sample valid reactions for this segment
            if valid_count > 0
                println("    Sample valid reactions:")
                for (i, reaction) in enumerate(local_reactions[1:min(3, end)])
                    stoich_str = join(reaction.stoichiometry, ",")
                    println("      $(i). Stoich: [$stoich_str], Change: $(reaction.total_change), Rate: $(round(reaction.rate, digits=5))")
                end
            end
            
            # Constraint check
            violations_count = count_constraint_violations(G_local)
            println("  Constraint violations: $violations_count")
        else
            println("  ✗ Segment processing failed")
        end
        
        push!(segment_results, (G_local, error, local_reactions))
    end
    
    println("\nSUMMARY:")
    println("Successfully processed $successful_segments/$(length(dmd_segments)) segments")
    println("Total system-valid reactions found: $total_valid_reactions")
    
    # Adaptive fusion
    if successful_segments >= 1 && total_valid_reactions > 0
        sorted_stoich, fused_reactions, reaction_stats = adaptive_reaction_fusion(
            segment_results, selected_states, species_names
        )
        
        G_combined = create_combined_generator(segment_results, n_states)
        
        return G_combined, sorted_stoich, fused_reactions, reaction_stats, successful_segments
    else
        println("⚠ No system-valid reactions found across all segments")
        return zeros(n_states, n_states), [], Dict(), Dict(), 0
    end
end

"""
    count_constraint_violations(G; tol=1e-6)

Quick constraint violation check.
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
    format_reaction(stoich, species_names)

Format reaction for display.
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

println("Complete Universal Multigrid DMD Module Loaded! 🔄🧬🎯")
println("Key features:")
println("  • ✅ Universal system support (MM, LV, Toggle, General)")
println("  • ✅ System-specific biochemical constraints")
println("  • ✅ Adaptive multigrid processing")
println("  • ✅ Conservation law enforcement")
println("  • ✅ No hardcoded reaction mechanisms")
println()
println("Main function:")
println("  multigrid_constrained_dmd(data, dt, states, species_names;")
println("                            system_type=\"mm\"|\"lotka_volterra\"|\"toggle_switch\"|\"general\")")
println()
println("System-specific constraints:")
println("  MM: Substrate + enzyme conservation laws")
println("  Lotka-Volterra: Universal constraints only")
println("  Toggle Switch: Gene regulatory constraints")
println("  General: Universal chemical principles only")
