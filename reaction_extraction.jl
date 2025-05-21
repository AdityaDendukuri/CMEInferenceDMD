# reaction_extraction.jl
# Functions for extracting and validating reactions from the generator matrix

using SparseArrays
using Statistics

"""
    extract_reactions_from_generator(G, selected_states, species_indices, species_names; threshold=1e-5, validate_reactions=true)

Extract elementary reactions from the generator matrix with improved filtering.

# Arguments
- `G`: Generator matrix
- `selected_states`: List of states in the reduced space
- `species_indices`: Indices of species to consider
- `species_names`: Names of species
- `threshold`: Relative threshold for identifying significant transitions
- `validate_reactions`: Whether to validate reactions using conservation principles

# Returns
- List of significant reaction stoichiometries
- Dictionary mapping stoichiometries to reaction instances
- Dictionary with statistics for each stoichiometry
"""
function extract_reactions_from_generator(G, selected_states, species_indices, species_names; threshold=1e-5, validate_reactions=true)
    # Initialize reaction list
    reactions = []
    
    # Convert G to sparse for efficiency
    G_sparse = sparse(G)
    rows, cols, vals = findnz(G_sparse)
    
    # Identify the maximum magnitude for scaling
    max_magnitude = maximum(abs.(vals))
    relative_threshold = threshold * max_magnitude
    
    println("Using reaction threshold: $relative_threshold ($(threshold*100)% of maximum magnitude $(max_magnitude))")
    
    for i in 1:length(vals)
        if abs(vals[i]) > relative_threshold && rows[i] != cols[i]  # Ignore diagonal elements
            # Get the states
            to_state = [selected_states[rows[i]]...]
            from_state = [selected_states[cols[i]]...]
            
            # Compute stoichiometry vector (state changes)
            stoichiometry = [(to_state[j] - from_state[j]) for j in 1:length(from_state)]
            
            # Filter for elementary reactions (limited molecule changes)
            total_change = sum(abs.(stoichiometry))
            if total_change <= 3  # Allow up to 3 molecular changes
                # Rate is the value in G
                rate = vals[i]
                
                # Add reaction to list
                push!(reactions, (
                    from_state=from_state,
                    to_state=to_state,
                    stoichiometry=stoichiometry,
                    rate=rate
                ))
            end
        end
    end
    
    println("Found $(length(reactions)) potential elementary reactions")
    
    # Group reactions by stoichiometry
    grouped_reactions = Dict()
    for r in reactions
        s_key = tuple(r.stoichiometry...)
        if haskey(grouped_reactions, s_key)
            push!(grouped_reactions[s_key], r)
        else
            grouped_reactions[s_key] = [r]
        end
    end
    
    # Calculate statistics for each stoichiometry pattern
    stoich_stats = Dict()
    for (stoich, rxns) in grouped_reactions
        # Calculate total rate
        total_rate = sum(abs(r.rate) for r in rxns)
        
        # Calculate average rate
        avg_rate = total_rate / length(rxns)
        
        # Calculate rate variance
        rate_var = var([abs(r.rate) for r in rxns], corrected=false)
        
        stoich_stats[stoich] = (
            total_rate=total_rate,
            avg_rate=avg_rate,
            rate_var=rate_var,
            count=length(rxns)
        )
    end
    
    # Sort stoichiometries by total rate
    sorted_stoich = sort(collect(keys(stoich_stats)), by=s -> stoich_stats[s].total_rate, rev=true)
    
    # Validate reactions if requested
    validated_stoich = copy(sorted_stoich)
    if validate_reactions
        # Check for conservation of mass/charge
        validated_stoich = filter(stoich -> is_valid_reaction(stoich, species_indices), sorted_stoich)
        println("Filtered out $(length(sorted_stoich) - length(validated_stoich)) invalid reactions")
    end
    
    # Keep only top stoichiometries after validation
    top_n = min(15, length(validated_stoich))  # Get top 15 reactions
    significant_stoich = validated_stoich[1:top_n]
    
    # Print top reactions found
    println("\nTop reactions identified by DMD (ranked by total rate):")
    for stoich in significant_stoich
        # Skip self-transitions
        if all(stoich .== 0)
            continue
        end
        
        # Get statistics
        stats = stoich_stats[stoich]
        
        # Format reaction string
        reaction_str = format_reaction(stoich, species_names)
        
        println("$reaction_str  (total rate ≈ $(round(stats.total_rate, digits=5)), avg rate ≈ $(round(stats.avg_rate, digits=5)), count: $(stats.count))")
    end
    
    return significant_stoich, grouped_reactions, stoich_stats
end

"""
    is_valid_reaction(stoich, species_indices)

Check if a reaction is valid based on conservation principles.

# Arguments
- `stoich`: Stoichiometry vector
- `species_indices`: Indices of species

# Returns
- Boolean indicating if the reaction is valid
"""
function is_valid_reaction(stoich, species_indices)
    # Skip if not enough species to match our model (Michaelis-Menten)
    if length(stoich) < 4
        return true  # Default to valid for unknown systems
    end
    
    # For Michaelis-Menten, S + E = SE, and SE = E + P
    # This means S should transform to P, E should be conserved, and SE should change in opposition to S/P
    
    # Calculate net change in each species
    s_change = stoich[1]  # S change
    e_change = stoich[2]  # E change
    se_change = stoich[3] # SE change
    p_change = stoich[4]  # P change
    
    # Rule 1: E should be approximately conserved (e_change + se_change = 0)
    e_conserved = abs(e_change + se_change) <= 1
    
    # Rule 2: S + SE should approximately equal P (s_change + se_change + p_change = 0)
    mass_conserved = abs(s_change + se_change + p_change) <= 1
    
    # Skip trivial cases with very small changes
    is_significant = sum(abs.(stoich)) >= 1
    
    # Special case: Allow for the basic MM reactions
    is_mm_reaction = false
    
    # S + E -> SE
    if s_change == -1 && e_change == -1 && se_change == 1 && p_change == 0
        is_mm_reaction = true
    end
    
    # SE -> S + E
    if s_change == 1 && e_change == 1 && se_change == -1 && p_change == 0
        is_mm_reaction = true
    end
    
    # SE -> E + P
    if s_change == 0 && e_change == 1 && se_change == -1 && p_change == 1
        is_mm_reaction = true
    end
    
    return (is_significant && (e_conserved || mass_conserved)) || is_mm_reaction
end

"""
    format_reaction(stoich, species_names)

Format a stoichiometry vector as a human-readable reaction string.

# Arguments
- `stoich`: Stoichiometry vector
- `species_names`: Names of species

# Returns
- Formatted reaction string
"""
function format_reaction(stoich, species_names)
    # Format stoichiometry vector to reaction string
    reactants = ""
    products = ""
    
    for i in 1:min(length(stoich), length(species_names))
        if stoich[i] < 0
            reactants *= "$(abs(stoich[i])) $(species_names[i]) + "
        elseif stoich[i] > 0
            products *= "$(stoich[i]) $(species_names[i]) + "
        end
    end
    
    # Remove trailing " + " if present
    if !isempty(reactants)
        reactants = reactants[1:end-3]
    else
        reactants = "∅"  # Empty set for no reactants
    end
    
    if !isempty(products)
        products = products[1:end-3]
    else
        products = "∅"  # Empty set for no products
    end
    
    return "$reactants → $products"
end

"""
    check_conservation_consistency(stoich, conservation_laws)

Check if a reaction is consistent with identified conservation laws.

# Arguments
- `stoich`: Stoichiometry vector
- `conservation_laws`: List of conservation law vectors

# Returns
- Boolean indicating if the reaction is consistent
- List of consistency scores for each conservation law
"""
function check_conservation_consistency(stoich, conservation_laws)
    consistency_scores = Float64[]
    
    for law in conservation_laws
        # Calculate how much the reaction changes the conserved quantity
        # For a consistent reaction, this should be close to zero
        if length(stoich) <= length(law)
            # Pad stoichiometry vector if needed
            padded_stoich = zeros(length(law))
            padded_stoich[1:length(stoich)] = stoich
            
            # Calculate dot product
            consistency = abs(dot(padded_stoich, law))
        else
            # Truncate conservation law if needed
            truncated_law = law[1:length(stoich)]
            
            # Calculate dot product
            consistency = abs(dot(stoich, truncated_law))
        end
        
        push!(consistency_scores, consistency)
    end
    
    # A reaction is consistent if all conservation laws are respected
    is_consistent = all(score < 1e-8 for score in consistency_scores)
    
    return is_consistent, consistency_scores
end

"""
    cross_validate_reaction_selection(G, reaction_scores, grouped_reactions, selected_states; max_reactions=15)

Perform cross-validation to find optimal number of reactions for reconstruction.

# Arguments
- `G`: Generator matrix
- `reaction_scores`: Dictionary of scores for each reaction
- `grouped_reactions`: Dictionary mapping stoichiometries to reactions
- `selected_states`: List of states in the reduced space
- `max_reactions`: Maximum number of reactions to consider

# Returns
- Optimal number of reactions
- List of cross-validation errors for each reaction set size
"""
function cross_validate_reaction_selection(G, reaction_scores, grouped_reactions, selected_states; max_reactions=15)
    # Sort reactions by score
    sorted_reactions = sort(collect(keys(reaction_scores)), by=s -> reaction_scores[s], rev=true)
    
    # Prepare results
    cv_errors = []
    
    println("Performing cross-validation...")
    
    # Try different reaction set sizes
    for k in 1:min(max_reactions, length(sorted_reactions))
        # Select top k reactions
        selected_k = sorted_reactions[1:k]
        
        # Create reconstruction
        G_recon = create_spectral_reconstruction(G, selected_k, grouped_reactions, selected_states)
        
        # Calculate spectral distance
        distance = calculate_spectral_distance(G, G_recon)
        
        push!(cv_errors, distance)
        println("  $k reactions: spectral error = $(round(distance, digits=5))")
    end
    
    # Find optimal size (minimum error)
    optimal_size = argmin(cv_errors)
    
    println("Optimal reaction set size: $optimal_size")
    
    return optimal_size, cv_errors
end

"""
    select_reactions_by_spectral_properties(G, grouped_reactions, selected_states, species_names)

Select reactions based on their spectral properties and importance for dynamics.

# Arguments
- `G`: Generator matrix
- `grouped_reactions`: Dictionary mapping stoichiometries to reactions
- `selected_states`: List of states in the reduced space
- `species_names`: Names of species

# Returns
- List of selected reaction stoichiometries sorted by importance
- Dictionary of scores for each reaction
"""
function select_reactions_by_spectral_properties(G, grouped_reactions, selected_states, species_names)
    # Calculate eigendecomposition
    λ, V_right = eigen(Matrix(G))
    V_left = inv(V_right)  # Left eigenvectors
    
    # Calculate reaction participation in dynamics
    reaction_participation = calculate_reaction_participation(G, grouped_reactions, selected_states, λ, V_right, V_left, species_names)
    
    # Group eigenvalues by their properties
    mode_groups = group_eigenvalues(λ)
    
    # Calculate weighted scores for each reaction
    reaction_scores = Dict()
    
    for (stoich, mode_scores) in reaction_participation
        # Weight by importance of different mode groups
        slow_modes = mode_groups["slow"]
        fast_modes = mode_groups["fast"]
        osc_modes = mode_groups["oscillatory"]
        
        # Skip if all mode groups are empty
        if isempty(slow_modes) && isempty(fast_modes) && isempty(osc_modes)
            reaction_scores[stoich] = 0.0
            continue
        end
        
        # Calculate scores for each mode group
        slow_score = isempty(slow_modes) ? 0.0 : sum(mode_scores[slow_modes]) / length(slow_modes)
        fast_score = isempty(fast_modes) ? 0.0 : sum(mode_scores[fast_modes]) / length(fast_modes)
        osc_score = isempty(osc_modes) ? 0.0 : sum(mode_scores[osc_modes]) / length(osc_modes)
        
        # Weighted combination of scores (emphasize slow and oscillatory modes)
        combined_score = 0.5 * slow_score + 0.2 * fast_score + 0.3 * osc_score
        
        reaction_scores[stoich] = combined_score
    end
    
    # Sort reactions by their scores
    sorted_reactions = sort(collect(keys(reaction_scores)), by=s -> reaction_scores[s], rev=true)
    
    return sorted_reactions, reaction_scores
end

"""
    group_eigenvalues(λ; slow_threshold=0.1, oscillation_ratio=0.5)

Group eigenvalues by their properties (slow, fast, oscillatory).

# Arguments
- `λ`: Array of eigenvalues
- `slow_threshold`: Threshold for classifying eigenvalues as "slow"
- `oscillation_ratio`: Ratio of imaginary to real part for classifying as "oscillatory"

# Returns
- Dictionary mapping eigenvalue groups to indices
"""
function group_eigenvalues(λ; slow_threshold=0.1, oscillation_ratio=0.5)
    # Initialize groups
    mode_groups = Dict(
        "slow" => Int[],
        "fast" => Int[],
        "oscillatory" => Int[]
    )
    
    # Ignore the first eigenvalue (stationary distribution)
    for i in 2:length(λ)
        # Skip undefined eigenvalues
        if isnan(λ[i]) || isinf(λ[i])
            continue
        end
        
        real_part = abs(real(λ[i]))
        imag_part = abs(imag(λ[i]))
        
        # Check if oscillatory (significant imaginary part)
        if imag_part > oscillation_ratio * real_part
            push!(mode_groups["oscillatory"], i)
        elseif real_part < slow_threshold
            push!(mode_groups["slow"], i)
        else
            push!(mode_groups["fast"], i)
        end
    end
    
    return mode_groups
end

"""
    calculate_reaction_participation(G, grouped_reactions, selected_states, λ, V_right, V_left, species_names)

Calculate how each reaction participates in different dynamic modes.

# Arguments
- `G`: Generator matrix
- `grouped_reactions`: Dictionary mapping stoichiometries to reactions
- `selected_states`: List of states in the reduced space
- `λ`: Eigenvalues of G
- `V_right`: Right eigenvectors of G
- `V_left`: Left eigenvectors of G
- `species_names`: Names of species

# Returns
- Dictionary mapping stoichiometries to arrays of mode participation scores
"""
function calculate_reaction_participation(G, grouped_reactions, selected_states, λ, V_right, V_left, species_names)
    # Initialize reaction participation dictionary
    reaction_participation = Dict()
    
    # For each reaction type (stoichiometry)
    for (stoich, rxns) in grouped_reactions
        # Initialize participation scores for each mode
        mode_scores = zeros(length(λ))
        
        # For each reaction instance
        for r in rxns
            # Get from and to states
            from_state = r.from_state
            to_state = r.to_state
            
            # Find indices in the reduced basis
            from_idx = findfirst(s -> all(s .== from_state), selected_states)
            to_idx = findfirst(s -> all(s .== to_state), selected_states)
            
            if from_idx !== nothing && to_idx !== nothing
                # Get reaction rate from generator
                rate = abs(G[to_idx, from_idx])
                
                # Calculate participation in each mode
                for i in 1:length(λ)
                    # Skip modes with eigenvalues that are too close to zero or undefined
                    if abs(λ[i]) < 1e-10 || isnan(λ[i]) || isinf(λ[i])
                        continue
                    end
                    
                    # Calculate participation using right and left eigenvectors
                    participation = abs(V_right[from_idx, i] * rate * V_left[i, to_idx])
                    mode_scores[i] += participation
                end
            end
        end
        
        # Store participation scores for this reaction
        reaction_participation[stoich] = mode_scores
    end
    
    return reaction_participation
end
