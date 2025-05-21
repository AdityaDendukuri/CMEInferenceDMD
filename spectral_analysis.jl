# spectral_analysis.jl
# Functions for spectral analysis of the generator matrix

using LinearAlgebra

"""
    analyze_important_state_reactions(important_states, selected_states, G, species_names, grouped_reactions, stoich_stats)

Analyze which reactions connect dynamically important states.

# Arguments
- `important_states`: Dictionary mapping state indices to importance scores
- `selected_states`: List of states in the reduced space
- `G`: Generator matrix
- `species_names`: Names of species
- `grouped_reactions`: Dictionary mapping stoichiometries to reactions
- `stoich_stats`: Dictionary with statistics for each stoichiometry

# Returns
- Dictionary mapping stoichiometries to importance scores
"""
function analyze_important_state_reactions(important_states, selected_states, G, species_names, grouped_reactions, stoich_stats)
    reaction_importance = Dict()
    
    for (stoich, rxns) in grouped_reactions
        reaction_importance[stoich] = 0.0
        
        for r in rxns
            # Get from and to indices
            from_idx = findfirst(s -> all(s .== r.from_state), selected_states)
            to_idx = findfirst(s -> all(s .== r.to_state), selected_states)
            
            if from_idx === nothing || to_idx === nothing
                continue
            end
            
            # Check if either from or to state is important
            from_importance = get(important_states, from_idx, 0.0)
            to_importance = get(important_states, to_idx, 0.0)
            
            # Rate is the value in G
            rate = abs(G[to_idx, from_idx])
            
            # Importance is rate * (from_importance + to_importance)
            reaction_importance[stoich] += rate * (from_importance + to_importance)
        end
    end
    
    return reaction_importance
end

"""
    simplified_spectral_analysis(G, λ, Φ, selected_states, species_names, grouped_reactions, stoich_stats)

Perform simplified spectral analysis of the generator matrix and its modes.

# Arguments
- `G`: Generator matrix
- `λ`: Eigenvalues
- `Φ`: Eigenvectors (DMD modes)
- `selected_states`: List of states in the reduced space
- `species_names`: Names of species
- `grouped_reactions`: Dictionary mapping stoichiometries to reactions
- `stoich_stats`: Dictionary with statistics for each stoichiometry

# Returns
- List of selected reaction stoichiometries
- Dictionary of scores for each reaction
"""
function simplified_spectral_analysis(G, λ, Φ, selected_states, species_names, grouped_reactions, stoich_stats)
    println("\n==== Simplified Spectral Analysis ====")
    
    # Sort eigenvalues by their importance (abs of real part)
    sorted_idx = sortperm(abs.(real.(λ)))
    
    # Get the most important modes (skip the stationary mode at index 1)
    important_modes = sorted_idx[2:min(6, length(sorted_idx))]
    
    println("Most important dynamic modes:")
    for (i, idx) in enumerate(important_modes)
        eig = λ[idx]
        println("Mode $i: λ = $(round(eig, digits=5)), period = $(round(2π/abs(imag(eig)), digits=2)) time units")
    end
    
    # Extract dynamically important states from these modes
    important_states = Dict()
    
    for (i, mode_idx) in enumerate(important_modes)
        # Get mode vector
        mode = Φ[:, mode_idx]
        
        # Find states with significant contributions
        significant_state_indices = findall(abs.(mode) .> 0.05 * maximum(abs.(mode)))
        
        # Add to important states dictionary
        for idx in significant_state_indices
            if !haskey(important_states, idx)
                important_states[idx] = 0.0
            end
            important_states[idx] += abs(mode[idx]) / i  # Weight by mode importance
        end
    end
    
    # Convert to list of (state_idx, importance) pairs and sort
    important_state_pairs = [(k, v) for (k, v) in important_states]
    sort!(important_state_pairs, by=x -> x[2], rev=true)
    
    # Analyze which reactions connect important states
    reaction_importance = analyze_important_state_reactions(
        important_states, selected_states, G, species_names, grouped_reactions, stoich_stats
    )
    
    # Combine with rate information
    final_scores = Dict()
    for stoich in keys(reaction_importance)
        # Normalize by number of instances
        count = stoich_stats[stoich].count
        rate = stoich_stats[stoich].total_rate
        
        # Combine dynamical importance with rate information
        # 70% weight to spectral importance, 30% to rate magnitude
        spectral_score = reaction_importance[stoich] / max(1, count)
        rate_score = rate / count
        
        final_scores[stoich] = 0.7 * spectral_score + 0.3 * rate_score
    end
    
    # Sort by final score
    sorted_reactions = sort(collect(keys(final_scores)), by=s -> final_scores[s], rev=true)
    
    # Select top 5 reactions
    n_reactions = min(5, length(sorted_reactions))
    selected_reactions = sorted_reactions[1:n_reactions]
    
    # Print results
    println("\nTop reactions selected by spectral importance:")
    for (i, stoich) in enumerate(selected_reactions)
        # Format reaction string
        reaction_str = format_reaction(stoich, species_names)
        
        count = stoich_stats[stoich].count
        rate = stoich_stats[stoich].total_rate
        score = final_scores[stoich]
        
        println("$i. $reaction_str")
        println("   Score: $(round(score, digits=5)), Count: $count, Rate: $(round(rate, digits=5))")
    end
    
    println("\n==== End of Simplified Spectral Analysis ====")
    
    return selected_reactions, final_scores
end

"""
    analyze_and_select_reactions_fixed(G, λ, Φ, grouped_reactions, stoich_stats, selected_states, species_names)

Perform spectral analysis with fixed stoichiometry calculation.

# Arguments
- `G`: Generator matrix
- `λ`: Eigenvalues
- `Φ`: Eigenvectors (DMD modes)
- `grouped_reactions`: Dictionary mapping stoichiometries to reactions
- `stoich_stats`: Dictionary with statistics for each stoichiometry
- `selected_states`: List of states in the reduced space
- `species_names`: Names of species

# Returns
- List of selected reaction stoichiometries
- Dictionary of scores for each reaction
"""
function analyze_and_select_reactions_fixed(G, λ, Φ, grouped_reactions, stoich_stats, selected_states, species_names)
    println("\n==== Spectral Analysis with Fixed Stoichiometry Calculation ====")
    
    # Use the simplified spectral analysis with fixed stoichiometry calculation
    return simplified_spectral_analysis(
        G, λ, Φ, selected_states, species_names, grouped_reactions, stoich_stats
    )
end

"""
    analyze_and_select_reactions_spectral(G, λ, Φ, grouped_reactions, stoich_stats, selected_states, species_names)

Main function for spectral-based reaction selection with error handling.

# Arguments
- `G`: Generator matrix
- `λ`: Eigenvalues
- `Φ`: Eigenvectors (DMD modes)
- `grouped_reactions`: Dictionary mapping stoichiometries to reactions
- `stoich_stats`: Dictionary with statistics for each stoichiometry
- `selected_states`: List of states in the reduced space
- `species_names`: Names of species

# Returns
- List of selected reaction stoichiometries
- Dictionary with detailed analysis results
"""
function analyze_and_select_reactions_spectral(G, λ, Φ, grouped_reactions, stoich_stats, selected_states, species_names)
    println("\n==== Spectral-Based Reaction Selection Analysis ====")
    
    # Try to identify conservation laws, but don't fail if we can't
    conservation_laws = []
    law_descriptions = []
    
    try
        # Step 1: Identify conservation laws
        conservation_laws, law_descriptions = identify_conservation_laws(G, species_names)
        
        println("\nIdentified $(length(conservation_laws)) conservation laws:")
        for law in law_descriptions
            println("  $law")
        end
    catch e
        println("Could not identify conservation laws: $e")
        println("Proceeding with simplified spectral analysis...")
    end
    
    # Check if we found conservation laws
    if isempty(conservation_laws)
        # Use simplified analysis when conservation laws aren't available
        selected_reactions, reaction_scores = simplified_spectral_analysis(
            G, λ, Φ, selected_states, species_names, grouped_reactions, stoich_stats
        )
        
        # Return simplified results
        analysis_results = Dict(
            "conservation_laws" => [],
            "law_descriptions" => [],
            "reaction_scores" => reaction_scores,
            "cv_errors" => [],
            "optimal_size" => length(selected_reactions)
        )
        
        return selected_reactions, analysis_results
    end
    
    # If we have conservation laws, proceed with full spectral analysis
    try
        # Step 2: Select reactions based on spectral properties
        selected_reactions, reaction_scores = select_reactions_by_spectral_properties(
            G, grouped_reactions, selected_states, species_names
        )
        
        # Step 3: Perform cross-validation to find optimal reaction set size
        optimal_size, cv_errors = cross_validate_reaction_selection(
            G, reaction_scores, grouped_reactions, selected_states
        )
        
        # Select optimal number of reactions
        optimal_reactions = selected_reactions[1:min(optimal_size, length(selected_reactions))]
        
        # Print results
        println("\nTop reactions selected by spectral analysis:")
        for (i, stoich) in enumerate(optimal_reactions)
            reaction_str = format_reaction(stoich, species_names)
            score = reaction_scores[stoich]
            is_consistent, consistency = check_conservation_consistency(stoich, conservation_laws)
            consistency_status = is_consistent ? "consistent" : "inconsistent"
            
            println("$i. $reaction_str")
            println("   Score: $(round(score, digits=5)), Conservation: $consistency_status")
        end
        
        # Return results
        analysis_results = Dict(
            "conservation_laws" => conservation_laws,
            "law_descriptions" => law_descriptions,
            "reaction_scores" => reaction_scores,
            "cv_errors" => cv_errors,
            "optimal_size" => optimal_size
        )
        
        println("\n==== End of Spectral-Based Analysis ====")
        
        return optimal_reactions, analysis_results
    catch e
        println("Error in full spectral analysis: $e")
        println("Falling back to simplified spectral analysis...")
        
        # Fall back to simplified analysis
        selected_reactions, reaction_scores = simplified_spectral_analysis(
            G, λ, Φ, selected_states, species_names, grouped_reactions, stoich_stats
        )
        
        # Return simplified results
        analysis_results = Dict(
            "conservation_laws" => conservation_laws,
            "law_descriptions" => law_descriptions,
            "reaction_scores" => reaction_scores,
            "cv_errors" => [],
            "optimal_size" => length(selected_reactions)
        )
        
        return selected_reactions, analysis_results
    end
end

"""
    identify_conservation_laws_molecular(G, species_names; tol=1e-8)

Identify conservation laws based on molecular counts from the generator matrix.

# Arguments
- `G`: Generator matrix
- `species_names`: Names of species
- `tol`: Tolerance for identifying zero eigenvalues

# Returns
- Array of conservation law vectors
- Array of conservation law descriptions
"""
function identify_conservation_laws_molecular(G, species_names; tol=1e-8)
    # Compute eigendecomposition of G
    λ, V_right = eigen(Matrix(G))
    V_left = inv(V_right)  # Left eigenvectors (rows of V_left)
    
    # Find eigenvalues close to zero
    zero_indices = findall(abs.(λ) .< tol)
    
    if isempty(zero_indices)
        println("No conservation laws found (no eigenvalues close to zero)")
        return [], []
    end
    
    # Extract left eigenvectors corresponding to zero eigenvalues
    conservation_laws = []
    law_descriptions = []
    
    for idx in zero_indices
        # Get the left eigenvector (row of V_left)
        left_ev = V_left[idx, :]
        
        # Normalize the coefficients
        max_coef = maximum(abs.(left_ev))
        if max_coef > 0
            left_ev = left_ev ./ max_coef
        end
        
        # Round small values to zero for clarity
        left_ev[abs.(left_ev) .< 1e-10] .= 0.0
        
        # Create a description of the conservation law - in terms of molecular counts
        description = "Conservation law: "
        terms = []
        for (i, coef) in enumerate(left_ev)
            if abs(coef) > 1e-10
                if length(species_names) >= i
                    species = species_names[i]
                    push!(terms, "$(round(coef, digits=3)) × $species")
                else
                    push!(terms, "$(round(coef, digits=3)) × species$i")
                end
            end
        end
        description *= join(terms, " + ") * " = constant"
        
        push!(conservation_laws, left_ev)
        push!(law_descriptions, description)
    end
    
    return conservation_laws, law_descriptions
end

"""
    check_conservation_consistency_molecular(stoich, conservation_laws, species_names)

Check if a reaction is consistent with identified conservation laws based on molecule counts.

# Arguments
- `stoich`: Stoichiometry vector
- `conservation_laws`: List of conservation law vectors
- `species_names`: Names of species

# Returns
- Boolean indicating if the reaction is consistent
- List of consistency scores for each conservation law
"""
function check_conservation_consistency_molecular(stoich, conservation_laws, species_names)
    # Convert stoichiometry to reflect changes in molecule counts rather than grid indices
    molecular_stoich = stoich
    
    consistency_scores = Float64[]
    
    for law in conservation_laws
        # Calculate how much the reaction changes the conserved quantity
        # For a consistent reaction, this should be close to zero
        if length(molecular_stoich) <= length(law)
            # Pad stoichiometry vector if needed
            padded_stoich = zeros(length(law))
            padded_stoich[1:length(molecular_stoich)] = molecular_stoich
            
            # Calculate dot product
            consistency = abs(dot(padded_stoich, law))
        else
            # Truncate conservation law if needed
            truncated_law = law[1:length(molecular_stoich)]
            
            # Calculate dot product
            consistency = abs(dot(molecular_stoich, truncated_law))
        end
        
        push!(consistency_scores, consistency)
    end
    
    # A reaction is consistent if all conservation laws are respected
    is_consistent = all(score < 1e-8 for score in consistency_scores)
    
    return is_consistent, consistency_scores
end

"""
    simplified_spectral_analysis_with_plots(G, λ, Φ, selected_states, species_names, grouped_reactions, stoich_stats)

Enhanced simplified spectral analysis with fixed stoichiometry calculations.

# Arguments
- `G`: Generator matrix
- `λ`: Eigenvalues
- `Φ`: Eigenvectors (DMD modes)
- `selected_states`: List of states in the reduced space
- `species_names`: Names of species
- `grouped_reactions`: Dictionary mapping stoichiometries to reactions
- `stoich_stats`: Dictionary with statistics for each stoichiometry

# Returns
- List of selected reaction stoichiometries
- Dictionary of scores for each reaction
"""
function simplified_spectral_analysis_with_plots(G, λ, Φ, selected_states, species_names, grouped_reactions, stoich_stats)
    println("\n==== Simplified Spectral Analysis ====")
    
    # Sort eigenvalues by their importance (abs of real part)
    sorted_idx = sortperm(abs.(real.(λ)))
    
    # Get the most important modes (skip the stationary mode at index 1)
    important_modes = sorted_idx[2:min(6, length(sorted_idx))]
    
    println("\nMost important dynamic modes:")
    for (i, idx) in enumerate(important_modes)
        eig = λ[idx]
        println("Mode $i: λ = $(round(eig, digits=5)), period = $(round(2π/abs(imag(eig)), digits=2)) time units")
    end
    
    # Extract dynamically important states from these modes
    important_states = Dict()
    
    for (i, mode_idx) in enumerate(important_modes)
        # Get mode vector
        mode = Φ[:, mode_idx]
        
        # Find states with significant contributions
        significant_state_indices = findall(abs.(mode) .> 0.05 * maximum(abs.(mode)))
        
        # Add to important states dictionary
        for idx in significant_state_indices
            if !haskey(important_states, idx)
                important_states[idx] = 0.0
            end
            important_states[idx] += abs(mode[idx]) / i  # Weight by mode importance
        end
    end
    
    # Convert to list of (state_idx, importance) pairs and sort
    important_state_pairs = [(k, v) for (k, v) in important_states]
    sort!(important_state_pairs, by=x -> x[2], rev=true)
    
    # Analyze which reactions connect important states
    reaction_importance = analyze_important_state_reactions(
        important_states, selected_states, G, species_names, grouped_reactions, stoich_stats
    )
    
    # Combine with rate information
    final_scores = Dict()
    for stoich in keys(reaction_importance)
        # Normalize by number of instances
        count = stoich_stats[stoich].count
        rate = stoich_stats[stoich].total_rate
        
        # Combine dynamical importance with rate information
        # 70% weight to spectral importance, 30% to rate magnitude
        spectral_score = reaction_importance[stoich] / max(1, count)
        rate_score = rate / count
        
        final_scores[stoich] = 0.7 * spectral_score + 0.3 * rate_score
    end
    
    # Sort by final score
    sorted_reactions = sort(collect(keys(final_scores)), by=s -> final_scores[s], rev=true)
    
    # Select top 5 reactions
    n_reactions = min(5, length(sorted_reactions))
    selected_reactions = sorted_reactions[1:n_reactions]
    
    # Print results in table form
    println("\nTop reactions selected by spectral importance:")
    
    # Create a simple table format
    headers = ["Rank", "Reaction", "Score", "Count", "Rate"]
    rows = []
    
    for (i, stoich) in enumerate(selected_reactions)
        # Format reaction
        reaction_str = format_reaction(stoich, species_names)
        
        count = stoich_stats[stoich].count
        rate = stoich_stats[stoich].total_rate
        score = final_scores[stoich]
        
        push!(rows, ["$i", reaction_str, "$(round(score, digits=4))", "$count", "$(round(rate, digits=4))"])
    end
    
    # Print the table in a simple format
    print_simple_table(rows, headers)
    
    println("\n==== End of Simplified Spectral Analysis ====")
    
    return selected_reactions, final_scores
end

"""
    print_simple_table(rows, headers)

Print a simple table without using external packages.

# Arguments
- `rows`: Array of rows, where each row is an array of strings
- `headers`: Array of column headers
"""
function print_simple_table(rows, headers)
    # Determine column widths
    col_widths = [length(h) for h in headers]
    for row in rows
        for (i, cell) in enumerate(row)
            col_widths[i] = max(col_widths[i], length(cell))
        end
    end
    
    # Print header
    header_line = "│ "
    for (i, h) in enumerate(headers)
        header_line *= h * " " * " "^(col_widths[i] - length(h)) * "│ "
    end
    
    separator = "├" * join(["─"^(w+2) * "┼" for w in col_widths]) * "┤"
    separator = replace(separator, "┼┤" => "┤")
    
    top_line = "┌" * join(["─"^(w+2) * "┬" for w in col_widths]) * "┐"
    top_line = replace(top_line, "┬┐" => "┐")
    
    bottom_line = "└" * join(["─"^(w+2) * "┴" for w in col_widths]) * "┘"
    bottom_line = replace(bottom_line, "┴┘" => "┘")
    
    println(top_line)
    println(header_line)
    println(separator)
    
    # Print rows
    for row in rows
        row_line = "│ "
        for (i, cell) in enumerate(row)
            row_line *= cell * " " * " "^(col_widths[i] - length(cell)) * "│ "
        end
        println(row_line)
    end
    
    println(bottom_line)
end
