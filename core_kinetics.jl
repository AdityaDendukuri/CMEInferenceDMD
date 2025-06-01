# core_kinetics.jl - SYSTEM-AGNOSTIC KINETICS ANALYSIS (FIXED)
# Mass-action kinetics analysis with proper rate constant estimation

using Statistics
using LinearAlgebra

"""
    calculate_propensity(state, stoichiometry)

Calculate mass-action propensity for a reaction at given state.
"""
function calculate_propensity(state, stoichiometry)
    propensity = 1.0
    
    for i in 1:length(stoichiometry)
        if stoichiometry[i] < 0  # Reactant species
            reactant_count = state[i]  # Use actual molecular count
            stoich_coeff = abs(stoichiometry[i])
            
            if reactant_count < stoich_coeff
                return 0.0  # Not enough molecules
            end
            
            # Mass-action propensity: n! / (n-k)! = n × (n-1) × ... × (n-k+1)
            for j in 0:(stoich_coeff-1)
                propensity *= (reactant_count - j)
            end
        end
    end
    
    return propensity
end

"""
    fit_propensity_model(transitions_by_stoich, stoichiometries)

Fit the full mass-action propensity model: a_j(X) = c_j * h_j(X) using regression.
"""
function fit_propensity_model(transitions_by_stoich, stoichiometries)
    println("Fitting propensity model using regression...")
    
    # Collect regression data
    observed_rates = Float64[]
    propensity_matrix = []
    reaction_labels = String[]
    
    for (stoich_idx, stoich) in enumerate(stoichiometries)
        stoich_tuple = tuple(stoich...)
        
        if haskey(transitions_by_stoich, stoich_tuple)
            transitions = transitions_by_stoich[stoich_tuple]
            
            for transition in transitions
                from_state = transition.from_state
                rate_observed = abs(transition.rate)
                
                # Build propensity vector for this state [h_1(X), h_2(X), ...]
                propensities = zeros(length(stoichiometries))
                propensities[stoich_idx] = calculate_propensity(from_state, stoich)
                
                if propensities[stoich_idx] > 0  # Valid propensity
                    push!(observed_rates, rate_observed)
                    push!(propensity_matrix, propensities)
                    push!(reaction_labels, format_reaction_string(stoich, ["S", "E", "SE", "P"]))
                end
            end
        end
    end
    
    if isempty(observed_rates)
        println("No valid transitions for regression")
        return [], 0.0, "Failed"
    end
    
    # Convert to matrix form: r = H * c + ε
    H = hcat(propensity_matrix...)'  # Design matrix (n_obs × n_reactions)
    r = observed_rates               # Response vector
    
    println("Regression setup: $(size(H, 1)) observations, $(size(H, 2)) reactions")


    # DEBUG: Check regression setup
println("DEBUG REGRESSION SETUP:")
println("  Observed rates range: $(minimum(observed_rates)) to $(maximum(observed_rates))")
println("  Propensity matrix shape: $(size(H))")
println("  Propensity ranges per reaction:")
for j in 1:size(H, 2)
    col = H[:, j]
    non_zero = col[col .> 0]
    if !isempty(non_zero)
        println("    Reaction $j: $(minimum(non_zero)) to $(maximum(non_zero)) ($(length(non_zero)) non-zero)")
    else
        println("    Reaction $j: no data")
    end
end

# Show a few example rows
println("  First 5 regression examples:")
for i in 1:min(5, length(observed_rates))
    println("    rate=$(round(observed_rates[i], digits=6)), propensities=$(H[i, :])")
end
    
    # Solve using non-negative least squares
    c_estimates = solve_nnls(H, r)
    
    # Compute fit diagnostics
    r_predicted = H * c_estimates
    residuals = r - r_predicted
    R_squared = 1 - sum(residuals.^2) / sum((r .- mean(r)).^2)
    rmse = sqrt(mean(residuals.^2))
    
    # Quality assessment
    if R_squared > 0.8
        quality = "Excellent"
    elseif R_squared > 0.6
        quality = "Good"
    elseif R_squared > 0.4
        quality = "Fair"
    else
        quality = "Poor"
    end
    
    println("Regression Results:")
    println("  R² = $(round(R_squared, digits=3))")
    println("  RMSE = $(round(rmse, digits=6))")
    println("  Quality: $quality")
    
    return c_estimates, R_squared, quality
end

"""
    solve_nnls(H, r)

Solve non-negative least squares: min ||Hc - r||² subject to c ≥ 0
Simple implementation using projected gradient descent.
"""
function solve_nnls(H, r)
    n_vars = size(H, 2)
    c = max.(0.0, H \ r)  # Initialize with unconstrained solution, projected to non-negative
    
    # Simple projected gradient descent for non-negative constraint
    learning_rate = 0.01
    max_iterations = 1000
    tolerance = 1e-8
    
    for iter in 1:max_iterations
        # Gradient of ||Hc - r||²
        gradient = H' * (H * c - r)
        
        # Gradient descent step
        c_new = c - learning_rate * gradient
        
        # Project to non-negative orthant
        c_new = max.(0.0, c_new)
        
        # Check convergence
        if norm(c_new - c) < tolerance
            break
        end
        
        c = c_new
    end
    
    return c
end


"""
    analyze_reaction_kinetics_per_reaction(sorted_stoichiometries, stoich_stats, species_names)

Analyze each reaction separately with proper propensity weighting.
"""
function analyze_reaction_kinetics_per_reaction(sorted_stoichiometries, stoich_stats, species_names)
    println("\n=== Mass-Action Kinetics Analysis (Per-Reaction) ===")
    
    rate_constants = []
    
    for stoich_tuple in sorted_stoichiometries
        if haskey(stoich_stats, stoich_tuple) && haskey(stoich_stats[stoich_tuple], :transitions)
            transitions = stoich_stats[stoich_tuple].transitions
            stoich_vec = collect(stoich_tuple)
            reaction_str = format_reaction_string(stoich_vec, species_names)
            
            # Single reaction regression: rate = k * propensity
            rates = Float64[]
            propensities = Float64[]
            
            for transition in transitions
                rate = abs(transition.rate)
                prop = calculate_propensity(transition.from_state, stoich_vec)
                if prop > 0
                    push!(rates, rate)
                    push!(propensities, prop)
                end
            end
            
            if length(rates) >= 3
                # Weighted least squares: k = (Σ w_i * r_i * p_i) / (Σ w_i * p_i²)
                # where w_i = p_i (weight by propensity for better signal)
                weights = propensities
                k_est = sum(weights .* rates .* propensities) / sum(weights .* propensities.^2)
                
                # Compute R² for this reaction
                predicted = k_est .* propensities
                R2 = 1 - sum((rates - predicted).^2) / sum((rates .- mean(rates)).^2)
                
                quality = R2 > 0.8 ? "Excellent" : (R2 > 0.6 ? "Good" : (R2 > 0.4 ? "Fair" : "Poor"))
                
                push!(rate_constants, (
                    stoichiometry = stoich_vec,
                    reaction_string = reaction_str,
                    rate_constant = k_est,
                    quality = quality,
                    observations = length(rates),
                    R_squared = R2
                ))
                
                println("$(reaction_str): k = $(round(k_est, digits=6)) ($(quality), R²=$(round(R2, digits=3)), n=$(length(rates)))")
                println("  Rate range: $(round(minimum(rates), digits=4)) to $(round(maximum(rates), digits=4))")
                println("  Propensity range: $(minimum(propensities)) to $(maximum(propensities))")
            end
        end
    end
    
    return rate_constants
end

"""
    organize_transitions_by_stoichiometry(sorted_stoichiometries, stoich_stats)

Organize transitions by stoichiometry for regression.
"""
function organize_transitions_by_stoichiometry(sorted_stoichiometries, stoich_stats)
    transitions_by_stoich = Dict()
    
    for stoich_tuple in sorted_stoichiometries
        if haskey(stoich_stats, stoich_tuple) && haskey(stoich_stats[stoich_tuple], :transitions)
            transitions_by_stoich[stoich_tuple] = stoich_stats[stoich_tuple].transitions
        end
    end
    
    return transitions_by_stoich
end

"""
    analyze_reaction_kinetics_regression(sorted_stoichiometries, stoich_stats, species_names)

Analyze kinetics using full propensity model regression.
"""
function analyze_reaction_kinetics_regression(sorted_stoichiometries, stoich_stats, species_names)
    println("\n=== Mass-Action Kinetics Regression Analysis ===")
    
    # Organize transitions by stoichiometry
    transitions_by_stoich = organize_transitions_by_stoichiometry(sorted_stoichiometries, stoich_stats)
    
    # Convert stoichiometries to vectors
    stoichiometry_vectors = [collect(s) for s in sorted_stoichiometries]
    
    # Fit the full propensity model
    c_estimates, R_squared, quality = fit_propensity_model(transitions_by_stoich, stoichiometry_vectors)
    
    # Create rate constant results
    rate_constants = []
    
    if !isempty(c_estimates)
        for (i, (stoich_tuple, c_est)) in enumerate(zip(sorted_stoichiometries, c_estimates))
            if c_est > 0
                stoich_vec = collect(stoich_tuple)
                reaction_str = format_reaction_string(stoich_vec, species_names)
                n_obs = haskey(stoich_stats, stoich_tuple) ? stoich_stats[stoich_tuple].count : 0
                
                push!(rate_constants, (
                    stoichiometry = stoich_vec,
                    reaction_string = reaction_str,
                    rate_constant = c_est,
                    quality = quality,
                    observations = n_obs,
                    R_squared = R_squared
                ))
                
                println("$(reaction_str): k = $(round(c_est, digits=6)) ($(quality))")
            end
        end
        
        println("\nRegression Summary:")
        println("  Rate constants estimated: $(length(rate_constants))")
        println("  Overall R² = $(round(R_squared, digits=3))")
        println("  Model quality: $quality")
    else
        println("Regression failed - no rate constants estimated")
    end
    
    return rate_constants
end

"""
    estimate_rate_constant_from_transitions(stoichiometry, transitions)

Estimate rate constant using propensity-weighted regression for better accuracy.
"""
function estimate_rate_constant_from_transitions(stoichiometry, transitions)
    rate_estimates = Float64[]
    propensity_weights = Float64[]
    
    for transition in transitions
        from_state = transition.from_state
        rate_observed = abs(transition.rate)  # Take absolute value for rate constant
        
        # Calculate propensity for this specific state
        propensity = calculate_propensity(from_state, stoichiometry)
        
        if propensity > 0
            k_estimate = rate_observed / propensity
            push!(rate_estimates, k_estimate)
            push!(propensity_weights, propensity)  # Weight by signal strength
            
            # DEBUG: Print first few estimates to see what's happening
            if length(rate_estimates) <= 3
                println("    DEBUG transition: state=$(from_state), rate=$(rate_observed), propensity=$(propensity), k=$(k_estimate), weight=$(propensity)")
            end
        end
    end
    
    if isempty(rate_estimates)
        return NaN, "No Data", 0
    end
    
    # Propensity-weighted average instead of median
    total_weight = sum(propensity_weights)
    k_final = sum(rate_estimates .* propensity_weights) / total_weight
    
    # Quality assessment based on signal strength distribution
    n_estimates = length(rate_estimates)
    if n_estimates >= 3
        # Check if high-propensity states dominate (good signal-to-noise)
        max_weight = maximum(propensity_weights)
        avg_weight = total_weight / n_estimates
        weight_ratio = max_weight / avg_weight
        
        # Also check consistency of estimates
        weighted_variance = sum(propensity_weights .* (rate_estimates .- k_final).^2) / total_weight
        cv = sqrt(weighted_variance) / k_final
        
        if weight_ratio > 5 && cv < 0.3
            quality = "Excellent"  # Dominated by high-signal states with good consistency
        elseif weight_ratio > 3 && cv < 0.5
            quality = "Good"       # Good signal dominance and consistency
        elseif weight_ratio > 2 || cv < 0.7
            quality = "Fair"       # Moderate signal or consistency
        else
            quality = "Poor"       # Low signal dominance and/or poor consistency
        end
    elseif n_estimates >= 2
        quality = "Limited"
    else
        quality = "Single"
    end
    
    # DEBUG: Show weighting effect
    if n_estimates >= 3
        median_k = median(rate_estimates)
        println("    WEIGHTED REGRESSION: median=$(round(median_k, digits=6)), weighted=$(round(k_final, digits=6)), ratio=$(round(k_final/median_k, digits=2))")
    end
    
    return k_final, quality, n_estimates
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
    analyze_reaction_kinetics(sorted_stoichiometries, stoich_stats, species_names)

Analyze kinetics using actual transition data (FIXED VERSION).
"""
function analyze_reaction_kinetics(sorted_stoichiometries, stoich_stats, species_names)
    println("\n=== Mass-Action Kinetics Analysis (FIXED) ===")
    
    rate_constants = []
    
    for stoich_tuple in sorted_stoichiometries
        stoich = collect(stoich_tuple)
        stats = stoich_stats[stoich_tuple]
        
        # Skip if too complex for elementary reaction
        total_change = sum(abs.(stoich))
        if total_change > 3
            continue
        end
        
        # Use actual transitions for rate estimation
        transitions = stats.transitions
        k, quality, n_estimates = estimate_rate_constant_from_transitions(stoich, transitions)
        
        if !isnan(k) && k > 0
            reaction_str = format_reaction_string(stoich, species_names)
            
            push!(rate_constants, (
                stoichiometry = stoich,
                reaction_string = reaction_str,
                rate_constant = k,
                quality = quality,
                observations = stats.count,
                n_estimates = n_estimates
            ))
            
            println("$(reaction_str): k = $(round(k, digits=6)) ($(quality), n=$(n_estimates))")
        end
    end
    
    if isempty(rate_constants)
        println("No rate constants could be estimated")
    else
        println("\nRate constant summary:")
        println("  Estimated constants: $(length(rate_constants))")
        
        # Quality distribution
        quality_counts = Dict()
        for rc in rate_constants
            quality_counts[rc.quality] = get(quality_counts, rc.quality, 0) + 1
        end
        
        for (quality, count) in sort(collect(quality_counts), by=x->x[1])
            println("  $(quality): $count")
        end
    end
    
    return rate_constants
end

"""
    extract_reactions_from_generator_fixed(G, selected_states; threshold=1e-5)

Extract reactions with proper individual transition data (FIXED VERSION).
"""
function extract_reactions_from_generator_fixed(G, selected_states; threshold=1e-5)
    # Extract individual transitions first
    individual_transitions = extract_individual_transitions(G, selected_states, threshold=threshold)
    
    # Group by stoichiometry and compute statistics
    sorted_stoich, stoich_stats = group_transitions_by_stoichiometry(individual_transitions)
    
    # Display results
    println("\nTop reactions found:")
    for (i, stoich) in enumerate(sorted_stoich[1:min(10, end)])
        stats = stoich_stats[stoich]
        reaction_str = format_reaction_string(collect(stoich), get(selected_states, 1, [1,2,3,4]))
        println("$i. $reaction_str (rate: $(round(stats.total_rate, digits=4)), n=$(stats.count))")
    end
    
    return sorted_stoich, stoich_stats
end

"""
    classify_reaction_types(stoichiometries, species_names)

Classify reactions by their stoichiometric patterns.
"""
function classify_reaction_types(stoichiometries, species_names)
    binding_reactions = []      # 2 reactants → 1 product
    dissociation_reactions = [] # 1 reactant → 2 products  
    conversion_reactions = []   # 1 reactant → 1 product
    birth_reactions = []        # ∅ → products
    death_reactions = []        # reactants → ∅
    
    for stoich in stoichiometries
        reactants = sum(s < 0 ? 1 : 0 for s in stoich)
        products = sum(s > 0 ? 1 : 0 for s in stoich)
        
        if reactants == 0 && products > 0
            push!(birth_reactions, stoich)
        elseif reactants > 0 && products == 0
            push!(death_reactions, stoich)
        elseif reactants == 2 && products == 1
            push!(binding_reactions, stoich)
        elseif reactants == 1 && products == 2
            push!(dissociation_reactions, stoich)
        elseif reactants == 1 && products == 1
            push!(conversion_reactions, stoich)
        end
    end
    
    return Dict(
        "binding" => binding_reactions,
        "dissociation" => dissociation_reactions,
        "conversion" => conversion_reactions,
        "birth" => birth_reactions,
        "death" => death_reactions
    )
end

"""
    run_kinetics_analysis(sorted_stoichiometries, stoich_stats, selected_states, species_names)

Run complete kinetics analysis with proper rate estimation (FIXED VERSION).
"""
function run_kinetics_analysis(sorted_stoichiometries, stoich_stats, selected_states, species_names)
    println("\n" * "="^50)
    println("KINETICS ANALYSIS (FIXED)")
    println("="^50)
    
    # Analyze reaction kinetics with real transition data
    #rate_constants = analyze_reaction_kinetics(sorted_stoichiometries, stoich_stats, species_names)
    #rate_constants = analyze_reaction_kinetics_regression(sorted_stoichiometries, stoich_stats, species_names)
    rate_constants = analyze_reaction_kinetics_per_reaction(sorted_stoichiometries, stoich_stats, species_names)
    
    # Classify reaction types
    reaction_types = classify_reaction_types(sorted_stoichiometries, species_names)
    
    # Summary statistics
    if !isempty(rate_constants)
        rates = [rc.rate_constant for rc in rate_constants]
        
        println("\nKinetics Summary:")
        println("  Rate constants computed: $(length(rates))")
        if !isempty(rates)
            println("  Rate range: $(round(minimum(rates), digits=6)) to $(round(maximum(rates), digits=6))")
            println("  Rate span: $(round(maximum(rates)/minimum(rates), digits=1))×")
        end
        
        # Quality distribution
        quality_counts = Dict()
        for rc in rate_constants
            quality_counts[rc.quality] = get(quality_counts, rc.quality, 0) + 1
        end
        
        println("  Quality distribution:")
        for (quality, count) in quality_counts
            println("    $(quality): $count")
        end
    end
    
    # Reaction type summary
    println("\nReaction Type Summary:")
    for (type, reactions) in reaction_types
        if !isempty(reactions)
            println("  $(titlecase(type)) reactions: $(length(reactions))")
        end
    end
    
    results = Dict(
        "rate_constants" => rate_constants,
        "reaction_types" => reaction_types,
        "quality_summary" => isempty(rate_constants) ? Dict() : 
                            Dict(rc.quality => get(Dict(), rc.quality, 0) + 1 for rc in rate_constants)
    )
    
    println("\nKinetics analysis completed!")
    return results
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



println("Core Kinetics Analysis Module (FIXED) Loaded! ⚗️")
println("System-agnostic functions:")
println("  extract_reactions_from_generator_fixed(G, states)")
println("  run_kinetics_analysis_fixed(stoich, stats, states, species)")
println("  estimate_rate_constant_from_transitions(stoich, transitions)")
println("  calculate_propensity(state, stoichiometry)")
println()
println("🔧 Key Fixes Applied:")
println("  ✅ Individual transition extraction")
println("  ✅ Real state-rate pairs")
println("  ✅ Proper propensity calculation")
println("  ✅ No more mock data")
