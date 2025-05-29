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
    estimate_rate_constant_from_transitions(stoichiometry, transitions)

Estimate rate constant from individual transitions with proper propensity calculation.
"""
function estimate_rate_constant_from_transitions(stoichiometry, transitions)
    rate_estimates = Float64[]
    
    for transition in transitions
        from_state = transition.from_state
        rate_observed = abs(transition.rate)  # Take absolute value for rate constant
        
        # Calculate propensity for this specific state
        propensity = calculate_propensity(from_state, stoichiometry)
        
        if propensity > 0
            k_estimate = rate_observed / propensity
            push!(rate_estimates, k_estimate)
            
            # DEBUG: Print first few estimates to see what's happening
            if length(rate_estimates) <= 3
                println("    DEBUG transition: state=$(from_state), rate=$(rate_observed), propensity=$(propensity), k=$(k_estimate)")
            end
        end
    end
    
    if isempty(rate_estimates)
        return NaN, "No Data", 0
    end
    
    # Use median for robustness against outliers
    k_final = median(rate_estimates)
    
    # Assess quality based on consistency
    n_estimates = length(rate_estimates)
    if n_estimates >= 3
        cv = std(rate_estimates) / mean(rate_estimates)
        if cv < 0.2
            quality = "Excellent"
        elseif cv < 0.5
            quality = "Good"
        elseif cv < 1.0
            quality = "Fair"
        else
            quality = "Poor"
        end
    elseif n_estimates >= 2
        quality = "Limited"
    else
        quality = "Single"
    end
    
    return k_final, quality, n_estimates
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
    rate_constants = analyze_reaction_kinetics(sorted_stoichiometries, stoich_stats, species_names)
    
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
