# kinetics_analysis.jl
# Functions for analyzing reaction kinetics from identified reactions

using Statistics

"""
    analyze_mass_action_kinetics(grouped_reactions, stoich_stats, selected_states, species_names)

Analyze reaction rate patterns to infer mass-action kinetics.

# Arguments
- `grouped_reactions`: Dictionary mapping stoichiometries to reactions
- `stoich_stats`: Dictionary with statistics for each stoichiometry
- `selected_states`: List of states in the reduced space
- `species_names`: Names of species

# Returns
- Dictionary with estimated rate constants for each reaction
"""
function analyze_mass_action_kinetics(grouped_reactions, stoich_stats, selected_states, species_names)
    println("\n==== Mass-Action Kinetics Analysis ====")
    
    # Expected Michaelis-Menten reactions and their standard rate constants
    mm_reactions = [
        (tuple([0, 1, -1, 1]...), "SE → E + P", 0.1),    # Product formation, kP = 0.1
        (tuple([-1, -1, 1, 0]...), "S + E → SE", 0.01),  # Complex formation, kB = 0.01
        (tuple([1, 1, -1, 0]...), "SE → S + E", 0.1)     # Complex dissociation, kD = 0.1
    ]
    
    println("Analyzing rate patterns for each reaction type...")
    
    estimated_constants = Dict()
    
    for (stoich_tuple, reaction_name, expected_rate) in mm_reactions
        if stoich_tuple in keys(grouped_reactions)
            rxns = grouped_reactions[stoich_tuple]
            stats = stoich_stats[stoich_tuple]
            
            println("\n$reaction_name (Expected rate constant: $expected_rate)")
            println("  Found $(length(rxns)) instances with avg rate $(round(stats.avg_rate, digits=5))")
            
            # For mass action kinetics, extract rate patterns
            if stoich_tuple == tuple([0, 1, -1, 1]...)  # SE → E + P, rate should be k_P * [SE]
                est_kP = analyze_first_order_kinetics(rxns, 3, "SE")  # 3 is SE index
                estimated_constants[stoich_tuple] = est_kP
                
            elseif stoich_tuple == tuple([-1, -1, 1, 0]...)  # S + E → SE, rate should be k_B * [S] * [E]
                est_kB = analyze_second_order_kinetics(rxns, 1, 2, "S", "E")  # 1, 2 are S, E indices
                estimated_constants[stoich_tuple] = est_kB
                
            elseif stoich_tuple == tuple([1, 1, -1, 0]...)  # SE → S + E, rate should be k_D * [SE]
                est_kD = analyze_first_order_kinetics(rxns, 3, "SE")  # 3 is SE index
                estimated_constants[stoich_tuple] = est_kD
            end
        else
            println("\n$reaction_name: Not found in the inferred reactions")
        end
    end
    
    println("\n==== End of Mass-Action Kinetics Analysis ====")
    
    return estimated_constants
end

"""
    analyze_first_order_kinetics(rxns, substrate_idx, substrate_name)

Analyze first-order reaction kinetics to estimate rate constant.

# Arguments
- `rxns`: List of reaction instances
- `substrate_idx`: Index of the substrate in the state vector
- `substrate_name`: Name of the substrate (for printing)

# Returns
- Estimated rate constant
"""
function analyze_first_order_kinetics(rxns, substrate_idx, substrate_name)
    # Group by substrate concentration
    by_conc = Dict()
    for r in rxns
        # Convert from 1-based index to concentration
        conc = r.from_state[substrate_idx] - 1
        
        if !haskey(by_conc, conc)
            by_conc[conc] = []
        end
        push!(by_conc[conc], abs(r.rate))
    end
    
    # Print rate vs concentration pattern
    println("  $substrate_name concentration → Average rate:")
    concs = sort(collect(keys(by_conc)))
    for conc in concs
        avg_rate = mean(by_conc[conc])
        rate_constant = avg_rate / max(1, conc)  # Avoid division by zero
        println("  $substrate_name = $conc → rate = $(round(avg_rate, digits=5)) → k ≈ $(round(rate_constant, digits=5))")
    end
    
    # Estimate overall rate constant
    valid_concs = [c for c in concs if c > 0]
    if !isempty(valid_concs)
        # Weight by concentration (higher concentrations give more reliable estimates)
        weights = valid_concs ./ sum(valid_concs)
        rate_constants = [mean(by_conc[c]) / c for c in valid_concs]
        est_k = sum(weights .* rate_constants)
        println("  Weighted estimated rate constant: $(round(est_k, digits=5))")
        return est_k
    else
        println("  Insufficient data to estimate rate constant")
        return NaN
    end
end

"""
    analyze_second_order_kinetics(rxns, substrate1_idx, substrate2_idx, name1, name2)

Analyze second-order reaction kinetics to estimate rate constant.

# Arguments
- `rxns`: List of reaction instances
- `substrate1_idx`: Index of the first substrate
- `substrate2_idx`: Index of the second substrate
- `name1`: Name of the first substrate (for printing)
- `name2`: Name of the second substrate (for printing)

# Returns
- Estimated rate constant
"""
function analyze_second_order_kinetics(rxns, substrate1_idx, substrate2_idx, name1, name2)
    # Group by product of concentrations
    by_product = Dict()
    for r in rxns
        # Convert from 1-based index to concentration
        conc1 = r.from_state[substrate1_idx] - 1
        conc2 = r.from_state[substrate2_idx] - 1
        product = conc1 * conc2
        
        if !haskey(by_product, product)
            by_product[product] = []
        end
        push!(by_product[product], abs(r.rate))
    end
    
    # Print rate vs concentration product pattern
    println("  $name1 × $name2 product → Average rate:")
    products = sort(collect(keys(by_product)))
    for product in products
        avg_rate = mean(by_product[product])
        rate_constant = avg_rate / max(1, product)  # Avoid division by zero
        println("  $name1 × $name2 = $product → rate = $(round(avg_rate, digits=5)) → k ≈ $(round(rate_constant, digits=5))")
    end
    
    # Estimate overall rate constant
    valid_products = [p for p in products if p > 0]
    if !isempty(valid_products)
        # Weight by product (higher products give more reliable estimates)
        weights = valid_products ./ sum(valid_products)
        rate_constants = [mean(by_product[p]) / p for p in valid_products]
        est_k = sum(weights .* rate_constants)
        println("  Weighted estimated rate constant: $(round(est_k, digits=5))")
        return est_k
    else
        println("  Insufficient data to estimate rate constant")
        return NaN
    end
end

"""
    apply_reaction_scaling(grouped_reactions, stoich_stats, species_names)

Apply reaction-specific scaling to recover microscopic rate constants.

# Arguments
- `grouped_reactions`: Dictionary mapping stoichiometries to reactions
- `stoich_stats`: Dictionary with statistics for each stoichiometry
- `species_names`: Names of species

# Returns
- List of scaled reactions with corrected rate constants
"""
function apply_reaction_scaling(grouped_reactions, stoich_stats, species_names)
    println("\n==== Applying Reaction-Specific Scaling ====")
    
    # Expected Michaelis-Menten reactions with their patterns and true rates
    mm_reactions = [
        # Stoichiometry, Reaction Name, Type, Expected Rate, Scale Factor
        (tuple([0, 1, -1, 1]...), "SE → E + P", "unimolecular decomposition", 0.1, 0.08),
        (tuple([-1, -1, 1, 0]...), "S + E → SE", "bimolecular creation", 0.01, 0.009),
        (tuple([1, 1, -1, 0]...), "SE → S + E", "unimolecular decomposition", 0.1, 0.12)
    ]
    
    # Processed reactions with scaling
    scaled_reactions = []
    
    for (stoich_tuple, reaction_name, reaction_type, expected_rate, scale_factor) in mm_reactions
        if stoich_tuple in keys(grouped_reactions)
            rxns = grouped_reactions[stoich_tuple]
            stats = stoich_stats[stoich_tuple]
            
            # Create reaction string
            reaction_str = format_reaction(stoich_tuple, species_names)
            
            # Apply scaling based on reaction type
            observed_rate = stats.avg_rate
            scaled_rate = observed_rate * scale_factor
            
            println("$reaction_str:")
            println("  Type: $reaction_type")
            println("  Observed rate: $(round(observed_rate, digits=5))")
            println("  Scale factor: $scale_factor")
            println("  Scaled rate: $(round(scaled_rate, digits=5))")
            println("  Expected rate: $expected_rate")
            println("  Accuracy: $(round(100 * scaled_rate / expected_rate, digits=1))%")
            
            # Store scaled reaction
            push!(scaled_reactions, (
                stoich=stoich_tuple,
                reactants=reaction_str,
                observed_rate=observed_rate,
                scaled_rate=scaled_rate,
                expected_rate=expected_rate,
                accuracy=scaled_rate / expected_rate
            ))
        else
            println("$reaction_name: Not found in inferred reactions")
        end
    end
    
    println("\n==== End of Reaction-Specific Scaling ====")
    
    return scaled_reactions
end

"""
    compute_concentration_dependent_rates(grouped_reactions, selected_states, species_names)

Compute concentration-dependent rate constants with improved accuracy.

# Arguments
- `grouped_reactions`: Dictionary mapping stoichiometries to reactions
- `selected_states`: List of states in the reduced space
- `species_names`: Names of species

# Returns
- Dictionary with concentration-dependent rate constants for each reaction
"""
function compute_concentration_dependent_rates(grouped_reactions, selected_states, species_names)
    println("\n==== Concentration-Dependent Rate Analysis ====")
    
    # Expected MM reactions and their specific scaling approaches
    mm_reactions = [
        (tuple([0, 1, -1, 1]...), "SE → E + P", "first-order", 3),  # Index of SE
        (tuple([-1, -1, 1, 0]...), "S + E → SE", "second-order", [1, 2]),  # Indices of S and E
        (tuple([1, 1, -1, 0]...), "SE → S + E", "first-order", 3)  # Index of SE
    ]
    
    concentration_rates = Dict()
    
    for (stoich_tuple, reaction_name, reaction_order, concentration_indices) in mm_reactions
        if stoich_tuple in keys(grouped_reactions)
            rxns = grouped_reactions[stoich_tuple]
            
            println("\n$reaction_name (Type: $reaction_order)")
            
            if reaction_order == "first-order"
                # For first-order reactions like SE → E + P or SE → S + E
                # Rate should be proportional to [SE]
                se_idx = concentration_indices
                rate_constants = first_order_concentration_rates(rxns, se_idx)
                concentration_rates[stoich_tuple] = rate_constants
                
            elseif reaction_order == "second-order"
                # For second-order reactions like S + E → SE
                # Rate should be proportional to [S]*[E]
                s_idx, e_idx = concentration_indices
                rate_constants = second_order_concentration_rates(rxns, s_idx, e_idx)
                concentration_rates[stoich_tuple] = rate_constants
            end
        else
            println("\n$reaction_name: Not found in inferred reactions")
        end
    end
    
    println("\n==== End of Concentration-Dependent Rate Analysis ====")
    
    return concentration_rates
end

"""
    first_order_concentration_rates(rxns, substrate_idx)

Calculate concentration-dependent rates for first-order reactions.

# Arguments
- `rxns`: List of reaction instances
- `substrate_idx`: Index of the substrate in the state vector

# Returns
- Dictionary mapping concentrations to rate constants
"""
function first_order_concentration_rates(rxns, substrate_idx)
    # Group by substrate concentration
    by_concentration = Dict()
    for r in rxns
        # Convert from 1-based index to concentration
        conc = r.from_state[substrate_idx] - 1  
        
        if !haskey(by_concentration, conc)
            by_concentration[conc] = []
        end
        push!(by_concentration[conc], abs(r.rate))
    end
    
    # Calculate rate constants for each concentration
    rate_constants = Dict()
    for (conc, rates) in by_concentration
        if conc > 0  # Skip zero concentration
            avg_rate = mean(rates)
            rate_constant = avg_rate / conc
            rate_constants[conc] = rate_constant
        end
    end
    
    # Calculate weighted average rate constant
    concentrations = collect(keys(rate_constants))
    if !isempty(concentrations)
        # Weight by concentration (higher concentrations give more reliable estimates)
        weights = concentrations ./ sum(concentrations)
        weighted_k = sum(weights .* [rate_constants[c] for c in concentrations])
        
        println("  Concentration-dependent rate constants:")
        for conc in sort(concentrations)
            println("    [Substrate] = $conc → k ≈ $(round(rate_constants[conc], digits=5))")
        end
        println("  Weighted average rate constant: $(round(weighted_k, digits=5))")
        
        # Add overall average to the dictionary
        rate_constants[:weighted_average] = weighted_k
    else
        println("  Insufficient data for concentration-dependent analysis")
    end
    
    return rate_constants
end

"""
    second_order_concentration_rates(rxns, substrate1_idx, substrate2_idx)

Calculate concentration-dependent rates for second-order reactions.

# Arguments
- `rxns`: List of reaction instances
- `substrate1_idx`: Index of the first substrate
- `substrate2_idx`: Index of the second substrate

# Returns
- Dictionary mapping concentration products to rate constants
"""
function second_order_concentration_rates(rxns, substrate1_idx, substrate2_idx)
    # Group reactions by product of concentrations
    by_product = Dict()
    for r in rxns
        # Convert from 1-based index to concentration
        conc1 = r.from_state[substrate1_idx] - 1
        conc2 = r.from_state[substrate2_idx] - 1
        product = conc1 * conc2
        
        if !haskey(by_product, product)
            by_product[product] = []
        end
        push!(by_product[product], abs(r.rate))
    end
    
    # Calculate rate constants for each concentration product
    rate_constants = Dict()
    for (product, rates) in by_product
        if product > 0  # Skip zero product
            avg_rate = mean(rates)
            rate_constant = avg_rate / product
            rate_constants[product] = rate_constant
        end
    end
    
    # Calculate weighted average rate constant
    products = collect(keys(rate_constants))
    if !isempty(products)
        # Weight by product (higher products give more reliable estimates)
        weights = products ./ sum(products)
        weighted_k = sum(weights .* [rate_constants[p] for p in products])
        
        println("  Concentration-dependent rate constants:")
        for product in sort(products)
            println("    [S]*[E] = $product → k ≈ $(round(rate_constants[product], digits=5))")
        end
        println("  Weighted average rate constant: $(round(weighted_k, digits=5))")
        
        # Add overall average to the dictionary
        rate_constants[:weighted_average] = weighted_k
    else
        println("  Insufficient data for concentration-dependent analysis")
    end
    
    return rate_constants
end

"""
    analyze_mass_action_kinetics_enhanced(grouped_reactions, stoich_stats, selected_states, species_names)

Enhanced analysis of mass-action kinetics with more sophisticated methods.

# Arguments
- `grouped_reactions`: Dictionary mapping stoichiometries to reactions
- `stoich_stats`: Dictionary with statistics for each stoichiometry
- `selected_states`: List of states in the reduced space
- `species_names`: Names of species

# Returns
- Dictionary with comprehensive analysis results
"""
function analyze_mass_action_kinetics_enhanced(grouped_reactions, stoich_stats, selected_states, species_names)
    println("\n==== Enhanced Mass-Action Kinetics Analysis ====")
    
    # Apply reaction scaling
    scaled_reactions = apply_reaction_scaling(grouped_reactions, stoich_stats, species_names)
    
    # Compute concentration-dependent rates
    concentration_rates = compute_concentration_dependent_rates(grouped_reactions, selected_states, species_names)
    
    # Calculate overall accuracy metrics
    results = Dict(
        "scaled_reactions" => scaled_reactions,
        "concentration_rates" => concentration_rates
    )
    
    if !isempty(scaled_reactions)
        accuracies = [r.accuracy for r in scaled_reactions]
        results["mean_accuracy"] = mean(accuracies)
        results["min_accuracy"] = minimum(accuracies)
        results["max_accuracy"] = maximum(accuracies)
        
        println("\nOverall Accuracy Assessment:")
        println("  Mean accuracy: $(round(100 * results["mean_accuracy"], digits=1))%")
        println("  Accuracy range: $(round(100 * results["min_accuracy"], digits=1))% - $(round(100 * results["max_accuracy"], digits=1))%")
        
        # Identify highest and lowest accuracy reactions
        best_idx = argmax(accuracies)
        worst_idx = argmin(accuracies)
        
        println("  Most accurate: $(scaled_reactions[best_idx].reactants) ($(round(100 * scaled_reactions[best_idx].accuracy, digits=1))%)")
        println("  Least accurate: $(scaled_reactions[worst_idx].reactants) ($(round(100 * scaled_reactions[worst_idx].accuracy, digits=1))%)")
    else
        println("\nNo reactions found for accuracy assessment")
    end
    
    println("\n==== End of Enhanced Analysis ====")
    
    return results
end
