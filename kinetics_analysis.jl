# kinetics_analysis.jl - KINETICS ANALYSIS MODULE
# Rate constant estimation and kinetic law determination

using Statistics
using LinearAlgebra

"""
    analyze_first_order_kinetics(reactions, substrate_idx, substrate_name)

Analyze first-order reaction kinetics (rate ∝ [substrate]).
"""
function analyze_first_order_kinetics(reactions, substrate_idx, substrate_name)
    println("  Analyzing $substrate_name → products (first-order)")
    
    # Group reactions by substrate concentration
    conc_groups = Dict()
    for r in reactions
        conc = max(0, r.from_state[substrate_idx] - 1)  # Convert to molecular count
        if conc > 0  # Only consider non-zero concentrations
            if !haskey(conc_groups, conc)
                conc_groups[conc] = []
            end
            push!(conc_groups[conc], abs(r.rate))
        end
    end
    
    if isempty(conc_groups)
        println("    No valid data for kinetic analysis")
        return NaN, NaN
    end
    
    # Calculate rate constants for each concentration
    rate_constants = Float64[]
    concentrations = Float64[]
    avg_rates = Float64[]
    
    for (conc, rates) in conc_groups
        avg_rate = mean(rates)
        rate_constant = avg_rate / conc  # k = rate / [substrate]
        
        push!(concentrations, Float64(conc))
        push!(avg_rates, Float64(avg_rate))
        push!(rate_constants, Float64(rate_constant))
    end
    
    # Estimate overall rate constant
    estimated_k = mean(rate_constants)
    k_std = std(rate_constants)
    
    # Calculate R² for linearity check
    if length(concentrations) > 2
        # Linear regression: rate = k * concentration
        X = hcat(ones(Float64, length(concentrations)), concentrations)
        β = X \ avg_rates
        predicted_rates = X * β
        ss_res = sum((avg_rates - predicted_rates).^2)
        ss_tot = sum((avg_rates .- mean(avg_rates)).^2)
        r_squared = 1 - ss_res / ss_tot
        
        println("    Rate constant k ≈ $(round(estimated_k, digits=5)) ± $(round(k_std, digits=5))")
        println("    Linearity R² = $(round(r_squared, digits=3))")
        
        if r_squared > 0.8
            println("    ✓ Good first-order behavior")
        else
            println("    ⚠ Poor linearity - may not be first-order")
        end
        
        return estimated_k, r_squared
    else
        println("    Rate constant k ≈ $(round(estimated_k, digits=5))")
        return estimated_k, NaN
    end
end

"""
    analyze_second_order_kinetics(reactions, sub1_idx, sub2_idx, name1, name2)

Analyze second-order reaction kinetics (rate ∝ [substrate1] × [substrate2]).
"""
function analyze_second_order_kinetics(reactions, sub1_idx, sub2_idx, name1, name2)
    println("  Analyzing $name1 + $name2 → products (second-order)")
    
    # Group reactions by concentration product
    product_groups = Dict()
    for r in reactions
        conc1 = max(0, r.from_state[sub1_idx] - 1)
        conc2 = max(0, r.from_state[sub2_idx] - 1)
        conc_product = conc1 * conc2
        
        if conc_product > 0  # Only consider non-zero products
            if !haskey(product_groups, conc_product)
                product_groups[conc_product] = []
            end
            push!(product_groups[conc_product], (abs(r.rate), conc1, conc2))
        end
    end
    
    if isempty(product_groups)
        println("    No valid data for kinetic analysis")
        return NaN, NaN
    end
    
    # Calculate rate constants
    rate_constants = Float64[]
    conc_products = Float64[]
    avg_rates = Float64[]
    
    for (conc_prod, rate_data) in product_groups
        rates = [rd[1] for rd in rate_data]
        avg_rate = mean(rates)
        rate_constant = avg_rate / conc_prod  # k = rate / ([A] × [B])
        
        push!(conc_products, Float64(conc_prod))
        push!(avg_rates, Float64(avg_rate))
        push!(rate_constants, Float64(rate_constant))
    end
    
    # Estimate overall rate constant
    estimated_k = mean(rate_constants)
    k_std = std(rate_constants)
    
    # Calculate R² for linearity check
    if length(conc_products) > 2
        # Linear regression: rate = k * [A] * [B]
        X = hcat(ones(Float64, length(conc_products)), conc_products)
        β = X \ avg_rates
        predicted_rates = X * β
        ss_res = sum((avg_rates - predicted_rates).^2)
        ss_tot = sum((avg_rates .- mean(avg_rates)).^2)
        r_squared = 1 - ss_res / ss_tot
        
        println("    Rate constant k ≈ $(round(estimated_k, digits=5)) ± $(round(k_std, digits=5))")
        println("    Linearity R² = $(round(r_squared, digits=3))")
        
        if r_squared > 0.8
            println("    ✓ Good second-order behavior")
        else
            println("    ⚠ Poor linearity - may not be second-order")
        end
        
        return estimated_k, r_squared
    else
        println("    Rate constant k ≈ $(round(estimated_k, digits=5))")
        return estimated_k, NaN
    end
end

"""
    analyze_mm_reaction_kinetics(grouped_reactions, stoich_stats)

Analyze kinetics of identified MM reactions and estimate rate constants.
"""
function analyze_mm_reaction_kinetics(grouped_reactions, stoich_stats)
    println("\n=== MM Reaction Kinetics Analysis ===")
    
    # Expected MM reactions with their properties
    mm_reactions = [
        (
            stoich = tuple([0, 1, -1, 1]...),      # SE → E + P
            name = "SE → E + P",
            order = "first",
            substrate_idx = 3,                      # SE is index 3
            substrate_name = "SE",
            expected_k = 0.1,
            description = "Product formation (kP)"
        ),
        (
            stoich = tuple([-1, -1, 1, 0]...),     # S + E → SE
            name = "S + E → SE", 
            order = "second",
            substrate_indices = [1, 2],             # S is index 1, E is index 2
            substrate_names = ["S", "E"],
            expected_k = 0.01,
            description = "Complex formation (kB)"
        ),
        (
            stoich = tuple([1, 1, -1, 0]...),      # SE → S + E
            name = "SE → S + E",
            order = "first", 
            substrate_idx = 3,                      # SE is index 3
            substrate_name = "SE",
            expected_k = 0.1,
            description = "Complex dissociation (kD)"
        )
    ]
    
    estimated_rates = Dict()
    kinetic_summary = []
    
    for reaction_info in mm_reactions
        stoich = reaction_info.stoich
        
        if stoich in keys(grouped_reactions)
            println("\n--- $(reaction_info.name) ---")
            rxns = grouped_reactions[stoich]
            println("Found $(length(rxns)) instances of this reaction")
            
            if reaction_info.order == "first"
                est_k, r_squared = analyze_first_order_kinetics(
                    rxns, reaction_info.substrate_idx, reaction_info.substrate_name
                )
            else  # second order
                est_k, r_squared = analyze_second_order_kinetics(
                    rxns, reaction_info.substrate_indices[1], reaction_info.substrate_indices[2],
                    reaction_info.substrate_names[1], reaction_info.substrate_names[2]
                )
            end
            
            if !isnan(est_k)
                expected_k = reaction_info.expected_k
                error_pct = abs(est_k - expected_k) / expected_k * 100
                
                println("    Expected: $(expected_k)")
                println("    Estimated: $(round(est_k, digits=5))")
                println("    Error: $(round(error_pct, digits=1))%")
                
                # Quality assessment
                quality = "Unknown"
                if !isnan(r_squared)
                    if r_squared > 0.9 && error_pct < 20
                        quality = "Excellent"
                    elseif r_squared > 0.7 && error_pct < 40
                        quality = "Good"
                    elseif r_squared > 0.5 && error_pct < 60
                        quality = "Fair"
                    else
                        quality = "Poor"
                    end
                end
                
                println("    Quality: $quality")
                
                estimated_rates[stoich] = est_k
                
                push!(kinetic_summary, (
                    reaction = reaction_info.name,
                    description = reaction_info.description,
                    expected = expected_k,
                    estimated = est_k,
                    error_pct = error_pct,
                    r_squared = r_squared,
                    quality = quality
                ))
                
            else
                println("    Could not estimate rate constant")
                push!(kinetic_summary, (
                    reaction = reaction_info.name,
                    description = reaction_info.description,
                    expected = reaction_info.expected_k,
                    estimated = NaN,
                    error_pct = NaN,
                    r_squared = NaN,
                    quality = "Failed"
                ))
            end
            
        else
            println("\n--- $(reaction_info.name) ---")
            println("✗ Reaction not found in dataset")
            push!(kinetic_summary, (
                reaction = reaction_info.name,
                description = reaction_info.description,
                expected = reaction_info.expected_k,
                estimated = NaN,
                error_pct = NaN,
                r_squared = NaN,
                quality = "Not Found"
            ))
        end
    end
    
    # Summary table
    println("\n=== Kinetics Summary ===")
    println("Reaction          | Expected | Estimated | Error  | Quality")
    println("-"^65)
    
    for summary in kinetic_summary
        reaction_short = split(summary.reaction, " → ")[1] * "→" * split(summary.reaction, " → ")[2]
        if length(reaction_short) > 16
            reaction_short = reaction_short[1:13] * "..."
        end
        
        expected_str = "$(summary.expected)"
        estimated_str = isnan(summary.estimated) ? "N/A" : "$(round(summary.estimated, digits=4))"
        error_str = isnan(summary.error_pct) ? "N/A" : "$(round(summary.error_pct, digits=1))%"
        
        println("$(rpad(reaction_short, 17)) | $(rpad(expected_str, 8)) | $(rpad(estimated_str, 9)) | $(rpad(error_str, 6)) | $(summary.quality)")
    end
    
    # Overall assessment
    successful_estimates = filter(s -> s.quality in ["Excellent", "Good", "Fair"], kinetic_summary)
    
    println("\n=== Overall Kinetic Analysis ===")
    println("Successfully estimated: $(length(successful_estimates))/$(length(kinetic_summary)) reactions")
    
    if !isempty(successful_estimates)
        avg_error = mean([s.error_pct for s in successful_estimates if !isnan(s.error_pct)])
        println("Average error: $(round(avg_error, digits=1))%")
        
        excellent_count = count(s -> s.quality == "Excellent", successful_estimates)
        good_count = count(s -> s.quality == "Good", successful_estimates)
        fair_count = count(s -> s.quality == "Fair", successful_estimates)
        
        println("Quality distribution: $excellent_count excellent, $good_count good, $fair_count fair")
        
        if avg_error < 30
            println("✓ Kinetic analysis shows good agreement with expected values")
        elseif avg_error < 50
            println("⚠ Kinetic analysis shows moderate agreement")
        else
            println("✗ Kinetic analysis shows poor agreement")
        end
    else
        println("✗ No successful kinetic estimates obtained")
    end
    
    return estimated_rates, kinetic_summary
end

"""
    analyze_concentration_effects(grouped_reactions, stoich_stats)

Analyze how reaction rates depend on species concentrations.
"""
function analyze_concentration_effects(grouped_reactions, stoich_stats)
    println("\n=== Concentration Effects Analysis ===")
    
    for (stoich, rxns) in grouped_reactions
        if length(rxns) < 5  # Need sufficient data
            continue
        end
        
        reaction_name = format_reaction(stoich, ["S", "E", "SE", "P"])
        println("\n--- $reaction_name ---")
        
        # Extract concentration data
        s_concs = Float64[max(0, r.from_state[1] - 1) for r in rxns]
        e_concs = Float64[max(0, r.from_state[2] - 1) for r in rxns]
        se_concs = Float64[max(0, r.from_state[3] - 1) for r in rxns]
        p_concs = Float64[max(0, r.from_state[4] - 1) for r in rxns]
        rates = Float64[abs(r.rate) for r in rxns]
        
        # Calculate correlations
        species_data = [
            ("S", s_concs),
            ("E", e_concs), 
            ("SE", se_concs),
            ("P", p_concs)
        ]
        
        println("Rate correlations with species concentrations:")
        for (species, concs) in species_data
            if var(concs) > 1e-10 && var(rates) > 1e-10
                correlation = cor(concs, rates)
                
                interpretation = ""
                if abs(correlation) > 0.8
                    interpretation = abs(correlation) > 0 ? "Strong positive" : "Strong negative" 
                elseif abs(correlation) > 0.5
                    interpretation = correlation > 0 ? "Moderate positive" : "Moderate negative"
                elseif abs(correlation) > 0.2
                    interpretation = correlation > 0 ? "Weak positive" : "Weak negative"
                else
                    interpretation = "No correlation"
                end
                
                println("  $species: $(round(correlation, digits=3)) ($interpretation)")
            else
                println("  $species: insufficient variation")
            end
        end
    end
end

"""
    validate_mm_rate_laws(grouped_reactions, stoich_stats)

Validate that identified reactions follow expected MM rate laws.
"""
function validate_mm_rate_laws(grouped_reactions, stoich_stats)
    println("\n=== MM Rate Law Validation ===")
    
    validations = []
    
    # SE → E + P should be first-order in SE
    se_to_ep = tuple([0, 1, -1, 1]...)
    if se_to_ep in keys(grouped_reactions)
        rxns = grouped_reactions[se_to_ep]
        
        # Check if rate ∝ [SE]
        se_concs = Float64[max(0, r.from_state[3] - 1) for r in rxns]
        rates = Float64[abs(r.rate) for r in rxns]
        
        if length(unique(se_concs)) > 2 && var(se_concs) > 1e-10
            correlation = cor(se_concs, rates)
            
            validation = (
                reaction = "SE → E + P",
                expected_law = "Rate ∝ [SE]",
                correlation = correlation,
                passes = abs(correlation) > 0.6,
                interpretation = abs(correlation) > 0.6 ? "Follows first-order kinetics" : "Deviates from first-order"
            )
            
            push!(validations, validation)
            println("SE → E + P: correlation with [SE] = $(round(correlation, digits=3))")
            println("  $(validation.interpretation)")
        end
    end
    
    # S + E → SE should be second-order
    s_e_to_se = tuple([-1, -1, 1, 0]...)
    if s_e_to_se in keys(grouped_reactions)
        rxns = grouped_reactions[s_e_to_se]
        
        s_concs = Float64[max(0, r.from_state[1] - 1) for r in rxns]
        e_concs = Float64[max(0, r.from_state[2] - 1) for r in rxns]
        se_products = Float64[s * e for (s, e) in zip(s_concs, e_concs)]
        rates = Float64[abs(r.rate) for r in rxns]
        
        if length(unique(se_products)) > 2 && var(se_products) > 1e-10
            correlation = cor(se_products, rates)
            
            validation = (
                reaction = "S + E → SE",
                expected_law = "Rate ∝ [S][E]",
                correlation = correlation,
                passes = abs(correlation) > 0.6,
                interpretation = abs(correlation) > 0.6 ? "Follows second-order kinetics" : "Deviates from second-order"
            )
            
            push!(validations, validation)
            println("S + E → SE: correlation with [S][E] = $(round(correlation, digits=3))")
            println("  $(validation.interpretation)")
        end
    end
    
    # Summary
    passing_validations = filter(v -> v.passes, validations)
    println("\nRate Law Validation Summary:")
    println("Passed: $(length(passing_validations))/$(length(validations)) tests")
    
    if length(passing_validations) == length(validations) && !isempty(validations)
        println("✓ All reactions follow expected MM rate laws")
    elseif !isempty(passing_validations)
        println("⚠ Some reactions deviate from expected rate laws")
    else
        println("✗ Reactions do not follow expected MM rate laws")
    end
    
    return validations
end

"""
    estimate_mm_parameters(estimated_rates)

Estimate overall MM parameters from individual rate constants.
"""
function estimate_mm_parameters(estimated_rates)
    println("\n=== MM Parameter Estimation ===")
    
    # Extract rate constants
    kB_stoich = tuple([-1, -1, 1, 0]...)  # S + E → SE
    kD_stoich = tuple([1, 1, -1, 0]...)   # SE → S + E  
    kP_stoich = tuple([0, 1, -1, 1]...)   # SE → E + P
    
    kB = get(estimated_rates, kB_stoich, NaN)
    kD = get(estimated_rates, kD_stoich, NaN)
    kP = get(estimated_rates, kP_stoich, NaN)
    
    println("Individual rate constants:")
    println("  kB (binding): $(isnan(kB) ? "not estimated" : round(kB, digits=5))")
    println("  kD (dissociation): $(isnan(kD) ? "not estimated" : round(kD, digits=5))")
    println("  kP (product formation): $(isnan(kP) ? "not estimated" : round(kP, digits=5))")
    
    # Calculate MM parameters if possible
    if !isnan(kB) && !isnan(kD) && !isnan(kP)
        # Km = (kD + kP) / kB
        Km = (kD + kP) / kB
        
        # Vmax = kP * [E_total] (we'd need to know total enzyme)
        # For now, just report kP as the catalytic rate constant
        kcat = kP
        
        # Catalytic efficiency
        kcat_Km = kcat / Km
        
        println("\nDerived MM parameters:")
        println("  Km ≈ $(round(Km, digits=3)) (Michaelis constant)")
        println("  kcat ≈ $(round(kcat, digits=5)) (catalytic rate constant)")
        println("  kcat/Km ≈ $(round(kcat_Km, digits=5)) (catalytic efficiency)")
        
        # Compare with expected values (if known)
        expected_Km = (0.1 + 0.1) / 0.01  # (kD + kP) / kB with true parameters
        expected_kcat = 0.1
        expected_efficiency = expected_kcat / expected_Km
        
        Km_error = abs(Km - expected_Km) / expected_Km * 100
        kcat_error = abs(kcat - expected_kcat) / expected_kcat * 100
        
        println("\nComparison with expected values:")
        println("  Km: expected $(expected_Km), error $(round(Km_error, digits=1))%")
        println("  kcat: expected $(expected_kcat), error $(round(kcat_error, digits=1))%")
        
        return Dict(
            "kB" => kB,
            "kD" => kD, 
            "kP" => kP,
            "Km" => Km,
            "kcat" => kcat,
            "kcat_Km" => kcat_Km
        )
    else
        println("\nCannot derive MM parameters - missing rate constants")
        return Dict(
            "kB" => kB,
            "kD" => kD,
            "kP" => kP
        )
    end
end

"""
    complete_kinetics_analysis(grouped_reactions, stoich_stats)

Run complete kinetics analysis pipeline.
"""
function complete_kinetics_analysis(grouped_reactions, stoich_stats)
    println("\n" * "="^50)
    println("COMPLETE KINETICS ANALYSIS")
    println("="^50)
    
    # 1. MM reaction kinetics
    estimated_rates, kinetic_summary = analyze_mm_reaction_kinetics(grouped_reactions, stoich_stats)
    
    # 2. Concentration effects
    analyze_concentration_effects(grouped_reactions, stoich_stats)
    
    # 3. Rate law validation
    rate_law_validations = validate_mm_rate_laws(grouped_reactions, stoich_stats)
    
    # 4. MM parameter estimation
    mm_parameters = estimate_mm_parameters(estimated_rates)
    
    println("\n" * "="^50)
    println("KINETICS ANALYSIS COMPLETE")
    println("="^50)
    
    return Dict(
        "estimated_rates" => estimated_rates,
        "kinetic_summary" => kinetic_summary,
        "rate_law_validations" => rate_law_validations,
        "mm_parameters" => mm_parameters
    )
end

println("Kinetics Analysis Module Loaded! ⚗️")
println("Usage: complete_kinetics_analysis(grouped_reactions, stoich_stats)")
