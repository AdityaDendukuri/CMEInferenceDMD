# kinetics.jl - CLEAN KINETICS ANALYSIS MODULE
# Rate constant estimation and MM parameter derivation

using Statistics
using LinearAlgebra

"""
    estimate_rate_constants(reaction_stats, species_names)

Estimate rate constants from reaction statistics.
"""
function estimate_rate_constants(reaction_stats, species_names)
    println("Estimating rate constants...")
    
    rate_estimates = []
    
    for (stoich, stats) in reaction_stats
        reaction_str = format_reaction(collect(stoich), species_names)
        
        # Basic rate estimation from total rate
        estimated_rate = stats.total_rate
        
        # Quality assessment based on variance and count
        quality = assess_rate_quality(stats)
        
        # Error estimation (rough)
        error_pct = stats.count > 1 ? (sqrt(stats.rate_var) / stats.avg_rate * 100) : NaN
        
        push!(rate_estimates, (
            stoichiometry = collect(stoich),
            reaction_string = reaction_str,
            estimated_rate = estimated_rate,
            avg_rate = stats.avg_rate,
            rate_variance = stats.rate_var,
            observation_count = stats.count,
            quality = quality,
            error_pct = error_pct
        ))
    end
    
    # Sort by quality and rate magnitude
    sort!(rate_estimates, by=r -> (r.quality == "Excellent" ? 3 : 
                                   r.quality == "Good" ? 2 : 
                                   r.quality == "Fair" ? 1 : 0, r.estimated_rate), rev=true)
    
    println("Rate constant estimates:")
    for (i, est) in enumerate(rate_estimates[1:min(10, end)])
        println("$i. $(est.reaction_string)")
        println("   Rate: $(round(est.estimated_rate, digits=5)) (quality: $(est.quality))")
        if !isnan(est.error_pct)
            println("   Error: ±$(round(est.error_pct, digits=1))%")
        end
    end
    
    return rate_estimates
end

"""
    assess_rate_quality(stats)

Assess the quality of rate constant estimation.
"""
function assess_rate_quality(stats)
    # Quality based on observation count and variance
    if stats.count >= 10 && stats.rate_var < 0.1 * stats.avg_rate^2
        return "Excellent"
    elseif stats.count >= 5 && stats.rate_var < 0.25 * stats.avg_rate^2
        return "Good"
    elseif stats.count >= 3 && stats.rate_var < 0.5 * stats.avg_rate^2
        return "Fair"
    else
        return "Poor"
    end
end

"""
    derive_mm_parameters(rate_estimates, species_names)

Derive Michaelis-Menten parameters from rate estimates.
"""
function derive_mm_parameters(rate_estimates, species_names)
    println("\nDeriving MM parameters...")
    
    if length(species_names) != 4
        println("MM parameter derivation requires exactly 4 species")
        return Dict()
    end
    
    # Expected MM reactions and their stoichiometries
    mm_reactions = Dict(
        "binding" => [-1, -1, 1, 0],      # S + E → SE (k₁)
        "dissociation" => [1, 1, -1, 0],  # SE → S + E (k₋₁)  
        "catalysis" => [0, 1, -1, 1]      # SE → E + P (k₂)
    )
    
    # Find rates for each MM reaction
    mm_rates = Dict()
    
    for (name, stoich) in mm_reactions
        for est in rate_estimates
            if est.stoichiometry == stoich && est.quality in ["Excellent", "Good", "Fair"]
                mm_rates[name] = est.estimated_rate
                println("Found $name: k = $(round(est.estimated_rate, digits=5))")
                break
            end
        end
    end
    
    # Derive MM parameters if we have sufficient rates
    mm_params = Dict()
    
    if haskey(mm_rates, "binding") && haskey(mm_rates, "dissociation") && haskey(mm_rates, "catalysis")
        k1 = mm_rates["binding"]
        k_minus1 = mm_rates["dissociation"] 
        k2 = mm_rates["catalysis"]
        
        # Classical MM parameters
        Km = (k_minus1 + k2) / k1  # Michaelis constant
        kcat = k2                   # Turnover number (catalytic rate constant)
        kcat_Km = k1                # Catalytic efficiency
        
        mm_params = Dict(
            "k1" => k1,
            "k_minus1" => k_minus1,
            "k2" => k2,
            "Km" => Km,
            "kcat" => kcat,
            "kcat_Km" => kcat_Km
        )
        
        println("\nMM Parameters:")
        println("  k₁ (binding) = $(round(k1, digits=5))")
        println("  k₋₁ (dissociation) = $(round(k_minus1, digits=5))")
        println("  k₂ (catalysis) = $(round(k2, digits=5))")
        println("  Km = $(round(Km, digits=3))")
        println("  kcat = $(round(kcat, digits=5))")
        println("  kcat/Km = $(round(kcat_Km, digits=5))")
        
    else
        missing_reactions = []
        for name in ["binding", "dissociation", "catalysis"]
            if !haskey(mm_rates, name)
                push!(missing_reactions, name)
            end
        end
        println("Cannot derive MM parameters - missing reactions: $(join(missing_reactions, ", "))")
    end
    
    return mm_params
end

"""
    analyze_reaction_kinetics(reaction_stats, species_names)

Analyze reaction kinetics and identify kinetic regimes.
"""
function analyze_reaction_kinetics(reaction_stats, species_names)
    println("\n=== Kinetic Analysis ===")
    
    # Get rate estimates
    rate_estimates = estimate_rate_constants(reaction_stats, species_names)
    
    # Analyze rate distribution
    rates = [est.estimated_rate for est in rate_estimates if est.quality != "Poor"]
    
    if !isempty(rates)
        println("\nRate distribution:")
        println("  Fastest rate: $(round(maximum(rates), digits=5))")
        println("  Slowest rate: $(round(minimum(rates), digits=5))")
        println("  Rate ratio: $(round(maximum(rates)/minimum(rates), digits=1))")
        
        # Identify potential rate-limiting steps
        mean_rate = mean(rates)
        slow_reactions = filter(est -> est.estimated_rate < 0.5 * mean_rate && est.quality != "Poor", rate_estimates)
        
        if !isempty(slow_reactions)
            println("\nPotential rate-limiting steps:")
            for rxn in slow_reactions[1:min(3, end)]
                println("  $(rxn.reaction_string) (rate: $(round(rxn.estimated_rate, digits=5)))")
            end
        end
    end
    
    # MM-specific analysis
    mm_params = derive_mm_parameters(rate_estimates, species_names)
    
    # Kinetic regime identification
    kinetic_regime = identify_kinetic_regime(mm_params)
    
    return Dict(
        "rate_estimates" => rate_estimates,
        "mm_parameters" => mm_params,
        "kinetic_regime" => kinetic_regime,
        "kinetic_summary" => rate_estimates
    )
end

"""
    identify_kinetic_regime(mm_params)

Identify the kinetic regime of the MM system.
"""
function identify_kinetic_regime(mm_params)
    if isempty(mm_params)
        return "Unknown"
    end
    
    k1 = get(mm_params, "k1", 0)
    k_minus1 = get(mm_params, "k_minus1", 0) 
    k2 = get(mm_params, "k2", 0)
    
    if k1 == 0 || k_minus1 == 0 || k2 == 0
        return "Incomplete"
    end
    
    # Analyze rate ratios - FIXED: Use > instead of >>
    if k_minus1 > 10 * k2  # k₋₁ > 10*k₂ (was k_minus1 >> k2)
        if k1 > 10 * k2
            return "Pre-equilibrium (rapid equilibrium)"
        else
            return "Mixed regime"
        end
    elseif k2 > 10 * k_minus1  # k₂ > 10*k₋₁ (was k2 >> k_minus1)
        return "Irreversible catalysis"
    else
        return "General MM kinetics"
    end
end

"""
    run_kinetics_analysis(data_dict)

Run complete kinetics analysis on reaction data.
"""
function run_kinetics_analysis(data_dict)
    println("\n" * "="^50)
    println("KINETICS ANALYSIS")
    println("="^50)
    
    # Check if reaction data is available
    if !haskey(data_dict, "reaction_stats")
        println("ERROR: Reaction statistics not found. Run DMD analysis first.")
        return data_dict
    end
    
    # Extract reaction data
    reaction_stats = data_dict["reaction_stats"]
    species_names = data_dict["species_names"]
    
    # Run kinetics analysis
    kinetics_results = analyze_reaction_kinetics(reaction_stats, species_names)
    
    # Add results to data dictionary
    data_dict["kinetics_analysis"] = kinetics_results
    
    println("\nKinetics analysis completed!")
    return data_dict
end

"""
    format_reaction(stoich, species_names)

Format stoichiometry vector as reaction string (duplicate from main for completeness).
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

println("Clean Kinetics Analysis Module Loaded! ⚗️")
println("Functions:")
println("  estimate_rate_constants(reaction_stats, species)")
println("  derive_mm_parameters(rate_estimates, species)")
println("  analyze_reaction_kinetics(reaction_stats, species)")
println("  run_kinetics_analysis(data_dict)")
println()
println("Kinetics analysis provides:")
println("  • Rate constant estimation with quality assessment")
println("  • MM parameter derivation (Km, kcat, etc.)")
println("  • Kinetic regime identification")
println("  • Rate-limiting step analysis")
