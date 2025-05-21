# example.jl
# Simple demonstration of the CRN inference framework

# Include the main script
include("main.jl")

"""
    run_mm_example()

Run Michaelis-Menten example with visualization.
"""
function run_mm_example()
    println("=== Michaelis-Menten Example ===")
    
    # Generate data and run inference
    results = infer_crn_michaelis_menten(100, 500)  # 100 trajectories, max 500 states
    
    # Extract key results
    significant_stoich = results["significant_stoichiometries"]
    grouped_reactions = results["grouped_reactions"]
    stoich_stats = results["stoich_stats"]
    species_names = ["S", "E", "SE", "P"]
    
    # Print top reactions
    println("\nTop reactions by rate:")
    for (i, stoich) in enumerate(significant_stoich[1:5])
        stats = stoich_stats[stoich]
        reaction_str = format_reaction(stoich, species_names)
        println("$i. $reaction_str (rate: $(round(stats.total_rate, digits=5)))")
    end
    
    # Print estimated rate constants
    println("\nEstimated rate constants:")
    scaled_reactions = results["kinetics_results"]["scaled_reactions"]
    for r in scaled_reactions
        println("$(r.reactants): $(round(r.scaled_rate, digits=5))")
    end
    
    return results
end

"""
    run_lv_example()

Run Lotka-Volterra example with visualization.
"""
function run_lv_example()
    println("=== Lotka-Volterra Example ===")
    
    # Generate data and run inference
    results = infer_crn_lotka_volterra(100, 500)  # 100 trajectories, max 500 states
    
    # Extract key results
    significant_stoich = results["significant_stoichiometries"]
    grouped_reactions = results["grouped_reactions"]
    stoich_stats = results["stoich_stats"]
    species_names = ["X", "Y"]  # prey, predator
    
    # Print top reactions
    println("\nTop reactions by rate:")
    for (i, stoich) in enumerate(significant_stoich[1:5])
        stats = stoich_stats[stoich]
        reaction_str = format_reaction(stoich, species_names)
        println("$i. $reaction_str (rate: $(round(stats.total_rate, digits=5)))")
    end
    
    return results
end

"""
    main()

Run both examples in sequence.
"""
function main()
    println("=== CRN Inference Examples ===")
    
    # Run Michaelis-Menten example
    mm_results = run_mm_example()
    
    println("\nPress Enter to continue to Lotka-Volterra example...")
    readline()
    
    # Run Lotka-Volterra example
    lv_results = run_lv_example()
    
    println("\n=== Examples Completed ===")
end

# Run the main function if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
