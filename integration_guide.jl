# integration_guide.jl - HOW TO USE THE FIXED MULTIGRID DMD

# Include the fixed module
include("multigrid_dmd_debug.jl")

"""
    test_fixed_multigrid(results, species_names)

Test the fixed multigrid DMD on your existing results.
"""
function test_fixed_multigrid(results, species_names)
    println("="^70)
    println("TESTING FIXED MULTIGRID DMD")
    println("="^70)
    
    # Extract data from your existing results
    if !haskey(results, "reduced_data") || !haskey(results, "selected_states")
        println("❌ Missing required data in results. Need 'reduced_data' and 'selected_states'")
        return nothing
    end
    
    reduced_data = results["reduced_data"]
    selected_states = results["selected_states"]
    dt = haskey(results, "dt") ? results["dt"] : 2.5  # Default from your setup
    
    println("Input data: $(size(reduced_data)) matrix, $(length(selected_states)) states")
    
    # Run the fixed algorithm
    G_combined, sorted_stoich, fused_reactions, reaction_stats, successful_segments = fixed_multigrid_constrained_dmd(
        reduced_data, dt, selected_states, species_names,
        segment_length=12,      # Longer segments for better stability
        overlap_fraction=0.3,   # Less overlap to reduce artifacts
        max_stoich_change=2     # Very strict elementary reaction filter
    )
    
    # Compare with expected MM reactions
    println("\n" * "="^50)
    println("MICHAELIS-MENTEN REACTION CHECK")
    println("="^50)
    
    expected_reactions = [
        ([-1, -1, 1, 0], "S + E → SE"),      # Binding
        ([1, 1, -1, 0], "SE → S + E"),       # Dissociation  
        ([0, 1, -1, 1], "SE → E + P")        # Catalysis
    ]
    
    found_count = 0
    for (expected_stoich, reaction_name) in expected_reactions
        stoich_tuple = tuple(expected_stoich...)
        
        if stoich_tuple in sorted_stoich
            stats = reaction_stats[stoich_tuple]
            println("✅ FOUND: $reaction_name")
            println("   Rate: $(round(stats.total_rate, digits=4))")
            println("   Confidence: $(round(stats.confidence, digits=3))")
            println("   Segments: $(stats.segments)")
            found_count += 1
        else
            println("❌ MISSING: $reaction_name")
        end
    end
    
    recovery_rate = (found_count / length(expected_reactions)) * 100
    println("\nMM Recovery Rate: $(round(recovery_rate, digits=1))%")
    
    # Overall assessment
    println("\n" * "="^50)
    println("OVERALL ASSESSMENT")
    println("="^50)
    
    if found_count >= 2
        println("🎉 SUCCESS: Fixed algorithm recovered $(found_count)/3 MM reactions!")
        println("   This is a major improvement over 0 elementary reactions before.")
    elseif found_count >= 1
        println("🔶 PARTIAL SUCCESS: Found $(found_count)/3 MM reactions")
        println("   Getting closer - may need parameter tuning")
    else
        println("🔍 STILL DEBUGGING: No MM reactions found")
        println("   Need to investigate state space or parameters further")
    end
    
    # Return enhanced results
    enhanced_results = copy(results)
    enhanced_results["multigrid_fixed"] = Dict(
        "generator" => G_combined,
        "reactions" => sorted_stoich,
        "reaction_stats" => reaction_stats,
        "successful_segments" => successful_segments,
        "recovery_rate" => recovery_rate
    )
    
    return enhanced_results
end

"""
    diagnose_state_space_issues(selected_states, species_names)

Diagnose potential state space issues that could cause spurious reactions.
"""
function diagnose_state_space_issues(selected_states, species_names)
    println("\n" * "="^60)
    println("STATE SPACE DIAGNOSTIC")
    println("="^60)
    
    if isempty(selected_states)
        println("❌ No states to analyze")
        return
    end
    
    # Convert to molecular counts for analysis
    molecular_states = []
    for state in selected_states
        mol_counts = [max(0, x-1) for x in state]
        push!(molecular_states, mol_counts)
    end
    
    # Analyze distributions
    n_species = length(species_names)
    println("Species distributions:")
    
    for i in 1:n_species
        counts = [mol_state[i] for mol_state in molecular_states]
        min_val = minimum(counts)
        max_val = maximum(counts)
        mean_val = mean(counts)
        
        println("  $(species_names[i]): $min_val to $max_val (mean: $(round(mean_val, digits=1)))")
        
        # Check for unrealistic ranges
        if max_val - min_val > 50
            println("    ⚠ WARNING: Very large range ($(max_val - min_val))")
        end
    end
    
    # Check for neighboring states
    println("\nNeighboring state analysis:")
    neighbor_distances = []
    
    for i in 1:min(100, length(molecular_states))  # Sample first 100
        state_i = molecular_states[i]
        
        # Find nearest neighbors
        min_distance = Inf
        for j in (i+1):min(i+20, length(molecular_states))  # Check next 20 states
            state_j = molecular_states[j]
            distance = sum(abs.(state_i - state_j))
            min_distance = min(min_distance, distance)
        end
        
        if min_distance < Inf
            push!(neighbor_distances, min_distance)
        end
    end
    
    if !isempty(neighbor_distances)
        avg_neighbor_distance = mean(neighbor_distances)
        println("  Average distance to nearest neighbor: $(round(avg_neighbor_distance, digits=1))")
        
        if avg_neighbor_distance > 10
            println("  ⚠ WARNING: States are very far apart on average")
            println("    This explains why DMD finds large stoichiometric changes")
            println("    Consider:")
            println("      • Using more states (smaller grid spacing)")
            println("      • Better state selection criteria")
            println("      • Different dimensionality reduction")
        else
            println("  ✅ State spacing looks reasonable")
        end
    end
    
    # Recommend fixes
    println("\n" * "="^40)
    println("RECOMMENDATIONS")
    println("="^40)
    
    total_states = length(selected_states)
    if total_states < 200
        println("1. 📈 INCREASE STATE COUNT: You have $total_states states")
        println("   Try increasing max_states to 500-1000 for finer resolution")
    end
    
    if !isempty(neighbor_distances) && mean(neighbor_distances) > 5
        println("2. 🎯 IMPROVE STATE SELECTION: States are too far apart")
        println("   The current state selection may be missing intermediate states")
    end
    
    println("3. 🔧 PARAMETER TUNING: Try these parameters:")
    println("   • segment_length=15 (longer segments)")
    println("   • max_stoich_change=2 (stricter elementary filter)")
    println("   • Higher state count in data generation")
end

"""
    run_complete_fixed_analysis(n_trajs=500, max_states=800)

Run complete analysis with the fixed multigrid approach.
"""
function run_complete_fixed_analysis(n_trajs=500, max_states=800)
    println("🔧 RUNNING COMPLETE ANALYSIS WITH FIXED MULTIGRID DMD")
    
    # Step 1: Generate high-resolution data
    println("\n1. Generating high-resolution MM data...")
    results = run_mm_inference(n_trajs, max_states, use_constrained_dmd=false)  # Start with basic approach
    species_names = ["S", "E", "SE", "P"]
    
    # Step 2: Diagnose state space
    println("\n2. Diagnosing state space...")
    diagnose_state_space_issues(results["selected_states"], species_names)
    
    # Step 3: Apply fixed multigrid
    println("\n3. Applying fixed multigrid DMD...")
    enhanced_results = test_fixed_multigrid(results, species_names)
    
    # Step 4: Compare approaches
    if enhanced_results !== nothing
        println("\n4. Comparison Summary:")
        
        original_reactions = length(results["significant_stoichiometries"])
        fixed_reactions = haskey(enhanced_results["multigrid_fixed"], "reactions") ? 
                         length(enhanced_results["multigrid_fixed"]["reactions"]) : 0
        
        println("  Original approach: $original_reactions reactions")
        println("  Fixed multigrid: $fixed_reactions elementary reactions")
        
        if haskey(enhanced_results, "multigrid_fixed")
            recovery = enhanced_results["multigrid_fixed"]["recovery_rate"]
            println("  MM recovery rate: $(round(recovery, digits=1))%")
            
            if recovery > 50
                println("  🎉 Fixed approach is working!")
            else
                println("  🔧 Still needs parameter tuning")
            end
        end
    end
    
    return enhanced_results
end

# Convenience functions
quick_fix_test = () -> test_fixed_multigrid(run_mm_inference(300, 500), ["S", "E", "SE", "P"])
diagnose_states = () -> diagnose_state_space_issues(run_mm_inference(300, 500)["selected_states"], ["S", "E", "SE", "P"])

println("🔧 Fixed Multigrid Integration Loaded!")
println()
println("Quick test functions:")
println("  quick_fix_test()           - Test fix on small problem")
println("  diagnose_states()          - Diagnose state space issues")
println("  run_complete_fixed_analysis() - Complete analysis with fix")
println()
println("Main functions:")
println("  test_fixed_multigrid(results, species_names)")
println("  diagnose_state_space_issues(selected_states, species_names)")
println()
println("To test the fix on your current results:")
println("  enhanced_results = test_fixed_multigrid(your_results, [\"S\", \"E\", \"SE\", \"P\"])")
