# integration.jl - INTEGRATION AND USAGE MODULE
# Simply includes existing modules and provides convenience functions

println("Loading MM CRN Inference System...")

# Include all existing modules
println("  Loading main.jl...")
include("main.jl")

println("  Loading flow_analysis.jl...")
include("flow_analysis.jl") 

println("  Loading mm_specific_analysis.jl...")
include("mm_specific_analysis.jl")

println("  Loading kinetics_analysis.jl...")
include("kinetics_analysis.jl")

println("  Loading constrained_dmd.jl...")
include("constrained_dmd.jl")

"""
    run_constrained_vs_unconstrained_comparison(n_trajs=500, max_states=500)

Compare constrained vs unconstrained DMD to demonstrate the fix for S → P problem.
"""
function run_constrained_vs_unconstrained_comparison(n_trajs=500, max_states=500)
    println("="^70)
    println("CONSTRAINED vs UNCONSTRAINED DMD COMPARISON")
    println("="^70)
    
    # Run unconstrained first
    println("\n🔓 UNCONSTRAINED DMD (Original - may have S → P problem)")
    println("="^50)
    unconstrained_results = run_mm_inference(n_trajs, max_states, use_constrained_dmd=false)
    
    # Run constrained
    println("\n🔒 CONSTRAINED DMD (Fixed - should eliminate S → P)")
    println("="^50) 
    constrained_results = run_mm_inference(n_trajs, max_states, use_constrained_dmd=true, λ_sparse=0.01)
    
    # Comparison summary
    println("\n📊 COMPARISON SUMMARY")
    println("="^40)
    
    println("Recovery rates:")
    println("  Unconstrained: $(round(unconstrained_results["recovery_rate"], digits=1))%")
    println("  Constrained:   $(round(constrained_results["recovery_rate"], digits=1))%")
    
    println("\nUnphysical reactions:")
    println("  Unconstrained: $(unconstrained_results["unphysical_count"]) found")
    println("  Constrained:   $(constrained_results["unphysical_count"]) found")
    
    println("\nTotal reactions identified:")
    println("  Unconstrained: $(length(unconstrained_results["significant_stoichiometries"]))")
    println("  Constrained:   $(length(constrained_results["significant_stoichiometries"]))")
    
    # Check specifically for S → P
    s_to_p_stoich = tuple([-1, 0, 0, 1]...)
    
    println("\nSpecific check for S → P reaction:")
    
    if s_to_p_stoich in keys(unconstrained_results["grouped_reactions"])
        s_to_p_rate = unconstrained_results["stoich_stats"][s_to_p_stoich].total_rate
        println("  Unconstrained: X S → P FOUND (rate: $(round(s_to_p_rate, digits=4)))")
    else
        println("  Unconstrained: ✓ S → P not found")
    end
    
    if s_to_p_stoich in keys(constrained_results["grouped_reactions"])
        s_to_p_rate = constrained_results["stoich_stats"][s_to_p_stoich].total_rate
        println("  Constrained:   X S → P FOUND (rate: $(round(s_to_p_rate, digits=4))) - CONSTRAINT FAILED!")
    else
        println("  Constrained:   ✓ S → P eliminated by constraints")
    end
    
    # Overall verdict
    println("\n🏆 VERDICT:")
    improvement = constrained_results["unphysical_count"] < unconstrained_results["unphysical_count"]
    recovery_maintained = constrained_results["recovery_rate"] >= (unconstrained_results["recovery_rate"] - 10)
    
    if improvement && recovery_maintained
        println("✓ Constrained DMD successfully eliminates unphysical reactions while maintaining recovery!")
    elseif improvement
        println("⚠ Constrained DMD eliminates unphysical reactions but reduces recovery rate")
    else
        println("✗ Constrained DMD implementation needs debugging")
    end
    
    return Dict(
        "unconstrained" => unconstrained_results,
        "constrained" => constrained_results,
        "improvement" => improvement,
        "recovery_maintained" => recovery_maintained
    )
end

"""
    show_results_overview(results)

Show a concise overview of results.
"""
function show_results_overview(results)
    println("\n" * "="^40)
    println("RESULTS OVERVIEW")
    println("="^40)
    
    println("Basic Info:")
    println("  Species: $(join(results["species_names"], ", "))")
    if haskey(results, "rank")
        println("  DMD rank: $(results["rank"])")
    end
    if haskey(results, "selected_states")
        println("  States analyzed: $(length(results["selected_states"]))")
    end
    
    println("\nTop 5 Reactions:")
    for (i, stoich) in enumerate(results["significant_stoichiometries"][1:min(5, end)])
        stats = results["stoich_stats"][stoich]
        reaction_str = format_reaction(stoich, results["species_names"])
        println("  $i. $reaction_str (rate: $(round(stats.total_rate, digits=4)))")
    end
    
    if haskey(results, "flow_analysis") && results["flow_analysis"] !== nothing
        flow_modes = results["flow_analysis"]["flow_modes"]
        println("\nFlow Analysis:")
        println("  Active modes: $(length(flow_modes))")
        if !isempty(flow_modes)
            println("  Dominant type: $(flow_modes[1].mode_type)")
        end
    end
    
    if haskey(results, "kinetics_analysis") && haskey(results["kinetics_analysis"], "mm_parameters")
        kinetics = results["kinetics_analysis"]
        if haskey(kinetics["mm_parameters"], "Km")
            params = kinetics["mm_parameters"]
            println("\nMM Parameters:")
            println("  Km ≈ $(round(params["Km"], digits=3))")
            println("  kcat ≈ $(round(params["kcat"], digits=5))")
        end
    end
end

# Convenience aliases
compare_dmd = () -> run_constrained_vs_unconstrained_comparison(300, 400)
basic_mm = () -> run_mm_inference(300, 400, use_constrained_dmd=false)
constrained_mm = () -> run_mm_inference(300, 400, use_constrained_dmd=true)
flow_mm = () -> run_with_flow_analysis(300, 400)
mm_analysis = () -> run_with_mm_analysis(300, 400)
quick_demo = () -> constrained_mm()  # Use constrained by default

println("\n" * "="^50)
println("🧬 FRESH MM CRN INFERENCE SYSTEM LOADED! 🧬")
println("="^50)
println()
println("🔒 Constrained DMD Functions:")
println("  compare_dmd()           - Compare constrained vs unconstrained")
println("  constrained_mm()        - Basic inference with constraints")
println("  basic_mm()             - Basic inference without constraints")
println()
println("Quick Start Functions:")
println("  quick_demo()           - Fast demo with constraints")
println("  flow_mm()              - With flow analysis")  
println("  mm_analysis()          - With MM-specific analysis")
println()
println("Analysis Functions:")
println("  run_constrained_vs_unconstrained_comparison(n_trajs, max_states)")
println("  show_results_overview(results)")
println()
println("Example Usage:")
println("  # Test the S → P fix:")
println("  comparison = compare_dmd()")
println("  ")
println("  # Quick demo:")
println("  results = quick_demo()")
println("  show_results_overview(results)")
println()
println("🚀 Ready to test constrained DMD!")
