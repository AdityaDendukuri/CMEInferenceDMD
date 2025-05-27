# integration.jl - INTEGRATION AND USAGE MODULE
# Combines all analyses and provides easy-to-use functions

# Load all modules
include("main.jl")
include("flow_analysis.jl") 
include("mm_specific_analysis.jl")
include("kinetics_analysis.jl")

"""
    run_complete_analysis(n_trajs=500, max_states=500)

Run the complete MM analysis pipeline with all components.
"""
function run_complete_analysis(n_trajs=500, max_states=500)
    println("="^70)
    println("COMPLETE MM CRN ANALYSIS PIPELINE")
    println("="^70)
    
    # Step 1: Basic CRN inference
    println("\n🧬 STEP 1: Basic CRN Inference")
    results = run_mm_inference(n_trajs, max_states)
    
    # Step 2: Flow field analysis
    println("\n🌊 STEP 2: Flow Field Analysis")
    flow_results = basic_flow_analysis(results, results["species_names"])
    if flow_results !== nothing
        results["flow_analysis"] = flow_results
    end
    
    # Step 3: MM-specific analysis
    println("\n🧪 STEP 3: MM-Specific Analysis")
    if flow_results !== nothing
        mm_results = complete_mm_analysis(results, results["species_names"])
        if mm_results !== nothing
            results["mm_flow_analysis"] = mm_results
        end
    end
    
    # Step 4: Kinetics analysis
    println("\n⚗️ STEP 4: Kinetics Analysis")
    kinetics_results = complete_kinetics_analysis(results["grouped_reactions"], results["stoich_stats"])
    results["kinetics_analysis"] = kinetics_results
    
    # Step 5: Final summary
    println("\n📊 STEP 5: Final Summary")
    generate_final_summary(results)
    
    println("\n" * "="^70)
    println("COMPLETE ANALYSIS FINISHED")
    println("="^70)
    
    return results
end

"""
    generate_final_summary(results)

Generate a comprehensive summary of all analyses.
"""
function generate_final_summary(results)
    println("\n" * "="^50)
    println("FINAL ANALYSIS SUMMARY")
    println("="^50)
    
    # Basic inference summary
    println("\n🧬 CRN Inference Results:")
    println("  DMD rank: $(results["rank"])")
    println("  Reactions identified: $(length(results["significant_stoichiometries"]))")
    println("  Conservation laws applied: ✓")
    
    # Expected MM reactions check
    expected_found = 0
    expected_reactions = [
        (tuple([0, 1, -1, 1]...), "SE → E + P"),
        (tuple([-1, -1, 1, 0]...), "S + E → SE"),
        (tuple([1, 1, -1, 0]...), "SE → S + E")
    ]
    
    println("\n  Expected MM reactions:")
    for (stoich, name) in expected_reactions
        if stoich in keys(results["grouped_reactions"])
            stats = results["stoich_stats"][stoich]
            println("    ✓ $name (rate: $(round(stats.total_rate, digits=4)))")
            expected_found += 1
        else
            println("    ✗ $name (not found)")
        end
    end
    
    reaction_recovery = expected_found / length(expected_reactions) * 100
    println("  Reaction recovery: $(round(reaction_recovery, digits=1))%")
    
    # Flow analysis summary
    if haskey(results, "flow_analysis") && results["flow_analysis"] !== nothing
        flow_modes = results["flow_analysis"]["flow_modes"]
        println("\n🌊 Flow Analysis Results:")
        println("  Valid flow modes: $(length(flow_modes))")
        
        if !isempty(flow_modes)
            dominant_mode = flow_modes[1]
            println("  Dominant mode: $(dominant_mode.mode_index) [$(dominant_mode.mode_type)]")
            println("  Max flow magnitude: $(round(maximum(dominant_mode.flow_magnitude), digits=4))")
        end
    end
    
    # MM-specific analysis summary
    if haskey(results, "mm_flow_analysis") && results["mm_flow_analysis"] !== nothing
        mm_analysis = results["mm_flow_analysis"]
        println("\n🧪 MM-Specific Analysis Results:")
        
        if haskey(mm_analysis, "mm_signature") && mm_analysis["mm_signature"] !== nothing
            mm_matches = filter(m -> get(m, :matches_mm, false), mm_analysis["mm_signature"])
            if !isempty(mm_matches)
                # Find best match manually
                best_match = mm_matches[1]
                for match in mm_matches
                    if get(match, :confidence, 0) > get(best_match, :confidence, 0)
                        best_match = match
                    end
                end
                
                println("  MM signature detected: ✓")
                println("  Best matching mode: $(best_match.mode_index)")
                println("  Confidence: $(round(best_match.confidence, digits=3))")
            else
                println("  MM signature detected: ✗")
            end
        end
        
        if haskey(mm_analysis, "mm_processes") && mm_analysis["mm_processes"] !== nothing
            processes = mm_analysis["mm_processes"]
            if !isempty(processes)
                dominant_processes = [p.dominant_process for p in processes[1:min(3, end)]]
                println("  Dominant processes: $(join(unique(dominant_processes), ", "))")
            end
        end
    end
    
    # Kinetics analysis summary
    if haskey(results, "kinetics_analysis")
        kinetics = results["kinetics_analysis"]
        println("\n⚗️ Kinetics Analysis Results:")
        
        if haskey(kinetics, "kinetic_summary")
            summary = kinetics["kinetic_summary"]
            successful = filter(s -> s.quality in ["Excellent", "Good", "Fair"], summary)
            println("  Rate constants estimated: $(length(successful))/$(length(summary))")
            
            if !isempty(successful)
                avg_error = mean([s.error_pct for s in successful if !isnan(s.error_pct)])
                println("  Average error: $(round(avg_error, digits=1))%")
                
                excellent_count = count(s -> s.quality == "Excellent", successful)
                good_count = count(s -> s.quality == "Good", successful)
                println("  Quality: $excellent_count excellent, $good_count good")
            end
        end
        
        if haskey(kinetics, "mm_parameters") && haskey(kinetics["mm_parameters"], "Km")
            params = kinetics["mm_parameters"]
            println("  MM parameters derived: ✓")
            println("    Km ≈ $(round(params["Km"], digits=3))")
            println("    kcat ≈ $(round(params["kcat"], digits=5))")
        end
    end
    
    # Overall assessment
    println("\n🎯 Overall Assessment:")
    
    score = 0
    max_score = 4
    
    # Reaction recovery
    if reaction_recovery > 80
        score += 1
        println("  ✓ Excellent reaction recovery ($(round(reaction_recovery, digits=1))%)")
    elseif reaction_recovery > 50
        println("  ⚠ Moderate reaction recovery ($(round(reaction_recovery, digits=1))%)")
    else
        println("  ✗ Poor reaction recovery ($(round(reaction_recovery, digits=1))%)")
    end
    
    # Flow analysis
    if haskey(results, "flow_analysis") && results["flow_analysis"] !== nothing
        score += 1
        println("  ✓ Flow field analysis successful")
    else
        println("  ✗ Flow field analysis failed")
    end
    
    # MM signature
    if haskey(results, "mm_flow_analysis") && 
       haskey(results["mm_flow_analysis"], "mm_signature") &&
       any(get(m, :matches_mm, false) for m in results["mm_flow_analysis"]["mm_signature"])
        score += 1
        println("  ✓ MM mechanism signature detected")
    else
        println("  ⚠ MM mechanism signature unclear")
    end
    
    # Kinetics
    if haskey(results, "kinetics_analysis") && 
       haskey(results["kinetics_analysis"], "kinetic_summary")
        successful = filter(s -> s.quality in ["Excellent", "Good"], 
                          results["kinetics_analysis"]["kinetic_summary"])
        if length(successful) >= 2
            score += 1
            println("  ✓ Kinetic analysis successful")
        else
            println("  ⚠ Kinetic analysis partially successful")
        end
    else
        println("  ✗ Kinetic analysis failed")
    end
    
    # Final grade
    grade_pct = score / max_score * 100
    if grade_pct >= 75
        grade = "Excellent"
        emoji = "🏆"
    elseif grade_pct >= 50
        grade = "Good"
        emoji = "👍"
    elseif grade_pct >= 25
        grade = "Fair" 
        emoji = "👌"
    else
        grade = "Poor"
        emoji = "😞"
    end
    
    println("\n$emoji Final Grade: $grade ($(round(grade_pct, digits=0))%)")
    
    if grade == "Excellent"
        println("  All major components working correctly!")
    elseif grade == "Good"
        println("  Most components working well with minor issues.")
    elseif grade == "Fair"
        println("  Some components working, others need improvement.")
    else
        println("  Major issues detected, analysis needs debugging.")
    end
end

"""
    quick_demo()

Run a quick demonstration with default parameters.
"""
function quick_demo()
    println("🚀 Running Quick MM Demo...")
    return run_complete_analysis(200, 300)  # Smaller for speed
end

"""
    full_analysis()

Run full analysis with high-quality parameters.
"""
function full_analysis()
    println("🔬 Running Full MM Analysis...")
    return run_complete_analysis(1000, 1000)  # High quality
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
    println("  DMD rank: $(results["rank"])")
    println("  States analyzed: $(length(results["selected_states"]))")
    
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
    
    if haskey(results, "kinetics_analysis")
        kinetics = results["kinetics_analysis"]
        if haskey(kinetics, "mm_parameters") && haskey(kinetics["mm_parameters"], "Km")
            params = kinetics["mm_parameters"]
            println("\nMM Parameters:")
            println("  Km ≈ $(round(params["Km"], digits=3))")
            println("  kcat ≈ $(round(params["kcat"], digits=5))")
        end
    end
end

# Convenience aliases
const basic_mm = () -> run_mm_inference(300, 400)
const flow_mm = () -> run_with_flow_analysis(300, 400)
const mm_analysis = () -> run_with_mm_analysis(300, 400)
const complete_mm = () -> run_complete_analysis(500, 500)

println("="^50)
println("🧬 FRESH MM CRN INFERENCE SYSTEM LOADED! 🧬")
println("="^50)
println()
println("Quick Start Functions:")
println("  quick_demo()     - Fast demo (200 trajs, 300 states)")
println("  basic_mm()       - Basic inference only")
println("  flow_mm()        - With flow analysis")  
println("  mm_analysis()    - With MM-specific analysis")
println("  complete_mm()    - Complete analysis pipeline")
println("  full_analysis()  - High-quality analysis (1000 trajs)")
println()
println("Analysis Functions:")
println("  run_complete_analysis(n_trajs, max_states)")
println("  show_results_overview(results)")
println("  generate_final_summary(results)")
println()
println("Example Usage:")
println("  results = complete_mm()")
println("  show_results_overview(results)")
println()
println("🚀 Ready to analyze Michaelis-Menten kinetics!")
