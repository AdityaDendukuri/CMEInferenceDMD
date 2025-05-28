# integration.jl - CLEAN INTEGRATION MODULE
# Brings together all components for complete analysis

# Load all modules
include("main.jl")
include("dmd.jl")
include("flow.jl")
include("kinetics.jl")

"""
    run_complete_mm_analysis(n_trajs=500, max_states=500, n_time_points=20; 
                            segment_length=8, overlap_fraction=0.3)

Run complete MM analysis pipeline with all components.
"""
function run_complete_mm_analysis(n_trajs=500, max_states=500, n_time_points=20; 
                                 segment_length=8, overlap_fraction=0.3,
                                 use_reachability=true, masking_strength=1.0,
                                 min_observations=2, confidence_threshold=0.8)
    
    println("="^70)
    println("COMPLETE MM ANALYSIS PIPELINE")
    println("="^70)
    
    # Step 1: Basic data processing
    println("\n🔬 STEP 1: Data Generation and Processing")
    data_dict = run_basic_mm_inference(n_trajs, max_states, n_time_points)
    
    # Step 2: DMD Analysis (Multigrid by default) with reachability options
    println("\n🔄 STEP 2: DMD Analysis")
    data_dict = run_dmd_analysis(data_dict, 
                                dmd_method=:multigrid, 
                                segment_length=segment_length, 
                                overlap_fraction=overlap_fraction,
                                use_reachability=use_reachability,
                                masking_strength=masking_strength,
                                min_observations=min_observations,
                                confidence_threshold=confidence_threshold)
    
    # Step 3: Flow Analysis
    println("\n🌊 STEP 3: Flow Field Analysis") 
    data_dict = run_flow_analysis(data_dict)
    
    # Step 4: Kinetics Analysis
    println("\n⚗️ STEP 4: Kinetics Analysis")
    data_dict = run_kinetics_analysis(data_dict)
    
    # Step 5: Summary
    println("\n📊 STEP 5: Analysis Summary")
    generate_analysis_summary(data_dict)
    
    println("\n" * "="^70)
    println("COMPLETE ANALYSIS FINISHED")
    println("="^70)
    
    return data_dict
end

"""
    generate_analysis_summary(data_dict)

Generate comprehensive summary of all analyses.
"""
function generate_analysis_summary(data_dict)
    println("\n" * "="^50)
    println("ANALYSIS SUMMARY")
    println("="^50)
    
    # Basic info
    println("\n🔬 Data Processing:")
    println("  Trajectories: $(length(data_dict["trajectories"]))")
    println("  Time points: $(length(data_dict["time_points"]))")
    println("  Selected states: $(length(data_dict["selected_states"]))")
    println("  Species: $(join(data_dict["species_names"], ", "))")
    
    # DMD results
    if haskey(data_dict, "significant_stoichiometries")
        println("\n🔄 DMD Analysis:")
        println("  Method: $(get(data_dict, "dmd_method", "unknown"))")
        println("  Reactions found: $(length(data_dict["significant_stoichiometries"]))")
        
        if haskey(data_dict, "successful_segments")
            println("  Successful segments: $(data_dict["successful_segments"])")
        end
        
        # Check MM reaction recovery
        mm_reactions_found = count_mm_reactions_found(data_dict["significant_stoichiometries"])
        println("  MM reactions recovered: $mm_reactions_found/3")
        
        # Top reactions
        println("\n  Top 5 reactions:")
        for (i, stoich) in enumerate(data_dict["significant_stoichiometries"][1:min(5, end)])
            if haskey(data_dict, "reaction_stats")
                stats = data_dict["reaction_stats"][stoich]
                reaction_str = format_reaction(collect(stoich), data_dict["species_names"])
                println("    $i. $reaction_str (rate: $(round(stats.total_rate, digits=4)))")
            end
        end
    end
    
    # Flow analysis results
    if haskey(data_dict, "flow_modes")
        println("\n🌊 Flow Analysis:")
        flow_modes = data_dict["flow_modes"]
        println("  Flow modes computed: $(length(flow_modes))")
        
        if !isempty(flow_modes)
            dominant_mode = flow_modes[1]
            println("  Dominant mode type: $(dominant_mode.mode_type)")
            println("  Decay time: $(round(dominant_mode.decay_time, digits=2)) time units")
        end
        
        # MM signature detection
        if haskey(data_dict, "mm_signature") && data_dict["mm_signature"] !== nothing
            mm_matches = filter(m -> get(m, :matches_mm, false), data_dict["mm_signature"])
            if !isempty(mm_matches)
                best_match = mm_matches[1]
                for match in mm_matches
                    if get(match, :confidence, 0) > get(best_match, :confidence, 0)
                        best_match = match
                    end
                end
                println("  MM signature detected: ✓ (confidence: $(round(best_match.confidence, digits=3)))")
            else
                println("  MM signature detected: ✗")
            end
        end
    end
    
    # Kinetics analysis results
    if haskey(data_dict, "kinetics_analysis")
        println("\n⚗️ Kinetics Analysis:")
        kinetics = data_dict["kinetics_analysis"]
        
        if haskey(kinetics, "kinetic_summary")
            summary = kinetics["kinetic_summary"]
            successful = filter(s -> s.quality in ["Excellent", "Good", "Fair"], summary)
            println("  Rate constants estimated: $(length(successful))/$(length(summary))")
            
            if !isempty(successful)
                avg_error = mean([s.error_pct for s in successful if !isnan(s.error_pct)])
                if !isnan(avg_error)
                    println("  Average error: $(round(avg_error, digits=1))%")
                end
                
                excellent_count = count(s -> s.quality == "Excellent", successful)
                good_count = count(s -> s.quality == "Good", successful)
                println("  Quality: $excellent_count excellent, $good_count good")
            end
        end
        
        # MM parameters
        if haskey(kinetics, "mm_parameters") && !isempty(kinetics["mm_parameters"])
            params = kinetics["mm_parameters"]
            println("  MM parameters derived: ✓")
            if haskey(params, "Km")
                println("    Km ≈ $(round(params["Km"], digits=3))")
            end
            if haskey(params, "kcat")
                println("    kcat ≈ $(round(params["kcat"], digits=5))")
            end
            if haskey(params, "kinetic_regime")
                println("    Regime: $(kinetics["kinetic_regime"])")
            end
        else
            println("  MM parameters derived: ✗")
        end
    end
    
    # Overall assessment
    println("\n🎯 Overall Assessment:")
    
    score = 0
    max_score = 4
    
    # DMD success
    if haskey(data_dict, "significant_stoichiometries") && !isempty(data_dict["significant_stoichiometries"])
        mm_recovery = count_mm_reactions_found(data_dict["significant_stoichiometries"])
        if mm_recovery >= 2
            score += 1
            println("  ✓ DMD analysis successful ($(mm_recovery)/3 MM reactions)")
        else
            println("  ⚠ DMD analysis partial ($(mm_recovery)/3 MM reactions)")
        end
    else
        println("  ✗ DMD analysis failed")
    end
    
    # Flow analysis success
    if haskey(data_dict, "flow_modes") && !isempty(data_dict["flow_modes"])
        score += 1
        println("  ✓ Flow analysis successful")
    else
        println("  ✗ Flow analysis failed")
    end
    
    # MM signature detection
    if haskey(data_dict, "mm_signature") && data_dict["mm_signature"] !== nothing
        mm_detected = any(get(m, :matches_mm, false) for m in data_dict["mm_signature"])
        if mm_detected
            score += 1
            println("  ✓ MM signature detected")
        else
            println("  ⚠ MM signature unclear")
        end
    else
        println("  ✗ MM signature analysis failed")
    end
    
    # Kinetics analysis success
    if haskey(data_dict, "kinetics_analysis") && haskey(data_dict["kinetics_analysis"], "mm_parameters")
        if !isempty(data_dict["kinetics_analysis"]["mm_parameters"])
            score += 1
            println("  ✓ Kinetics analysis successful")
        else
            println("  ⚠ Kinetics analysis partial")
        end
    else
        println("  ✗ Kinetics analysis failed")
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
    count_mm_reactions_found(stoichiometries)

Count how many expected MM reactions were found.
"""
function count_mm_reactions_found(stoichiometries)
    expected_mm = [
        [-1, -1, 1, 0],  # S + E → SE
        [1, 1, -1, 0],   # SE → S + E
        [0, 1, -1, 1]    # SE → E + P
    ]
    
    count = 0
    for expected in expected_mm
        if tuple(expected...) in stoichiometries
            count += 1
        end
    end
    
    return count
end

"""
    run_quick_mm_test(n_trajs=200, max_states=300; use_reachability=true, masking_strength=1.0)

Quick test with smaller parameters for rapid iteration.
"""
function run_quick_mm_test(n_trajs=200, max_states=300; use_reachability=true, masking_strength=1.0)
    println("🚀 Running Quick MM Test...")
    return run_complete_mm_analysis(n_trajs, max_states, 15, 
                                   segment_length=6, overlap_fraction=0.4,
                                   use_reachability=use_reachability, 
                                   masking_strength=masking_strength)
end

"""
    run_high_quality_mm_analysis(n_trajs=1000, max_states=1000; use_reachability=true, masking_strength=1.0)

High-quality analysis with larger parameters.
"""
function run_high_quality_mm_analysis(n_trajs=1000, max_states=1000; use_reachability=true, masking_strength=1.0)
    println("🔬 Running High-Quality MM Analysis...")
    return run_complete_mm_analysis(n_trajs, max_states, 30, 
                                   segment_length=10, overlap_fraction=0.3,
                                   use_reachability=use_reachability,
                                   masking_strength=masking_strength)
end

"""
    show_results_overview(data_dict)

Show concise overview of analysis results.
"""
function show_results_overview(data_dict)
    println("\n" * "="^40)
    println("RESULTS OVERVIEW")
    println("="^40)
    
    println("System: MM kinetics ($(join(data_dict["species_names"], ", ")))")
    
    if haskey(data_dict, "dmd_method")
        println("DMD Method: $(data_dict["dmd_method"])")
    end
    
    if haskey(data_dict, "significant_stoichiometries")
        println("Reactions Found: $(length(data_dict["significant_stoichiometries"]))")
        
        println("\nTop 3 Reactions:")
        for (i, stoich) in enumerate(data_dict["significant_stoichiometries"][1:min(3, end)])
            if haskey(data_dict, "reaction_stats")
                stats = data_dict["reaction_stats"][stoich]
                reaction_str = format_reaction(collect(stoich), data_dict["species_names"])
                println("  $i. $reaction_str (rate: $(round(stats.total_rate, digits=4)))")
            end
        end
    end
    
    if haskey(data_dict, "flow_modes") && !isempty(data_dict["flow_modes"])
        println("\nFlow Analysis: $(length(data_dict["flow_modes"])) modes")
        dominant_mode = data_dict["flow_modes"][1]
        println("  Dominant: $(dominant_mode.mode_type)")
    end
    
    if haskey(data_dict, "kinetics_analysis") && haskey(data_dict["kinetics_analysis"], "mm_parameters")
        params = data_dict["kinetics_analysis"]["mm_parameters"]
        if !isempty(params)
            println("\nMM Parameters:")
            if haskey(params, "Km")
                println("  Km ≈ $(round(params["Km"], digits=3))")
            end
            if haskey(params, "kcat")
                println("  kcat ≈ $(round(params["kcat"], digits=5))")
            end
        end
    end
end

# Convenience aliases
const quick_mm = run_quick_mm_test
const complete_mm = run_complete_mm_analysis  
const high_quality_mm = run_high_quality_mm_analysis

println("="^70)
println("🧬 CLEAN MM CRN INFERENCE SYSTEM LOADED! 🧬")
println("="^70)
println()
println("🚀 Quick Start Functions:")
println("  quick_mm()              - Fast test (200 trajs, 300 states)")
println("  complete_mm()           - Standard analysis (500 trajs, 500 states)")
println("  high_quality_mm()       - High-quality analysis (1000 trajs, 1000 states)")
println()
println("📊 Analysis Functions:")
println("  run_complete_mm_analysis(n_trajs, max_states, n_times)")
println("  show_results_overview(results)")
println("  generate_analysis_summary(results)")
println()
println("🔧 Components:")
println("  • main.jl     - Data generation and processing")
println("  • dmd.jl      - Multigrid DMD analysis (default)")
println("  • flow.jl     - Flow field and MM pattern analysis")
println("  • kinetics.jl - Rate estimation and MM parameters")
println()
println("✨ Features:")
println("  • Clean modular architecture")
println("  • Multigrid DMD as default method")
println("  • Comprehensive flow analysis")
println("  • MM-specific pattern detection")
println("  • Automatic kinetics analysis")
println("  • Quality assessment and grading")
println()
println("Example Usage:")
println("  results = quick_mm()")
println("  show_results_overview(results)")
println()
println("🎯 Ready for clean, comprehensive MM analysis!")
