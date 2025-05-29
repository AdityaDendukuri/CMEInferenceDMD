# mm_system.jl - MICHAELIS-MENTEN SPECIFIC IMPLEMENTATION
# System-specific functions for MM kinetics

using Catalyst
using JumpProcesses
using DifferentialEquations
using ProgressMeter

# Load core modules
include("core_data.jl")
include("core_dmd.jl")
include("core_flow.jl")
include("core_kinetics.jl")

"""
    generate_mm_trajectories(n_trajs=500)

Generate Michaelis-Menten trajectory data.
"""
function generate_mm_trajectories(n_trajs=500)
    println("Generating MM trajectory data...")
    
    # Define the Michaelis-Menten reaction network
    rn = @reaction_network begin
        kB, S + E --> SE
        kD, SE --> S + E
        kP, SE --> P + E
    end
    
    # Parameters
    u0_integers = [:S => 50, :E => 10, :SE => 1, :P => 1]
    tspan = (0.0, 200.0)
    ps = [:kB => 0.01, :kD => 0.1, :kP => 0.1]
    
    # Create jump problem
    jinput = JumpInputs(rn, u0_integers, tspan, ps)
    jprob = JumpProblem(jinput)
    
    # Generate trajectories
    ssa_trajs = []
    @showprogress desc="Generating MM trajectories..." for i in 1:n_trajs 
        push!(ssa_trajs, solve(jprob, SSAStepper()))
    end
    
    println("Generated $(length(ssa_trajs)) MM trajectories")
    return ssa_trajs, rn
end

"""
    get_mm_system_params()

Get MM system parameters.
"""
function get_mm_system_params()
    return Dict(
        "n_species" => 4,
        "species_names" => ["S", "E", "SE", "P"],
        "time_range" => (0.0, 200.0),  # Match trajectory generation span
        "expected_reactions" => [
            [-1, -1, 1, 0],   # S + E → SE
            [1, 1, -1, 0],    # SE → S + E  
            [0, 1, -1, 1]     # SE → E + P
        ],
        "true_params" => Dict(
            "kB" => 0.01,
            "kD" => 0.1, 
            "kP" => 0.1
        )
    )
end

"""
    filter_mm_reactions(sorted_stoichiometries, stoich_stats, species_names)

Apply MM-specific conservation laws and filters.
"""
function filter_mm_reactions(sorted_stoichiometries, stoich_stats, species_names)
    println("\n🔍 Applying MM Conservation Laws")
    println("="^40)
    
    valid_reactions = []
    invalid_reactions = []
    
    for stoich in sorted_stoichiometries
        stoich_vec = collect(stoich)
        stats = stoich_stats[stoich]
        reaction_str = format_reaction_string(stoich_vec, species_names)
        
        is_valid = true
        violation_reason = ""
        
        # MM Conservation Law 1: Substrate conservation (S + SE + P = constant)
        substrate_change = stoich_vec[1] + stoich_vec[3] + stoich_vec[4]  # ΔS + ΔSE + ΔP
        if abs(substrate_change) > 0
            is_valid = false
            violation_reason = "Violates substrate conservation (S + SE + P ≠ constant)"
        end
        
        # MM Conservation Law 2: Enzyme conservation (E + SE = constant)
        enzyme_change = stoich_vec[2] + stoich_vec[3]  # ΔE + ΔSE
        if abs(enzyme_change) > 0
            is_valid = false
            violation_reason = "Violates enzyme conservation (E + SE ≠ constant)"
        end
        
        # MM Physical Law: No direct S → P conversion
        if stoich_vec[1] < 0 && stoich_vec[4] > 0 && stoich_vec[2] == 0 && stoich_vec[3] == 0
            is_valid = false
            violation_reason = "Direct substrate-to-product conversion impossible"
        end
        
        if is_valid
            push!(valid_reactions, stoich)
            println("  ✓ $reaction_str")
        else
            push!(invalid_reactions, (stoich, violation_reason))
            println("  ✗ $reaction_str - $violation_reason")
        end
    end
    
    println("\nMM filtering results:")
    println("  Valid reactions: $(length(valid_reactions))")
    println("  Invalid reactions: $(length(invalid_reactions))")
    
    # Create filtered stats
    filtered_stats = Dict()
    for stoich in valid_reactions
        filtered_stats[stoich] = stoich_stats[stoich]
    end
    
    return valid_reactions, filtered_stats, invalid_reactions
end

"""
    analyze_mm_flow_patterns(flow_modes, selected_states, species_names)

Analyze MM-specific flow patterns.
"""
function analyze_mm_flow_patterns(flow_modes, selected_states, species_names)
    println("\n=== MM Flow Pattern Analysis ===")
    
    if length(species_names) != 4
        println("MM analysis requires exactly 4 species [S, E, SE, P]")
        return nothing
    end
    
    mm_analysis = []
    
    for mode in flow_modes[1:min(3, length(flow_modes))]
        flow_magnitude = mode.flow_magnitude
        n_valid = min(length(flow_magnitude), length(selected_states))
        
        # MM process categorization
        substrate_binding_flow = 0.0
        complex_dissociation_flow = 0.0
        product_formation_flow = 0.0
        enzyme_recycling_flow = 0.0
        
        for i in 1:n_valid
            state = selected_states[i]
            if length(state) >= 4
                s, e, se, p = [max(0, x) for x in state[1:4]]
                flow = flow_magnitude[i]
                
                # Categorize by MM process signatures
                if s > 10 && e > 2 && se < 8
                    substrate_binding_flow += flow
                end
                
                if se > 3 && (s > 0 || e > 0)
                    complex_dissociation_flow += flow
                end
                
                if p > s && e > se
                    product_formation_flow += flow
                end
                
                if e > 5 && se < 3
                    enzyme_recycling_flow += flow
                end
            end
        end
        
        total_flow = sum(flow_magnitude[1:n_valid])
        
        # Calculate percentages
        percentages = Dict(
            "substrate_binding" => total_flow > 0 ? (substrate_binding_flow/total_flow*100) : 0.0,
            "complex_dissociation" => total_flow > 0 ? (complex_dissociation_flow/total_flow*100) : 0.0,
            "product_formation" => total_flow > 0 ? (product_formation_flow/total_flow*100) : 0.0,
            "enzyme_recycling" => total_flow > 0 ? (enzyme_recycling_flow/total_flow*100) : 0.0
        )
        
        push!(mm_analysis, (
            mode_index = mode.mode_index,
            mode_type = mode.mode_type,
            percentages = percentages
        ))
        
        println("\nMode $(mode.mode_index) [$(mode.mode_type)]:")
        for (process, pct) in percentages
            if pct > 5.0
                println("  $(process): $(round(pct, digits=1))%")
            end
        end
    end
    
    return mm_analysis
end

"""
    assess_mm_recovery(valid_reactions, expected_reactions)

Assess how well MM reactions were recovered.
"""
function assess_mm_recovery(valid_reactions, expected_reactions)
    println("\n=== MM Recovery Assessment ===")
    
    recovery_count = 0
    
    println("Expected MM reactions:")
    for (i, expected) in enumerate(expected_reactions)
        expected_tuple = tuple(expected...)
        found = expected_tuple in valid_reactions
        
        if found
            recovery_count += 1
            println("  ✓ $(format_reaction_string(expected, ["S", "E", "SE", "P"]))")
        else
            println("  ✗ $(format_reaction_string(expected, ["S", "E", "SE", "P"]))")
        end
    end
    
    recovery_rate = (recovery_count / length(expected_reactions)) * 100
    println("\nMM Recovery Rate: $(round(recovery_rate, digits=1))% ($(recovery_count)/$(length(expected_reactions)))")
    
    return recovery_rate
end

"""
    run_mm_analysis(analysis_params=Dict())

Run complete MM analysis pipeline.
"""
function run_mm_analysis(analysis_params=Dict())
    println("="^70)
    println("MICHAELIS-MENTEN ANALYSIS PIPELINE")
    println("="^70)
    
    # Get system parameters
    system_params = get_mm_system_params()
    
    # Set default analysis parameters
    default_params = Dict(
        "n_trajs" => 500,
        "max_states" => 500,
        "n_time_points" => 20,
        "segment_length" => 8,
        "overlap_fraction" => 0.3,
        "use_reachability" => true,
        "masking_strength" => 1.0
    )
    
    # Merge with user parameters
    analysis_params = merge(default_params, analysis_params)
    
    # Step 1: Basic data processing
    println("\n🔬 STEP 1: Data Processing")
    data_dict = run_basic_data_processing(generate_mm_trajectories, system_params, analysis_params)
    
    # Step 2: DMD Analysis
    println("\n🔄 STEP 2: DMD Analysis")
    
    # Compute reachability matrix if requested
    reachability_matrix = nothing
    if analysis_params["use_reachability"]
        println("Computing reachability matrix...")
        reachability_matrix, _, _ = compute_reachability_matrix(
            data_dict["trajectories"], 
            data_dict["selected_states"]
        )
    end
    
    # Run multigrid DMD
    G_combined, λ_combined, Φ_combined, successful_segments = run_multigrid_dmd(
        data_dict["probability_matrix"],
        data_dict["dt"],
        data_dict["selected_states"],
        segment_length=analysis_params["segment_length"],
        overlap_fraction=analysis_params["overlap_fraction"],
        use_reachability=analysis_params["use_reachability"],
        reachability_matrix=reachability_matrix,
        masking_strength=analysis_params["masking_strength"]
    )
    
    # DEBUG: Check generator matrix and time step
    println("\n🔧 GENERATOR MATRIX DEBUG:")
    println("  dt = $(data_dict["dt"])")
    println("  Max |G| entry = $(maximum(abs.(G_combined)))")
    println("  Expected rate scale: For MM with k~0.01-0.1, expect G entries ~0.01-1.0")
    println("  Actual vs Expected ratio: $(maximum(abs.(G_combined)) / 0.1)")
    println("  Sample G entries:")
    for i in 1:min(3, size(G_combined, 1))
        for j in 1:min(3, size(G_combined, 2))
            if abs(G_combined[i,j]) > 1e-8
                println("    G[$i,$j] = $(G_combined[i,j])")
            end
        end
    end
    println("  Time range: $(data_dict["time_points"][1]) to $(data_dict["time_points"][end])")
    println("  Matrix shape: $(size(data_dict["probability_matrix"]))")
    
    # DEBUG: Check if the issue is in trajectory generation
    println("\n🔧 TRAJECTORY VALIDATION:")
    sample_traj = data_dict["trajectories"][1]
    println("  Sample trajectory length: $(length(sample_traj.t))")
    println("  Initial state: $(sample_traj.u[1])")
    println("  Final state: $(sample_traj.u[end])")
    println("  Time span: $(sample_traj.t[1]) to $(sample_traj.t[end])")
    
    # Check if trajectories have reasonable transition rates
    n_transitions = length(sample_traj.t) - 1
    total_time = sample_traj.t[end] - sample_traj.t[1]
    avg_rate = n_transitions / total_time
    println("  Average transition rate: $avg_rate transitions/time")
    println("  Expected MM rate scale: ~0.01*S*E + 0.1*SE + 0.1*SE ~ 1-10 transitions/time")
    
    # Extract reactions
    sorted_stoichiometries, stoich_stats = extract_reactions_from_generator(
        G_combined, data_dict["selected_states"]
    )
    
    # Apply MM-specific filtering
    valid_reactions, filtered_stats, invalid_reactions = filter_mm_reactions(
        sorted_stoichiometries, stoich_stats, data_dict["species_names"]
    )
    
    # Step 3: Flow Analysis
    println("\n🌊 STEP 3: Flow Analysis")
    flow_results = run_flow_analysis(λ_combined, Φ_combined, data_dict["selected_states"])
    
    # MM-specific flow pattern analysis
    mm_flow_analysis = analyze_mm_flow_patterns(
        flow_results["flow_modes"], 
        data_dict["selected_states"], 
        data_dict["species_names"]
    )
    
    # Step 4: Kinetics Analysis
    println("\n⚗️ STEP 4: Kinetics Analysis")
    kinetics_results = run_kinetics_analysis(
        valid_reactions, filtered_stats, 
        data_dict["selected_states"], data_dict["species_names"]
    )
    
    # Step 5: MM Recovery Assessment
    println("\n📊 STEP 5: Recovery Assessment")
    recovery_rate = assess_mm_recovery(valid_reactions, system_params["expected_reactions"])
    
    # Compile final results
    results = merge(data_dict, Dict(
        "generator" => G_combined,
        "eigenvalues" => λ_combined,
        "modes" => Φ_combined,
        "successful_segments" => successful_segments,
        "sorted_stoichiometries" => sorted_stoichiometries,
        "valid_reactions" => valid_reactions,
        "invalid_reactions" => invalid_reactions,
        "stoich_stats" => filtered_stats,
        "flow_results" => flow_results,
        "mm_flow_analysis" => mm_flow_analysis,
        "kinetics_results" => kinetics_results,
        "recovery_rate" => recovery_rate,
        "reachability_matrix" => reachability_matrix
    ))
    
    println("\n" * "="^70)
    println("MM ANALYSIS COMPLETED")
    println("Recovery Rate: $(round(recovery_rate, digits=1))%")
    println("Valid Reactions: $(length(valid_reactions))")
    println("Spurious Reactions Eliminated: $(length(invalid_reactions))")
    println("="^70)
    
    return results
end

# Convenience functions
const run_mm = run_mm_analysis
const quick_mm = () -> run_mm_analysis(Dict("n_trajs" => 200, "max_states" => 300))

println("="^60)
println("🧬 MM SYSTEM MODULE LOADED! 🧬")
println("="^60)
println()
println("Main Functions:")
println("  run_mm_analysis(params)     - Complete MM analysis")
println("  run_mm()                    - Default MM analysis")
println("  quick_mm()                  - Fast MM test")
println()
println("System Parameters:")
println("  Species: S, E, SE, P")
println("  Expected reactions: 3")
println("  Conservation laws: Substrate + Enzyme")
println()
println("Example:")
println("  results = run_mm()")
println("  results = quick_mm()")
