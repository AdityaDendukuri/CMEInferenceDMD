# lv_system.jl - LOTKA-VOLTERRA SPECIFIC IMPLEMENTATION
# System-specific functions for predator-prey dynamics

using Catalyst
using JumpProcesses
using StableRNGs

# Load core modules
include("core_data.jl")
include("core_dmd.jl")
include("core_flow.jl")
include("core_kinetics.jl")

"""
    generate_lv_trajectories(n_trajs=500)

Generate Lotka-Volterra trajectory data.
"""
function generate_lv_trajectories(n_trajs=500)
    println("Generating Lotka-Volterra trajectories...")
    
    # Define the Lotka-Volterra reaction network
    rn = @reaction_network begin
        k₁, X --> 2X      # Prey birth
        k₂, X + Y --> 2Y  # Predation
        k₃, Y --> 0       # Predator death
    end
    
    # Parameters
    u0_integers = [:X => 50, :Y => 100]
    ps = [:k₁ => 1.0, :k₂ => 0.005, :k₃ => 0.6]
    tspan = (0.0, 30.0)
    
    # Create jump problem
    jinput = JumpInputs(rn, u0_integers, tspan, ps)
    jprob = JumpProblem(jinput, rng = StableRNG(123))
    
    # Generate trajectories
    ssa_trajs = []
    @showprogress desc="Generating LV trajectories..." for i in 1:n_trajs 
        push!(ssa_trajs, solve(jprob, SSAStepper()))
    end
    
    println("Generated $(length(ssa_trajs)) LV trajectories")
    return ssa_trajs, rn
end

"""
    get_lv_system_params()

Get LV system parameters.
"""
function get_lv_system_params()
    return Dict(
        "n_species" => 2,
        "species_names" => ["X", "Y"],  # X = prey, Y = predator
        "time_range" => (0.0, 25.0),
        "expected_reactions" => [
            [1, 0],     # X → 2X (net: +X)
            [-1, 1],    # X + Y → 2Y (net: -X, +Y)  
            [0, -1]     # Y → ∅ (net: -Y)
        ],
        "true_params" => Dict(
            "k1" => 1.0,    # Prey birth rate
            "k2" => 0.005,  # Predation rate
            "k3" => 0.6     # Predator death rate
        )
    )
end

"""
    filter_lv_reactions(sorted_stoichiometries, stoich_stats, species_names)

Apply LV-specific physical laws and filters.
"""
function filter_lv_reactions(sorted_stoichiometries, stoich_stats, species_names)
    println("\n🔍 Applying LV Physical Laws")
    println("="^40)
    
    valid_reactions = []
    invalid_reactions = []
    
    for stoich in sorted_stoichiometries
        stoich_vec = collect(stoich)
        stats = stoich_stats[stoich]
        reaction_str = format_reaction_string(stoich_vec, species_names)
        
        is_valid = true
        violation_reason = ""
        
        # LV Physical Law 1: No direct species transformation (X ↔ Y impossible)
        if length(stoich_vec) >= 2
            x_change = stoich_vec[1]
            y_change = stoich_vec[2]
            
            # Direct transformation: one species decreases, other increases by same amount
            if (x_change > 0 && y_change < 0 && x_change == -y_change) || 
               (x_change < 0 && y_change > 0 && -x_change == y_change)
                is_valid = false
                violation_reason = "Direct species transformation (X ↔ Y) is impossible"
            end
        end
        
        # LV Physical Law 2: No large creation from nothing
        if all(x >= 0 for x in stoich_vec)  # Pure creation
            total_created = sum(stoich_vec)
            if total_created > 2
                is_valid = false
                violation_reason = "Large creation from nothing violates conservation"
            end
        end
        
        # LV Physical Law 3: Reasonable stoichiometric coefficients
        max_change = maximum(abs.(stoich_vec))
        if max_change > 3
            is_valid = false
            violation_reason = "Stoichiometry too large for elementary reaction"
        end
        
        if is_valid
            push!(valid_reactions, stoich)
            println("  ✓ $reaction_str")
        else
            push!(invalid_reactions, (stoich, violation_reason))
            println("  ✗ $reaction_str - $violation_reason")
        end
    end
    
    println("\nLV filtering results:")
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
    analyze_lv_flow_patterns(flow_modes, selected_states, species_names)

Analyze LV-specific flow patterns (oscillatory behavior).
"""
function analyze_lv_flow_patterns(flow_modes, selected_states, species_names)
    println("\n=== LV Flow Pattern Analysis ===")
    
    if length(species_names) != 2
        println("LV analysis requires exactly 2 species [X, Y]")
        return nothing
    end
    
    lv_analysis = []
    
    for mode in flow_modes[1:min(3, length(flow_modes))]
        flow_magnitude = mode.flow_magnitude
        n_valid = min(length(flow_magnitude), length(selected_states))
        
        # LV process categorization
        prey_birth_flow = 0.0        # High X, low Y regions
        predation_flow = 0.0         # High X and Y regions
        predator_death_flow = 0.0    # High Y, low X regions
        oscillatory_flow = 0.0       # Intermediate regions
        
        for i in 1:n_valid
            state = selected_states[i]
            if length(state) >= 2
                x, y = [max(0, val) for val in state[1:2]]
                flow = flow_magnitude[i]
                
                # Categorize by LV process signatures
                if x > 30 && y < 50  # High prey, low predator
                    prey_birth_flow += flow
                elseif x > 20 && y > 50  # Both high (predation region)
                    predation_flow += flow
                elseif x < 30 && y > 70  # Low prey, high predator
                    predator_death_flow += flow
                else  # Intermediate regions (transition zones)
                    oscillatory_flow += flow
                end
            end
        end
        
        total_flow = sum(flow_magnitude[1:n_valid])
        
        # Calculate percentages
        percentages = Dict(
            "prey_birth" => total_flow > 0 ? (prey_birth_flow/total_flow*100) : 0.0,
            "predation" => total_flow > 0 ? (predation_flow/total_flow*100) : 0.0,
            "predator_death" => total_flow > 0 ? (predator_death_flow/total_flow*100) : 0.0,
            "oscillatory_transition" => total_flow > 0 ? (oscillatory_flow/total_flow*100) : 0.0
        )
        
        # Check for oscillatory signature
        eigenvalue = mode.eigenvalue
        is_oscillatory = abs(imag(eigenvalue)) > 0.01
        
        push!(lv_analysis, (
            mode_index = mode.mode_index,
            mode_type = mode.mode_type,
            eigenvalue = eigenvalue,
            is_oscillatory = is_oscillatory,
            oscillation_period = is_oscillatory ? 2π / abs(imag(eigenvalue)) : Inf,
            percentages = percentages
        ))
        
        println("\nMode $(mode.mode_index) [$(mode.mode_type)]:")
        if is_oscillatory
            period = 2π / abs(imag(eigenvalue))
            println("  🌀 Oscillatory (period: $(round(period, digits=2)))")
        end
        for (process, pct) in percentages
            if pct > 5.0
                println("  $(process): $(round(pct, digits=1))%")
            end
        end
    end
    
    return lv_analysis
end

"""
    assess_lv_recovery(valid_reactions, expected_reactions)

Assess how well LV reactions were recovered.
"""
function assess_lv_recovery(valid_reactions, expected_reactions)
    println("\n=== LV Recovery Assessment ===")
    
    recovery_count = 0
    
    println("Expected LV reactions:")
    for (i, expected) in enumerate(expected_reactions)
        expected_tuple = tuple(expected...)
        found = expected_tuple in valid_reactions
        
        if found
            recovery_count += 1
            println("  ✓ $(format_reaction_string(expected, ["X", "Y"]))")
        else
            println("  ✗ $(format_reaction_string(expected, ["X", "Y"]))")
        end
    end
    
    recovery_rate = (recovery_count / length(expected_reactions)) * 100
    println("\nLV Recovery Rate: $(round(recovery_rate, digits=1))% ($(recovery_count)/$(length(expected_reactions)))")
    
    return recovery_rate
end

"""
    select_lv_states_specialized(trajectories, n_species; max_states=800)

Specialized state selection for LV oscillatory dynamics.
"""
function select_lv_states_specialized(trajectories, n_species; max_states=800)
    println("Selecting LV states with oscillatory-aware method...")
    
    # Collect states along actual trajectory paths (not just endpoints)
    trajectory_states = Set()
    phase_space_coverage = Dict()
    
    # Sample more trajectories for better phase space coverage
    for traj in trajectories[1:min(200, length(trajectories))]
        # Sample states more densely along trajectories
        n_samples = min(50, length(traj.t))
        sample_indices = round.(Int, range(1, length(traj.t), length=n_samples))
        
        for idx in sample_indices
            state = [traj.u[idx][j] for j in 1:n_species]
            state_tuple = tuple(state...)
            push!(trajectory_states, state_tuple)
            
            # Track phase space regions
            if n_species >= 2
                x, y = state[1], state[2]
                # Coarse-grain phase space into regions
                region = (div(x, 10), div(y, 10))
                phase_space_coverage[region] = get(phase_space_coverage, region, 0) + 1
            end
        end
    end
    
    trajectory_states = collect(trajectory_states)
    println("Found $(length(trajectory_states)) states in LV trajectory flow")
    
    # Select states with bias toward well-covered phase space regions
    n_select = min(max_states, length(trajectory_states))
    
    if length(trajectory_states) <= max_states
        selected_states = trajectory_states
    else
        # Weight by phase space coverage to ensure good representation
        state_weights = []
        for state_tuple in trajectory_states
            if length(state_tuple) >= 2
                x, y = state_tuple[1], state_tuple[2]
                region = (div(x, 10), div(y, 10))
                weight = get(phase_space_coverage, region, 1)
                push!(state_weights, weight)
            else
                push!(state_weights, 1)
            end
        end
        
        # Select with weighted sampling (higher weight = higher probability)
        sorted_indices = sortperm(state_weights, rev=true)
        selected_states = trajectory_states[sorted_indices[1:n_select]]
    end
    
    println("Selected $(length(selected_states)) LV states with phase space weighting")
    
    # Convert format
    return [collect(state) for state in selected_states]
end

"""
    run_lv_analysis(analysis_params=Dict())

Run complete LV analysis pipeline.
"""
function run_lv_analysis(analysis_params=Dict())
    println("="^70)
    println("LOTKA-VOLTERRA ANALYSIS PIPELINE")
    println("="^70)
    
    # Get system parameters
    system_params = get_lv_system_params()
    
    # Set default analysis parameters (tuned for oscillatory systems)
    default_params = Dict(
        "n_trajs" => 500,
        "max_states" => 800,  # Larger for phase space coverage
        "n_time_points" => 25,  # More points to capture oscillations
        "segment_length" => 6,  # Shorter segments for local dynamics
        "overlap_fraction" => 0.5,  # More overlap for continuity
        "use_reachability" => true,
        "masking_strength" => 0.8,  # Less aggressive for oscillatory systems
        "use_specialized_selection" => true
    )
    
    # Merge with user parameters
    analysis_params = merge(default_params, analysis_params)
    
    # Step 1: Basic data processing with LV-specific modifications
    println("\n🦌 STEP 1: LV Data Processing")
    
    # Generate trajectories
    trajectories, reaction_network = generate_lv_trajectories(analysis_params["n_trajs"])
    
    # Create modified data dict with LV-specific state selection
    n_species = system_params["n_species"]
    species_names = system_params["species_names"]
    time_range = system_params["time_range"]
    
    time_points = range(time_range[1], time_range[2], length=analysis_params["n_time_points"])
    dt = time_points[2] - time_points[1]
    
    # Process trajectories
    histograms = process_trajectories_to_histograms(trajectories, time_points, n_species)
    
    # Use specialized state selection for LV
    if analysis_params["use_specialized_selection"]
        selected_states = select_lv_states_specialized(trajectories, n_species, 
                                                     max_states=analysis_params["max_states"])
    else
        selected_states = select_states_from_trajectories(trajectories, n_species, 
                                                        max_states=analysis_params["max_states"])
    end
    
    prob_matrix = convert_histograms_to_matrix(histograms, selected_states)
    capture_rate = diagnose_data_quality(trajectories, selected_states, n_species)
    
    data_dict = Dict(
        "trajectories" => trajectories,
        "reaction_network" => reaction_network,
        "time_points" => time_points,
        "dt" => dt,
        "species_names" => species_names,
        "n_species" => n_species,
        "histograms" => histograms,
        "selected_states" => selected_states,
        "probability_matrix" => prob_matrix,
        "capture_rate" => capture_rate,
        "system_params" => system_params
    )
    
    # Step 2: DMD Analysis
    println("\n🔄 STEP 2: LV DMD Analysis")
    
    # Compute reachability matrix if requested
    reachability_matrix = nothing
    if analysis_params["use_reachability"]
        println("Computing reachability matrix...")
        reachability_matrix, _, _ = compute_reachability_matrix(
            data_dict["trajectories"], 
            data_dict["selected_states"],
            min_observations=1,  # Lower threshold for oscillatory systems
            confidence_threshold=0.6  # Lower threshold
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
    
    # Extract reactions
    sorted_stoichiometries, stoich_stats = extract_reactions_from_generator(
        G_combined, data_dict["selected_states"]
    )
    
    # Apply LV-specific filtering
    valid_reactions, filtered_stats, invalid_reactions = filter_lv_reactions(
        sorted_stoichiometries, stoich_stats, data_dict["species_names"]
    )
    
    # Step 3: Flow Analysis
    println("\n🌊 STEP 3: LV Flow Analysis")
    flow_results = run_flow_analysis(λ_combined, Φ_combined, data_dict["selected_states"])
    
    # LV-specific flow pattern analysis
    lv_flow_analysis = analyze_lv_flow_patterns(
        flow_results["flow_modes"], 
        data_dict["selected_states"], 
        data_dict["species_names"]
    )
    
    # Step 4: Kinetics Analysis
    println("\n⚗️ STEP 4: LV Kinetics Analysis")
    kinetics_results = run_kinetics_analysis(
        valid_reactions, filtered_stats, 
        data_dict["selected_states"], data_dict["species_names"]
    )
    
    # Step 5: LV Recovery Assessment
    println("\n📊 STEP 5: LV Recovery Assessment")
    recovery_rate = assess_lv_recovery(valid_reactions, system_params["expected_reactions"])
    
    # Check for oscillatory behavior
    has_oscillations = any(get(analysis, :is_oscillatory, false) for analysis in lv_flow_analysis)
    
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
        "lv_flow_analysis" => lv_flow_analysis,
        "kinetics_results" => kinetics_results,
        "recovery_rate" => recovery_rate,
        "has_oscillations" => has_oscillations,
        "reachability_matrix" => reachability_matrix
    ))
    
    println("\n" * "="^70)
    println("LV ANALYSIS COMPLETED")
    println("Recovery Rate: $(round(recovery_rate, digits=1))%")
    println("Valid Reactions: $(length(valid_reactions))")
    println("Oscillatory Behavior: $(has_oscillations ? "✓ Detected" : "✗ Not detected")")
    println("Spurious Reactions Eliminated: $(length(invalid_reactions))")
    println("="^70)
    
    return results
end

# Convenience functions
const run_lv = run_lv_analysis
const quick_lv = () -> run_lv_analysis(Dict("n_trajs" => 200, "max_states" => 400))

println("="^60)
println("🦌 LV SYSTEM MODULE LOADED! 🐺")
println("="^60)
println()
println("Main Functions:")
println("  run_lv_analysis(params)     - Complete LV analysis")
println("  run_lv()                    - Default LV analysis")
println("  quick_lv()                  - Fast LV test")
println()
println("System Parameters:")
println("  Species: X (prey), Y (predator)")
println("  Expected reactions: 3")
println("  Physical laws: No direct species transformation")
println("  Specialized: Oscillatory-aware state selection")
println()
println("Example:")
println("  results = run_lv()")
println("  results = quick_lv()")
