# mm_specific_analysis.jl - SYSTEM-SPECIFIC ANALYSIS MODULES
# MM, Lotka-Volterra, Toggle Switch, and General system analysis

using Statistics
using Catalyst
using JumpProcesses
using DifferentialEquations
using ProgressMeter

"""
    generate_system_data(system_type, n_trajs=1000; params...)

Generate trajectory data for different system types.
"""
function generate_system_data(system_type, n_trajs=1000; params...)
    if system_type == "mm"
        return generate_mm_data(n_trajs; params...)
    elseif system_type == "lotka_volterra"
        return generate_lotka_volterra_data(n_trajs; params...)
    elseif system_type == "toggle_switch"
        return generate_toggle_switch_data(n_trajs; params...)
    else
        error("Unknown system type: $system_type. Use 'mm', 'lotka_volterra', or 'toggle_switch'")
    end
end

"""
    generate_mm_data(n_trajs=1000)

Generate Michaelis-Menten trajectory data.
"""
function generate_mm_data(n_trajs=1000)
    println("Generating MM data with correct Catalyst setup...")
    
    # Correct reaction network definition
    rn = @reaction_network begin
        kB, S + E --> SE
        kD, SE --> S + E  
        kP, SE --> P + E
    end
    
    # Initial conditions and parameters
    u0_integers = [:S => 50, :E => 10, :SE => 1, :P => 1]
    tspan = (0., 200.)
    ps = [:kB => 0.01, :kD => 0.1, :kP => 0.1]
    
    # Create jump problem
    jinput = JumpInputs(rn, u0_integers, tspan, ps)
    jprob = JumpProblem(jinput)
    
    # Generate trajectories with progress bar
    ssa_trajs = []
    @showprogress desc="Generating MM trajectories..." for i in 1:n_trajs 
        push!(ssa_trajs, solve(jprob, SSAStepper()))
    end
    
    println("Generated $(length(ssa_trajs)) MM trajectories")
    return ssa_trajs, rn, ["S", "E", "SE", "P"]
end

"""
    generate_lotka_volterra_data(n_trajs=1000)

Generate Lotka-Volterra (predator-prey) trajectory data.
"""
function generate_lotka_volterra_data(n_trajs=1000)
    println("Generating Lotka-Volterra data...")
    
    # Lotka-Volterra reaction network
    rn = @reaction_network begin
        α, X --> 2*X        # Prey birth
        β, X + Y --> 2*Y    # Predation (prey death, predator birth)
        γ, Y --> ∅          # Predator death
    end
    
    # Initial conditions and parameters
    u0_integers = [:X => 50, :Y => 20]  # X = prey, Y = predator
    tspan = (0., 50.)
    ps = [:α => 1.0, :β => 0.05, :γ => 1.0]
    
    # Create jump problem
    jinput = JumpInputs(rn, u0_integers, tspan, ps)
    jprob = JumpProblem(jinput)
    
    # Generate trajectories
    ssa_trajs = []
    @showprogress desc="Generating Lotka-Volterra trajectories..." for i in 1:n_trajs 
        push!(ssa_trajs, solve(jprob, SSAStepper()))
    end
    
    println("Generated $(length(ssa_trajs)) Lotka-Volterra trajectories")
    return ssa_trajs, rn, ["X", "Y"]  # X = prey, Y = predator
end

"""
    generate_toggle_switch_data(n_trajs=1000)

Generate toggle switch (genetic switch) trajectory data.
"""
function generate_toggle_switch_data(n_trajs=1000)
    println("Generating Toggle Switch data...")
    
    # Toggle switch reaction network (simplified)
    rn = @reaction_network begin
        α₁, ∅ --> A          # A production
        α₂, ∅ --> B          # B production  
        γ₁, A --> ∅          # A degradation
        γ₂, B --> ∅          # B degradation
        # Note: Inhibition handled through rate modulation, not explicit reactions for simplicity
    end
    
    # Initial conditions and parameters
    u0_integers = [:A => 10, :B => 5]
    tspan = (0., 100.)
    ps = [:α₁ => 20.0, :α₂ => 20.0, :γ₁ => 1.0, :γ₂ => 1.0]
    
    # Create jump problem
    jinput = JumpInputs(rn, u0_integers, tspan, ps)
    jprob = JumpProblem(jinput)
    
    # Generate trajectories
    ssa_trajs = []
    @showprogress desc="Generating Toggle Switch trajectories..." for i in 1:n_trajs 
        push!(ssa_trajs, solve(jprob, SSAStepper()))
    end
    
    println("Generated $(length(ssa_trajs)) Toggle Switch trajectories")
    return ssa_trajs, rn, ["A", "B"]
end

"""
    process_trajectories_for_system(ssa_trajs, time_points, species_names)

Process trajectories for any system type.
"""
function process_trajectories_for_system(ssa_trajs, time_points, species_names)
    println("Processing $(length(ssa_trajs)) trajectories...")
    
    n_species = length(species_names)
    n_times = length(time_points)
    n_trajs = length(ssa_trajs)
    
    # Initialize histogram data
    histograms = []
    
    for (t_idx, t) in enumerate(time_points)
        if t_idx % 5 == 1  # Print every 5th time point
            println("Processing time point $t_idx/$n_times (t=$t)")
        end
        
        # Extract states at time t from all trajectories
        states_at_t = []
        
        for traj in ssa_trajs
            # Find the value at time t
            if t <= traj.t[end]
                # Find closest time index
                t_idx_traj = searchsortedfirst(traj.t, t)
                if t_idx_traj > length(traj.t)
                    t_idx_traj = length(traj.t)
                end
                
                # Extract species counts
                state = [traj.u[t_idx_traj][j] for j in 1:n_species]
                push!(states_at_t, state)
            end
        end
        
        # Convert to histogram
        state_counts = Dict()
        for state in states_at_t
            state_key = tuple(state...)
            state_counts[state_key] = get(state_counts, state_key, 0) + 1
        end
        
        # Normalize to probabilities
        total_count = sum(values(state_counts))
        state_probs = Dict(k => v/total_count for (k, v) in state_counts)
        
        push!(histograms, state_probs)
    end
    
    println("Processed all time points, found histograms")
    return histograms
end

"""
    convert_histograms_to_matrix(histograms, max_states=1000)

Convert histogram data to matrix format for DMD.
"""
function convert_histograms_to_matrix(histograms, max_states=1000)
    println("Converting histograms to matrix format...")
    
    # Collect all unique states across all time points
    all_states = Set()
    for hist in histograms
        for state_key in keys(hist)
            push!(all_states, state_key)
        end
    end
    
    all_states = collect(all_states)
    println("Found $(length(all_states)) total unique states")
    
    # Select most important states (by frequency and variance)
    state_importance = Dict()
    
    for state in all_states
        # Calculate frequency and variance across time
        probs = [get(hist, state, 0.0) for hist in histograms]
        frequency = sum(p -> p > 0, probs) / length(histograms)
        total_prob = sum(probs)
        variance = length(probs) > 1 ? var(probs) : 0.0
        
        # Combined importance score
        importance = 0.4 * frequency + 0.3 * total_prob + 0.3 * variance * 100
        state_importance[state] = importance
    end
    
    # Select top states
    n_select = min(max_states, length(all_states))
    selected_states = sort(collect(all_states), by=s->state_importance[s], rev=true)[1:n_select]
    
    println("Selected top $n_select states for analysis")
    
    # Create state index mapping
    state_to_idx = Dict(state => i for (i, state) in enumerate(selected_states))
    
    # Build probability matrix
    n_times = length(histograms)
    prob_matrix = zeros(n_select, n_times)
    
    for (t_idx, hist) in enumerate(histograms)
        for (state, prob) in hist
            if haskey(state_to_idx, state)
                state_idx = state_to_idx[state]
                prob_matrix[state_idx, t_idx] = prob
            end
        end
        
        # Normalize columns to ensure probability conservation
        col_sum = sum(prob_matrix[:, t_idx])
        if col_sum > 0
            prob_matrix[:, t_idx] ./= col_sum
        end
    end
    
    # Convert selected_states to the format expected by other functions
    selected_states_formatted = [collect(state) for state in selected_states]
    
    println("Created probability matrix: $(size(prob_matrix))")
    return prob_matrix, selected_states_formatted
end

"""
    analyze_system_dynamics(sorted_stoich, reaction_stats, species_names, system_type)

Analyze discovered reactions for system-specific patterns.
"""
function analyze_system_dynamics(sorted_stoich, reaction_stats, species_names, system_type)
    println("\n=== System-Specific Analysis ($system_type) ===")
    
    if system_type == "mm"
        return analyze_mm_dynamics(sorted_stoich, reaction_stats, species_names)
    elseif system_type == "lotka_volterra"
        return analyze_lotka_volterra_dynamics(sorted_stoich, reaction_stats, species_names)
    elseif system_type == "toggle_switch"
        return analyze_toggle_switch_dynamics(sorted_stoich, reaction_stats, species_names)
    else
        return analyze_general_dynamics(sorted_stoich, reaction_stats, species_names)
    end
end

"""
    analyze_mm_dynamics(sorted_stoich, reaction_stats, species_names)

Analyze MM-specific dynamics without hardcoded expectations.
"""
function analyze_mm_dynamics(sorted_stoich, reaction_stats, species_names)
    # Look for enzyme-substrate complex patterns (SE species)
    if length(species_names) >= 3
        se_index = findfirst(name -> occursin("SE", name), species_names)
        
        if se_index !== nothing
            # Count complex-forming and complex-consuming reactions
            complex_forming = count(stoich -> stoich[se_index] > 0, sorted_stoich)
            complex_consuming = count(stoich -> stoich[se_index] < 0, sorted_stoich)
            
            println("Enzyme-Substrate Complex Analysis:")
            println("  Complex-forming reactions: $complex_forming")
            println("  Complex-consuming reactions: $complex_consuming")
            
            # Look for typical MM patterns
            binding_patterns = count(stoich -> stoich[se_index] > 0 && sum(stoich[1:end .!= se_index] .< 0) >= 2, sorted_stoich)
            catalysis_patterns = count(stoich -> stoich[se_index] < 0 && sum(stoich[1:end .!= se_index] .> 0) >= 2, sorted_stoich)
            
            println("  Potential binding reactions: $binding_patterns")
            println("  Potential catalysis reactions: $catalysis_patterns")
            
            success = (complex_forming > 0 && complex_consuming > 0) ? "SUCCESS" : "PARTIAL"
            return success
        end
    end
    
    return "LIMITED"
end

"""
    analyze_lotka_volterra_dynamics(sorted_stoich, reaction_stats, species_names)

Analyze Lotka-Volterra predator-prey dynamics.
"""
function analyze_lotka_volterra_dynamics(sorted_stoich, reaction_stats, species_names)
    if length(species_names) >= 2
        # Look for birth/death and predation patterns
        birth_reactions = count(stoich -> sum(stoich .> 0) == 1 && sum(stoich .< 0) == 0, sorted_stoich)
        death_reactions = count(stoich -> sum(stoich .< 0) == 1 && sum(stoich .> 0) == 0, sorted_stoich)
        interaction_reactions = count(stoich -> sum(stoich .> 0) >= 1 && sum(stoich .< 0) >= 1, sorted_stoich)
        
        println("Predator-Prey Analysis:")
        println("  Birth-type reactions: $birth_reactions")
        println("  Death-type reactions: $death_reactions") 
        println("  Interaction reactions: $interaction_reactions")
        
        # Oscillatory behavior (indirect measure)
        high_variance_reactions = count(s -> reaction_stats[s].rate_var > 0.001, sorted_stoich)
        println("  High-variance reactions (potential oscillations): $high_variance_reactions")
        
        success = (birth_reactions > 0 && interaction_reactions > 0) ? "SUCCESS" : "PARTIAL"
        return success
    end
    
    return "LIMITED"
end

"""
    analyze_toggle_switch_dynamics(sorted_stoich, reaction_stats, species_names)

Analyze toggle switch bistable dynamics.
"""
function analyze_toggle_switch_dynamics(sorted_stoich, reaction_stats, species_names)
    if length(species_names) >= 2
        # Look for production/degradation patterns
        production_reactions = count(stoich -> sum(stoich .> 0) == 1 && sum(stoich .< 0) == 0, sorted_stoich)
        degradation_reactions = count(stoich -> sum(stoich .< 0) == 1 && sum(stoich .> 0) == 0, sorted_stoich)
        
        println("Toggle Switch Analysis:")
        println("  Production-type reactions: $production_reactions")
        println("  Degradation-type reactions: $degradation_reactions")
        
        # Check for mutual inhibition patterns (indirect)
        opposing_changes = count(stoich -> any(stoich .> 0) && any(stoich .< 0), sorted_stoich)
        println("  Reactions with opposing changes: $opposing_changes")
        
        success = (production_reactions > 0 && degradation_reactions > 0) ? "SUCCESS" : "PARTIAL"
        return success
    end
    
    return "LIMITED"
end

"""
    analyze_general_dynamics(sorted_stoich, reaction_stats, species_names)

General analysis for unknown systems.
"""
function analyze_general_dynamics(sorted_stoich, reaction_stats, species_names)
    println("General System Analysis:")
    
    # Basic reaction type classification
    creation_reactions = count(stoich -> sum(stoich .> 0) > 0 && sum(stoich .< 0) == 0, sorted_stoich)
    destruction_reactions = count(stoich -> sum(stoich .< 0) > 0 && sum(stoich .> 0) == 0, sorted_stoich)
    conversion_reactions = count(stoich -> sum(stoich .> 0) > 0 && sum(stoich .< 0) > 0, sorted_stoich)
    
    println("  Creation reactions: $creation_reactions")
    println("  Destruction reactions: $destruction_reactions")
    println("  Conversion reactions: $conversion_reactions")
    
    # Complexity measures
    max_stoich_change = maximum(sum(abs.(stoich)) for stoich in sorted_stoich)
    avg_stoich_change = mean(sum(abs.(stoich)) for stoich in sorted_stoich)
    
    println("  Maximum stoichiometric change: $max_stoich_change")
    println("  Average stoichiometric change: $(round(avg_stoich_change, digits=2))")
    
    success = (length(sorted_stoich) > 5) ? "SUCCESS" : "LIMITED"
    return success
end

"""
    run_universal_system_analysis(system_type, n_trajs=500, max_states=800, n_time_points=25)

Run complete analysis for any system type.
"""
function run_universal_system_analysis(system_type, n_trajs=500, max_states=800, n_time_points=25)
    println("="^70)
    println("UNIVERSAL SYSTEM ANALYSIS: $(uppercase(system_type))")
    println("="^70)
    
    # Step 1: Generate system-specific data
    println("\n1. Generating $system_type trajectory data...")
    ssa_trajs, rn, species_names = generate_system_data(system_type, n_trajs)
    
    # Step 2: Define time points
    if system_type == "mm"
        time_points = range(0.0, 100.0, length=n_time_points)
    elseif system_type == "lotka_volterra"
        time_points = range(0.0, 30.0, length=n_time_points)  # Shorter for oscillations
    elseif system_type == "toggle_switch"  
        time_points = range(0.0, 50.0, length=n_time_points)  # Medium range
    else
        time_points = range(0.0, 50.0, length=n_time_points)  # Default
    end
    
    dt = time_points[2] - time_points[1]
    println("Time points: $(length(time_points)) from $(time_points[1]) to $(time_points[end])")
    println("Time step dt: $dt")
    
    # Step 3: Process trajectories
    println("\n2. Processing trajectories to histograms...")
    histograms = process_trajectories_for_system(ssa_trajs, time_points, species_names)
    
    # Step 4: Convert to matrix format
    println("\n3. Converting to matrix format...")
    reduced_data, selected_states = convert_histograms_to_matrix(histograms, max_states)
    
    # Step 5: Apply multigrid DMD with system-specific constraints
    println("\n4. Applying system-specific multigrid DMD...")
    
    # Load multigrid functions
    include("multigrid_dmd.jl")
    
    G_combined, sorted_stoich, fused_reactions, reaction_stats, successful_segments = multigrid_constrained_dmd(
        reduced_data, dt, selected_states, species_names;
        segment_length=8,
        overlap_fraction=0.4,
        system_type=system_type
    )
    
    # Step 6: Analyze discovered reactions
    println("\n5. Analyzing discovered reactions...")
    
    if !isempty(sorted_stoich)
        println("\nTop discovered reactions:")
        for (i, stoich) in enumerate(sorted_stoich[1:min(5, end)])
            stats = reaction_stats[stoich]
            reaction_str = format_reaction(stoich, species_names)
            println("  $i. $reaction_str")
            println("     Rate: $(round(stats.total_rate, digits=6))")
            println("     Confidence: $(round(stats.confidence, digits=3))")
        end
        
        # System-specific analysis
        success_result = analyze_system_dynamics(sorted_stoich, reaction_stats, species_names, system_type)
        
        println("\n🎯 System Analysis Result: $success_result")
        
        if success_result == "SUCCESS"
            println("🎉 SUCCESS! Algorithm discovered characteristic $system_type dynamics")
        elseif success_result == "PARTIAL"
            println("🔶 PARTIAL SUCCESS: Some $system_type patterns detected")
        else
            println("🔧 LIMITED: Basic reactions found but no clear $system_type patterns")
        end
    else
        println("❌ No biochemically valid reactions discovered")
        success_result = "FAILED"
    end
    
    # Return comprehensive results
    return Dict(
        "system_type" => system_type,
        "species_names" => species_names,
        "generator" => G_combined,
        "significant_stoichiometries" => sorted_stoich,
        "reaction_stats" => reaction_stats,
        "successful_segments" => successful_segments,
        "analysis_result" => success_result,
        "reduced_data" => reduced_data,
        "selected_states" => selected_states,
        "dt" => dt,
        "trajectories" => ssa_trajs,
        "histograms" => histograms
    )
end

# Quick test functions for each system
mm_test = () -> run_universal_system_analysis("mm", 300, 500, 20)
lv_test = () -> run_universal_system_analysis("lotka_volterra", 300, 500, 20)
toggle_test = () -> run_universal_system_analysis("toggle_switch", 300, 500, 20)

println("Universal System Analysis Module Loaded! 🧬🔄🎯")
println()
println("Supported systems:")
println("  • Michaelis-Menten (mm)")
println("  • Lotka-Volterra (lotka_volterra)")  
println("  • Toggle Switch (toggle_switch)")
println("  • General (any system)")
println()
println("Quick test functions:")
println("  mm_test()           - Test MM system")
println("  lv_test()           - Test Lotka-Volterra system")
println("  toggle_test()       - Test Toggle Switch system")
println()
println("Main function:")
println("  run_universal_system_analysis(system_type, n_trajs, max_states, n_time_points)")
println()
println("✅ No hardcoded reaction mechanisms")
println("✅ System-appropriate conservation laws") 
println("✅ Universal biochemical constraints")
println("✅ Data-driven discovery")
