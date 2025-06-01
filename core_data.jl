# core_data.jl - SYSTEM-AGNOSTIC DATA PROCESSING
# Common data processing functions

using Statistics
using LinearAlgebra

"""
    process_trajectories_to_histograms(trajectories, time_points, n_species)

Convert trajectories to probability histograms at specified time points.
"""
function process_trajectories_to_histograms(trajectories, time_points, n_species)
    println("Processing $(length(trajectories)) trajectories...")
    
    n_times = length(time_points)
    histograms = []
    
    for (t_idx, t) in enumerate(time_points)
        if t_idx % 5 == 1
            println("  Processing time point $t_idx/$n_times (t=$t)")
        end
        
        # Extract states at time t from all trajectories
        states_at_t = []
        
        for traj in trajectories
            if t <= traj.t[end]
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
    
    println("Generated $(length(histograms)) histograms")
    return histograms
end

"""
    select_states_from_trajectories(trajectories, n_species; max_states=800, selection_method=:transition_frequency)

Select important states from trajectory data.
"""
function select_states_from_trajectories(trajectories, n_species; max_states=800, selection_method=:transition_frequency)
    println("Selecting states using method: $selection_method")
    
    if selection_method == :transition_frequency
        return select_by_transition_frequency(trajectories, n_species, max_states)
    elseif selection_method == :histogram_frequency
        # Would need histograms - simplified for now
        return select_by_transition_frequency(trajectories, n_species, max_states)
    else
        error("Unknown selection method: $selection_method")
    end
end

"""
    select_by_transition_frequency(trajectories, n_species, max_states)

Select states based on transition participation frequency.
"""
function select_by_transition_frequency(trajectories, n_species, max_states)
    # Collect all states and count transition participation
    trajectory_states = Set()
    transition_counts = Dict()
    
    for traj in trajectories[1:min(100, length(trajectories))]  # Sample for efficiency
        for i in 1:length(traj.t)
            state = [traj.u[i][j] for j in 1:n_species]
            state_tuple = tuple(state...)
            push!(trajectory_states, state_tuple)
        end
        
        # Count transitions
        for i in 1:(length(traj.t)-1)
            state1 = tuple([traj.u[i][j] for j in 1:n_species]...)
            state2 = tuple([traj.u[i+1][j] for j in 1:n_species]...)
            
            transition_counts[state1] = get(transition_counts, state1, 0) + 1
            transition_counts[state2] = get(transition_counts, state2, 0) + 1
        end
    end
    
    trajectory_states = collect(trajectory_states)
    println("Found $(length(trajectory_states)) states in trajectory flow")
    
    # Select by transition participation
    n_select = min(max_states, length(trajectory_states))
    
    if length(trajectory_states) <= max_states
        selected_states = trajectory_states
    else
        selected_states = sort(trajectory_states, 
                              by=s -> get(transition_counts, s, 0), 
                              rev=true)[1:n_select]
    end
    
    println("Selected $(length(selected_states)) states")
    
    # Convert to format expected by downstream functions
    return [collect(state) for state in selected_states]
end

"""
    convert_histograms_to_matrix(histograms, selected_states)

Convert histogram data to probability matrix format.
"""
function convert_histograms_to_matrix(histograms, selected_states)
    println("Converting histograms to probability matrix...")
    
    # Create state index mapping
    state_to_idx = Dict()
    for (i, state) in enumerate(selected_states)
        if isa(state, Vector)
            state_to_idx[tuple(state...)] = i
        else
            state_to_idx[state] = i
        end
    end
    
    # Build probability matrix
    n_states = length(selected_states)
    n_times = length(histograms)
    prob_matrix = zeros(n_states, n_times)
    
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
    
    println("Created probability matrix: $(size(prob_matrix))")
    return prob_matrix
end

"""
    diagnose_data_quality(trajectories, selected_states, n_species)

Diagnose data quality and coverage.
"""
function diagnose_data_quality(trajectories, selected_states, n_species)
    println("\n=== Data Quality Diagnosis ===")
    
    # Basic statistics
    n_trajs = length(trajectories)
    avg_length = mean(length(traj.t) for traj in trajectories)
    
    println("Basic Statistics:")
    println("  Trajectories: $n_trajs")
    println("  Average trajectory length: $(round(avg_length, digits=1)) time points")
    println("  Selected states: $(length(selected_states))")
    
    # Coverage analysis
    state_to_idx = Dict(Tuple(state) => i for (i, state) in enumerate(selected_states))
    
    total_steps = 0
    captured_steps = 0
    
    for traj in trajectories[1:min(10, length(trajectories))]  # Sample for diagnosis
        for i in 1:(length(traj.t)-1)
            state1 = [traj.u[i][j] for j in 1:n_species]
            state2 = [traj.u[i+1][j] for j in 1:n_species]
            
            total_steps += 1
            
            if haskey(state_to_idx, Tuple(state1)) && haskey(state_to_idx, Tuple(state2))
                captured_steps += 1
            end
        end
    end
    
    capture_rate = captured_steps / total_steps
    println("Coverage Analysis:")
    println("  Transition capture rate: $(round(capture_rate*100, digits=2))%")
    
    if capture_rate < 0.1
        println("  ⚠️ Warning: Very low capture rate - consider increasing state selection or reducing max_states")
    elseif capture_rate > 0.8
        println("  ✅ Good coverage")
    else
        println("  ⚠️ Moderate coverage - results may be incomplete")
    end
    
    return capture_rate
end

"""
    create_analysis_pipeline(trajectory_generator, system_params)

Create a standardized analysis pipeline.
"""
function create_analysis_pipeline(trajectory_generator, system_params)
    return Dict(
        "generate_trajectories" => trajectory_generator,
        "system_params" => system_params,
        "n_species" => system_params["n_species"],
        "species_names" => system_params["species_names"],
        "time_range" => system_params["time_range"],
        "expected_reactions" => get(system_params, "expected_reactions", [])
    )
end

"""
    run_basic_data_processing(trajectory_generator, system_params, analysis_params)

Run basic data processing pipeline.
"""
function run_basic_data_processing(trajectory_generator, system_params, analysis_params)
    println("\n" * "="^60)
    println("BASIC DATA PROCESSING")
    println("="^60)
    
    # Extract parameters
    n_trajs = get(analysis_params, "n_trajs", 500)
    max_states = get(analysis_params, "max_states", 500)
    n_time_points = get(analysis_params, "n_time_points", 20)
    
    n_species = system_params["n_species"]
    species_names = system_params["species_names"]
    time_range = system_params["time_range"]
    
    # Generate trajectories
    println("\n1. Generating trajectory data...")
    trajectories, reaction_network = trajectory_generator(n_trajs)
    
    # Define time points
    time_points = range(time_range[1], time_range[2], length=n_time_points)
    dt = time_points[2] - time_points[1]
    
    # Process trajectories
    println("\n2. Processing trajectories...")
    histograms = process_trajectories_to_histograms(trajectories, time_points, n_species)
    
    # Select important states
    println("\n3. Selecting important states...")
    selected_states = select_states_from_trajectories(trajectories, n_species, max_states=max_states)
    
    # Convert to matrix format
    println("\n4. Converting to matrix format...")
    prob_matrix = convert_histograms_to_matrix(histograms, selected_states)

    # DEBUG: Check probability matrix changes (key for DMD rate recovery)
    println("\n🔧 PROBABILITY MATRIX RATE DEBUG:")
    println("  Matrix shape: $(size(prob_matrix))")
    println("  Column sums (should be ~1.0): $(round.(sum(prob_matrix, dims=1)[1:5], digits=4))")
    println("  Probability changes between time points (these drive DMD rates):")
    max_changes = []
    for t in 1:min(5, size(prob_matrix, 2)-1)
        max_change = maximum(abs.(prob_matrix[:, t+1] - prob_matrix[:, t]))
        avg_change = mean(abs.(prob_matrix[:, t+1] - prob_matrix[:, t]))
        push!(max_changes, max_change)
        println("    t$t → t$(t+1): max change = $(round(max_change, digits=6)), avg change = $(round(avg_change, digits=6))")
    end
    println("  Overall max probability change: $(maximum(max_changes))")
    println("  Expected: For MM kinetics with dt=$(time_points[2]-time_points[1]), expect changes ~0.01-0.1")
    
    # Diagnose data quality
    println("\n5. Diagnosing data quality...")
    capture_rate = diagnose_data_quality(trajectories, selected_states, n_species)
    
    # Create result dictionary
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
    
    println("\nBasic data processing completed!")
    return data_dict
end

println("Core Data Processing Module Loaded! 📊")
println("System-agnostic functions:")
println("  process_trajectories_to_histograms(trajs, times, n_species)")
println("  select_states_from_trajectories(trajs, n_species)")
println("  convert_histograms_to_matrix(histograms, states)")
println("  diagnose_data_quality(trajs, states, n_species)")
println("  run_basic_data_processing(generator, sys_params, analysis_params)")
