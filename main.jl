# main.jl - CLEAN MAIN MODULE
# Core trajectory generation and data processing

using Catalyst
using JumpProcesses
using DifferentialEquations
using ProgressMeter
using Statistics
using LinearAlgebra
using SparseArrays

"""
    generate_mm_trajectories(n_trajs=500)

Generate Michaelis-Menten trajectory data using Catalyst.
"""
function generate_mm_trajectories(n_trajs=500)
    println("Generating MM trajectory data...")
    
    # Define the Michaelis-Menten reaction network
    rn = @reaction_network begin
        kB, S + E --> SE
        kD, SE --> S + E
        kP, SE --> P + E
    end
    
    # Initial conditions and parameters
    u0_integers = [:S => 50, :E => 10, :SE => 1, :P => 1]
    tspan = (0.0, 200.0)
    ps = [:kB => 0.01, :kD => 0.1, :kP => 0.1]
    
    # Create the jump problem
    jinput = JumpInputs(rn, u0_integers, tspan, ps)
    jprob = JumpProblem(jinput)
    
    # Generate trajectories
    ssa_trajs = []
    @showprogress desc="Generating trajectories..." for i in 1:n_trajs 
        push!(ssa_trajs, solve(jprob, SSAStepper()))
    end
    
    println("Generated $(length(ssa_trajs)) trajectories")
    return ssa_trajs, rn
end

"""
    process_trajectories_to_histograms(ssa_trajs, time_points, species_names=["S", "E", "SE", "P"])

Process trajectories into probability histograms at specified time points.
"""
function process_trajectories_to_histograms(ssa_trajs, time_points, species_names=["S", "E", "SE", "P"])
    println("Processing $(length(ssa_trajs)) trajectories...")
    
    n_species = length(species_names)
    n_times = length(time_points)
    histograms = []
    
    for (t_idx, t) in enumerate(time_points)
        if t_idx % 5 == 1
            println("Processing time point $t_idx/$n_times (t=$t)")
        end
        
        # Extract states at time t from all trajectories
        states_at_t = []
        
        for traj in ssa_trajs
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
    select_important_states(histograms, max_states=1000)

Select the most dynamically important states from histogram data.
"""
function select_important_states(histograms, max_states=1000)
    println("Selecting important states...")
    
    # Collect all unique states
    all_states = Set()
    for hist in histograms
        for state_key in keys(hist)
            push!(all_states, state_key)
        end
    end
    
    all_states = collect(all_states)
    println("Found $(length(all_states)) total unique states")
    
    # Calculate importance scores
    state_importance = Dict()
    
    for state in all_states
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
    return selected_states
end

"""
    convert_to_probability_matrix(histograms, selected_states)

Convert histogram data to probability matrix format for DMD.
"""
function convert_to_probability_matrix(histograms, selected_states)
    println("Converting to probability matrix...")
    
    # Create state index mapping
    state_to_idx = Dict(state => i for (i, state) in enumerate(selected_states))
    
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
    
    # Convert selected_states to consistent format
    selected_states_formatted = [collect(state) for state in selected_states]
    
    println("Created probability matrix: $(size(prob_matrix))")
    return prob_matrix, selected_states_formatted
end

"""
    extract_reactions_from_generator(G, selected_states, species_names; threshold=1e-5)

Extract elementary reactions from generator matrix.
"""
function extract_reactions_from_generator(G, selected_states, species_names; threshold=1e-5)
    println("Extracting reactions from generator matrix...")
    
    reactions = []
    n_states = size(G, 1)
    
    # Find significant transitions
    for i in 1:n_states
        for j in 1:n_states
            if i != j && abs(G[i, j]) > threshold
                if i <= length(selected_states) && j <= length(selected_states)
                    from_state = selected_states[j]
                    to_state = selected_states[i]
                    
                    # Compute stoichiometry
                    stoichiometry = to_state - from_state
                    
                    # Filter for elementary reactions (≤3 molecule changes)
                    total_change = sum(abs.(stoichiometry))
                    if 0 < total_change <= 3
                        push!(reactions, (
                            from_state = from_state,
                            to_state = to_state,
                            stoichiometry = stoichiometry,
                            rate = abs(G[i, j])
                        ))
                    end
                end
            end
        end
    end
    
    println("Found $(length(reactions)) potential elementary reactions")
    
    # Group reactions by stoichiometry
    grouped_reactions = Dict()
    for r in reactions
        s_key = tuple(r.stoichiometry...)
        if haskey(grouped_reactions, s_key)
            push!(grouped_reactions[s_key], r)
        else
            grouped_reactions[s_key] = [r]
        end
    end
    
    # Calculate statistics for each stoichiometry
    stoich_stats = Dict()
    for (stoich, rxns) in grouped_reactions
        total_rate = sum(r.rate for r in rxns)
        avg_rate = total_rate / length(rxns)
        rate_var = var([r.rate for r in rxns], corrected=false)
        
        stoich_stats[stoich] = (
            total_rate = total_rate,
            avg_rate = avg_rate,
            rate_var = rate_var,
            count = length(rxns)
        )
    end
    
    # Sort by total rate
    sorted_stoich = sort(collect(keys(stoich_stats)), 
                        by=s -> stoich_stats[s].total_rate, rev=true)
    
    # Display results
    println("\nTop reactions found:")
    for (i, stoich) in enumerate(sorted_stoich[1:min(10, end)])
        stats = stoich_stats[stoich]
        reaction_str = format_reaction(stoich, species_names)
        println("$i. $reaction_str (rate: $(round(stats.total_rate, digits=4)))")
    end
    
    return sorted_stoich, grouped_reactions, stoich_stats
end

"""
    format_reaction(stoich, species_names)

Format stoichiometry vector as reaction string.
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

"""
    run_basic_mm_inference(n_trajs=500, max_states=500, n_time_points=20)

Run basic MM inference pipeline without advanced analysis.
"""
function run_basic_mm_inference(n_trajs=500, max_states=500, n_time_points=20)
    println("="^70)
    println("BASIC MM INFERENCE PIPELINE")
    println("="^70)
    
    # Generate trajectory data
    println("\n1. Generating trajectory data...")
    ssa_trajs, rn = generate_mm_trajectories(n_trajs)
    
    # Define time points
    time_points = range(0.0, 50.0, length=n_time_points)
    dt = time_points[2] - time_points[1]
    species_names = ["S", "E", "SE", "P"]
    
    # Process trajectories
    println("\n2. Processing trajectories...")
    histograms = process_trajectories_to_histograms(ssa_trajs, time_points, species_names)
    
    # Select important states  
    println("\n3. Selecting important states...")
    selected_states = select_important_states(histograms, max_states)
    
    # Convert to matrix format
    println("\n4. Converting to matrix format...")
    prob_matrix, selected_states_formatted = convert_to_probability_matrix(histograms, selected_states)
    
    println("\nBasic processing completed!")
    println("Ready for DMD analysis with $(size(prob_matrix)) probability matrix")
    
    return Dict(
        "trajectories" => ssa_trajs,
        "time_points" => time_points,
        "dt" => dt,
        "species_names" => species_names,
        "histograms" => histograms,
        "selected_states" => selected_states_formatted,
        "probability_matrix" => prob_matrix,
        "reaction_network" => rn
    )
end

println("Clean Main Module Loaded! 🧹")
println("Functions:")
println("  generate_mm_trajectories(n_trajs)")
println("  process_trajectories_to_histograms(trajs, times)")
println("  select_important_states(histograms, max_states)")
println("  convert_to_probability_matrix(histograms, states)")
println("  extract_reactions_from_generator(G, states, species)")
println("  run_basic_mm_inference(n_trajs, max_states, n_times)")
println()
println("This provides clean data processing for all DMD methods!")
