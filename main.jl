# main.jl
# Main script for inferring Chemical Reaction Networks via DMD

# Include all necessary files
include("data_generation.jl")
include("data_processing.jl")
include("dmd_analysis.jl")
include("reaction_extraction.jl")
include("kinetics_analysis.jl")
include("spectral_analysis.jl")
include("visualization.jl")

"""
    infer_crn_from_trajectories(ssa_trajs, species_names; max_dim=1000, grid_sizes=nothing, validate_reactions=true)

Infer Chemical Reaction Network from trajectory data.

# Arguments
- `ssa_trajs`: Array of trajectory solutions
- `species_names`: Names of species
- `max_dim`: Maximum dimension for reduced state space
- `grid_sizes`: Array of grid sizes for discretization (defaults to automatic selection)
- `validate_reactions`: Whether to validate reactions using conservation principles

# Returns
- Dictionary containing inference results
"""
function infer_crn_from_trajectories(ssa_trajs, species_names; max_dim=1000, grid_sizes=nothing, validate_reactions=true)
    # Get sample trajectory to determine dimensionality
    sample_traj = ssa_trajs[1]
    n_species = length(sample_traj.u[1])
    species_indices = 1:n_species
    
    # Auto-determine grid sizes if not provided
    if grid_sizes === nothing
        grid_sizes = []
        for sp_idx in species_indices
            # Get max count across all trajectories
            max_count = 0
            for traj in ssa_trajs
                for u in traj.u
                    max_count = max(max_count, u[sp_idx])
                end
            end
            # Set grid size to max_count + buffer
            push!(grid_sizes, max_count + 10)
        end
    end
    
    # Define time points for analysis
    t_max = maximum([traj.t[end] for traj in ssa_trajs])
    time_points = range(0.0, t_max, length=20)
    dt = time_points[2] - time_points[1]
    
    # Process trajectories to sparse format
    println("Processing trajectories to sparse format...")
    sparse_probs = process_trajectories_to_sparse(ssa_trajs, species_indices, grid_sizes, time_points)
    
    # Reduce to important states
    println("Reducing to important states...")
    reduced_data, selected_states = reduce_sparse_data(sparse_probs, grid_sizes, max_dim)
    
    # Apply DMD
    println("Applying DMD to reduced data...")
    G, λ, Φ, A, r = apply_dmd(reduced_data, dt, svd_rank_threshold=1e-12)
    
    # Extract reactions
    println("Extracting reaction information...")
    significant_stoich, grouped_reactions, stoich_stats = extract_reactions_from_generator(
        G, selected_states, species_indices, species_names, 
        threshold=1e-5, validate_reactions=validate_reactions
    )
    
    # Analyze DMD eigenvalues
    println("\nAnalyzing DMD eigenvalues...")
    
    # Convert discrete-time eigenvalues to continuous-time
    cont_eigs = log.(Complex.(λ)) ./ dt
    
    # Sort by real part (largest to smallest)
    sorted_idx = sortperm(real.(cont_eigs), rev=true)
    
    # Display top eigenvalues
    println("Top 5 eigenvalues of the CME generator:")
    for i in 1:min(5, length(cont_eigs))
        idx = sorted_idx[i]
        println("λ$i = $(round(cont_eigs[idx], digits=4))")
    end
    
    # Spectral analysis
    println("\nPerforming spectral analysis...")
    spectral_reactions, spectral_scores = analyze_and_select_reactions_fixed(
        G, λ, Φ, grouped_reactions, stoich_stats, selected_states, species_names
    )
    
    # Kinetics analysis
    println("\nAnalyzing reaction kinetics...")
    kinetics_results = analyze_mass_action_kinetics_enhanced(
        grouped_reactions, stoich_stats, selected_states, species_names
    )
    
    # Print success message
    println("\nCRN inference completed successfully!")
    println("DMD rank: $r")
    
    # Return combined results
    return Dict(
        "significant_stoichiometries" => significant_stoich,
        "grouped_reactions" => grouped_reactions,
        "stoich_stats" => stoich_stats,
        "generator" => G,
        "eigenvalues" => λ,
        "DMD_modes" => Φ,
        "DMD_operator" => A,
        "selected_states" => selected_states,
        "sparse_probs" => sparse_probs,
        "reduced_data" => reduced_data,
        "rank" => r,
        "spectral_selected_reactions" => spectral_reactions,
        "spectral_scores" => spectral_scores,
        "kinetics_results" => kinetics_results
    )
end

"""
    infer_crn_michaelis_menten(n_trajs=500, max_dim=1000)

Infer Chemical Reaction Network for a Michaelis-Menten enzyme kinetics system.

# Arguments
- `n_trajs`: Number of trajectories to generate
- `max_dim`: Maximum dimension for reduced state space

# Returns
- Dictionary containing inference results
"""
function infer_crn_michaelis_menten(n_trajs=500, max_dim=1000)
    # Generate Michaelis-Menten data
    println("Generating Michaelis-Menten trajectory data...")
    ssa_trajs, rn = generate_mm_trajectories(n_trajs)
    
    # Define species names
    species_names = ["S", "E", "SE", "P"]
    
    # Define grid sizes - optimized for typical MM dynamics
    grid_sizes = [32, 16, 8, 32]
    
    # Run inference
    results = infer_crn_from_trajectories(
        ssa_trajs, species_names, 
        max_dim=max_dim, 
        grid_sizes=grid_sizes, 
        validate_reactions=true
    )
    
    # Add reaction network
    results["reaction_network"] = rn
    results["trajectories"] = ssa_trajs
    
    # Verify inferred reactions against known Michaelis-Menten reactions
    println("\nVerifying inferred reactions against known Michaelis-Menten reactions...")
    
    expected_stoichiometries = [
        [0, 1, -1, 1],    # SE → E + P
        [-1, -1, 1, 0],   # S + E → SE 
        [1, 1, -1, 0]     # SE → S + E
    ]
    
    found_count = 0
    
    for expected in expected_stoichiometries
        expected_tuple = tuple(expected...)
        
        # Check if found
        if expected_tuple in keys(results["grouped_reactions"])
            stats = results["stoich_stats"][expected_tuple]
            reaction_str = format_reaction(expected_tuple, species_names)
            
            println("$reaction_str : ✓ found (rate ≈ $(round(stats.total_rate, digits=5)))")
            found_count += 1
        elseif tuple(-1 .* expected...) in keys(results["grouped_reactions"])
            # Check for reverse reaction
            reverse_tuple = tuple(-1 .* expected...)
            stats = results["stoich_stats"][reverse_tuple]
            reaction_str = format_reaction(expected_tuple, species_names)
            
            println("$reaction_str : ✓ found reversed (rate ≈ $(round(stats.total_rate, digits=5)))")
            found_count += 1
        else
            reaction_str = format_reaction(expected_tuple, species_names)
            println("$reaction_str : ✗ not found")
        end
    end
    
    println("Found $found_count out of $(length(expected_stoichiometries)) expected reactions")
    
    return results
end

"""
    infer_crn_toggle_switch(n_trajs=500, max_dim=1000)

Infer Chemical Reaction Network for a genetic toggle switch system.

# Arguments
- `n_trajs`: Number of trajectories to generate
- `max_dim`: Maximum dimension for reduced state space

# Returns
- Dictionary containing inference results
"""
function infer_crn_toggle_switch(n_trajs=500, max_dim=1000)
    # Generate toggle switch data
    println("Generating toggle switch trajectory data...")
    ssa_trajs, rn = generate_toggle_switch_trajectories(n_trajs)
    
    # Define species names
    species_names = ["P₁", "P₂"]
    
    # Run inference
    results = infer_crn_from_trajectories(
        ssa_trajs, species_names, 
        max_dim=max_dim, 
        validate_reactions=false  # Toggle switch has complex regulation, disable validation
    )
    
    # Add reaction network
    results["reaction_network"] = rn
    results["trajectories"] = ssa_trajs
    
    return results
end

"""
    infer_crn_lotka_volterra(n_trajs=500, max_dim=1000)

Infer Chemical Reaction Network for a Lotka-Volterra (predator-prey) system.

# Arguments
- `n_trajs`: Number of trajectories to generate
- `max_dim`: Maximum dimension for reduced state space

# Returns
- Dictionary containing inference results
"""
function infer_crn_lotka_volterra(n_trajs=500, max_dim=1000)
    # Generate Lotka-Volterra data
    println("Generating Lotka-Volterra trajectory data...")
    ssa_trajs, rn = generate_lotka_volterra_trajectories(n_trajs)
    
    # Define species names
    species_names = ["X", "Y"]  # prey, predator
    
    # Run inference
    results = infer_crn_from_trajectories(
        ssa_trajs, species_names, 
        max_dim=max_dim, 
        validate_reactions=false  # Simple validation won't work for these reactions
    )
    
    # Add reaction network
    results["reaction_network"] = rn
    results["trajectories"] = ssa_trajs
    
    # Verify inferred reactions against known Lotka-Volterra reactions
    println("\nVerifying inferred reactions against known Lotka-Volterra reactions...")
    
    expected_stoichiometries = [
        [1, 0],    # X → X + X (prey reproduction)
        [-1, 1],   # X + Y → Y + Y (predator reproduction through prey consumption)
        [0, -1]    # Y → ∅ (predator death)
    ]
    
    found_count = 0
    
    for expected in expected_stoichiometries
        expected_tuple = tuple(expected...)
        
        # Check if found
        if expected_tuple in keys(results["grouped_reactions"])
            stats = results["stoich_stats"][expected_tuple]
            reaction_str = format_reaction(expected_tuple, species_names)
            
            println("$reaction_str : ✓ found (rate ≈ $(round(stats.total_rate, digits=5)))")
            found_count += 1
        else
            reaction_str = format_reaction(expected_tuple, species_names)
            println("$reaction_str : ✗ not found")
        end
    end
    
    println("Found $found_count out of $(length(expected_stoichiometries)) expected reactions")
    
    return results
end

"""
    visualize_all_results(results, species_names)

Generate comprehensive visualizations of the inference results.

# Arguments
- `results`: Dictionary containing inference results
- `species_names`: Names of species

# Returns
- Nothing
"""
function visualize_all_results(results, species_names)
    # Get data from results
    G = results["generator"]
    λ = results["eigenvalues"]
    Φ = results["DMD_modes"]
    selected_states = results["selected_states"]
    grouped_reactions = results["grouped_reactions"]
    stoich_stats = results["stoich_stats"]
    sparse_probs = results["sparse_probs"]
    time_points = range(0.0, maximum([traj.t[end] for traj in results["trajectories"]]), length=length(sparse_probs))
    
    # 1. Visualize eigenvalues
    println("\nVisualizing eigenvalue distribution...")
    visualize_eigenvalues(λ, title="Generator Eigenvalues")
    
    # 2. Visualize reaction scores
    if haskey(results, "spectral_scores") && !isempty(results["spectral_scores"])
        println("\nVisualizing reaction scores...")
        visualize_reaction_scores(keys(results["spectral_scores"]), results["spectral_scores"], species_names)
    end
    
    # 3. Visualize trajectory data
    println("\nVisualizing sample trajectories...")
    species_indices = 1:length(species_names)
    visualize_trajectory_data(results["trajectories"][1:5], species_indices, species_names)
    
    # 4. Visualize histogram evolution
    println("\nVisualizing probability distribution evolution...")
    visualize_histogram_evolution(sparse_probs, selected_states, species_indices, species_names, time_points)
    
    # 5. Visualize reaction rates
    println("\nVisualizing reaction rate patterns...")
    visualize_reaction_rates(grouped_reactions, species_names)
    
    # 6. Try to visualize conservation laws if available
    try
        conservation_laws, law_descriptions = identify_conservation_laws(G, species_names)
        if !isempty(conservation_laws)
            println("\nVisualizing conservation laws...")
            visualize_conservation_laws(conservation_laws, species_names)
        end
    catch e
        println("Could not identify conservation laws: $e")
    end
end

"""
    run_example(system="mm", n_trajs=500, max_dim=1000, visualize=true)

Run the CRN inference pipeline on a specified example system.

# Arguments
- `system`: System to analyze ("mm" for Michaelis-Menten, "ts" for toggle switch, "lv" for Lotka-Volterra)
- `n_trajs`: Number of trajectories to generate
- `max_dim`: Maximum dimension for reduced state space
- `visualize`: Whether to generate visualizations

# Returns
- Dictionary containing inference results
"""
function run_example(system="mm", n_trajs=500, max_dim=1000, visualize=true)
    # Select system
    if system == "mm"
        println("Running Michaelis-Menten example...")
        results = infer_crn_michaelis_menten(n_trajs, max_dim)
        species_names = ["S", "E", "SE", "P"]
    elseif system == "ts"
        println("Running Toggle Switch example...")
        results = infer_crn_toggle_switch(n_trajs, max_dim)
        species_names = ["P₁", "P₂"]
    elseif system == "lv"
        println("Running Lotka-Volterra example...")
        results = infer_crn_lotka_volterra(n_trajs, max_dim)
        species_names = ["X", "Y"]
    else
        error("Unknown system: $system. Choose from 'mm', 'ts', or 'lv'.")
    end
    
    # Generate visualizations if requested
    if visualize
        visualize_all_results(results, species_names)
    end
    
    return results
end

"""
    main()

Main function to demonstrate the CRN inference pipeline.
"""
function main()
    println("=== Chemical Reaction Network Inference via DMD ===")
    println("\nThis script demonstrates inference of reaction networks from trajectory data.")
    println("Choose a system to analyze:")
    println("1. Michaelis-Menten enzyme kinetics")
    println("2. Genetic toggle switch")
    println("3. Lotka-Volterra predator-prey")
    
    # Get user input with default
    print("\nEnter your choice (1-3, default: 1): ")
    choice_str = readline()
    choice = isempty(choice_str) ? 1 : parse(Int, choice_str)
    
    # Map choice to system
    system = ["mm", "ts", "lv"][choice]
    
    # Run example
    results = run_example(system)
    
    println("\n=== CRN Inference Complete ===")
    println("Results dictionary contains the following keys:")
    for key in keys(results)
        println("- $key")
    end
    
    return results
end

# Run the main function if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
