# visualization.jl
# Functions for visualizing results from DMD analysis

using UnicodePlots

"""
    visualize_eigenvalues(λ; title="Eigenvalue Distribution")

Visualize the eigenvalue distribution of the generator matrix.

# Arguments
- `λ`: Eigenvalues
- `title`: Title for the plot

# Returns
- The created plot
"""
function visualize_eigenvalues(λ; title="Eigenvalue Distribution")
    # Convert to real/imag components
    real_parts = real.(λ)
    imag_parts = imag.(λ)
    
    # Create scatter plot
    plt = scatterplot(
        real_parts, 
        imag_parts,
        title = title,
        xlabel = "Re(λ)",
        ylabel = "Im(λ)",
        width = 60,
        height = 20
    )
    
    # Mark the origin
    plt = annotate!(plt, 0, 0, "×")
    
    # Display plot
    println(plt)
    
    # Also create a zoomed version around the origin for the most important eigenvalues
    # Find appropriate zoom level
    important_eigs = filter(e -> abs(real(e)) < 1.0 && abs(imag(e)) < 2.0, λ)
    
    if !isempty(important_eigs)
        min_real = maximum([-1.0, minimum(real.(important_eigs)) * 1.2])
        max_real = minimum([0.2, maximum(real.(important_eigs)) * 1.2])
        min_imag = maximum([-2.0, minimum(imag.(important_eigs)) * 1.2])
        max_imag = minimum([2.0, maximum(imag.(important_eigs)) * 1.2])
        
        # Create zoomed plot
        zoom_plt = scatterplot(
            real_parts,
            imag_parts,
            title = "Important Eigenvalues (Zoomed)",
            xlabel = "Re(λ)",
            ylabel = "Im(λ)",
            width = 60,
            height = 20,
            xlim = (min_real, max_real),
            ylim = (min_imag, max_imag)
        )
        
        # Mark the origin
        zoom_plt = annotate!(zoom_plt, 0, 0, "×")
        
        println(zoom_plt)
    end
    
    return plt
end

"""
    visualize_reaction_scores(reactions, scores, species_names; top_n=15, title="Reaction Scores")

Visualize the scores of reactions to identify top reactions.

# Arguments
- `reactions`: List of reaction stoichiometries
- `scores`: Dictionary mapping stoichiometries to scores
- `species_names`: Names of species
- `top_n`: Number of top reactions to show
- `title`: Title for the plot

# Returns
- The created plot
"""
function visualize_reaction_scores(reactions, scores, species_names; top_n=15, title="Reaction Scores")
    # Format reactions as strings
    reaction_strs = []
    score_values = []
    
    # Convert to array of (reaction_str, score) pairs
    pairs = []
    for (stoich, score) in scores
        # Format reaction
        reaction_str = format_reaction(stoich, species_names)
        push!(pairs, (reaction_str, score))
    end
    
    # Sort by score
    sort!(pairs, by=x -> x[2], rev=true)
    
    # Take top_n
    pairs = pairs[1:min(top_n, length(pairs))]
    
    # Extract into separate arrays
    reaction_strs = [p[1] for p in pairs]
    score_values = [p[2] for p in pairs]
    
    # Normalize scores for better display
    max_score = maximum(score_values)
    normalized_scores = score_values ./ max_score
    
    # Create horizontal bar chart
    plt = barplot(
        reaction_strs,
        normalized_scores,
        title = title,
        xlabel = "Normalized Score",
        ylabel = "Reaction",
        width = 60,
        height = 25
    )
    
    println(plt)
    return plt
end

"""
    visualize_cv_results(cv_errors; title="Cross-Validation Error")

Visualize cross-validation results to find optimal reaction set size.

# Arguments
- `cv_errors`: Array of cross-validation errors
- `title`: Title for the plot

# Returns
- The created plot
"""
function visualize_cv_results(cv_errors; title="Cross-Validation Error")
    # Create x-axis (number of reactions)
    n_reactions = collect(1:length(cv_errors))
    
    # Find optimal size
    optimal_size = argmin(cv_errors)
    
    # Create line plot
    plt = lineplot(
        n_reactions, 
        cv_errors,
        title = title,
        xlabel = "Number of Reactions",
        ylabel = "Spectral Error",
        width = 60,
        height = 15
    )
    
    # Mark optimal point
    plt = annotate!(plt, optimal_size, cv_errors[optimal_size], "★")
    
    println(plt)
    
    # Also create a bar chart showing the error reduction
    if length(cv_errors) > 1
        error_reduction = [cv_errors[1] - cv_errors[i] for i in 1:length(cv_errors)]
        error_reduction = error_reduction ./ maximum(error_reduction)
        
        red_plt = barplot(
            n_reactions,
            error_reduction,
            title = "Error Reduction by Reaction Count",
            xlabel = "Number of Reactions",
            ylabel = "Normalized Reduction",
            width = 60,
            height = 15
        )
        
        # Mark optimal point
        red_plt = annotate!(red_plt, optimal_size, error_reduction[optimal_size], "★")
        
        println(red_plt)
    end
    
    return plt
end

"""
    visualize_mode_contributions(reaction_participation, top_reactions, mode_groups)

Visualize how different reactions participate in different types of dynamics.

# Arguments
- `reaction_participation`: Dictionary mapping stoichiometries to mode participation scores
- `top_reactions`: List of top reaction stoichiometries
- `mode_groups`: Dictionary grouping modes by type (slow, fast, oscillatory)

# Returns
- Nothing
"""
function visualize_mode_contributions(reaction_participation, top_reactions, mode_groups)
    # For each top reaction, show its participation in different mode groups
    for (i, stoich) in enumerate(top_reactions)
        if !haskey(reaction_participation, stoich)
            continue
        end
        
        # Get participation scores
        mode_scores = reaction_participation[stoich]
        
        # Normalize scores
        max_score = maximum(mode_scores)
        if max_score > 0
            mode_scores = mode_scores ./ max_score
        end
        
        # Create data for grouped bar chart
        group_names = ["Slow", "Fast", "Oscillatory"]
        group_scores = zeros(3)
        
        for (j, group) in enumerate(["slow", "fast", "oscillatory"])
            indices = mode_groups[group]
            if !isempty(indices)
                group_scores[j] = sum(mode_scores[indices]) / length(indices)
            end
        end
        
        # Create bar chart
        plt = barplot(
            group_names,
            group_scores,
            title = "Mode Contribution for Reaction $i",
            xlabel = "Mode Group",
            ylabel = "Normalized Participation",
            width = 40,
            height = 10
        )
        
        println(plt)
    end
end

"""
    visualize_conservation_laws(conservation_laws, species_names)

Visualize identified conservation laws.

# Arguments
- `conservation_laws`: List of conservation law vectors
- `species_names`: Names of species

# Returns
- Nothing
"""
function visualize_conservation_laws(conservation_laws, species_names)
    if isempty(conservation_laws)
        println("No conservation laws to visualize.")
        return
    end
    
    for (i, law) in enumerate(conservation_laws)
        # Format conservation law
        terms = []
        for (j, coef) in enumerate(law)
            if abs(coef) > 1e-10
                if j <= length(species_names)
                    push!(terms, "$(round(coef, digits=3)) × $(species_names[j])")
                else
                    push!(terms, "$(round(coef, digits=3)) × species$j")
                end
            end
        end
        
        law_str = join(terms, " + ") * " = constant"
        
        # Create bar chart of coefficients
        plt = barplot(
            species_names[1:min(length(law), length(species_names))],
            law[1:min(length(law), length(species_names))],
            title = "Conservation Law $i",
            xlabel = "Species",
            ylabel = "Coefficient",
            width = 50,
            height = 10
        )
        
        println(plt)
        println("Law $i: $law_str")
    end
end

"""
    visualize_reaction_rates(grouped_reactions, species_names)

Visualize reaction rate patterns for kinetics analysis.

# Arguments
- `grouped_reactions`: Dictionary mapping stoichiometries to reactions
- `species_names`: Names of species

# Returns
- Array of created plots
"""
function visualize_reaction_rates(grouped_reactions, species_names)
    println("\nPreparing reaction rate visualization data...")
    
    # Expected MM reactions
    mm_stoich = [
        tuple([0, 1, -1, 1]...),    # SE → E + P
        tuple([-1, -1, 1, 0]...),   # S + E → SE
        tuple([1, 1, -1, 0]...)     # SE → S + E
    ]
    
    reaction_names = [
        "SE → E + P",    # Product formation
        "S + E → SE",    # Complex formation
        "SE → S + E"     # Complex dissociation
    ]
    
    # Prepare plot data
    plot_data = []
    
    for (i, stoich) in enumerate(mm_stoich)
        if stoich in keys(grouped_reactions)
            rxns = grouped_reactions[stoich]
            
            if i == 1  # SE → E + P
                x_values = []
                y_values = []
                
                for r in rxns
                    se_idx = 3  # Index of SE
                    se_conc = r.from_state[se_idx] - 1  # Convert to concentration
                    rate = abs(r.rate)
                    
                    push!(x_values, se_conc)
                    push!(y_values, rate)
                end
                
                push!(plot_data, (
                    x_values=x_values,
                    y_values=y_values,
                    x_label="[SE] Concentration",
                    y_label="Reaction Rate",
                    title=reaction_names[i],
                    expected_slope=0.1  # kP
                ))
                
            elseif i == 2  # S + E → SE
                x_values = []
                y_values = []
                
                for r in rxns
                    s_idx = 1  # Index of S
                    e_idx = 2  # Index of E
                    s_conc = r.from_state[s_idx] - 1  # Convert to concentration
                    e_conc = r.from_state[e_idx] - 1  # Convert to concentration
                    product = s_conc * e_conc
                    rate = abs(r.rate)
                    
                    push!(x_values, product)
                    push!(y_values, rate)
                end
                
                push!(plot_data, (
                    x_values=x_values,
                    y_values=y_values,
                    x_label="[S]*[E] Product",
                    y_label="Reaction Rate",
                    title=reaction_names[i],
                    expected_slope=0.01  # kB
                ))
                
            elseif i == 3  # SE → S + E
                x_values = []
                y_values = []
                
                for r in rxns
                    se_idx = 3  # Index of SE
                    se_conc = r.from_state[se_idx] - 1  # Convert to concentration
                    rate = abs(r.rate)
                    
                    push!(x_values, se_conc)
                    push!(y_values, rate)
                end
                
                push!(plot_data, (
                    x_values=x_values,
                    y_values=y_values,
                    x_label="[SE] Concentration",
                    y_label="Reaction Rate",
                    title=reaction_names[i],
                    expected_slope=0.1  # kD
                ))
            end
        end
    end
    
    # Create plots using UnicodePlots
    plots = []
    for data in plot_data
        # Skip if no data points
        if isempty(data.x_values)
            continue
        end
        
        # Create scatter plot
        p = scatterplot(
            data.x_values, 
            data.y_values,
            xlabel=data.x_label,
            ylabel=data.y_label,
            title=data.title,
            width=60,
            height=20
        )
        
        # Add best-fit line if possible
        if length(data.x_values) > 1
            # Filter out zeros to avoid distortion
            valid_idx = findall(x -> x > 0, data.x_values)
            if !isempty(valid_idx)
                x_valid = data.x_values[valid_idx]
                y_valid = data.y_values[valid_idx]
                
                # Simple linear regression through origin
                slope = sum(x_valid .* y_valid) / sum(x_valid .^ 2)
                
                # Add linear regression line as annotation
                println("$p\nFitted slope (rate constant): k ≈ $(round(slope, digits=5))")
                println("Expected slope: k = $(data.expected_slope)")
                println("Accuracy: $(round(100 * slope / data.expected_slope, digits=1))%")
            end
        end
        
        push!(plots, p)
    end
    
    return plots
end

"""
    visualize_trajectory_data(ssa_trajs, species_indices, species_names; n_trajs=5, time_range=(0.0, 50.0))

Visualize sample trajectories from the data.

# Arguments
- `ssa_trajs`: Array of trajectory solutions
- `species_indices`: Indices of species to plot
- `species_names`: Names of species
- `n_trajs`: Number of trajectories to plot
- `time_range`: Range of time to plot

# Returns
- Array of created plots
"""
function visualize_trajectory_data(ssa_trajs, species_indices, species_names; n_trajs=5, time_range=(0.0, 50.0))
    # Use only a subset of trajectories
    n_trajs = min(n_trajs, length(ssa_trajs))
    trajs_subset = ssa_trajs[1:n_trajs]
    
    plots = []
    
    # Plot each species separately
    for (i, sp_idx) in enumerate(species_indices)
        if i > length(species_names)
            species_name = "Species $sp_idx"
        else
            species_name = species_names[i]
        end
        
        plt = lineplot(
            title="$species_name Trajectories",
            xlabel="Time",
            ylabel="Count",
            width=60,
            height=15
        )
        
        # Add each trajectory
        for (j, traj) in enumerate(trajs_subset)
            # Find points within time range
            time_mask = (traj.t .>= time_range[1]) .& (traj.t .<= time_range[2])
            
            if !any(time_mask)
                continue
            end
            
            times = traj.t[time_mask]
            counts = [traj.u[k][sp_idx] for k in findall(time_mask)]
            
            # Add line for this trajectory
            plt = lineplot!(plt, times, counts, label="Traj $j")
        end
        
        println(plt)
        push!(plots, plt)
    end
    
    return plots
end

"""
    visualize_histogram_evolution(sparse_probs, selected_states, species_indices, species_names, time_points)

Visualize the evolution of probability distributions over time.

# Arguments
- `sparse_probs`: Array of sparse probability distributions
- `selected_states`: List of selected states in the reduced space
- `species_indices`: Indices of species to visualize
- `species_names`: Names of species
- `time_points`: Time points corresponding to snapshots

# Returns
- Nothing
"""
function visualize_histogram_evolution(sparse_probs, selected_states, species_indices, species_names, time_points)
    n_snapshots = min(5, length(sparse_probs))  # Show at most 5 snapshots
    snapshot_indices = round.(Int, range(1, length(sparse_probs), length=n_snapshots))
    
    # For each species, extract its marginal distribution at different times
    for (sp_i, sp_idx) in enumerate(species_indices)
        if sp_i > length(species_names)
            species_name = "Species $sp_idx"
        else
            species_name = species_names[sp_i]
        end
        
        println("\nEvolution of $species_name distribution:")
        
        for t_idx in snapshot_indices
            time = time_points[t_idx]
            
            # Extract indices and values for this snapshot
            indices = sparse_probs[t_idx].indices
            values = sparse_probs[t_idx].values
            
            # Compute marginal distribution for this species
            sp_counts = Dict()
            for (i, idx) in enumerate(indices)
                sp_count = idx[sp_i] - 1  # Convert from 1-based index to count
                
                if !haskey(sp_counts, sp_count)
                    sp_counts[sp_count] = 0.0
                end
                
                sp_counts[sp_count] += values[i]
            end
            
            # Convert to arrays for plotting
            counts = collect(keys(sp_counts))
            probs = [sp_counts[c] for c in counts]
            
            # Skip if empty
            if isempty(counts)
                continue
            end
            
            # Create bar plot
            plt = barplot(
                counts,
                probs,
                title="$species_name at t = $time",
                xlabel="Count",
                ylabel="Probability",
                width=60,
                height=10
            )
            
            println(plt)
        end
    end
end
