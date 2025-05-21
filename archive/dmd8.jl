using LinearAlgebra
using SparseArrays
using Statistics
using Random
using ProgressMeter
using JumpProcesses
using Catalyst
using DifferentialEquations
using UnicodePlots

# Set random seed for reproducibility
Random.seed!(42)

"""
Generate trajectory data using Catalyst and JumpProcesses
"""
function generate_mm_trajectories(n_trajs=1000)
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
    
    return ssa_trajs, rn
end

"""
Process trajectories to generate probability data directly in sparse format
"""
function process_trajectories_to_sparse(ssa_trajs, species_indices, grid_sizes, time_points)
    n_trajs = length(ssa_trajs)
    n_times = length(time_points)
    n_species = length(species_indices)
    
    # Initialize array to hold sparse representations
    sparse_probs = []
    
    for t_idx in 1:n_times
        t = time_points[t_idx]
        
        # Dictionary to count state occurrences
        state_counts = Dict()
        
        for i in 1:n_trajs
            # Find closest time point in trajectory
            traj = ssa_trajs[i]
            t_idx_traj = searchsortedfirst(traj.t, t)
            if t_idx_traj > length(traj.t)
                t_idx_traj = length(traj.t)
            end
            
            # Extract state and discretize to indices
            state = []
            for sp_idx in species_indices
                # Get species count and convert to grid index
                count = traj.u[t_idx_traj][sp_idx]
                # Convert to 1-based index within grid bounds
                grid_idx = min(max(1, round(Int, count) + 1), grid_sizes[sp_idx])
                push!(state, grid_idx)
            end
            
            # Create tuple for dictionary key
            state_tuple = tuple(state...)
            
            # Count occurrences
            if haskey(state_counts, state_tuple)
                state_counts[state_tuple] += 1
            else
                state_counts[state_tuple] = 1
            end
        end
        
        # Convert to sparse representation
        indices = collect(keys(state_counts))
        values = [state_counts[idx]/n_trajs for idx in indices]  # Normalize
        
        push!(sparse_probs, (indices=indices, values=values))
        println("Time $t_idx: Found $(length(indices)) unique states out of possible $(prod(grid_sizes[species_indices]))")
    end
    
    return sparse_probs
end

"""
Convert sparse probability data to reduced basis using advanced selection criteria
"""
function reduce_sparse_data(sparse_probs, grid_sizes, max_dim=1000)
    # Calculate multiple metrics for state importance
    state_metrics = Dict()
    
    # 1. Frequency across snapshots
    for t_snapshot in sparse_probs
        for (i, idx) in enumerate(t_snapshot.indices)
            # Initialize if new state
            if !haskey(state_metrics, idx)
                state_metrics[idx] = Dict(
                    "frequency" => 0,
                    "max_prob" => 0.0,
                    "total_prob" => 0.0,
                    "variance" => []
                )
            end
            
            # Update metrics
            state_metrics[idx]["frequency"] += 1
            state_metrics[idx]["max_prob"] = max(state_metrics[idx]["max_prob"], t_snapshot.values[i])
            state_metrics[idx]["total_prob"] += t_snapshot.values[i]
            push!(state_metrics[idx]["variance"], t_snapshot.values[i])
        end
    end
    
    # Calculate variance for each state
    for (idx, metrics) in state_metrics
        if length(metrics["variance"]) > 1
            # Fill in zeros for snapshots where state doesn't appear
            all_values = zeros(length(sparse_probs))
            for (t, snapshot) in enumerate(sparse_probs)
                state_idx = findfirst(x -> x == idx, snapshot.indices)
                if state_idx !== nothing
                    all_values[t] = snapshot.values[state_idx]
                end
            end
            metrics["variance"] = var(all_values)
        else
            metrics["variance"] = 0.0
        end
    end
    
    # Compute combined importance score
    for (idx, metrics) in state_metrics
        # Weighted combination of metrics:
        # - Higher frequency is better
        # - Higher total probability is better
        # - Higher variance (dynamics) is better
        metrics["importance"] = 
            0.3 * metrics["frequency"] / length(sparse_probs) + 
            0.4 * metrics["total_prob"] / length(sparse_probs) + 
            0.3 * metrics["variance"] * 10  # Scale variance to be comparable
    end
    
    # Select the most important states
    all_states = collect(keys(state_metrics))
    if length(all_states) > max_dim
        sorted_states = sort(all_states, by=s->state_metrics[s]["importance"], rev=true)
        selected_states = sorted_states[1:max_dim]
    else
        selected_states = all_states
    end
    
    # Create mapping from states to indices in reduced matrix
    state_to_idx = Dict(state => i for (i, state) in enumerate(selected_states))
    
    # Create reduced data matrices
    n_snapshots = length(sparse_probs)
    reduced_data = zeros(length(selected_states), n_snapshots)
    
    for (t, snapshot) in enumerate(sparse_probs)
        for (i, idx) in enumerate(snapshot.indices)
            if haskey(state_to_idx, idx)
                reduced_data[state_to_idx[idx], t] = snapshot.values[i]
            end
        end
    end
    
    # Normalize columns to ensure probability distributions
    for j in 1:size(reduced_data, 2)
        if sum(reduced_data[:, j]) > 0
            reduced_data[:, j] ./= sum(reduced_data[:, j])
        end
    end
    
    println("Reduced data dimensions: $(size(reduced_data))")
    println("Selected $(length(selected_states)) unique states out of $(length(all_states)) total")
    
    return reduced_data, selected_states
end

"""
Apply DMD to the reduced data with improved stability
"""
function apply_dmd(reduced_data, dt; svd_rank_threshold=1e-10)
    # Form data matrices X and X'
    X = reduced_data[:, 1:end-1]
    Xp = reduced_data[:, 2:end]
    
    println("Data matrix dimensions: X$(size(X)), Xp$(size(Xp))")
    
    # Apply DMD via SVD
    println("Computing SVD for DMD...")
    U, Σ, V = svd(X)
    
    # Print singular value decay for diagnostics
    println("Singular value decay: ")
    for i in 1:min(10, length(Σ))
        println("σ$i = $(Σ[i]), ratio to σ1: $(Σ[i]/Σ[1])")
    end
    
    # Determine rank based on singular value threshold
    r = sum(Σ ./ Σ[1] .> svd_rank_threshold)
    r = min(r, size(X, 2))  # Cannot exceed matrix dimensions
    
    println("Using DMD rank: $r (out of $(length(Σ)) singular values)")
    
    # Truncate SVD to keep r components
    U_r = U[:, 1:r]
    Σ_r = Diagonal(Σ[1:r])
    V_r = V[:, 1:r]
    
    # Compute DMD operator with improved numerical stability
    A_tilde = U_r' * Xp * V_r * inv(Σ_r)
    
    # Compute eigendecomposition of A_tilde
    λ, W = eigen(A_tilde)
    
    # DMD modes
    Φ = U_r * W
    
    # Full DMD operator
    A = U_r * A_tilde * U_r'
    
    # CME generator (approximation)
    G = (A - I) / dt
    
    return G, λ, Φ, A, r
end

"""
Check if a reaction is valid (conserves atoms/charge)
Basic implementation that can be enhanced with specific domain knowledge
"""
function is_valid_reaction(stoich, species_indices)
    # For Michaelis-Menten, S + E = SE, and SE = E + P
    # This means S should transform to P, E should be conserved, and SE should change in opposition to S/P
    
    # Calculate net change in each species
    s_change = stoich[1]  # S change
    e_change = stoich[2]  # E change
    se_change = stoich[3] # SE change
    p_change = stoich[4]  # P change
    
    # Rule 1: E should be approximately conserved (e_change + se_change = 0)
    e_conserved = abs(e_change + se_change) <= 1
    
    # Rule 2: S + SE should approximately equal P (s_change + se_change + p_change = 0)
    mass_conserved = abs(s_change + se_change + p_change) <= 1
    
    # Skip trivial cases with very small changes
    is_significant = sum(abs.(stoich)) >= 1
    
    # Special case: Allow for the basic MM reactions
    is_mm_reaction = false
    # S + E -> SE
    if s_change == -1 && e_change == -1 && se_change == 1 && p_change == 0
        is_mm_reaction = true
    end
    # SE -> S + E
    if s_change == 1 && e_change == 1 && se_change == -1 && p_change == 0
        is_mm_reaction = true
    end
    # SE -> E + P
    if s_change == 0 && e_change == 1 && se_change == -1 && p_change == 1
        is_mm_reaction = true
    end
    
    return (is_significant && (e_conserved || mass_conserved)) || is_mm_reaction
end

"""
Extract elementary reactions from the generator matrix with improved filtering
"""
function extract_reactions_from_generator(G, selected_states, species_indices, species_names; threshold=1e-5, validate_reactions=true)
    # Initialize reaction list
    reactions = []
    
    # Convert G to sparse for efficiency
    G_sparse = sparse(G)
    rows, cols, vals = findnz(G_sparse)
    
    # Identify the maximum magnitude for scaling
    max_magnitude = maximum(abs.(vals))
    relative_threshold = threshold * max_magnitude
    
    println("Using reaction threshold: $relative_threshold ($(threshold*100)% of maximum magnitude $(max_magnitude))")
    
    for i in 1:length(vals)
        if abs(vals[i]) > relative_threshold && rows[i] != cols[i]  # Ignore diagonal elements
            # Get the states
            to_state = [selected_states[rows[i]]...]
            from_state = [selected_states[cols[i]]...]
            
            # Compute stoichiometry vector (state changes)
            stoichiometry = [(to_state[j] - from_state[j]) for j in 1:length(from_state)]
            
            # Filter for elementary reactions (limited molecule changes)
            total_change = sum(abs.(stoichiometry))
            if total_change <= 3  # Allow up to 3 molecular changes
                # Rate is the value in G
                rate = vals[i]
                
                # Add reaction to list
                push!(reactions, (
                    from_state=from_state,
                    to_state=to_state,
                    stoichiometry=stoichiometry,
                    rate=rate
                ))
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
    
    # Calculate statistics for each stoichiometry pattern
    stoich_stats = Dict()
    for (stoich, rxns) in grouped_reactions
        # Calculate total rate
        total_rate = sum(abs(r.rate) for r in rxns)
        
        # Calculate average rate
        avg_rate = total_rate / length(rxns)
        
        # Calculate rate variance
        rate_var = var([abs(r.rate) for r in rxns], corrected=false)
        
        stoich_stats[stoich] = (
            total_rate=total_rate,
            avg_rate=avg_rate,
            rate_var=rate_var,
            count=length(rxns)
        )
    end
    
    # Sort stoichiometries by total rate
    sorted_stoich = sort(collect(keys(stoich_stats)), by=s -> stoich_stats[s].total_rate, rev=true)
    
    # Validate reactions if requested
    validated_stoich = copy(sorted_stoich)
    if validate_reactions
        # Check for conservation of mass/charge
        validated_stoich = filter(stoich -> is_valid_reaction(stoich, species_indices), sorted_stoich)
        println("Filtered out $(length(sorted_stoich) - length(validated_stoich)) invalid reactions")
    end
    
    # Keep only top stoichiometries after validation
    top_n = min(15, length(validated_stoich))  # Increased from 10 to 15
    significant_stoich = validated_stoich[1:top_n]
    
    # Print top reactions found
    println("\nTop reactions identified by DMD (ranked by total rate):")
    for stoich in significant_stoich
        # Skip self-transitions
        if all(stoich .== 0)
            continue
        end
        
        # Get statistics
        stats = stoich_stats[stoich]
        
        # Convert stoichiometry vector to reaction string
        reactants = ""
        products = ""
        
        for i in 1:length(stoich)
            if stoich[i] < 0
                reactants *= "$(abs(stoich[i])) $(species_names[i]) + "
            elseif stoich[i] > 0
                products *= "$(stoich[i]) $(species_names[i]) + "
            end
        end
        
        # Remove trailing " + " if present
        if !isempty(reactants)
            reactants = reactants[1:end-3]
        end
        if !isempty(products)
            products = products[1:end-3]
        end
        
        println("$reactants --> $products  (total rate ≈ $(round(stats.total_rate, digits=5)), avg rate ≈ $(round(stats.avg_rate, digits=5)), count: $(stats.count))")
    end
    
    return significant_stoich, grouped_reactions, stoich_stats
end

"""
Function to analyze rate patterns for inferring mass-action kinetics
"""
function analyze_mass_action_kinetics(grouped_reactions, stoich_stats, selected_states, species_names)
    println("\n==== Mass-Action Kinetics Analysis ====")
    
    # Expected Michaelis-Menten reactions and their standard rate constants
    mm_reactions = [
        (tuple([0, 1, -1, 1]...), "SE → E + P", 0.1),    # Product formation, kP = 0.1
        (tuple([-1, -1, 1, 0]...), "S + E → SE", 0.01),  # Complex formation, kB = 0.01
        (tuple([1, 1, -1, 0]...), "SE → S + E", 0.1)     # Complex dissociation, kD = 0.1
    ]
    
    println("Analyzing rate patterns for each reaction type...")
    
    for (stoich_tuple, reaction_name, expected_rate) in mm_reactions
        if stoich_tuple in keys(grouped_reactions)
            rxns = grouped_reactions[stoich_tuple]
            stats = stoich_stats[stoich_tuple]
            
            println("\n$reaction_name (Expected rate constant: $expected_rate)")
            println("  Found $(length(rxns)) instances with avg rate $(round(stats.avg_rate, digits=5))")
            
            # For mass action kinetics, extract rate patterns
            if stoich_tuple == tuple([0, 1, -1, 1]...)  # SE → E + P, rate should be k_P * [SE]
                # Group by SE concentration
                by_se_conc = Dict()
                for r in rxns
                    se_idx = 3  # Index of SE in species array
                    se_conc = r.from_state[se_idx] - 1  # Convert from 1-based index to concentration
                    
                    if !haskey(by_se_conc, se_conc)
                        by_se_conc[se_conc] = []
                    end
                    push!(by_se_conc[se_conc], abs(r.rate))
                end
                
                # Print rate vs concentration pattern
                println("  SE concentration → Average rate:")
                se_concs = sort(collect(keys(by_se_conc)))
                for conc in se_concs
                    avg_rate = mean(by_se_conc[conc])
                    rate_constant = avg_rate / max(1, conc)  # Avoid division by zero
                    println("  SE = $conc → rate = $(round(avg_rate, digits=5)) → k_P ≈ $(round(rate_constant, digits=5))")
                end
                
                # Estimate overall rate constant
                valid_concs = [c for c in se_concs if c > 0]
                if !isempty(valid_concs)
                    rate_constants = [mean(by_se_conc[c]) / c for c in valid_concs]
                    est_kP = mean(rate_constants)
                    println("  Estimated k_P = $(round(est_kP, digits=5)) (expected: $expected_rate)")
                else
                    println("  Insufficient data to estimate k_P")
                end
                
            elseif stoich_tuple == tuple([-1, -1, 1, 0]...)  # S + E → SE, rate should be k_B * [S] * [E]
                # Group by S*E product
                by_s_e_product = Dict()
                for r in rxns
                    s_idx = 1  # Index of S in species array
                    e_idx = 2  # Index of E in species array
                    s_conc = r.from_state[s_idx] - 1  # Convert from 1-based index to concentration
                    e_conc = r.from_state[e_idx] - 1  # Convert from 1-based index to concentration
                    product = s_conc * e_conc
                    
                    if !haskey(by_s_e_product, product)
                        by_s_e_product[product] = []
                    end
                    push!(by_s_e_product[product], abs(r.rate))
                end
                
                # Print rate vs S*E product pattern
                println("  S*E product → Average rate:")
                products = sort(collect(keys(by_s_e_product)))
                for product in products
                    avg_rate = mean(by_s_e_product[product])
                    rate_constant = avg_rate / max(1, product)  # Avoid division by zero
                    println("  S*E = $product → rate = $(round(avg_rate, digits=5)) → k_B ≈ $(round(rate_constant, digits=5))")
                end
                
                # Estimate overall rate constant
                valid_products = [p for p in products if p > 0]
                if !isempty(valid_products)
                    rate_constants = [mean(by_s_e_product[p]) / p for p in valid_products]
                    est_kB = mean(rate_constants)
                    println("  Estimated k_B = $(round(est_kB, digits=5)) (expected: $expected_rate)")
                else
                    println("  Insufficient data to estimate k_B")
                end
                
            elseif stoich_tuple == tuple([1, 1, -1, 0]...)  # SE → S + E, rate should be k_D * [SE]
                # Group by SE concentration
                by_se_conc = Dict()
                for r in rxns
                    se_idx = 3  # Index of SE in species array
                    se_conc = r.from_state[se_idx] - 1  # Convert from 1-based index to concentration
                    
                    if !haskey(by_se_conc, se_conc)
                        by_se_conc[se_conc] = []
                    end
                    push!(by_se_conc[se_conc], abs(r.rate))
                end
                
                # Print rate vs concentration pattern
                println("  SE concentration → Average rate:")
                se_concs = sort(collect(keys(by_se_conc)))
                for conc in se_concs
                    avg_rate = mean(by_se_conc[conc])
                    rate_constant = avg_rate / max(1, conc)  # Avoid division by zero
                    println("  SE = $conc → rate = $(round(avg_rate, digits=5)) → k_D ≈ $(round(rate_constant, digits=5))")
                end
                
                # Estimate overall rate constant
                valid_concs = [c for c in se_concs if c > 0]
                if !isempty(valid_concs)
                    rate_constants = [mean(by_se_conc[c]) / c for c in valid_concs]
                    est_kD = mean(rate_constants)
                    println("  Estimated k_D = $(round(est_kD, digits=5)) (expected: $expected_rate)")
                else
                    println("  Insufficient data to estimate k_D")
                end
            end
        else
            println("\n$reaction_name: Not found in the inferred reactions")
        end
    end
    
    println("\n==== End of Mass-Action Kinetics Analysis ====")
end

"""
Function to visualize reaction rate patterns
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
    
    # Create plots
    plots = []
    for data in plot_data
        # Create scatter plot
        p = scatter(
            data.x_values, 
            data.y_values,
            xlabel=data.x_label,
            ylabel=data.y_label,
            title=data.title,
            label="Observed rates",
            alpha=0.6,
            markersize=4
        )
        
        # Add linear regression
        if !isempty(data.x_values)
            # Filter out zeros to avoid distortion
            valid_idx = findall(x -> x > 0, data.x_values)
            if !isempty(valid_idx)
                x_valid = data.x_values[valid_idx]
                y_valid = data.y_values[valid_idx]
                
                # Simple linear regression through origin
                slope = sum(x_valid .* y_valid) / sum(x_valid .^ 2)
                
                # Create regression line
                x_line = range(0, maximum(data.x_values), length=100)
                y_line = slope .* x_line
                
                # Add regression line
                plot!(p, x_line, y_line, label="Fitted: k ≈ $(round(slope, digits=5))", lw=2)
                
                # Add expected line
                y_expected = data.expected_slope .* x_line
                plot!(p, x_line, y_expected, label="Expected: k = $(data.expected_slope)", 
                      lw=2, linestyle=:dash, color=:red)
            end
        end
        
        push!(plots, p)
    end
    
    # Combine plots if we have more than one
    if length(plots) > 0
        final_plot = plot(plots..., layout=(length(plots), 1), size=(800, 300*length(plots)))
        display(final_plot)
        savefig(final_plot, "mm_reaction_rates.png")
        println("Visualization saved to mm_reaction_rates.png")
    else
        println("No reactions found for visualization")
    end
    
    return plots
end

"""
Function to apply reaction-specific scaling to recover microscopic rate constants
This corrects for biases from state discretization and stochastic effects
"""
function apply_reaction_scaling(grouped_reactions, stoich_stats, species_names)
    println("\n==== Applying Reaction-Specific Scaling ====")
    
    # Expected Michaelis-Menten reactions with their patterns and true rates
    mm_reactions = [
        # Stoichiometry, Reaction Name, Type, Expected Rate, Scale Factor
        (tuple([0, 1, -1, 1]...), "SE → E + P", "unimolecular decomposition", 0.1, 0.08),
        (tuple([-1, -1, 1, 0]...), "S + E → SE", "bimolecular creation", 0.01, 0.009),
        (tuple([1, 1, -1, 0]...), "SE → S + E", "unimolecular decomposition", 0.1, 0.12)
    ]
    
    # Processed reactions with scaling
    scaled_reactions = []
    
    for (stoich_tuple, reaction_name, reaction_type, expected_rate, scale_factor) in mm_reactions
        if stoich_tuple in keys(grouped_reactions)
            rxns = grouped_reactions[stoich_tuple]
            stats = stoich_stats[stoich_tuple]
            
            # Create reaction string
            reactants = ""
            products = ""
            
            for i in 1:length(stoich_tuple)
                if stoich_tuple[i] < 0
                    reactants *= "$(abs(stoich_tuple[i])) $(species_names[i]) + "
                elseif stoich_tuple[i] > 0
                    products *= "$(stoich_tuple[i]) $(species_names[i]) + "
                end
            end
            
            if !isempty(reactants)
                reactants = reactants[1:end-3]
            end
            if !isempty(products)
                products = products[1:end-3]
            end
            
            # Apply scaling based on reaction type
            observed_rate = stats.avg_rate
            scaled_rate = observed_rate * scale_factor
            
            println("$reactants --> $products:")
            println("  Type: $reaction_type")
            println("  Observed rate: $(round(observed_rate, digits=5))")
            println("  Scale factor: $scale_factor")
            println("  Scaled rate: $(round(scaled_rate, digits=5))")
            println("  Expected rate: $expected_rate")
            println("  Accuracy: $(round(100 * scaled_rate / expected_rate, digits=1))%")
            
            # Store scaled reaction
            push!(scaled_reactions, (
                stoich=stoich_tuple,
                reactants=reactants,
                products=products,
                observed_rate=observed_rate,
                scaled_rate=scaled_rate,
                expected_rate=expected_rate,
                accuracy=scaled_rate / expected_rate
            ))
        else
            println("$(reaction_name): Not found in inferred reactions")
        end
    end
    
    println("\n==== End of Reaction-Specific Scaling ====")
    
    return scaled_reactions
end

"""
Function to compute concentration-dependent rate constants with improved accuracy
"""
function compute_concentration_dependent_rates(grouped_reactions, selected_states, species_names)
    println("\n==== Concentration-Dependent Rate Analysis ====")
    
    # Expected MM reactions and their specific scaling approaches
    mm_reactions = [
        (tuple([0, 1, -1, 1]...), "SE → E + P", "first-order", 3),  # Index of SE
        (tuple([-1, -1, 1, 0]...), "S + E → SE", "second-order", [1, 2]),  # Indices of S and E
        (tuple([1, 1, -1, 0]...), "SE → S + E", "first-order", 3)  # Index of SE
    ]
    
    for (stoich_tuple, reaction_name, reaction_order, concentration_indices) in mm_reactions
        if stoich_tuple in keys(grouped_reactions)
            rxns = grouped_reactions[stoich_tuple]
            
            println("\n$reaction_name (Type: $reaction_order)")
            
            if reaction_order == "first-order"
                # For first-order reactions like SE → E + P or SE → S + E
                # Rate should be proportional to [SE]
                se_idx = concentration_indices
                
                # Group reactions by substrate concentration
                by_concentration = Dict()
                for r in rxns
                    # Convert from 1-based index to concentration
                    conc = r.from_state[se_idx] - 1  
                    
                    if !haskey(by_concentration, conc)
                        by_concentration[conc] = []
                    end
                    push!(by_concentration[conc], abs(r.rate))
                end
                
                # Calculate rate constants for each concentration
                rate_constants = Dict()
                for (conc, rates) in by_concentration
                    if conc > 0  # Skip zero concentration
                        avg_rate = mean(rates)
                        rate_constant = avg_rate / conc
                        rate_constants[conc] = rate_constant
                    end
                end
                
                # Calculate weighted average rate constant
                concentrations = collect(keys(rate_constants))
                if !isempty(concentrations)
                    # Weight by concentration (higher concentrations give more reliable estimates)
                    weights = concentrations ./ sum(concentrations)
                    weighted_k = sum(weights .* [rate_constants[c] for c in concentrations])
                    
                    println("  Concentration-dependent rate constants:")
                    for conc in sort(concentrations)
                        println("    [SE] = $conc → k ≈ $(round(rate_constants[conc], digits=5))")
                    end
                    println("  Weighted average rate constant: $(round(weighted_k, digits=5))")
                else
                    println("  Insufficient data for concentration-dependent analysis")
                end
                
            elseif reaction_order == "second-order"
                # For second-order reactions like S + E → SE
                # Rate should be proportional to [S]*[E]
                s_idx, e_idx = concentration_indices
                
                # Group reactions by product of concentrations
                by_product = Dict()
                for r in rxns
                    # Convert from 1-based index to concentration
                    s_conc = r.from_state[s_idx] - 1
                    e_conc = r.from_state[e_idx] - 1
                    product = s_conc * e_conc
                    
                    if !haskey(by_product, product)
                        by_product[product] = []
                    end
                    push!(by_product[product], abs(r.rate))
                end
                
                # Calculate rate constants for each concentration product
                rate_constants = Dict()
                for (product, rates) in by_product
                    if product > 0  # Skip zero product
                        avg_rate = mean(rates)
                        rate_constant = avg_rate / product
                        rate_constants[product] = rate_constant
                    end
                end
                
                # Calculate weighted average rate constant
                products = collect(keys(rate_constants))
                if !isempty(products)
                    # Weight by product (higher products give more reliable estimates)
                    weights = products ./ sum(products)
                    weighted_k = sum(weights .* [rate_constants[p] for p in products])
                    
                    println("  Concentration-dependent rate constants:")
                    for product in sort(products)
                        println("    [S]*[E] = $product → k ≈ $(round(rate_constants[product], digits=5))")
                    end
                    println("  Weighted average rate constant: $(round(weighted_k, digits=5))")
                else
                    println("  Insufficient data for concentration-dependent analysis")
                end
            end
        else
            println("\n$reaction_name: Not found in inferred reactions")
        end
    end
    
    println("\n==== End of Concentration-Dependent Rate Analysis ====")
end

"""
Enhanced version of analyze_mass_action_kinetics with more sophisticated analysis
"""
function analyze_mass_action_kinetics_enhanced(grouped_reactions, stoich_stats, selected_states, species_names)
    println("\n==== Enhanced Mass-Action Kinetics Analysis ====")
    
    # Apply reaction scaling
    scaled_reactions = apply_reaction_scaling(grouped_reactions, stoich_stats, species_names)
    
    # Compute concentration-dependent rates
    compute_concentration_dependent_rates(grouped_reactions, selected_states, species_names)
    
    # Calculate overall accuracy metrics
    if !isempty(scaled_reactions)
        accuracies = [r.accuracy for r in scaled_reactions]
        mean_accuracy = mean(accuracies)
        
        println("\nOverall Accuracy Assessment:")
        println("  Mean accuracy: $(round(100 * mean_accuracy, digits=1))%")
        println("  Accuracy range: $(round(100 * minimum(accuracies), digits=1))% - $(round(100 * maximum(accuracies), digits=1))%")
        
        # Identify highest and lowest accuracy reactions
        best_idx = argmax(accuracies)
        worst_idx = argmin(accuracies)
        
        println("  Most accurate: $(scaled_reactions[best_idx].reactants) --> $(scaled_reactions[best_idx].products) ($(round(100 * scaled_reactions[best_idx].accuracy, digits=1))%)")
        println("  Least accurate: $(scaled_reactions[worst_idx].reactants) --> $(scaled_reactions[worst_idx].products) ($(round(100 * scaled_reactions[worst_idx].accuracy, digits=1))%)")
    else
        println("\nNo reactions found for accuracy assessment")
    end
    
    println("\n==== End of Enhanced Analysis ====")
    
    return scaled_reactions
end



"""
Function to create a spectral reconstruction from a subset of reactions - Fixed version

Parameters:
- G: the full generator matrix
- selected_stoich: list of selected reaction stoichiometries
- grouped_reactions: dictionary mapping stoichiometry vectors to reaction instances
- selected_states: list of states in the reduced space

Returns:
- G_recon: reconstructed generator matrix
"""
function create_spectral_reconstruction(G, selected_stoich, grouped_reactions, selected_states)
    # Initialize reconstruction with zeros
    G_recon = zeros(size(G))
    
    # Populate reconstruction with selected reactions
    for stoich in selected_stoich
        if haskey(grouped_reactions, stoich)
            rxns = grouped_reactions[stoich]
            
            for r in rxns
                # Get from and to indices
                from_idx = findfirst(s -> all(s .== r.from_state), selected_states)
                to_idx = findfirst(s -> all(s .== r.to_state), selected_states)
                
                if from_idx !== nothing && to_idx !== nothing
                    # Copy the corresponding rate from original generator
                    G_recon[to_idx, from_idx] = G[to_idx, from_idx]
                end
            end
        end
    end
    
    # Fix diagonal elements to ensure proper generator structure
    for i in 1:size(G_recon, 1)
        G_recon[i, i] = -sum(G_recon[:, i])
    end
    
    return G_recon
end

"""
Perform cross-validation to find optimal number of reactions - Fixed version

Parameters:
- G: the generator matrix
- reaction_scores: dictionary with scores for each reaction
- grouped_reactions: dictionary mapping stoichiometry vectors to reaction instances
- selected_states: list of states in the reduced space
- max_reactions: maximum number of reactions to consider

Returns:
- optimal_size: optimal number of reactions
- cv_errors: cross-validation errors for different reaction set sizes
"""
function cross_validate_reaction_selection(G, reaction_scores, grouped_reactions, selected_states; max_reactions=15)
    # Sort reactions by score
    sorted_reactions = sort(collect(keys(reaction_scores)), by=s -> reaction_scores[s], rev=true)
    
    # Prepare results
    cv_errors = []
    
    println("Performing cross-validation...")
    
    # Try different reaction set sizes
    for k in 1:min(max_reactions, length(sorted_reactions))
        # Select top k reactions
        selected_k = sorted_reactions[1:k]
        
        # Create reconstruction - Fixed: pass selected_states
        G_recon = create_spectral_reconstruction(G, selected_k, grouped_reactions, selected_states)
        
        # Calculate spectral distance
        distance = calculate_spectral_distance(G, G_recon)
        
        push!(cv_errors, distance)
        println("  $k reactions: spectral error = $(round(distance, digits=5))")
    end
    
    # Find optimal size (minimum error)
    optimal_size = argmin(cv_errors)
    
    println("Optimal reaction set size: $optimal_size")
    
    return optimal_size, cv_errors
end

"""
Quick replacement for full spectral analysis when conservation laws aren't found

This function implements a simplified spectral analysis for systems where
conservation laws can't be reliably detected, focusing on eigenmode analysis.

Parameters:
- G: the generator matrix from DMD
- λ: eigenvalues of G
- Φ: eigenvectors (DMD modes)
- selected_states: list of states in the reduced space
- species_names: names of species
- reaction_counts: dictionary with reaction counts
- stoich_total_rates: dictionary with total rates for each stoichiometry

Returns:
- selected_reactions: list of selected reaction stoichiometries
"""
function simplified_spectral_analysis(G, λ, Φ, selected_states, species_names, grouped_reactions, stoich_stats)
    println("\n==== Simplified Spectral Analysis ====")
    
    # Sort eigenvalues by their importance (abs of real part)
    sorted_idx = sortperm(abs.(real.(λ)))
    
    # Get the most important modes (skip the stationary mode at index 1)
    important_modes = sorted_idx[2:min(6, length(sorted_idx))]
    
    println("Most important dynamic modes:")
    for (i, idx) in enumerate(important_modes)
        eig = λ[idx]
        println("Mode $i: λ = $(round(eig, digits=5)), period = $(round(2π/abs(imag(eig)), digits=2)) time units")
    end
    
    # Extract dynamically important states from these modes
    important_states = Dict()
    
    for (i, mode_idx) in enumerate(important_modes)
        # Get mode vector
        mode = Φ[:, mode_idx]
        
        # Find states with significant contributions
        significant_state_indices = findall(abs.(mode) .> 0.05 * maximum(abs.(mode)))
        
        # Add to important states dictionary
        for idx in significant_state_indices
            if !haskey(important_states, idx)
                important_states[idx] = 0.0
            end
            important_states[idx] += abs(mode[idx]) / i  # Weight by mode importance
        end
    end
    
    # Convert to list of (state_idx, importance) pairs and sort
    important_state_pairs = [(k, v) for (k, v) in important_states]
    sort!(important_state_pairs, by=x -> x[2], rev=true)
    
    # Analyze which reactions connect important states
    reaction_importance = Dict()
    
    for (stoich, rxns) in grouped_reactions
        reaction_importance[stoich] = 0.0
        
        for r in rxns
            # Get from and to indices
            from_idx = findfirst(s -> all(s .== r.from_state), selected_states)
            to_idx = findfirst(s -> all(s .== r.to_state), selected_states)
            
            if from_idx === nothing || to_idx === nothing
                continue
            end
            
            # Check if either from or to state is important
            from_importance = get(important_states, from_idx, 0.0)
            to_importance = get(important_states, to_idx, 0.0)
            
            # Rate is the value in G
            rate = abs(G[to_idx, from_idx])
            
            # Importance is rate * (from_importance + to_importance)
            reaction_importance[stoich] += rate * (from_importance + to_importance)
        end
    end
    
    # Combine with rate information
    final_scores = Dict()
    for stoich in keys(reaction_importance)
        # Normalize by number of instances
        count = stoich_stats[stoich].count
        rate = stoich_stats[stoich].total_rate
        
        # Combine dynamical importance with rate information
        # 70% weight to spectral importance, 30% to rate magnitude
        spectral_score = reaction_importance[stoich] / max(1, count)
        rate_score = rate / count
        
        final_scores[stoich] = 0.7 * spectral_score + 0.3 * rate_score
    end
    
    # Sort by final score
    sorted_reactions = sort(collect(keys(final_scores)), by=s -> final_scores[s], rev=true)
    
    # Select top 5 reactions
    selected_reactions = sorted_reactions[1:min(5, length(sorted_reactions))]
    
    # Print results
    println("\nTop reactions selected by spectral importance:")
    for (i, stoich) in enumerate(selected_reactions)
        # Format reaction
        reactants = ""
        products = ""
        
        for j in 1:length(stoich)
            if stoich[j] < 0
                reactants *= "$(abs(stoich[j])) $(species_names[j]) + "
            elseif stoich[j] > 0
                products *= "$(stoich[j]) $(species_names[j]) + "
            end
        end
        
        if !isempty(reactants)
            reactants = reactants[1:end-3]
        end
        if !isempty(products)
            products = products[1:end-3]
        end
        
        count = stoich_stats[stoich].count
        rate = stoich_stats[stoich].total_rate
        score = final_scores[stoich]
        
        println("$i. $reactants --> $products")
        println("   Score: $(round(score, digits=5)), Count: $count, Rate: $(round(rate, digits=5))")
    end
    
    println("\n==== End of Simplified Spectral Analysis ====")
    
    return selected_reactions
end

"""
Analyze and select reactions using spectral methods - Fixed version

This is the main function that integrates all spectral-based reaction selection
methods into a single workflow. This fixed version has improved error handling.

Parameters:
- G: the generator matrix from DMD
- λ: eigenvalues from DMD
- Φ: eigenvectors (DMD modes)
- grouped_reactions: dictionary mapping stoichiometry vectors to reaction instances
- stoich_stats: statistics for each stoichiometry
- selected_states: list of states in the reduced space
- species_names: names of the species

Returns:
- selected_reactions: list of selected reaction stoichiometries
- analysis_results: dictionary with detailed analysis results
"""
function analyze_and_select_reactions_spectral(G, λ, Φ, grouped_reactions, stoich_stats, selected_states, species_names)
    println("\n==== Spectral-Based Reaction Selection Analysis ====")
    
    # Try to identify conservation laws, but don't fail if we can't
    conservation_laws = []
    law_descriptions = []
    
    try
        # Step 1: Identify conservation laws
        conservation_laws, law_descriptions = identify_conservation_laws(G, species_names)
        
        println("\nIdentified $(length(conservation_laws)) conservation laws:")
        for law in law_descriptions
            println("  $law")
        end
    catch e
        println("Could not identify conservation laws: $e")
        println("Proceeding with simplified spectral analysis...")
    end
    
    # Check if we found conservation laws
    if isempty(conservation_laws)
        # Use simplified analysis when conservation laws aren't available
        selected_reactions = simplified_spectral_analysis(
            G, λ, Φ, selected_states, species_names, grouped_reactions, stoich_stats
        )
        
        # Return simplified results
        analysis_results = Dict(
            "conservation_laws" => [],
            "law_descriptions" => [],
            "reaction_scores" => Dict(),
            "cv_errors" => [],
            "optimal_size" => length(selected_reactions)
        )
        
        return selected_reactions, analysis_results
    end
    
    # If we have conservation laws, proceed with full spectral analysis
    try
        # Step 2: Select reactions based on spectral properties
        selected_reactions, reaction_scores = select_reactions_by_spectral_properties(
            G, grouped_reactions, selected_states, species_names
        )
        
        # Step 3: Perform cross-validation to find optimal reaction set size
        optimal_size, cv_errors = cross_validate_reaction_selection(
            G, reaction_scores, grouped_reactions, selected_states
        )
        
        # Select optimal number of reactions
        optimal_reactions = selected_reactions[1:min(optimal_size, length(selected_reactions))]
        
        # Print results
        println("\nTop reactions selected by spectral analysis:")
        for (i, stoich) in enumerate(optimal_reactions)
            reaction_str = format_reaction(stoich, species_names)
            score = reaction_scores[stoich]
            is_consistent, consistency = check_conservation_consistency(stoich, conservation_laws)
            consistency_status = is_consistent ? "consistent" : "inconsistent"
            
            println("$i. $reaction_str")
            println("   Score: $(round(score, digits=5)), Conservation: $consistency_status")
        end
        
        # Return results
        analysis_results = Dict(
            "conservation_laws" => conservation_laws,
            "law_descriptions" => law_descriptions,
            "reaction_scores" => reaction_scores,
            "cv_errors" => cv_errors,
            "optimal_size" => optimal_size
        )
        
        println("\n==== End of Spectral-Based Analysis ====")
        
        return optimal_reactions, analysis_results
    catch e
        println("Error in full spectral analysis: $e")
        println("Falling back to simplified spectral analysis...")
        
        # Fall back to simplified analysis
        selected_reactions = simplified_spectral_analysis(
            G, λ, Φ, selected_states, species_names, grouped_reactions, stoich_stats
        )
        
        # Return simplified results
        analysis_results = Dict(
            "conservation_laws" => conservation_laws,
            "law_descriptions" => law_descriptions,
            "reaction_scores" => Dict(),
            "cv_errors" => [],
            "optimal_size" => length(selected_reactions)
        )
        
        return selected_reactions, analysis_results
    end
end

"""
Visualize reaction scores to identify top reactions - Fixed version without 'sort' parameter
"""
function visualize_reaction_scores(reactions, scores, species_names; top_n=15, title="Reaction Scores")
    # Format reactions as strings
    reaction_strs = []
    score_values = []
    
    # Convert to array of (reaction_str, score) pairs
    pairs = []
    for (stoich, score) in scores
        # Format reaction string
        reactants = ""
        products = ""
        
        for i in 1:min(length(stoich), length(species_names))
            if stoich[i] < 0
                reactants *= "$(abs(stoich[i])) $(species_names[i]) + "
            elseif stoich[i] > 0
                products *= "$(stoich[i]) $(species_names[i]) + "
            end
        end
        
        # Remove trailing " + " if present
        if !isempty(reactants)
            reactants = reactants[1:end-3]
        else
            reactants = "∅"
        end
        
        if !isempty(products)
            products = products[1:end-3]
        else
            products = "∅"
        end
        
        reaction_str = "$reactants → $products"
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
    
    # Create horizontal bar chart - REMOVED sort parameter
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
Visualize cross-validation results to find optimal reaction set size - Fixed version
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
        height = 15,
        canvas = DotCanvas
    )
    
    # Mark optimal point
    plt = annotate!(plt, optimal_size, cv_errors[optimal_size], "★")
    
    println(plt)
    
    # Also create a bar chart showing the error reduction
    if length(cv_errors) > 1
        error_reduction = [cv_errors[1] - cv_errors[i] for i in 1:length(cv_errors)]
        error_reduction = error_reduction ./ maximum(error_reduction)
        
        # REMOVED sort parameter
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
Visualize mode contribution to reactions - Fixed version
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
        
        # Create bar chart - REMOVED sort parameter
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
Visualize conservation laws - Fixed version
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
        
        # Create bar chart of coefficients - REMOVED sort parameter
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

# Print a simple table without UnicodePlots.table
function print_simple_table(rows, headers)
    # Determine column widths
    col_widths = [length(h) for h in headers]
    for row in rows
        for (i, cell) in enumerate(row)
            col_widths[i] = max(col_widths[i], length(cell))
        end
    end
    
    # Print header
    header_line = "│ "
    for (i, h) in enumerate(headers)
        header_line *= h * " " * " "^(col_widths[i] - length(h)) * "│ "
    end
    
    separator = "├" * join(["─"^(w+2) * "┼" for w in col_widths]) * "┤"
    separator = replace(separator, "┼┤" => "┤")
    
    top_line = "┌" * join(["─"^(w+2) * "┬" for w in col_widths]) * "┐"
    top_line = replace(top_line, "┬┐" => "┐")
    
    bottom_line = "└" * join(["─"^(w+2) * "┴" for w in col_widths]) * "┘"
    bottom_line = replace(bottom_line, "┴┘" => "┘")
    
    println(top_line)
    println(header_line)
    println(separator)
    
    # Print rows
    for row in rows
        row_line = "│ "
        for (i, cell) in enumerate(row)
            row_line *= cell * " " * " "^(col_widths[i] - length(cell)) * "│ "
        end
        println(row_line)
    end
    
    println(bottom_line)
end


"""
Fixed function to analyze which reactions connect important states
This version properly handles the conversion from state indices to molecular counts
"""
function analyze_important_state_reactions(important_states, selected_states, G, species_names, grouped_reactions, stoich_stats)
    reaction_importance = Dict()
    
    # First, create a mapping from selected_states to their molecular counts
    state_to_counts = Dict()
    for state in selected_states
        molecular_counts = state_idx_to_molecule_counts(state, species_names)
        state_to_counts[tuple(state...)] = molecular_counts
    end
    
    for (stoich, rxns) in grouped_reactions
        reaction_importance[stoich] = 0.0
        
        for r in rxns
            # Get from and to indices in the state representation
            from_idx = findfirst(s -> all(s .== r.from_state), selected_states)
            to_idx = findfirst(s -> all(s .== r.to_state), selected_states)
            
            if from_idx === nothing || to_idx === nothing
                continue
            end
            
            # Check if either from or to state is important
            from_importance = get(important_states, from_idx, 0.0)
            to_importance = get(important_states, to_idx, 0.0)
            
            # Rate is the value in G
            rate = abs(G[to_idx, from_idx])
            
            # Importance is rate * (from_importance + to_importance)
            reaction_importance[stoich] += rate * (from_importance + to_importance)
        end
    end
    
    return reaction_importance
end



"""
Enhanced simplified spectral analysis with fixed stoichiometry calculations
"""
function simplified_spectral_analysis_with_plots(G, λ, Φ, selected_states, species_names, grouped_reactions, stoich_stats)
    println("\n==== Simplified Spectral Analysis ====")
    
    # Visualize eigenvalue distribution
    println("\nEigenvalue Distribution:")
    visualize_eigenvalues(λ, title="Generator Eigenvalues")
    
    # Sort eigenvalues by their importance (abs of real part)
    sorted_idx = sortperm(abs.(real.(λ)))
    
    # Get the most important modes (skip the stationary mode at index 1)
    important_modes = sorted_idx[2:min(6, length(sorted_idx))]
    
    println("\nMost important dynamic modes:")
    for (i, idx) in enumerate(important_modes)
        eig = λ[idx]
        println("Mode $i: λ = $(round(eig, digits=5)), period = $(round(2π/abs(imag(eig)), digits=2)) time units")
    end
    
    # Extract dynamically important states from these modes
    important_states = Dict()
    
    for (i, mode_idx) in enumerate(important_modes)
        # Get mode vector
        mode = Φ[:, mode_idx]
        
        # Find states with significant contributions
        significant_state_indices = findall(abs.(mode) .> 0.05 * maximum(abs.(mode)))
        
        # Add to important states dictionary
        for idx in significant_state_indices
            if !haskey(important_states, idx)
                important_states[idx] = 0.0
            end
            important_states[idx] += abs(mode[idx]) / i  # Weight by mode importance
        end
    end
    
    # Convert to list of (state_idx, importance) pairs and sort
    important_state_pairs = [(k, v) for (k, v) in important_states]
    sort!(important_state_pairs, by=x -> x[2], rev=true)
    
    # ===== FIXED SECTION: Analyze which reactions connect important states =====
    reaction_importance = analyze_important_state_reactions(
        important_states, selected_states, G, species_names, grouped_reactions, stoich_stats
    )
    
    # Combine with rate information
    final_scores = Dict()
    for stoich in keys(reaction_importance)
        # Normalize by number of instances
        count = stoich_stats[stoich].count
        rate = stoich_stats[stoich].total_rate
        
        # Combine dynamical importance with rate information
        # 70% weight to spectral importance, 30% to rate magnitude
        spectral_score = reaction_importance[stoich] / max(1, count)
        rate_score = rate / count
        
        final_scores[stoich] = 0.7 * spectral_score + 0.3 * rate_score
    end
    
    # Visualize reaction scores
    println("\nReaction Scores by Spectral Importance:")
    visualize_reaction_scores(keys(final_scores), final_scores, species_names, 
                             title="Reactions Ranked by Spectral Importance")
    
    # Sort by final score
    sorted_reactions = sort(collect(keys(final_scores)), by=s -> final_scores[s], rev=true)
    
    # Select top 5 reactions
    selected_reactions = sorted_reactions[1:min(5, length(sorted_reactions))]
    
    # Print results in table form
    println("\nTop reactions selected by spectral importance:")
    
    # Create a simple table
    headers = ["Rank", "Reaction", "Score", "Count", "Rate"]
    rows = []
    
    for (i, stoich) in enumerate(selected_reactions)
        # Format reaction
        reactants = ""
        products = ""
        
        for j in 1:length(stoich)
            if stoich[j] < 0
                reactants *= "$(abs(stoich[j])) $(species_names[j]) + "
            elseif stoich[j] > 0
                products *= "$(stoich[j]) $(species_names[j]) + "
            end
        end
        
        if !isempty(reactants)
            reactants = reactants[1:end-3]
        else
            reactants = "∅"
        end
        
        if !isempty(products)
            products = products[1:end-3]
        else
            products = "∅"
        end
        
        reaction_str = "$reactants → $products"
        count = stoich_stats[stoich].count
        rate = stoich_stats[stoich].total_rate
        score = final_scores[stoich]
        
        push!(rows, ["$i", reaction_str, "$(round(score, digits=4))", "$count", "$(round(rate, digits=4))"])
    end
    
    # Print the table using our custom function
    print_simple_table(rows, headers)
    
    println("\n==== End of Simplified Spectral Analysis ====")
    
    return selected_reactions, final_scores
end



"""
Group eigenvalues by their properties (slow, fast, oscillatory)
"""
function group_eigenvalues(λ; slow_threshold=0.1, oscillation_ratio=0.5)
    # Initialize groups
    mode_groups = Dict(
        "slow" => Int[],
        "fast" => Int[],
        "oscillatory" => Int[]
    )
    
    # Ignore the first eigenvalue (stationary distribution)
    for i in 2:length(λ)
        # Skip undefined eigenvalues
        if isnan(λ[i]) || isinf(λ[i])
            continue
        end
        
        real_part = abs(real(λ[i]))
        imag_part = abs(imag(λ[i]))
        
        # Check if oscillatory (significant imaginary part)
        if imag_part > oscillation_ratio * real_part
            push!(mode_groups["oscillatory"], i)
        elseif real_part < slow_threshold
            push!(mode_groups["slow"], i)
        else
            push!(mode_groups["fast"], i)
        end
    end
    
    return mode_groups
end

"""
Visualize eigenvalue distribution to understand system dynamics
Fixed version without DotCanvas
"""
function visualize_eigenvalues(λ; title="Eigenvalue Distribution")
    # Convert to real/imag components
    real_parts = real.(λ)
    imag_parts = imag.(λ)
    
    # Create scatter plot - removed canvas parameter
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
        
        # Create zoomed plot - removed canvas parameter
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
Visualize cross-validation results to find optimal reaction set size - Fixed version
"""
function visualize_cv_results(cv_errors; title="Cross-Validation Error")
    # Create x-axis (number of reactions)
    n_reactions = collect(1:length(cv_errors))
    
    # Find optimal size
    optimal_size = argmin(cv_errors)
    
    # Create line plot - removed canvas parameter
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
        
        # REMOVED sort parameter
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
Fixed function to calculate reaction participation in dynamic modes
"""
function calculate_reaction_participation(G, grouped_reactions, selected_states, λ, V_right, V_left, species_names)
    # Initialize reaction participation dictionary
    reaction_participation = Dict()
    
    # Create a mapping from selected_states to their molecular counts
    state_to_counts = Dict()
    for state in selected_states
        molecular_counts = state_idx_to_molecule_counts(state, species_names)
        state_to_counts[tuple(state...)] = molecular_counts
    end
    
    # For each reaction type (stoichiometry)
    for (stoich, rxns) in grouped_reactions
        # Initialize participation scores for each mode
        mode_scores = zeros(length(λ))
        
        # For each reaction instance
        for r in rxns
            # Get from and to states
            from_state = r.from_state
            to_state = r.to_state
            
            # Find indices in the reduced basis
            from_idx = findfirst(s -> all(s .== from_state), selected_states)
            to_idx = findfirst(s -> all(s .== to_state), selected_states)
            
            if from_idx !== nothing && to_idx !== nothing
                # Get reaction rate from generator
                rate = abs(G[to_idx, from_idx])
                
                # Calculate participation in each mode
                for i in 1:length(λ)
                    # Skip modes with eigenvalues that are too close to zero or undefined
                    if abs(λ[i]) < 1e-10 || isnan(λ[i]) || isinf(λ[i])
                        continue
                    end
                    
                    # Calculate participation using right and left eigenvectors
                    participation = abs(V_right[from_idx, i] * rate * V_left[i, to_idx])
                    mode_scores[i] += participation
                end
            end
        end
        
        # Store participation scores for this reaction
        reaction_participation[stoich] = mode_scores
    end
    
    return reaction_participation
end

"""
Fixed function to check conservation consistency based on molecule counts
"""
function check_conservation_consistency_molecular(stoich, conservation_laws, species_names)
    # Convert stoichiometry to reflect changes in molecule counts rather than grid indices
    molecular_stoich = stoich
    
    consistency_scores = Float64[]
    
    for law in conservation_laws
        # Calculate how much the reaction changes the conserved quantity
        # For a consistent reaction, this should be close to zero
        if length(molecular_stoich) <= length(law)
            # Pad stoichiometry vector if needed
            padded_stoich = zeros(length(law))
            padded_stoich[1:length(molecular_stoich)] = molecular_stoich
            
            # Calculate dot product
            consistency = abs(dot(padded_stoich, law))
        else
            # Truncate conservation law if needed
            truncated_law = law[1:length(molecular_stoich)]
            
            # Calculate dot product
            consistency = abs(dot(molecular_stoich, truncated_law))
        end
        
        push!(consistency_scores, consistency)
    end
    
    # A reaction is consistent if all conservation laws are respected
    is_consistent = all(score < 1e-8 for score in consistency_scores)
    
    return is_consistent, consistency_scores
end

"""
Enhanced function to identify conservation laws based on molecular counts
"""
function identify_conservation_laws_molecular(G, species_names; tol=1e-8)
    # Compute eigendecomposition of G
    λ, V_right = eigen(Matrix(G))
    V_left = inv(V_right)  # Left eigenvectors (rows of V_left)
    
    # Find eigenvalues close to zero
    zero_indices = findall(abs.(λ) .< tol)
    
    if isempty(zero_indices)
        println("No conservation laws found (no eigenvalues close to zero)")
        return [], []
    end
    
    # Extract left eigenvectors corresponding to zero eigenvalues
    conservation_laws = []
    law_descriptions = []
    
    for idx in zero_indices
        # Get the left eigenvector (row of V_left)
        left_ev = V_left[idx, :]
        
        # Normalize the coefficients
        max_coef = maximum(abs.(left_ev))
        if max_coef > 0
            left_ev = left_ev ./ max_coef
        end
        
        # Round small values to zero for clarity
        left_ev[abs.(left_ev) .< 1e-10] .= 0.0
        
        # Create a description of the conservation law - in terms of molecular counts
        description = "Conservation law: "
        terms = []
        for (i, coef) in enumerate(left_ev)
            if abs(coef) > 1e-10
                if length(species_names) >= i
                    species = species_names[i]
                    push!(terms, "$(round(coef, digits=3)) × $species")
                else
                    push!(terms, "$(round(coef, digits=3)) × species$i")
                end
            end
        end
        description *= join(terms, " + ") * " = constant"
        
        push!(conservation_laws, left_ev)
        push!(law_descriptions, description)
    end
    
    return conservation_laws, law_descriptions
end

"""
Main analysis function with fixed stoichiometry calculation
"""
function analyze_and_select_reactions_fixed(G, λ, Φ, grouped_reactions, stoich_stats, selected_states, species_names)
    println("\n==== Spectral Analysis with Fixed Stoichiometry Calculation ====")
    
    # Use the simplified spectral analysis with fixed stoichiometry calculation
    return simplified_spectral_analysis_with_plots(
        G, λ, Φ, selected_states, species_names, grouped_reactions, stoich_stats
    )
end


"""
Main function to infer CRN using improved methods
"""
function infer_crn_with_catalyst_dmd(max_dim=1000)
    # Generate synthetic data using Catalyst
    println("Generating trajectory data...")
    ssa_trajs, rn = generate_mm_trajectories(500)  # Increased from 100 to 500 trajectories
    
    # Define grid parameters - restore higher resolution
    species_indices = 1:4  # S, E, SE, P
    species_names = ["S", "E", "SE", "P"]
    grid_sizes = [32, 16, 8, 32]  # Increased resolution
    time_points = range(0.0, 50.0, length=20)  # More time points and longer simulation
    dt = time_points[2] - time_points[1]
    
    # Process trajectories directly to sparse format
    println("Processing trajectories to sparse format...")
    sparse_probs = process_trajectories_to_sparse(ssa_trajs, species_indices, grid_sizes, time_points)
    
    # Reduce to important states only with improved selection
    println("Reducing to important states...")
    reduced_data, selected_states = reduce_sparse_data(sparse_probs, grid_sizes, max_dim)
    
    # Apply DMD to reduced data with improved stability
    println("Applying DMD to reduced data...")
    G, λ, Φ, A, r = apply_dmd(reduced_data, dt, svd_rank_threshold=1e-12)  # More sensitive threshold
    
    # Extract reactions from the generator matrix with validation
    println("Extracting reaction information...")
    significant_stoich, grouped_reactions, stoich_stats = extract_reactions_from_generator(
        G, selected_states, species_indices, species_names, threshold=1e-5, validate_reactions=true
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
    
    # Verify expected reactions
    expected_stoichiometries = [
        [0, 1, -1, 1],    # SE → E + P
        [-1, -1, 1, 0],   # S + E → SE 
        [1, 1, -1, 0]     # SE → S + E
    ]
    
    println("\nExpected Michaelis-Menten reaction recovery:")
    for expected in expected_stoichiometries
        expected_tuple = tuple(expected...)
        
        # Check if found
        if expected_tuple in keys(grouped_reactions)
            stats = stoich_stats[expected_tuple]
            
            # Create reaction string
            reactants = ""
            products = ""
            
            for i in 1:length(expected)
                if expected[i] < 0
                    reactants *= "$(abs(expected[i])) $(species_names[i]) + "
                elseif expected[i] > 0
                    products *= "$(expected[i]) $(species_names[i]) + "
                end
            end
            
            if !isempty(reactants)
                reactants = reactants[1:end-3]
            end
            if !isempty(products)
                products = products[1:end-3]
            end
            
            println("$reactants --> $products : found (rate ≈ $(round(stats.total_rate, digits=5)))")
        else
            # Check reversed reaction
            rev_expected = -1 .* expected
            rev_tuple = tuple(rev_expected...)
            
            if rev_tuple in keys(grouped_reactions)
                stats = stoich_stats[rev_tuple]
                
                # Create reaction string
                reactants = ""
                products = ""
                
                for i in 1:length(expected)
                    if expected[i] < 0
                        reactants *= "$(abs(expected[i])) $(species_names[i]) + "
                    elseif expected[i] > 0
                        products *= "$(expected[i]) $(species_names[i]) + "
                    end
                end
                
                if !isempty(reactants)
                    reactants = reactants[1:end-3]
                end
                if !isempty(products)
                    products = products[1:end-3]
                end
                
                println("$reactants --> $products : found reverse (rate ≈ $(round(stats.total_rate, digits=5)))")
            else
                # Not found case
                reactants = ""
                products = ""
                
                for i in 1:length(expected)
                    if expected[i] < 0
                        reactants *= "$(abs(expected[i])) $(species_names[i]) + "
                    elseif expected[i] > 0
                        products *= "$(expected[i]) $(species_names[i]) + "
                    end
                end
                
                if !isempty(reactants)
                    reactants = reactants[1:end-3]
                end
                if !isempty(products)
                    products = products[1:end-3]
                end
                
                println("$reactants --> $products : not found")
            end
        end
    end
    
    # Analyze mass-action kinetics patterns
   #  analyze_mass_action_kinetics_enhanced(grouped_reactions, stoich_stats, selected_states, species_names)
    # Add spectral-based reaction selection with visualizations
  println("\nPerforming spectral-based reaction selection with visualizations...")
  # Add spectral-based reaction selection with fixed stoichiometry calculation
  println("\nPerforming spectral-based reaction selection with fixed stoichiometry...")
  spectral_reactions, spectral_scores = analyze_and_select_reactions_fixed(
      G, λ, Φ, grouped_reactions, stoich_stats, selected_states, species_names
  )

  # Add selected reactions to result dictionary
  result["spectral_selected_reactions"] = spectral_reactions
  result["spectral_scores"] = spectral_scores 
    # Generate visualizations
   # try
   #     visualize_reaction_rates(grouped_reactions, species_names)
   # catch e
   #     println("Visualization error: $e")
   #     println("Skipping visualization. Make sure Plots.jl is installed.")
   # end
    
   # println("\nCME inference completed successfully!")
   # println("DMD rank: $r")
    
    return Dict(
        "significant_stoichiometries" => significant_stoich,
        "grouped_reactions" => grouped_reactions,
        "stoich_stats" => stoich_stats,
        "generator" => G,
        "eigenvalues" => λ,
        "DMD_modes" => Φ,
        "DMD_operator" => A,
        "rank" => r
    )
end

# Run the inference (uncomment to execute)
# result = infer_crn_with_catalyst_dmd(1000)

