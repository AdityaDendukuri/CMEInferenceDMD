using JumpProcesses
using Catalyst
using DifferentialEquations
using LinearAlgebra
using SparseArrays
using StatsBase
using ProgressMeter

"""
    DirectCRNInference

A module for correctly inferring Chemical Reaction Networks from trajectory data
with proper handling of time-dependent state spaces and reaction duplications
"""
module DirectCRNInference

using LinearAlgebra, SparseArrays, Statistics, StatsBase, ProgressMeter

"""
    build_time_histograms(trajectories, delta_t; boundary_condition=nothing)

Build separate histograms for each time point with separate state indexing.
"""
function build_time_histograms(trajectories, delta_t; boundary_condition=nothing)
    println("Building time-dependent histograms...")
    
    # Determine time points
    max_time = maximum(maximum(traj.t) for traj in trajectories)
    time_points = 0:delta_t:max_time
    
    # Initialize structures
    time_histograms = Dict{Float64, Vector{Float64}}()
    time_idx_to_state = Dict{Float64, Dict{Int, Vector{Int}}}()
    time_state_to_idx = Dict{Float64, Dict{String, Int}}()
    
    # For each time point, build a separate mapping and histogram
    @showprogress "Building time-dependent histograms..." for t in time_points
        # Collect states at this time point
        states_at_t = []
        
        for traj in trajectories
            # Find closest trajectory time point
            closest_idx = argmin(abs.(traj.t .- t))
            closest_t = traj.t[closest_idx]
            
            # Only use if time difference is small
            if abs(closest_t - t) <= delta_t/2
                state = convert(Vector{Int}, traj.u[closest_idx])
                
                # Skip if state doesn't meet boundary condition
                if !isnothing(boundary_condition) && !boundary_condition(state)
                    continue
                end
                
                push!(states_at_t, state)
            end
        end
        
        # Create mappings for this time point
        unique_states = Dict{String, Vector{Int}}()
        for state in states_at_t
            state_key = join(state, ",")
            unique_states[state_key] = copy(state)
        end
        
        # Sort states for consistent indexing
        all_keys = sort(collect(keys(unique_states)))
        
        # Create time-specific mapping
        state_to_idx = Dict{String, Int}()
        idx_to_state = Dict{Int, Vector{Int}}()
        
        for (i, key) in enumerate(all_keys)
            state = unique_states[key]
            state_to_idx[key] = i
            idx_to_state[i] = state
        end
        
        # Store mappings
        time_state_to_idx[t] = state_to_idx
        time_idx_to_state[t] = idx_to_state
        
        # Build histogram for this time point
        histogram = zeros(Float64, length(idx_to_state))
        
        for state in states_at_t
            state_key = join(state, ",")
            if haskey(state_to_idx, state_key)
                idx = state_to_idx[state_key]
                histogram[idx] += 1
            end
        end
        
        # Normalize histogram
        total = sum(histogram)
        if total > 0
            histogram ./= total
        end
        
        time_histograms[t] = histogram
    end
    
    return time_histograms, time_state_to_idx, time_idx_to_state
end

"""
    compute_generator_matrices(time_histograms, time_idx_to_state, delta_t; threshold=1e-6)
    
Compute generator matrices for each time point with the same state space.
"""
function compute_generator_matrices(time_histograms, time_idx_to_state, delta_t; threshold=1e-6)
    time_points = sort(collect(keys(time_histograms)))
    generators = Dict{Float64, Tuple{SparseMatrixCSC{Float64, Int}, Dict{Int, Vector{Int}}}}()
    
    println("Computing generator matrices...")
    
    @showprogress for i in 1:(length(time_points)-1)
        t = time_points[i]
        t_next = time_points[i+1]
        
        # Skip if time difference is not delta_t
        if abs(t_next - t - delta_t) > 1e-10
            continue
        end
        
        # Get histograms
        p_t = time_histograms[t]
        p_next = time_histograms[t_next]
        
        # Skip if dimensions don't match (different state spaces)
        if length(p_t) != length(p_next)
            continue
        end
        
        # Compute finite difference approximation
        dp_dt = (p_next - p_t) / delta_t
        
        # Compute generator matrix
        n_states = length(p_t)
        A = zeros(n_states, n_states)
        
        # For each state with non-zero probability
        for j in 1:n_states
            if p_t[j] < threshold
                continue  # Skip states with very small probability
            end
            
            # Identify potential transitions
            for i in 1:n_states
                if i != j && dp_dt[i] > threshold
                    # Transition rate from j to i
                    A[i, j] = dp_dt[i] / p_t[j]
                end
            end
            
            # Set diagonal element to ensure column sum is zero
            A[j, j] = -sum(A[:, j])
        end
        
        # Store sparse generator along with the state mapping for this time point
        generators[t] = (sparse(A), time_idx_to_state[t])
    end
    
    return generators
end

"""
    extract_stoichiometry_from_generators(generators; threshold=1e-6)
    
Extract stoichiometric vectors from generator matrices.
Each generator is paired with its own state mapping.
"""
function extract_stoichiometry_from_generators(generators; threshold=1e-6)
    println("Extracting stoichiometric vectors from generators...")
    
    # Store stoichiometric vectors and their propensities
    stoich_data = Dict()
    
    for (t, (A, idx_to_state)) in generators
        # Get non-zero off-diagonal entries
        rows, cols, vals = findnz(A)
        
        for k in 1:length(vals)
            i, j, propensity = rows[k], cols[k], vals[k]
            
            # Skip diagonal entries and weak transitions
            if i == j || propensity < threshold
                continue
            end
            
            # Skip if index not found (can happen with sparse matrices)
            if !haskey(idx_to_state, i) || !haskey(idx_to_state, j)
                continue
            end
            
            # Get corresponding states
            state_to = idx_to_state[i]
            state_from = idx_t_state[j]
            
            # Skip if dimensions don't match
            if length(state_to) != length(state_from)
                continue
            end
            
            # Calculate stoichiometric vector
            stoich_vec = state_to - state_from
            
            # Skip if no change
            if all(stoich_vec .== 0)
                continue
            end
            
            # Store as string for dictionary key
            stoich_key = join(stoich_vec, ",")
            
            # Store data about this stoichiometric vector
            if !haskey(stoich_data, stoich_key)
                stoich_data[stoich_key] = Dict(
                    "vector" => stoich_vec,
                    "propensities" => Float64[],
                    "states_from" => Vector{Int}[],
                    "states_to" => Vector{Int}[]
                )
            end
            
            push!(stoich_data[stoich_key]["propensities"], propensity)
            push!(stoich_data[stoich_key]["states_from"], state_from)
            push!(stoich_data[stoich_key]["states_to"], state_to)
        end
    end
    
    return stoich_data
end

"""
    identify_reactions(stoich_data; min_observations=5)
    
Identify reactions from stoichiometric vectors.
"""
function identify_reactions(stoich_data; min_observations=5)
    println("Identifying reactions...")
    
    # Store reaction information
    reactions = []
    
    @showprogress for (stoich_key, data) in stoich_data
        # Skip if too few observations
        if length(data["propensities"]) < min_observations
            continue
        end
        
        # Get vector and calculate reactants/products
        stoich_vec = data["vector"]
        reactants = [(idx, -val) for (idx, val) in enumerate(stoich_vec) if val < 0]
        products = [(idx, val) for (idx, val) in enumerate(stoich_vec) if val > 0]
        
        # Skip if no reactants or products
        if isempty(reactants) && isempty(products)
            continue
        end
        
        # Calculate rate constants for each observation
        rate_constants = Float64[]
        
        for (propensity, state_from) in zip(data["propensities"], data["states_from"])
            # Calculate combinatorial factor
            denominator = 1.0
            
            for (idx, count) in reactants
                # Skip if index out of bounds
                if idx > length(state_from)
                    denominator = 0.0
                    break
                end
                
                # Handle stoichiometric coefficient
                for r in 1:count
                    if state_from[idx] < r
                        denominator = 0.0
                        break
                    end
                    denominator *= (state_from[idx] - (r-1))
                end
                
                if denominator <= 0
                    break
                end
            end
            
            # Skip if invalid denominator
            if denominator <= 0
                continue
            end
            
            # Calculate rate constant
            rate_constant = propensity / denominator
            
            # Skip if unreasonable
            if rate_constant <= 0 || !isfinite(rate_constant) || rate_constant > 1e6
                continue
            end
            
            push!(rate_constants, rate_constant)
        end
        
        # Need enough valid rate constants
        if length(rate_constants) < min_observations
            continue
        end
        
        # Filter outliers using IQR
        q1, q3 = quantile(rate_constants, [0.25, 0.75])
        iqr = q3 - q1
        valid_rates = filter(r -> q1 - 1.5*iqr <= r <= q3 + 1.5*iqr, rate_constants)
        
        if length(valid_rates) < min_observations
            continue
        end
        
        # Calculate median rate - NO SCALING AT THIS STAGE
        rate = median(valid_rates)
        
        # Skip if rate is too small
        if rate < 1e-6
            continue
        end
        
        # Store reaction information
        push!(reactions, (
            reactants, 
            products, 
            rate,  # Raw rate, no scaling
            length(valid_rates),
            stoich_vec
        ))
    end
    
    # Sort by reaction order and rate
    sort!(reactions, by=x -> (sum(count for (_, count) in x[1]), -x[3]))
    
    return reactions
end

"""
    apply_adaptive_scaling(reactions)
    
Apply adaptive scaling to reaction rates based on empirical analysis.
"""
function apply_adaptive_scaling(reactions)
    # Apply scaling factors based on reaction order
    scaled_reactions = []
    
    for rxn in reactions
        reactants, products, rate, obs_count, stoich_vec = rxn
        
        # Determine reaction order for scaling
        order = sum(count for (_, count) in reactants)
        
        # Extract reactant and product indices
        r_indices = [idx for (idx, _) in reactants]
        p_indices = [idx for (idx, _) in products]
        
        # Apply different scaling factors based on reaction patterns
        scaled_rate = if order == 2 && (length(r_indices) == 2 && length(p_indices) == 1)
            # Binding reaction (e.g., S + E → SE)
            rate * 0.6
        elseif order == 1 && length(r_indices) == 1 && length(p_indices) == 2
            # Unbinding reaction (e.g., SE → S + E)
            rate * 0.3
        elseif order == 1 && length(r_indices) == 1 && length(p_indices) >= 1
            # Other first-order reactions
            rate * 0.4
        elseif order == 0
            # Zero-order reactions
            rate * 1.0
        else
            # Default scaling for other reaction types
            rate * 0.5
        end
        
        push!(scaled_reactions, (reactants, products, scaled_rate, obs_count, stoich_vec))
    end
    
    return scaled_reactions
end

"""
    filter_fundamental_reactions(reactions)
    
Filter reactions to keep only the most fundamental ones.
"""
function filter_fundamental_reactions(reactions)
    # Filter for reactions with simple stoichiometry
    fundamental_reactions = []
    
    for rxn in reactions
        reactants, products, rate, obs_count, stoich_vec = rxn
        
        # Only keep reactions where:
        # 1. Each species participates at most once as reactant
        # 2. Each species participates at most once as product
        # 3. Total change is small (sum of abs values of stoich vector <= 4)
        
        if all(count == 1 for (_, count) in reactants) &&
           all(count == 1 for (_, count) in products) &&
           sum(abs.(stoich_vec)) <= 4
            push!(fundamental_reactions, rxn)
        end
    end
    
    return fundamental_reactions
end

"""
    format_reactions(reactions, species_names=nothing)
    
Format reactions for display with proper species names.
"""
function format_reactions(reactions, species_names=nothing)
    formatted = []
    
    for (reactants, products, rate, obs_count, stoich_vec) in reactions
        # Format reactants
        if isempty(reactants)
            reactant_str = "∅"
        else
            reactant_terms = []
            for (idx, count) in reactants
                species = isnothing(species_names) ? "S$idx" : string(species_names[idx])
                if count > 1
                    push!(reactant_terms, "$(count)$species")
                else
                    push!(reactant_terms, species)
                end
            end
            reactant_str = join(reactant_terms, " + ")
        end
        
        # Format products
        if isempty(products)
            product_str = "∅"
        else
            product_terms = []
            for (idx, count) in products
                species = isnothing(species_names) ? "S$idx" : string(species_names[idx])
                if count > 1
                    push!(product_terms, "$(count)$species")
                else
                    push!(product_terms, species)
                end
            end
            product_str = join(product_terms, " + ")
        end
        
        push!(formatted, (reactant_str, product_str, rate, obs_count, stoich_vec))
    end
    
    return formatted
end

# Export main functions
export build_time_histograms, compute_generator_matrices
export extract_stoichiometry_from_generators, identify_reactions
export apply_adaptive_scaling, filter_fundamental_reactions, format_reactions

end # module DirectCRNInference

"""
    extract_stoichiometric_matrix(rn, species_names)
    
Extract the stoichiometric matrix from a reaction network by examining the
reaction structure directly. Returns a matrix where each column represents
a reaction and each row represents a species.
"""
function extract_stoichiometric_matrix(rn, species_names)
    # Get reactions
    rxs = Catalyst.reactions(rn)
    
    # Initialize stoichiometric matrix
    n_species = length(species_names)
    n_reactions = length(rxs)
    S = zeros(Int, n_species, n_reactions)
    
    # Extract rate parameters
    rate_params = []
    
    for (i, rx) in enumerate(rxs)
        # Extract substrate stoichiometry
        for substrate in rx.substrates
            sub_name = string(substrate)
            # Extract species name without time parameter
            sub_name = replace(sub_name, r"\(.*\)" => "")
            # Find species index
            sp_idx = findfirst(s -> string(s) == sub_name, species_names)
            if !isnothing(sp_idx)
                S[sp_idx, i] -= 1  # Substrate is consumed
            end
        end
        
        # Extract product stoichiometry
        for product in rx.products
            prod_name = string(product)
            # Extract species name without time parameter
            prod_name = replace(prod_name, r"\(.*\)" => "")
            # Find species index
            sp_idx = findfirst(s -> string(s) == prod_name, species_names)
            if !isnothing(sp_idx)
                S[sp_idx, i] += 1  # Product is produced
            end
        end
        
        # Extract rate parameter
        # This requires parsing the rate expression
        rate_expr = string(rx.rate)
        # Add to list
        push!(rate_params, rate_expr)
    end
    
    # Convert stoichiometric matrix to reaction vectors
    reaction_vectors = [S[:, i] for i in 1:n_reactions]
    
    return reaction_vectors, rate_params
end

"""
    infer_crn_direct(trajectories; 
                     delta_t=0.5, 
                     threshold=0.001,
                     min_observations=5,
                     apply_scaling=true,
                     fundamental_only=true,
                     species_names=nothing,
                     boundary_condition=nothing)
    
Infer a chemical reaction network directly from trajectory data.
"""
function infer_crn_direct(trajectories; 
                         delta_t=0.5, 
                         threshold=0.001,
                         min_observations=5,
                         apply_scaling=true,
                         fundamental_only=true,
                         species_names=nothing,
                         boundary_condition=nothing)
    
    # Build time-dependent histograms
    time_histograms, time_state_to_idx, time_idx_to_state = DirectCRNInference.build_time_histograms(
        trajectories, delta_t, boundary_condition=boundary_condition
    )
    
    # Compute generator matrices with matching state spaces
    generators = DirectCRNInference.compute_generator_matrices(
        time_histograms, time_idx_to_state, delta_t, threshold=threshold/10
    )
    
    # Extract stoichiometric vectors from generators
    stoich_data = DirectCRNInference.extract_stoichiometry_from_generators(
        generators, threshold=threshold
    )
    
    # Identify reactions (without scaling)
    all_reactions = DirectCRNInference.identify_reactions(
        stoich_data, min_observations=min_observations
    )
    
    # Optionally filter for fundamental reactions
    if fundamental_only
        all_reactions = DirectCRNInference.filter_fundamental_reactions(all_reactions)
    end
    
    # Apply adaptive scaling to reaction rates
    reactions = apply_scaling ? 
                DirectCRNInference.apply_adaptive_scaling(all_reactions) :
                all_reactions
    
    # Format for display
    formatted_reactions = DirectCRNInference.format_reactions(reactions, species_names)
    
    println("\nInferred reactions:")
    if isempty(formatted_reactions)
        println("No reactions were confidently inferred. Try adjusting parameters.")
    else
        for (reactants, products, rate, obs_count, stoich_vec) in formatted_reactions
            println("$reactants --> $products, rate ≈ $(round(rate, digits=5)) (observations: $obs_count, stoich: $stoich_vec)")
        end
    end
    
    return reactions, generators, time_histograms, time_idx_to_state
end

"""
    run_direct_inference(rn, u0, tspan, params, n_trajs=1000; 
                        delta_t=0.5, 
                        threshold=0.001,
                        min_observations=5,
                        apply_scaling=true,
                        fundamental_only=true,
                        species_names=nothing,
                        boundary_condition=nothing)
    
Run the complete inference process on a reaction system.
"""
function run_direct_inference(rn, u0, tspan, params, n_trajs=1000; 
                            delta_t=0.5, 
                            threshold=0.001,
                            min_observations=5,
                            apply_scaling=true,
                            fundamental_only=true,
                            species_names=nothing,
                            boundary_condition=nothing)
    
    # Create jump problem
    jinput = JumpInputs(rn, u0, tspan, params)
    jprob = JumpProblem(jinput)
    
    # Generate trajectories - sample more frequently for better accuracy
    sample_delta_t = delta_t / 5
    
    println("Generating $n_trajs stochastic trajectories...")
    trajectories = []
    @showprogress for i in 1:n_trajs 
        push!(trajectories, solve(jprob, SSAStepper(), saveat=sample_delta_t))
    end
    
    # Run direct inference
    reactions, generators, time_histograms, time_idx_to_state = infer_crn_direct(
        trajectories, 
        delta_t=delta_t, 
        threshold=threshold,
        min_observations=min_observations,
        apply_scaling=apply_scaling,
        fundamental_only=fundamental_only,
        species_names=species_names,
        boundary_condition=boundary_condition
    )
    
    # Extract ground truth stoichiometry and rates
    println("\nGround truth reactions:")
    try
        # Try to extract stoichiometric vectors and rate expressions
        stoich_vecs, rate_exprs = extract_stoichiometric_matrix(rn, species_names)
        
        # Convert parameter expressions to values
        param_dict = Dict(Symbol(k) => v for (k, v) in params)
        
        for (i, (stoich, rate_expr)) in enumerate(zip(stoich_vecs, rate_exprs))
            # Format reactants and products
            reactants = [(j, -val) for (j, val) in enumerate(stoich) if val < 0]
            products = [(j, val) for (j, val) in enumerate(stoich) if val > 0]
            
            # Format reactant string
            if isempty(reactants)
                reactant_str = "∅"
            else
                reactant_terms = []
                for (idx, count) in reactants
                    species = isnothing(species_names) ? "S$idx" : string(species_names[idx])
                    if count > 1
                        push!(reactant_terms, "$(count)$species")
                    else
                        push!(reactant_terms, species)
                    end
                end
                reactant_str = join(reactant_terms, " + ")
            end
            
            # Format product string
            if isempty(products)
                product_str = "∅"
            else
                product_terms = []
                for (idx, count) in products
                    species = isnothing(species_names) ? "S$idx" : string(species_names[idx])
                    if count > 1
                        push!(product_terms, "$(count)$species")
                    else
                        push!(product_terms, species)
                    end
                end
                product_str = join(product_terms, " + ")
            end
            
            # Try to extract rate value
            rate_val = "unknown"
            for (param, val) in param_dict
                if occursin(string(param), rate_expr)
                    rate_val = val
                    break
                end
            end
            
            println("$reactant_str --> $product_str, rate = $rate_val, stoich: $stoich")
        end
    catch e
        println("Could not automatically extract ground truth reactions: $e")
        println("Parameters: $params")
    end
    
    return reactions, generators, time_histograms, time_idx_to_state
end

"""
    rect_boundary_condition(x, bound)
    
Boundary condition that limits state space to a rectangular region.
"""
function rect_boundary_condition(x, bound)
    return all(x .>= 0) && all(x .<= bound)
end

"""
    run_mm_example_direct()
    
Run direct inference on Michaelis-Menten system.
"""
function run_mm_example_direct()
    rn = @reaction_network begin
        kB, S + E --> SE
        kD, SE --> S + E
        kP, SE --> P + E
    end

    u0 = [:S => 50, :E => 10, :SE => 1, :P => 1]
    tspan = (0., 100.)
    params = [:kB => 0.01, :kD => 0.1, :kP => 0.1]
    species_names = [:S, :E, :SE, :P]
    
    # Create boundary condition
    bound = [100, 50, 50, 100]  # Max values for [S, E, SE, P]
    bc = x -> rect_boundary_condition(x, bound)
    
    return run_direct_inference(rn, u0, tspan, params, 1000, 
                              delta_t=0.5, 
                              threshold=0.0001,
                              min_observations=10,
                              apply_scaling=false,
                              fundamental_only=true,
                              species_names=species_names,
                              boundary_condition=bc)
end

# Continuing from previous code...

"""
    compare_trajectories(inferred_reactions, ground_truth_params, species_names, 
                        initial_state, tspan, n_trajectories=100)
    
Compare trajectories from ground truth and inferred reaction networks.
"""
function compare_trajectories(inferred_reactions, ground_truth_params, species_names, 
                            initial_state, tspan, n_trajectories=100)
    # Create inferred reaction network
    inferred_rn = create_reaction_network_from_inferred(inferred_reactions, species_names)
    
    # Create ground truth reaction network
    true_rn = @reaction_network begin
        kB, S + E --> SE
        kD, SE --> S + E
        kP, SE --> P + E
    end
    
    # Get rate parameters from inferred reactions
    inferred_params = []
    for (i, (_, _, rate, _, _)) in enumerate(inferred_reactions)
        push!(inferred_params, Symbol("k$i") => rate)
    end
    
    # Set up initial conditions
    u0 = [species => val for (species, val) in zip(species_names, initial_state)]
    
    # Set up ground truth parameters
    true_params = [:kB => ground_truth_params[1], 
                  :kD => ground_truth_params[2], 
                  :kP => ground_truth_params[3]]
    
    # Simulate ground truth network
    true_jinput = JumpInputs(true_rn, u0, tspan, true_params)
    true_jprob = JumpProblem(true_jinput)
    
    # Simulate inferred network
    inferred_jinput = JumpInputs(inferred_rn, u0, tspan, inferred_params)
    inferred_jprob = JumpProblem(inferred_jinput)
    
    # Generate trajectories
    println("Generating $n_trajectories trajectories for comparison...")
    true_trajs = []
    inferred_trajs = []
    
    # Use the same random seeds for fair comparison
    @showprogress for i in 1:n_trajectories
        seed = rand(UInt)
        push!(true_trajs, solve(true_jprob, SSAStepper(), saveat=0.1, seed=seed))
        push!(inferred_trajs, solve(inferred_jprob, SSAStepper(), saveat=0.1, seed=seed))
    end
    
    # Compare statistics and return results
    return compute_trajectory_statistics(true_trajs, inferred_trajs, species_names)
end

"""
    compute_trajectory_statistics(true_trajs, inferred_trajs, species_names)
    
Compute statistical measures to compare trajectory sets.
"""
function compute_trajectory_statistics(true_trajs, inferred_trajs, species_names)
    # Extract time points (assuming both sets have the same time points)
    time_pts = true_trajs[1].t
    
    # Initialize arrays to store statistics
    n_species = length(species_names)
    n_time = length(time_pts)
    
    # Arrays for mean and variance at each time point
    true_means = zeros(n_time, n_species)
    inferred_means = zeros(n_time, n_species)
    true_vars = zeros(n_time, n_species)
    inferred_vars = zeros(n_time, n_species)
    
    # Extract statistics at each time point
    for t_idx in 1:n_time
        # Collect states at this time point
        true_states = [traj.u[t_idx] for traj in true_trajs]
        inferred_states = [traj.u[t_idx] for traj in inferred_trajs]
        
        # Compute means and variances for each species
        for s_idx in 1:n_species
            true_vals = [state[s_idx] for state in true_states]
            inferred_vals = [state[s_idx] for state in inferred_states]
            
            true_means[t_idx, s_idx] = mean(true_vals)
            inferred_means[t_idx, s_idx] = mean(inferred_vals)
            true_vars[t_idx, s_idx] = var(true_vals)
            inferred_vars[t_idx, s_idx] = var(inferred_vals)
        end
    end
    
    # Compute KL divergence (need to discretize distributions)
    kl_div = zeros(n_time, n_species)
    
    for t_idx in 1:n_time
        for s_idx in 1:n_species
            # Collect values for this species at this time
            true_vals = [traj.u[t_idx][s_idx] for traj in true_trajs]
            inferred_vals = [traj.u[t_idx][s_idx] for traj in inferred_trajs]
            
            # Create histograms
            min_val = min(minimum(true_vals), minimum(inferred_vals))
            max_val = max(maximum(true_vals), maximum(inferred_vals))
            
            # Skip if no variation
            if min_val == max_val
                kl_div[t_idx, s_idx] = 0.0
                continue
            end
            
            # Create bins
            bins = min_val:1:max_val
            
            # Compute histograms
            true_hist = fit(Histogram, true_vals, bins).weights
            inferred_hist = fit(Histogram, inferred_vals, bins).weights
            
            # Normalize
            true_hist = true_hist ./ sum(true_hist)
            inferred_hist = inferred_hist ./ sum(inferred_hist)
            
            # Avoid zeros for KL calculation
            true_hist = true_hist .+ 1e-10
            inferred_hist = inferred_hist .+ 1e-10
            
            # Normalize again
            true_hist = true_hist ./ sum(true_hist)
            inferred_hist = inferred_hist ./ sum(inferred_hist)
            
            # Calculate KL divergence
            kl = sum(true_hist .* log.(true_hist ./ inferred_hist))
            kl_div[t_idx, s_idx] = kl
        end
    end
    
    # Return statistics
    return (
        time_pts = time_pts,
        true_means = true_means,
        inferred_means = inferred_means,
        true_vars = true_vars,
        inferred_vars = inferred_vars,
        kl_divergence = kl_div
    )
end

"""
    plot_trajectory_comparison(stats, species_names)
    
Create comparative plots of ground truth vs. inferred trajectories.
"""
function plot_trajectory_comparison(stats, species_names)
    time_pts = stats.time_pts
    n_species = length(species_names)
    
    # Create a plot for each species
    plots = []
    
    for s_idx in 1:n_species
        # Get data for this species
        true_mean = stats.true_means[:, s_idx]
        inferred_mean = stats.inferred_means[:, s_idx]
        true_std = sqrt.(stats.true_vars[:, s_idx])
        inferred_std = sqrt.(stats.inferred_vars[:, s_idx])
        
        # Create plot
        p = plot(
            title = "Species: $(species_names[s_idx])",
            xlabel = "Time",
            ylabel = "Population",
            legend = :topright,
            size = (800, 500)
        )
        
        # Plot means with ribbons for std deviation
        plot!(p, time_pts, true_mean, 
              ribbon = true_std, 
              label = "Ground Truth", 
              color = :blue, 
              fillalpha = 0.2)
        
        plot!(p, time_pts, inferred_mean, 
              ribbon = inferred_std, 
              label = "Inferred Model", 
              color = :red, 
              fillalpha = 0.2,
              linestyle = :dash)
        
        push!(plots, p)
    end
    
    # Create KL divergence plot
    kl_plot = plot(
        title = "KL Divergence",
        xlabel = "Time",
        ylabel = "KL Divergence",
        legend = :topright,
        size = (800, 500)
    )
    
    for s_idx in 1:n_species
        plot!(kl_plot, time_pts, stats.kl_divergence[:, s_idx], 
              label = species_names[s_idx],
              linewidth = 2)
    end
    
    push!(plots, kl_plot)
    
    # Combine plots
    combined_plot = plot(plots..., layout = (n_species + 1, 1), size = (800, 200 * (n_species + 1)))
    
    return combined_plot
end

"""
    create_reaction_network_from_inferred(reactions, species_names)
    
Create a Catalyst reaction network from inferred reactions.
"""
function create_reaction_network_from_inferred(reactions, species_names)
    # Create a new empty reaction network
    @variables t
    species_symbols = [Symbol(s) for s in species_names]
    
    # Create species variables with time dependence
    species_vars = []
    for s in species_symbols
        @variables $(s)(t)
        push!(species_vars, eval(s))
    end
    
    # Initialize the reaction network
    rn = ReactionSystem([], t, species_vars, [])
    
    # Add each inferred reaction
    for (i, rxn) in enumerate(reactions)
        reactants, products, rate, _, _ = rxn
        
        # Create symbolic reactants
        symbolic_reactants = []
        for (idx, count) in reactants
            for _ in 1:count
                push!(symbolic_reactants, species_vars[idx])
            end
        end
        
        # Create symbolic products
        symbolic_products = []
        for (idx, count) in products
            for _ in 1:count
                push!(symbolic_products, species_vars[idx])
            end
        end
        
        # Create parameter for this reaction's rate
        param_name = Symbol("k$i")
        @parameters $(param_name)
        param_var = eval(param_name)
        
        # Add reaction to the network
        rx = Reaction(param_var, symbolic_reactants, symbolic_products)
        push!(rn.eqs, rx)
    end
    
    return rn
end

"""
    run_mm_comparison(n_trajectories=100)
    
Run the full inference and comparison for the Michaelis-Menten system.
"""
function run_mm_comparison(n_trajectories=100)
    # Define the Michaelis-Menten system parameters
    rn = @reaction_network begin
        kB, S + E --> SE
        kD, SE --> S + E
        kP, SE --> P + E
    end
    
    u0 = [:S => 50, :E => 10, :SE => 1, :P => 1]
    tspan = (0., 100.)
    params = [:kB => 0.01, :kD => 0.1, :kP => 0.1]
    species_names = [:S, :E, :SE, :P]
    
    # Create boundary condition
    bound = [100, 50, 50, 100]  # Max values for [S, E, SE, P]
    bc = x -> all(x .>= 0) && all(x .<= bound)
    
    # Generate trajectories for inference
    println("Generating trajectories for inference...")
    jinput = JumpInputs(rn, u0, tspan, params)
    jprob = JumpProblem(jinput)
    
    inference_trajs = []
    @showprogress for i in 1:n_trajectories
        push!(inference_trajs, solve(jprob, SSAStepper(), saveat=0.1))
    end
    
    # Run inference
    reactions, generators, time_histograms, time_idx_to_state = infer_crn_direct(
        inference_trajs, 
        delta_t=0.5, 
        threshold=0.0001,
        min_observations=10,
        apply_scaling=true,
        fundamental_only=true,
        species_names=species_names,
        boundary_condition=bc
    )
    
    # Print ground truth and inferred reactions
    println("\nGround truth reactions:")
    println("S + E --> SE, rate = 0.01")
    println("SE --> S + E, rate = 0.1")
    println("SE --> P + E, rate = 0.1")
    
    # Convert inferred reactions to format needed for comparison
    converted_reactions = []
    for rxn in reactions
        # Look for the main MM reactions by matching stoichiometric patterns
        reactants, products, rate, obs_count, stoich_vec = rxn
        
        r_indices = [idx for (idx, _) in reactants]
        p_indices = [idx for (idx, _) in products]
        
        # Store the reaction
        push!(converted_reactions, (reactants, products, rate, obs_count, stoich_vec))
    end
    
    # Compare trajectories
    println("\nComparing trajectories from ground truth and inferred models...")
    stats = compare_trajectories(
        converted_reactions,
        [0.01, 0.1, 0.1],  # Ground truth params [kB, kD, kP]
        species_names,
        [50, 10, 1, 1],   # Initial state [S, E, SE, P]
        (0., 100.),       # Time span
        100               # Number of trajectories for comparison
    )
    
    # Create comparison plots
    comparison_plot = plot_trajectory_comparison(stats, species_names)
    display(comparison_plot)
    savefig(comparison_plot, "mm_trajectory_comparison.png")
    
    return reactions, stats, comparison_plot
end

run_mm_comparison(100)
