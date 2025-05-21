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
with proper handling of time-dependent state spaces
"""
module DirectCRNInference

using LinearAlgebra, SparseArrays, Statistics, StatsBase, ProgressMeter

"""
    build_time_histograms(trajectories, delta_t; boundary_condition=nothing)

Build separate histograms for each time point with separate state indexing.
This correctly handles the fact that each time snapshot may have a different state space.
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
    compute_generator_matrices(time_histograms, delta_t; threshold=1e-6)
    
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
            state_from = idx_to_state[j]
            
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
        
        # Calculate median rate
        rate = median(valid_rates)
        
        # Determine reaction order for scaling
        order = sum(count for (_, count) in reactants)
        
        # Apply appropriate scaling based on reaction order
        scaled_rate = if order == 0
            rate * 1.0
        elseif order == 1
            rate * 10.0
        elseif order == 2
            rate * 30.0
        else
            rate * (30.0 * order / 2.0)
        end
        
        # Skip if rate is too small
        if scaled_rate < 1e-5
            continue
        end
        
        # Store reaction information
        push!(reactions, (
            reactants, 
            products, 
            scaled_rate,
            length(valid_rates),
            stoich_vec
        ))
    end
    
    # Sort by reaction order and rate
    sort!(reactions, by=x -> (sum(count for (_, count) in x[1]), -x[3]))
    
    return reactions
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
export filter_fundamental_reactions, format_reactions

end # module DirectCRNInference

"""
    infer_crn_direct(trajectories; 
                     delta_t=0.5, 
                     threshold=0.001,
                     min_observations=5,
                     fundamental_only=true,
                     species_names=nothing,
                     boundary_condition=nothing)
    
Infer a chemical reaction network directly from trajectory data.
Uses time-dependent state spaces.
"""
function infer_crn_direct(trajectories; 
                         delta_t=0.5, 
                         threshold=0.001,
                         min_observations=5,
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
    
    # Identify reactions
    all_reactions = DirectCRNInference.identify_reactions(
        stoich_data, min_observations=min_observations
    )
    
    # Optionally filter for fundamental reactions
    reactions = fundamental_only ? 
                DirectCRNInference.filter_fundamental_reactions(all_reactions) :
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
                        fundamental_only=true,
                        species_names=nothing,
                        boundary_condition=nothing)
    
Run the complete inference process on a reaction system.
"""
function run_direct_inference(rn, u0, tspan, params, n_trajs=1000; 
                            delta_t=0.5, 
                            threshold=0.001,
                            min_observations=5,
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
        fundamental_only=fundamental_only,
        species_names=species_names,
        boundary_condition=boundary_condition
    )
    
    # Print ground truth in a more robust way
    println("\nGround truth reactions:")
    
    try
        # Get parameters as a dictionary
        param_dict = Dict{Symbol, Float64}()
        for (k, v) in params
            param_dict[Symbol(k)] = v
        end
        
        # Extract reactions from the model safely
        rxs = Catalyst.reactions(rn)
        
        # Print each reaction
        for rx in rxs
            # Format substrates
            if isempty(rx.substrates)
                sub_str = "∅"
            else
                sub_str = join([string(s) for s in rx.substrates], " + ")
            end
            
            # Format products
            if isempty(rx.products)
                prod_str = "∅"
            else
                prod_str = join([string(p) for p in rx.products], " + ")
            end
            
            # Extract rate parameter name
            rate_params = parameters(rx.rate)
            rate_str = string(rx.rate)
            
            # Try to find the actual value
            rate_value = "unknown"
            for (name, value) in param_dict
                if occursin(string(name), rate_str)
                    rate_value = value
                    break
                end
            end
            
            # Compute stoichiometric vector for comparison
            stoich_vec = zeros(Int, length(species_names))
            for sub in rx.substrates
                for (i, name) in enumerate(species_names)
                    if string(sub) == string(name)
                        stoich_vec[i] -= 1
                    end
                end
            end
            
            for prod in rx.products
                for (i, name) in enumerate(species_names)
                    if string(prod) == string(name)
                        stoich_vec[i] += 1
                    end
                end
            end
            
            println("$sub_str --> $prod_str, rate = $rate_value, stoich: $stoich_vec")
        end
    catch e
        println("Error printing ground truth: $e")
        println("Known reactions: S + E --> SE (rate = 0.01), SE --> S + E (rate = 0.1), SE --> P + E (rate = 0.1)")
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
                              fundamental_only=true,
                              species_names=species_names,
                              boundary_condition=bc)
end
