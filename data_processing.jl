# data_processing.jl
# Functions for processing trajectory data into formats suitable for DMD analysis

using Statistics
using ProgressMeter

"""
    process_trajectories_to_sparse(ssa_trajs, species_indices, grid_sizes, time_points)

Process trajectory data into sparse histogram representation for DMD analysis.

# Arguments
- `ssa_trajs`: Array of trajectory solutions
- `species_indices`: Indices of species to include
- `grid_sizes`: Array of grid sizes for discretization of each species
- `time_points`: Array of time points to sample

# Returns
- Array of sparse probability distributions (histograms)
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
    reduce_sparse_data(sparse_probs, grid_sizes, max_dim=1000)

Reduce the dimensionality of sparse probability data by selecting important states.

# Arguments
- `sparse_probs`: Array of sparse probability distributions
- `grid_sizes`: Array of grid sizes for each species
- `max_dim`: Maximum dimension of the reduced state space

# Returns
- Matrix of reduced probability data
- Array of selected states
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
    state_idx_to_molecule_counts(state_idx, species_names)

Convert grid indices to approximate molecular counts.

# Arguments
- `state_idx`: State index vector (grid coordinates)
- `species_names`: Names of species (for context)

# Returns
- Array of approximate molecular counts
"""
function state_idx_to_molecule_counts(state_idx, species_names)
    # Convert from 1-based grid indices to molecule counts
    # Grid index of 1 corresponds to 0 molecules
    return [idx - 1 for idx in state_idx]
end

"""
    molecule_counts_to_state_idx(counts, grid_sizes)

Convert molecular counts to grid indices with bounds checking.

# Arguments
- `counts`: Array of molecular counts
- `grid_sizes`: Array of grid sizes for each species

# Returns
- Array of grid indices
"""
function molecule_counts_to_state_idx(counts, grid_sizes)
    # Convert from molecule counts to 1-based grid indices
    # With bounds checking
    indices = []
    for (count, grid_size) in zip(counts, grid_sizes)
        # Convert count to index (count 0 => index 1)
        idx = min(max(1, round(Int, count) + 1), grid_size)
        push!(indices, idx)
    end
    return indices
end
