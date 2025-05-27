# flow_analysis.jl - FLOW FIELD ANALYSIS MODULE
# Analyzes probability flux patterns λᵢ × φᵢ to identify dynamically important states

using LinearAlgebra
using Statistics

"""
    compute_flow_fields(results; top_modes=6)

Core flow field analysis - computes λᵢ × φᵢ for each DMD mode.
"""
function compute_flow_fields(results; top_modes=6)
    println("\n=== Flow Field Analysis ===")
    
    λ = results["eigenvalues"]
    Φ = results["DMD_modes"]
    dt = results["dt"]
    
    valid_modes = []
    
    for i in 1:length(λ)
        # Skip problematic eigenvalues
        if abs(λ[i]) < 1e-12 || !isfinite(λ[i])
            continue
        end
        
        try
            # Convert to continuous time eigenvalue
            cont_eig = log(Complex(λ[i])) / dt
            
            if !isfinite(cont_eig)
                continue
            end
            
            # Compute flow field: λᵢ × φᵢ
            eigenvector = Φ[:, i]
            flow_field = cont_eig .* eigenvector
            flow_magnitude = abs.(flow_field)
            
            if !all(isfinite.(flow_magnitude))
                continue
            end
            
            # Classify mode temporally
            mode_type = classify_mode_type(cont_eig)
            
            # Calculate timescales
            decay_time = real(cont_eig) != 0 ? -1/real(cont_eig) : Inf
            osc_period = abs(imag(cont_eig)) > 1e-10 ? 2π/abs(imag(cont_eig)) : Inf
            
            mode_data = (
                mode_index = i,
                eigenvalue = cont_eig,
                eigenvector = eigenvector,
                flow_magnitude = flow_magnitude,
                flow_direction = angle.(flow_field),
                mode_type = mode_type,
                decay_time = decay_time,
                oscillation_period = osc_period
            )
            
            push!(valid_modes, mode_data)
            
        catch e
            continue
        end
    end
    
    if !isempty(valid_modes)
        # Sort by maximum flow magnitude
        flow_maxes = [maximum(m.flow_magnitude) for m in valid_modes]
        sorted_indices = sortperm(flow_maxes, rev=true)
        selected_modes = valid_modes[sorted_indices[1:min(top_modes, length(valid_modes))]]
        
        println("Found $(length(selected_modes)) valid flow modes:")
        for mode in selected_modes
            max_flow = maximum(mode.flow_magnitude)
            println("Mode $(mode.mode_index) [$(mode.mode_type)]: λ=$(round(mode.eigenvalue, digits=4)), max_flow=$(round(max_flow, digits=4))")
        end
        
        return selected_modes
    else
        println("No valid flow modes found")
        return []
    end
end

"""
    classify_mode_type(eigenvalue)

Classify mode by temporal characteristics.
"""
function classify_mode_type(λ)
    real_part = real(λ)
    imag_part = imag(λ)
    
    if real_part > 1e-10
        return "unstable"
    elseif abs(imag_part) > 0.5 * abs(real_part)
        return abs(real_part) < 0.5 ? "slow_oscillatory" : "fast_oscillatory"
    else
        return abs(real_part) < 0.5 ? "slow_decay" : "fast_decay"
    end
end

"""
    identify_active_states(flow_modes, selected_states, species_names; top_n=15)

Find the most dynamically active states across all flow modes.
"""
function identify_active_states(flow_modes, selected_states, species_names; top_n=15)
    if isempty(flow_modes)
        return
    end
    
    println("\n=== Most Active States Analysis ===")
    
    for (mode_num, mode) in enumerate(flow_modes[1:min(3, length(flow_modes))])
        flow_magnitude = mode.flow_magnitude
        
        # Find top states for this mode
        top_indices = sortperm(flow_magnitude, rev=true)[1:min(top_n, length(flow_magnitude))]
        
        println("\nMode $(mode.mode_index) [$(mode.mode_type)]:")
        println("Eigenvalue: $(round(mode.eigenvalue, digits=4))")
        if mode.decay_time < Inf
            println("Decay time: $(round(mode.decay_time, digits=2)) time units")
        end
        if mode.oscillation_period < Inf
            println("Oscillation period: $(round(mode.oscillation_period, digits=2)) time units")
        end
        
        println("\nTop active states:")
        println("Rank | State (S,E,SE,P) | Flow Magnitude | Description")
        println("-"^60)
        
        for (rank, idx) in enumerate(top_indices[1:min(10, length(top_indices))])
            if idx <= length(selected_states)
                state = selected_states[idx]
                mol_counts = [max(0, s-1) for s in state]  # Convert to molecular counts
                flow_mag = flow_magnitude[idx]
                
                # Biological interpretation
                s, e, se, p = mol_counts
                total_substrate = s + se + p
                conversion = total_substrate > 0 ? p / total_substrate : 0.0
                
                description = ""
                if conversion < 0.2
                    description = "Early (substrate-rich)"
                elseif conversion > 0.8
                    description = "Late (product-rich)"
                elseif se > 5
                    description = "Complex-dominated"
                else
                    description = "Intermediate"
                end
                
                state_str = "(" * join(mol_counts, ",") * ")"
                println("$(lpad(rank,4)) | $(rpad(state_str,12)) | $(rpad(round(flow_mag, digits=4),13)) | $description")
            end
        end
    end
end

"""
    analyze_flow_connectivity(flow_modes, selected_states, G; threshold=0.1)

Analyze how flow connects different regions of state space.
"""
function analyze_flow_connectivity(flow_modes, selected_states, G; threshold=0.1)
    if isempty(flow_modes)
        return
    end
    
    println("\n=== Flow Connectivity Analysis ===")
    
    for (mode_num, mode) in enumerate(flow_modes[1:min(2, length(flow_modes))])
        flow_magnitude = mode.flow_magnitude
        max_flow = maximum(flow_magnitude)
        active_threshold = threshold * max_flow
        
        # Identify highly active states
        active_indices = findall(flow_magnitude .> active_threshold)
        
        if length(active_indices) < 2
            continue
        end
        
        println("\nMode $(mode.mode_index) Connectivity:")
        println("Active states: $(length(active_indices)) (threshold: $(round(active_threshold, digits=4)))")
        
        # Check connections between active states
        connections = 0
        strong_connections = 0
        
        G_sparse = sparse(G)
        
        for i in active_indices
            for j in active_indices
                if i != j && abs(G[i, j]) > 1e-6
                    connections += 1
                    if abs(G[i, j]) > 1e-4
                        strong_connections += 1
                    end
                end
            end
        end
        
        total_possible = length(active_indices) * (length(active_indices) - 1)
        connectivity_ratio = total_possible > 0 ? connections / total_possible : 0.0
        
        println("Connections: $connections/$total_possible ($(round(connectivity_ratio*100, digits=1))%)")
        println("Strong connections: $strong_connections")
        
        # Flow coherence (how similar are flow directions in connected states)
        if connections > 0
            flow_directions = mode.flow_direction
            coherence_sum = 0.0
            pair_count = 0
            
            for i in active_indices
                for j in active_indices
                    if i != j && abs(G[i, j]) > 1e-6
                        angle_diff = abs(flow_directions[i] - flow_directions[j])
                        angle_diff = min(angle_diff, 2π - angle_diff)  # Circular distance
                        coherence_sum += cos(angle_diff)
                        pair_count += 1
                    end
                end
            end
            
            if pair_count > 0
                coherence = coherence_sum / pair_count
                println("Flow coherence: $(round(coherence, digits=3)) (1=perfectly aligned, -1=opposed)")
            end
        end
    end
end

"""
    compare_flow_modes(flow_modes, selected_states; correlation_threshold=0.5)

Compare flow patterns across different modes.
"""
function compare_flow_modes(flow_modes, selected_states; correlation_threshold=0.5)
    if length(flow_modes) < 2
        return
    end
    
    println("\n=== Flow Mode Comparison ===")
    
    n_modes = min(4, length(flow_modes))
    n_states = length(selected_states)
    
    # Create flow magnitude matrix
    flow_matrix = zeros(n_states, n_modes)
    for (i, mode) in enumerate(flow_modes[1:n_modes])
        n_valid = min(length(mode.flow_magnitude), n_states)
        flow_matrix[1:n_valid, i] = mode.flow_magnitude[1:n_valid]
    end
    
    # Calculate correlations between modes
    println("Mode Correlations:")
    println("Mode Pair | Correlation | Interpretation")
    println("-"^45)
    
    for i in 1:n_modes
        for j in (i+1):n_modes
            if var(flow_matrix[:, i]) > 1e-10 && var(flow_matrix[:, j]) > 1e-10
                corr = cor(flow_matrix[:, i], flow_matrix[:, j])
                
                interpretation = ""
                if abs(corr) > 0.8
                    interpretation = "Highly coupled"
                elseif abs(corr) > 0.5
                    interpretation = "Moderately coupled"
                elseif abs(corr) < 0.1
                    interpretation = "Independent"
                else
                    interpretation = "Weakly coupled"
                end
                
                mode_i = flow_modes[i].mode_index
                mode_j = flow_modes[j].mode_index
                println("$mode_i vs $mode_j  |  $(rpad(round(corr, digits=3), 11)) | $interpretation")
            end
        end
    end
    
    # Find states active in multiple modes
    active_threshold = 0.1
    multi_active_states = []
    
    for i in 1:n_states
        active_count = sum(flow_matrix[i, :] .> active_threshold * maximum(flow_matrix))
        if active_count >= 2
            state = selected_states[i]
            mol_counts = [max(0, s-1) for s in state]
            flows = flow_matrix[i, :]
            push!(multi_active_states, (mol_counts, flows, active_count))
        end
    end
    
    if !isempty(multi_active_states)
        println("\nStates active in multiple modes:")
        println("State | Flow Magnitudes | Active Count")
        println("-"^45)
        
        for (state, flows, count) in multi_active_states[1:min(10, end)]
            state_str = "(" * join(state, ",") * ")"
            flows_str = join([lpad(string(round(f, digits=3)), 7) for f in flows], " ")
            println("$(rpad(state_str,12)) | $flows_str | $count")
        end
    end
end

"""
    basic_flow_analysis(results, species_names)

Run basic flow field analysis pipeline.
"""
function basic_flow_analysis(results, species_names)
    println("\n" * "="^50)
    println("FLOW FIELD ANALYSIS")
    println("="^50)
    
    # Compute flow fields
    flow_modes = compute_flow_fields(results, top_modes=6)
    
    if isempty(flow_modes)
        println("No flow modes found - skipping flow analysis")
        return nothing
    end
    
    # Analyze active states
    identify_active_states(flow_modes, results["selected_states"], species_names)
    
    # Analyze connectivity
    analyze_flow_connectivity(flow_modes, results["selected_states"], results["generator"])
    
    # Compare modes
    compare_flow_modes(flow_modes, results["selected_states"])
    
    println("\n" * "="^50)
    println("FLOW ANALYSIS COMPLETE")
    println("="^50)
    
    return Dict("flow_modes" => flow_modes)
end

"""
    run_with_flow_analysis(n_trajs=500, max_states=500)

Run complete MM inference with flow analysis.
"""
function run_with_flow_analysis(n_trajs=500, max_states=500)
    # Run basic inference
    results = run_mm_inference(n_trajs, max_states)
    
    # Add flow analysis
    flow_results = basic_flow_analysis(results, results["species_names"])
    
    if flow_results !== nothing
        results["flow_analysis"] = flow_results
    end
    
    return results
end

println("Flow Analysis Module Loaded! 🌊")
println("Usage: run_with_flow_analysis(n_trajs, max_states)")
