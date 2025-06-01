# core_flow.jl - SYSTEM-AGNOSTIC FLOW ANALYSIS
# Flow field computation and timescale analysis

using Statistics
using LinearAlgebra

"""
    FlowMode

Generic structure for flow mode information.
"""
struct FlowMode
    mode_index::Int
    eigenvalue::ComplexF64
    mode_vector::Vector{ComplexF64}
    flow_magnitude::Vector{Float64}
    decay_time::Float64
    oscillation_period::Float64
    mode_type::String
end

"""
    compute_flow_modes(eigenvalues, modes, selected_states)

Compute flow modes from DMD eigenvalues and eigenvectors (system-agnostic).
"""
function compute_flow_modes(eigenvalues, modes, selected_states)
    println("Computing flow modes...")
    
    flow_modes = FlowMode[]
    n_modes = min(length(eigenvalues), size(modes, 2))
    
    for i in 1:n_modes
        λ = eigenvalues[i]
        φ = modes[:, i]
        
        # Compute flow field: λ × φ
        flow_field = λ .* φ
        flow_magnitude = abs.(flow_field)
        
        # Compute timescales
        decay_time = real(λ) != 0 ? -1.0 / real(λ) : Inf
        oscillation_period = imag(λ) != 0 ? 2π / abs(imag(λ)) : Inf
        
        # Classify mode type
        mode_type = classify_mode_type(λ)
        
        push!(flow_modes, FlowMode(
            i, λ, φ, flow_magnitude, decay_time, oscillation_period, mode_type
        ))
    end
    
    # Sort by flow magnitude
    sort!(flow_modes, by=m -> maximum(m.flow_magnitude), rev=true)
    
    println("Computed $(length(flow_modes)) flow modes")
    return flow_modes
end

"""
    classify_mode_type(eigenvalue)

Classify DMD mode based on eigenvalue properties.
"""
function classify_mode_type(λ)
    if abs(imag(λ)) < 1e-6
        # Real eigenvalue
        if real(λ) < -1e-6
            return "Decay"
        elseif real(λ) > 1e-6
            return "Growth"
        else
            return "Steady"
        end
    else
        # Complex eigenvalue
        if real(λ) < -1e-6
            return "Damped Oscillation"
        elseif real(λ) > 1e-6
            return "Growing Oscillation"
        else
            return "Pure Oscillation"
        end
    end
end

"""
    compute_state_importance(flow_modes, selected_states)

Compute importance scores for states based on flow analysis.
"""
function compute_state_importance(flow_modes, selected_states)
    n_states = length(selected_states)
    importance_scores = zeros(n_states)
    
    for mode in flow_modes
        # Weight by eigenvalue magnitude
        weight = abs(mode.eigenvalue)
        
        # Add weighted flow contribution
        n_valid = min(length(mode.flow_magnitude), n_states)
        importance_scores[1:n_valid] .+= weight .* mode.flow_magnitude[1:n_valid]
    end
    
    # Normalize
    if maximum(importance_scores) > 0
        importance_scores ./= maximum(importance_scores)
    end
    
    return importance_scores
end

"""
    analyze_timescales(flow_modes)

Analyze different timescales in the flow modes.
"""
function analyze_timescales(flow_modes)
    println("\n=== Flow Timescale Analysis ===")
    
    # Classify modes by timescale
    fast_modes = []      # τ < 5 time units
    medium_modes = []    # 5 ≤ τ < 20 time units  
    slow_modes = []      # τ ≥ 20 time units
    
    for mode in flow_modes
        if mode.decay_time < 5
            push!(fast_modes, mode)
        elseif mode.decay_time < 20
            push!(medium_modes, mode)
        else
            push!(slow_modes, mode)
        end
    end
    
    println("Timescale Classification:")
    println("  Fast modes (τ < 5): $(length(fast_modes))")
    println("  Medium modes (5 ≤ τ < 20): $(length(medium_modes))")
    println("  Slow modes (τ ≥ 20): $(length(slow_modes))")
    
    return Dict(
        "fast" => fast_modes,
        "medium" => medium_modes,
        "slow" => slow_modes
    )
end

"""
    detect_oscillatory_behavior(flow_modes; threshold=0.01)

Detect oscillatory behavior in flow modes.
"""
function detect_oscillatory_behavior(flow_modes; threshold=0.01)
    oscillatory_modes = []
    
    for mode in flow_modes
        if abs(imag(mode.eigenvalue)) > threshold
            push!(oscillatory_modes, mode)
        end
    end
    
    if !isempty(oscillatory_modes)
        println("\n🌀 Oscillatory Behavior Detected:")
        for mode in oscillatory_modes[1:min(3, end)]
            period = mode.oscillation_period
            println("  Mode $(mode.mode_index): Period = $(round(period, digits=2)) time units")
        end
        
        return true
    else
        println("\n➡️ No significant oscillatory behavior detected")
        return false
    end
end

"""
    run_flow_analysis(eigenvalues, modes, selected_states)

Run complete flow analysis (system-agnostic).
"""
function run_flow_analysis(eigenvalues, modes, selected_states)
    println("\n" * "="^50)
    println("FLOW FIELD ANALYSIS")
    println("="^50)
    
    # Compute flow modes
    flow_modes = compute_flow_modes(eigenvalues, modes, selected_states)
    
    # Compute state importance
    importance_scores = compute_state_importance(flow_modes, selected_states)
    
    # Timescale analysis
    timescale_analysis = analyze_timescales(flow_modes)
    
    # Oscillatory behavior detection
    has_oscillations = detect_oscillatory_behavior(flow_modes)
    
    results = Dict(
        "flow_modes" => flow_modes,
        "state_importance" => importance_scores,
        "timescale_analysis" => timescale_analysis,
        "has_oscillations" => has_oscillations
    )
    
    println("\nFlow analysis completed!")
    return results
end

"""
    format_reaction_string(stoich, species_names)

Format stoichiometry as readable reaction string.
"""
function format_reaction_string(stoich, species_names)
    reactants = String[]
    products = String[]
    
    for (i, coeff) in enumerate(stoich)
        if i <= length(species_names)
            species = species_names[i]
            if coeff < 0
                abs_coeff = abs(coeff)
                if abs_coeff == 1
                    push!(reactants, species)
                else
                    push!(reactants, "$abs_coeff $species")
                end
            elseif coeff > 0
                if coeff == 1
                    push!(products, species)
                else
                    push!(products, "$coeff $species")
                end
            end
        end
    end
    
    reactant_str = isempty(reactants) ? "∅" : join(reactants, " + ")
    product_str = isempty(products) ? "∅" : join(products, " + ")
    
    return "$reactant_str → $product_str"
end

"""
    compute_dmd_flow_field(probability_matrix, time_points)

Compute flow field directly from probability evolution (DMD-derived).
"""
function compute_dmd_flow_field(probability_matrix, time_points)
    n_states, n_times = size(probability_matrix)
    
    if n_times < 2
        return zeros(n_states, 1)
    end
    
    # Compute time derivatives: dP/dt ≈ (P(t+1) - P(t)) / dt
    flow_field = zeros(n_states, n_times-1)
    
    for t in 1:(n_times-1)
        dt = time_points[t+1] - time_points[t]
        flow_field[:, t] = (probability_matrix[:, t+1] - probability_matrix[:, t]) / dt
    end
    
    return flow_field
end

"""
    compute_mechanistic_flow(states, reactions, rate_constants, probability_dist)

Compute mechanistic flow field from known reactions and rate constants.
"""
function compute_mechanistic_flow(states, reactions, rate_constants, probability_dist)
    n_states = length(states)
    mechanistic_flow = zeros(n_states)
    
    for (r, reaction) in enumerate(reactions)
        k_r = rate_constants[r]
        
        # For each reaction, compute its contribution to flow
        for j in 1:n_states
            from_state = states[j]
            prob_j = probability_dist[j]
            
            if prob_j > 1e-10  # Only consider states with significant probability
                # Check all possible transitions from state j
                for i in 1:n_states
                    to_state = states[i]
                    
                    # Check if this transition corresponds to the reaction
                    stoich_change = to_state - from_state
                    if stoich_change == reaction
                        propensity = calculate_propensity(from_state, reaction)
                        if propensity > 0
                            # Net flux: out of state j, into state i
                            flux = k_r * propensity * prob_j
                            mechanistic_flow[j] -= flux  # Outflow from j
                            mechanistic_flow[i] += flux  # Inflow to i
                        end
                    end
                end
            end
        end
    end

    # Normalize by total system reactivity to match probability derivative scale
    total_reactivity = 0.0
    for j in 1:length(states)
        if probability_dist[j] > 1e-10
            state_reactivity = 0.0
            for (r, reaction) in enumerate(reactions)
                prop = calculate_propensity(states[j], reaction)
                state_reactivity += rate_constants[r] * prop
            end
            total_reactivity += state_reactivity * probability_dist[j]
        end
    end
    
    # Scale mechanistic flow to match probability derivative scale
    if total_reactivity > 0
        mechanistic_flow ./= total_reactivity
    end
    
    
    return mechanistic_flow
end

"""
    debug_flow_objective(rate_constants, states, reactions, probability_matrix, time_points, dmd_flow)

Debug the flow objective function to understand optimization behavior.
"""
function debug_flow_objective(rate_constants, states, reactions, probability_matrix, time_points, dmd_flow)
    println("\n🔧 DEBUGGING FLOW OBJECTIVE:")
    println("  Rate constants: $(round.(rate_constants, digits=6))")
    
    n_times = size(dmd_flow, 2)
    total_error = 0.0
    reaction_contributions = zeros(length(reactions))
    
    for t in 1:min(3, n_times)  # Debug first few time points
        prob_dist = probability_matrix[:, t]
        obs_flow = dmd_flow[:, t]
        
        println("\n  Time point $t:")
        println("    Probability dist: sum=$(round(sum(prob_dist), digits=4)), max=$(round(maximum(prob_dist), digits=4))")
        println("    Observed flow: range=$(round(minimum(obs_flow), digits=6)) to $(round(maximum(obs_flow), digits=6))")
        
        # Compute mechanistic flow with detailed breakdown
        total_mech_flow = zeros(length(states))
        
        for (r, reaction) in enumerate(reactions)
            k_r = rate_constants[r]
            reaction_flow = zeros(length(states))
            
            for j in 1:length(states)
                prob_j = prob_dist[j]
                if prob_j > 1e-10
                    from_state = states[j]
                    
                    for i in 1:length(states)
                        to_state = states[i]
                        stoich_change = to_state - from_state
                        
                        if stoich_change == reaction
                            propensity = calculate_propensity(from_state, reaction)
                            if propensity > 0
                                flux = k_r * propensity * prob_j
                                reaction_flow[j] -= flux
                                reaction_flow[i] += flux
                            end
                        end
                    end
                end
            end
            
            reaction_magnitude = norm(reaction_flow)
            reaction_contributions[r] += reaction_magnitude
            total_mech_flow .+= reaction_flow
            
            reaction_str = format_reaction_string(reaction, ["S", "E", "SE", "P"])
            println("    Reaction $r ($(reaction_str)): k=$(round(k_r, digits=6)), flow_mag=$(round(reaction_magnitude, digits=6))")
        end
        
        # Compare total flows
        mech_magnitude = norm(total_mech_flow)
        obs_magnitude = norm(obs_flow)
        error = norm(obs_flow - total_mech_flow)^2
        
        println("    Mechanistic flow magnitude: $(round(mech_magnitude, digits=6))")
        println("    Observed flow magnitude: $(round(obs_magnitude, digits=6))")
        println("    Error: $(round(error, digits=6))")
        
        if obs_magnitude > 0 && mech_magnitude > 0
            correlation = dot(obs_flow, total_mech_flow) / (obs_magnitude * mech_magnitude)
            println("    Flow correlation: $(round(correlation, digits=3))")
        end
        
        total_error += error
    end
    
    println("\n  Overall reaction contributions:")
    for (r, contrib) in enumerate(reaction_contributions)
        reaction_str = format_reaction_string(reactions[r], ["S", "E", "SE", "P"])
        println("    $(reaction_str): total_magnitude=$(round(contrib, digits=6))")
    end
    
    return total_error
end

"""
    debug_optimization_path(states, reactions, probability_matrix, time_points, dmd_flow)

Debug the optimization path to see why certain parameters go to zero.
"""
function debug_optimization_path(states, reactions, probability_matrix, time_points, dmd_flow)
    println("\n🔍 DEBUGGING OPTIMIZATION PATH:")
    
    # Test individual reactions
    for (r, reaction) in enumerate(reactions)
        println("\n  Testing reaction $r alone:")
        reaction_str = format_reaction_string(reaction, ["S", "E", "SE", "P"])
        println("    $(reaction_str)")
        
        # Test with only this reaction active
        test_rates = zeros(length(reactions))
        
        for k_test in [0.001, 0.01, 0.1, 1.0]
            test_rates[r] = k_test
            error = compute_flow_error(test_rates, states, reactions, probability_matrix, time_points, dmd_flow)
            println("      k=$(k_test): error=$(round(error, digits=6))")
        end
        
        test_rates[r] = 0.0  # Reset
    end
    
    # Test pairwise interactions
    println("\n  Testing pairwise interactions:")
    for r1 in 1:length(reactions)
        for r2 in (r1+1):length(reactions)
            reaction1_str = format_reaction_string(reactions[r1], ["S", "E", "SE", "P"])
            reaction2_str = format_reaction_string(reactions[r2], ["S", "E", "SE", "P"])
            
            test_rates = zeros(length(reactions))
            test_rates[r1] = 0.01
            test_rates[r2] = 0.01
            
            error = compute_flow_error(test_rates, states, reactions, probability_matrix, time_points, dmd_flow)
            println("    $(reaction1_str) + $(reaction2_str): error=$(round(error, digits=6))")
        end
    end
end

function compute_flow_error(rate_constants, states, reactions, probability_matrix, time_points, dmd_flow)
    total_error = 0.0
    n_valid = 0
    
    for t in 1:size(dmd_flow, 2)
        prob_dist = probability_matrix[:, t]
        if sum(prob_dist .> 1e-8) >= 3
            mech_flow = compute_mechanistic_flow(states, reactions, rate_constants, prob_dist)
            obs_flow = dmd_flow[:, t]
            error = norm(obs_flow - mech_flow)^2
            total_error += error
            n_valid += 1
        end
    end
    
    return n_valid > 0 ? total_error / n_valid : 1e6
end

"""
    fit_rate_constants_via_flow(states, reactions, probability_matrix, time_points)

Fit rate constants by minimizing flow field discrepancy.
"""
function fit_rate_constants_via_flow(states, reactions, probability_matrix, time_points)
    println("Fitting rate constants via flow field matching...")
    
    # Compute observed flow from probability evolution
    dmd_flow = compute_dmd_flow_field(probability_matrix, time_points)
    n_states, n_times = size(dmd_flow)
    
    println("  DMD flow field computed: $(size(dmd_flow))")
    println("  Flow magnitude range: $(round(minimum(dmd_flow), digits=6)) to $(round(maximum(dmd_flow), digits=6))")
    
    # Objective function: minimize flow discrepancy across all time points
    function flow_objective(rate_constants)
        total_error = 0.0
        n_valid_times = 0
        
        for t in 1:n_times
            prob_dist = probability_matrix[:, t]
            
            # Skip if probability distribution is too sparse
            if sum(prob_dist .> 1e-8) < 3
                continue
            end
            
            # Compute mechanistic flow with current rate constants
            mech_flow = compute_mechanistic_flow(states, reactions, rate_constants, prob_dist)
            
            # Flow discrepancy at this time point
            obs_flow = dmd_flow[:, t]
            error = norm(obs_flow - mech_flow)^2
            total_error += error
            n_valid_times += 1
        end
        
        return n_valid_times > 0 ? total_error / n_valid_times : 1e6
    end

    # Add this after computing dmd_flow and before optimization:
    
    # Initial guess based on expected MM rates
    initial_guess = [0.01, 0.1, 0.1]  # Rough MM estimates
    if length(reactions) != length(initial_guess)
        initial_guess = ones(length(reactions)) * 0.05
    end
    
    println("  Optimizing $(length(reactions)) rate constants...")
    println("  Initial guess: $(round.(initial_guess, digits=4))")
println("=== DEBUGGING FLOW OBJECTIVE ===")
debug_flow_objective(initial_guess, states, reactions, probability_matrix, time_points, dmd_flow)
debug_optimization_path(states, reactions, probability_matrix, time_points, dmd_flow)
    
    # Simple optimization using coordinate descent
    best_params = copy(initial_guess)
    best_error = flow_objective(best_params)
    
    # Coordinate descent optimization
    for iteration in 1:100
        improved = false
        
        for i in 1:length(best_params)
            # Try scaling this parameter
            for scale in [0.5, 0.8, 1.2, 2.0]
                test_params = copy(best_params)
                test_params[i] *= scale
                
                # Ensure non-negative
                if test_params[i] > 0
                    error = flow_objective(test_params)
                    
                    if error < best_error
                        best_params = test_params
                        best_error = error
                        improved = true
                    end
                end
            end
        end
        
        if !improved
            break
        end
    end
    
    println("  Optimization completed. Final error: $(round(best_error, digits=6))")
    
    return best_params, best_error
end

"""
    validate_flow_consistency(states, reactions, rate_constants, probability_matrix, time_points)

Validate consistency between DMD and mechanistic flow fields.
"""
function validate_flow_consistency(states, reactions, rate_constants, probability_matrix, time_points)
    println("Validating flow field consistency...")
    
    # Compute both flow fields
    dmd_flow = compute_dmd_flow_field(probability_matrix, time_points)
    n_states, n_times = size(dmd_flow)
    
    correlations = Float64[]
    rmse_values = Float64[]
    
    for t in 1:n_times
        prob_dist = probability_matrix[:, t]
        mech_flow = compute_mechanistic_flow(states, reactions, rate_constants, prob_dist)
        obs_flow = dmd_flow[:, t]
        
        # Skip if flows are too small
        if norm(obs_flow) > 1e-8 && norm(mech_flow) > 1e-8
            # Normalize for comparison
            obs_normalized = obs_flow ./ norm(obs_flow)
            mech_normalized = mech_flow ./ norm(mech_flow)
            
            # Compute metrics
            correlation = dot(obs_normalized, mech_normalized)
            rmse = norm(obs_normalized - mech_normalized) / sqrt(n_states)
            
            push!(correlations, correlation)
            push!(rmse_values, rmse)
        end
    end
    
    if !isempty(correlations)
        avg_correlation = mean(correlations)
        avg_rmse = mean(rmse_values)
        
        # Quality assessment
        if avg_correlation > 0.8 && avg_rmse < 0.3
            quality = "Excellent"
        elseif avg_correlation > 0.6 && avg_rmse < 0.5
            quality = "Good"
        elseif avg_correlation > 0.4 && avg_rmse < 0.7
            quality = "Fair"
        else
            quality = "Poor"
        end
        
        println("  Flow consistency: $(quality)")
        println("  Average correlation: $(round(avg_correlation, digits=3))")
        println("  Average RMSE: $(round(avg_rmse, digits=3))")
        
        return quality, avg_correlation, avg_rmse
    else
        println("  Could not validate - insufficient flow data")
        return "Unknown", 0.0, 1.0
    end
end

"""
    run_mechanistic_flow_analysis(states, probability_matrix, time_points, expected_reactions)

Complete mechanistic flow analysis pipeline.
"""
function run_mechanistic_flow_analysis(states, probability_matrix, time_points, expected_reactions)
    println("\n" * "="^60)
    println("MECHANISTIC FLOW ANALYSIS")
    println("="^60)
    
    # Convert expected reactions to stoichiometry vectors
    reactions = [collect(rxn) for rxn in expected_reactions]
    
    # Fit rate constants via flow matching
    fitted_params, fit_error = fit_rate_constants_via_flow(states, reactions, probability_matrix, time_points)
    
    # Validate flow consistency
    quality, correlation, rmse = validate_flow_consistency(states, reactions, fitted_params, probability_matrix, time_points)
    
    # Create results
    results = []
    species_names = ["S", "E", "SE", "P"]  # Default for MM
    
    for (i, (reaction, k_est)) in enumerate(zip(reactions, fitted_params))
        reaction_str = format_reaction_string(reaction, species_names)
        
        push!(results, (
            stoichiometry = reaction,
            reaction_string = reaction_str,
            rate_constant = k_est,
            quality = quality,
            fit_error = fit_error,
            correlation = correlation
        ))
        
        println("$(reaction_str): k = $(round(k_est, digits=6)) ($(quality))")
    end
    
    println("\nFlow Analysis Summary:")
    println("  Rate constants fitted: $(length(results))")
    println("  Flow consistency: $(quality) (r=$(round(correlation, digits=3)))")
    println("  Fit error: $(round(fit_error, digits=6))")
    
    return Dict(
        "rate_constants" => results,
        "flow_quality" => quality,
        "flow_correlation" => correlation,
        "fit_error" => fit_error
    )
end

println("Core Flow Analysis Module Loaded! 🌊")
println("System-agnostic functions:")
println("  compute_flow_modes(eigenvalues, modes, states)")
println("  run_flow_analysis(eigenvalues, modes, states)")
println("  analyze_timescales(flow_modes)")
println("  detect_oscillatory_behavior(flow_modes)")
