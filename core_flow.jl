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

println("Core Flow Analysis Module Loaded! 🌊")
println("System-agnostic functions:")
println("  compute_flow_modes(eigenvalues, modes, states)")
println("  run_flow_analysis(eigenvalues, modes, states)")
println("  analyze_timescales(flow_modes)")
println("  detect_oscillatory_behavior(flow_modes)")
