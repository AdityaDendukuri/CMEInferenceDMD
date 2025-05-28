# flow.jl - CLEAN FLOW ANALYSIS MODULE
# Spatiotemporal flow field analysis and MM-specific pattern detection

using Statistics
using LinearAlgebra

"""
    FlowMode

Structure to hold flow mode information.
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

Compute flow modes from DMD eigenvalues and eigenvectors.
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
        mode_type = classify_mode_type(λ, decay_time, oscillation_period)
        
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
    classify_mode_type(eigenvalue, decay_time, oscillation_period)

Classify DMD mode based on eigenvalue properties.
"""
function classify_mode_type(λ, decay_time, osc_period)
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
    compute_flow_importance(flow_modes, selected_states)

Compute importance scores for states based on flow analysis.
"""
function compute_flow_importance(flow_modes, selected_states)
    n_states = length(selected_states)
    importance_scores = zeros(n_states)
    
    for mode in flow_modes
        # Weight by eigenvalue magnitude
        weight = abs(mode.eigenvalue)
        
        # Add weighted flow contribution
        importance_scores .+= weight .* mode.flow_magnitude
    end
    
    # Normalize
    if maximum(importance_scores) > 0
        importance_scores ./= maximum(importance_scores)
    end
    
    return importance_scores
end

"""
    analyze_mm_flow_patterns(flow_modes, selected_states, species_names)

Analyze MM-specific flow patterns without hardcoded expectations.
"""
function analyze_mm_flow_patterns(flow_modes, selected_states, species_names)
    println("\n=== MM Flow Pattern Analysis ===")
    
    if length(species_names) != 4
        println("MM analysis requires exactly 4 species [S, E, SE, P]")
        return nothing
    end
    
    mm_analysis = []
    
    for mode in flow_modes[1:min(3, length(flow_modes))]
        flow_magnitude = mode.flow_magnitude
        n_valid = min(length(flow_magnitude), length(selected_states))
        
        # MM process categorization
        substrate_binding_flow = 0.0      # S + E → SE regions
        complex_dissociation_flow = 0.0   # SE → S + E regions
        product_formation_flow = 0.0      # SE → E + P regions
        enzyme_recycling_flow = 0.0       # Free enzyme regions
        substrate_depletion_flow = 0.0    # Low substrate regions
        
        for i in 1:n_valid
            state = selected_states[i]
            if length(state) >= 4
                s, e, se, p = [max(0, x) for x in state[1:4]]
                flow = flow_magnitude[i]
                
                # Categorize states by MM process signatures
                if s > 10 && e > 2 && se < 8
                    substrate_binding_flow += flow
                end
                
                if se > 3 && (s > 0 || e > 0)
                    complex_dissociation_flow += flow
                end
                
                if p > s && e > se && (s + se + p) > 20
                    product_formation_flow += flow
                end
                
                if e > 5 && se < 3
                    enzyme_recycling_flow += flow
                end
                
                if s < 5 && (s + se + p) > 20
                    substrate_depletion_flow += flow
                end
            end
        end
        
        total_flow = sum(flow_magnitude[1:n_valid])
        
        # Calculate percentages
        percentages = Dict(
            "substrate_binding" => total_flow > 0 ? (substrate_binding_flow/total_flow*100) : 0.0,
            "complex_dissociation" => total_flow > 0 ? (complex_dissociation_flow/total_flow*100) : 0.0,
            "product_formation" => total_flow > 0 ? (product_formation_flow/total_flow*100) : 0.0,
            "enzyme_recycling" => total_flow > 0 ? (enzyme_recycling_flow/total_flow*100) : 0.0,
            "substrate_depletion" => total_flow > 0 ? (substrate_depletion_flow/total_flow*100) : 0.0
        )
        
        # Find dominant process
        dominant_process = "unknown"
        dominant_value = 0.0
        for (process, value) in percentages
            if value > dominant_value
                dominant_process = process
                dominant_value = value
            end
        end
        
        # Biological interpretation
        interpretation = get_mm_interpretation(dominant_process)
        
        push!(mm_analysis, (
            mode_index = mode.mode_index,
            mode_type = mode.mode_type,
            eigenvalue = mode.eigenvalue,
            decay_time = mode.decay_time,
            oscillation_period = mode.oscillation_period,
            dominant_process = dominant_process,
            dominant_percentage = dominant_value,
            interpretation = interpretation,
            percentages = percentages
        ))
        
        println("\nMode $(mode.mode_index) [$(mode.mode_type)]:")
        println("  Eigenvalue: $(round(mode.eigenvalue, digits=4))")
        println("  Decay time: $(round(mode.decay_time, digits=2)) time units")
        println("  Dominant process: $dominant_process ($(round(dominant_value, digits=1))%)")
        println("  → $interpretation")
    end
    
    return mm_analysis
end

"""
    get_mm_interpretation(dominant_process)

Get biological interpretation for MM process.
"""
function get_mm_interpretation(process)
    interpretations = Dict(
        "substrate_binding" => "Mode captures initial substrate-enzyme encounter",
        "complex_dissociation" => "Mode captures SE complex equilibration",
        "product_formation" => "Mode captures catalytic conversion",
        "enzyme_recycling" => "Mode captures enzyme turnover cycle",
        "substrate_depletion" => "Mode captures substrate consumption dynamics"
    )
    
    return get(interpretations, process, "Mode captures unknown MM dynamics")
end

"""
    detect_mm_signature(mm_analysis)

Detect MM mechanism signature from flow analysis.
"""
function detect_mm_signature(mm_analysis)
    if mm_analysis === nothing || isempty(mm_analysis)
        return nothing
    end
    
    println("\n=== MM Mechanism Signature Detection ===")
    
    signature_matches = []
    
    for result in mm_analysis
        percentages = result.percentages
        
        # MM signature scoring
        product_pct = percentages["product_formation"]
        binding_pct = percentages["substrate_binding"]
        dissoc_pct = percentages["complex_dissociation"]
        recycling_pct = percentages["enzyme_recycling"]
        
        # Calculate MM signature score
        mm_score = 0.0
        
        if product_pct > 30
            mm_score += 0.4 * (product_pct / 100)
        end
        
        if binding_pct > 20
            mm_score += 0.3 * (binding_pct / 100)
        end
        
        if dissoc_pct > 15
            mm_score += 0.2 * (dissoc_pct / 100)
        end
        
        if recycling_pct > 10
            mm_score += 0.1 * (recycling_pct / 100)
        end
        
        # Penalty for unbalanced patterns
        if product_pct < 10 && dissoc_pct > 60
            mm_score *= 0.5
        end
        
        matches_mm = mm_score > 0.6
        
        push!(signature_matches, (
            mode_index = result.mode_index,
            matches_mm = matches_mm,
            mm_score = mm_score,
            confidence = mm_score,
            dominant_process = result.dominant_process,
            interpretation = result.interpretation
        ))
        
        status = matches_mm ? "✓ MATCHES" : "⚠ PARTIAL"
        println("Mode $(result.mode_index): $status MM signature")
        println("  MM Score: $(round(mm_score, digits=3))/1.0")
        println("  Key processes: Product $(round(product_pct, digits=1))%, Binding $(round(binding_pct, digits=1))%, Dissociation $(round(dissoc_pct, digits=1))%")
    end
    
    # Overall assessment
    mm_modes = filter(m -> m.matches_mm, signature_matches)
    
    if !isempty(mm_modes)
        best_match = mm_modes[1]
        for mode in mm_modes
            if mode.confidence > best_match.confidence
                best_match = mode
            end
        end
        
        println("\n✓ MM mechanism detected!")
        println("  Best mode: $(best_match.mode_index) (confidence: $(round(best_match.confidence, digits=3)))")
        println("  This confirms classical Michaelis-Menten kinetics")
    else
        println("\n⚠ No strong MM signature detected")
    end
    
    return signature_matches
end

"""
    analyze_flow_timescales(flow_modes)

Analyze different timescales in the flow modes.
"""
function analyze_flow_timescales(flow_modes)
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
    run_flow_analysis(data_dict)

Run complete flow analysis on DMD results.
"""
function run_flow_analysis(data_dict)
    println("\n" * "="^50)
    println("FLOW FIELD ANALYSIS")
    println("="^50)
    
    # Check if DMD results are available
    if !haskey(data_dict, "eigenvalues") || !haskey(data_dict, "modes")
        println("ERROR: DMD results not found. Run DMD analysis first.")
        return data_dict
    end
    
    # Extract DMD data
    eigenvalues = data_dict["eigenvalues"]
    modes = data_dict["modes"]
    selected_states = data_dict["selected_states"]
    species_names = data_dict["species_names"]
    
    # Compute flow modes
    flow_modes = compute_flow_modes(eigenvalues, modes, selected_states)
    
    # Compute state importance
    importance_scores = compute_flow_importance(flow_modes, selected_states)
    
    # MM-specific analysis
    mm_analysis = analyze_mm_flow_patterns(flow_modes, selected_states, species_names)
    mm_signature = detect_mm_signature(mm_analysis)
    
    # Timescale analysis
    timescale_analysis = analyze_flow_timescales(flow_modes)
    
    # Add results to data dictionary
    data_dict["flow_modes"] = flow_modes
    data_dict["state_importance"] = importance_scores
    data_dict["mm_flow_analysis"] = mm_analysis
    data_dict["mm_signature"] = mm_signature
    data_dict["timescale_analysis"] = timescale_analysis
    
    println("\nFlow analysis completed!")
    return data_dict
end

println("Clean Flow Analysis Module Loaded! 🌊")
println("Functions:")
println("  compute_flow_modes(eigenvalues, modes, states)")
println("  analyze_mm_flow_patterns(flow_modes, states, species)")
println("  detect_mm_signature(mm_analysis)")
println("  run_flow_analysis(data_dict)")
println()
println("Flow analysis provides:")
println("  • Spatiotemporal flow field computation")
println("  • MM-specific pattern detection")
println("  • Timescale separation analysis")
println("  • State importance scoring")
