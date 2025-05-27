# mm_specific_analysis.jl - MICHAELIS-MENTEN SPECIFIC ANALYSIS MODULE
# MM-specific flow field analysis and mechanism detection

using Statistics

"""
    categorize_mm_processes(flow_modes, selected_states, species_names)

Analyze MM-specific kinetic processes in flow modes.
Requires exactly 4 species: [S, E, SE, P]
"""
function categorize_mm_processes(flow_modes, selected_states, species_names)
    if length(species_names) != 4 || isempty(flow_modes)
        println("MM analysis requires exactly 4 species [S, E, SE, P]")
        return nothing
    end
    
    println("\n=== MM Process Categorization ===")
    
    mode_results = []
    
    for mode in flow_modes[1:min(3, length(flow_modes))]
        flow_magnitude = mode.flow_magnitude
        n_valid = min(length(flow_magnitude), length(selected_states))
        
        # MM-specific process counters
        substrate_binding_flow = 0.0      # S + E → SE
        complex_dissociation_flow = 0.0   # SE → S + E
        product_formation_flow = 0.0      # SE → E + P
        enzyme_recycling_flow = 0.0       # Free enzyme states
        substrate_depletion_flow = 0.0    # Low substrate states
        
        for i in 1:n_valid
            state = selected_states[i]
            if length(state) >= 4
                s, e, se, p = [max(0, x-1) for x in state[1:4]]  # Convert to molecular counts
                flow = flow_magnitude[i]
                
                total_substrate = s + se + p  # Conservation of substrate material
                total_enzyme = e + se         # Conservation of enzyme
                
                # Categorize states by MM process signatures
                
                # Substrate binding: High S, E available, moderate SE
                if s > 10 && e > 2 && se < 8
                    substrate_binding_flow += flow
                end
                
                # Complex dissociation: High SE, products of dissociation
                if se > 3 && (s > 0 || e > 0)
                    complex_dissociation_flow += flow
                end
                
                # Product formation: SE conversion to P, enzyme release
                if p > s && e > se && total_substrate > 20
                    product_formation_flow += flow
                end
                
                # Enzyme recycling: High free enzyme, low complex
                if e > 5 && se < 3
                    enzyme_recycling_flow += flow
                end
                
                # Substrate depletion: Low remaining substrate
                if s < 5 && total_substrate > 20
                    substrate_depletion_flow += flow
                end
            end
        end
        
        total_flow = sum(flow_magnitude[1:n_valid])
        
        # Calculate percentages
        substrate_bind_pct = total_flow > 0 ? (substrate_binding_flow/total_flow*100) : 0.0
        complex_diss_pct = total_flow > 0 ? (complex_dissociation_flow/total_flow*100) : 0.0
        product_form_pct = total_flow > 0 ? (product_formation_flow/total_flow*100) : 0.0
        enzyme_recyc_pct = total_flow > 0 ? (enzyme_recycling_flow/total_flow*100) : 0.0
        substrate_depl_pct = total_flow > 0 ? (substrate_depletion_flow/total_flow*100) : 0.0
        
        # Find dominant process
        processes = [
            ("substrate_binding", substrate_bind_pct),
            ("complex_dissociation", complex_diss_pct),
            ("product_formation", product_form_pct),
            ("enzyme_recycling", enzyme_recyc_pct),
            ("substrate_depletion", substrate_depl_pct)
        ]
        
        dominant_process, dominant_value = processes[1]
        for (process, value) in processes
            if value > dominant_value
                dominant_process = process
                dominant_value = value
            end
        end
        
        println("\nMode $(mode.mode_index) [$(mode.mode_type)]:")
        println("  Eigenvalue: $(round(mode.eigenvalue, digits=4))")
        println("  Decay time: $(round(mode.decay_time, digits=2)) time units")
        if mode.oscillation_period < Inf
            println("  Oscillation period: $(round(mode.oscillation_period, digits=2)) time units")
        end
        
        println("  MM Process Distribution:")
        println("    Substrate binding (S+E→SE): $(round(substrate_bind_pct, digits=1))%")
        println("    Complex dissociation (SE→S+E): $(round(complex_diss_pct, digits=1))%")
        println("    Product formation (SE→E+P): $(round(product_form_pct, digits=1))%")
        println("    Enzyme recycling: $(round(enzyme_recyc_pct, digits=1))%")
        println("    Substrate depletion: $(round(substrate_depl_pct, digits=1))%")
        println("  Dominant process: $dominant_process ($(round(dominant_value, digits=1))%)")
        
        # Biological interpretation
        interpretation = ""
        if dominant_process == "substrate_binding"
            interpretation = "Mode captures initial substrate-enzyme encounter"
        elseif dominant_process == "complex_dissociation"
            interpretation = "Mode captures SE complex equilibration"
        elseif dominant_process == "product_formation"
            interpretation = "Mode captures catalytic conversion"
        elseif dominant_process == "enzyme_recycling"
            interpretation = "Mode captures enzyme turnover cycle"
        else
            interpretation = "Mode captures substrate consumption dynamics"
        end
        println("  → $interpretation")
        
        # Store results
        push!(mode_results, (
            mode_index = mode.mode_index,
            mode_type = mode.mode_type,
            eigenvalue = mode.eigenvalue,
            decay_time = mode.decay_time,
            oscillation_period = mode.oscillation_period,
            dominant_process = dominant_process,
            dominant_percentage = dominant_value,
            interpretation = interpretation,
            percentages = Dict(
                "substrate_binding" => substrate_bind_pct,
                "complex_dissociation" => complex_diss_pct,
                "product_formation" => product_form_pct,
                "enzyme_recycling" => enzyme_recyc_pct,
                "substrate_depletion" => substrate_depl_pct
            )
        ))
    end
    
    return mode_results
end

"""
    detect_mm_mechanism_signature(flow_modes, selected_states, species_names)

Test if flow patterns match expected MM mechanism signatures.
"""
function detect_mm_mechanism_signature(flow_modes, selected_states, species_names)
    if length(species_names) != 4 || isempty(flow_modes)
        return nothing
    end
    
    println("\n=== MM Mechanism Signature Detection ===")
    
    # Get MM process analysis
    mm_processes = categorize_mm_processes(flow_modes, selected_states, species_names)
    
    if mm_processes === nothing
        return nothing
    end
    
    mm_signature_matches = []
    
    for result in mm_processes
        percentages = result.percentages
        
        # Classic MM signature expectations:
        # 1. Product formation should be significant (>30%)
        # 2. Either substrate binding or complex dissociation should be present
        # 3. Enzyme recycling should be detectable
        
        product_pct = percentages["product_formation"]
        binding_pct = percentages["substrate_binding"]
        dissoc_pct = percentages["complex_dissociation"]
        recycling_pct = percentages["enzyme_recycling"]
        
        # Calculate MM signature score
        mm_score = 0.0
        
        # Product formation is key for MM
        if product_pct > 30
            mm_score += 0.4 * (product_pct / 100)
        end
        
        # Substrate binding indicates initial step
        if binding_pct > 20
            mm_score += 0.3 * (binding_pct / 100)
        end
        
        # Complex dissociation indicates reversibility
        if dissoc_pct > 15
            mm_score += 0.2 * (dissoc_pct / 100)
        end
        
        # Enzyme recycling indicates turnover
        if recycling_pct > 10
            mm_score += 0.1 * (recycling_pct / 100)
        end
        
        # Penalty for unbalanced patterns
        if product_pct < 10 && dissoc_pct > 60
            mm_score *= 0.5  # Probably just SE equilibration
        end
        
        matches_mm = mm_score > 0.6
        confidence = mm_score
        
        push!(mm_signature_matches, (
            mode_index = result.mode_index,
            matches_mm = matches_mm,
            mm_score = mm_score,
            confidence = confidence,
            dominant_process = result.dominant_process,
            interpretation = result.interpretation
        ))
        
        status = matches_mm ? "✓ MATCHES" : "⚠ PARTIAL"
        println("Mode $(result.mode_index): $status MM signature")
        println("  MM Score: $(round(mm_score, digits=3))/1.0")
        println("  Key processes: Product $(round(product_pct, digits=1))%, Binding $(round(binding_pct, digits=1))%, Dissociation $(round(dissoc_pct, digits=1))%")
        println("  $(result.interpretation)")
    end
    
    # Overall MM assessment
    println("\n=== Overall MM Assessment ===")
    mm_modes = filter(m -> m.matches_mm, mm_signature_matches)
    
    if !isempty(mm_modes)
        # Find best match manually
        best_match = mm_modes[1]
        for mode in mm_modes
            if mode.confidence > best_match.confidence
                best_match = mode
            end
        end
        
        println("✓ MM mechanism detected!")
        println("  Best mode: $(best_match.mode_index) (confidence: $(round(best_match.confidence, digits=3)))")
        println("  Dominant process: $(best_match.dominant_process)")
        println("  This confirms classical Michaelis-Menten kinetics")
        
        # Additional insights
        if best_match.dominant_process == "product_formation"
            println("  → System shows strong catalytic conversion signature")
        elseif best_match.dominant_process == "complex_dissociation"
            println("  → System shows SE equilibration signature (fast pre-equilibrium)")
        elseif best_match.dominant_process == "substrate_binding"
            println("  → System shows substrate encounter-limited kinetics")
        end
        
    else
        println("⚠ No strong MM signature detected")
        println("  Possible explanations:")
        println("  - System in different kinetic regime (pre-equilibrium, product inhibition)")
        println("  - Multiple competing pathways")
        println("  - Insufficient temporal resolution")
        
        # Show what was found instead
        if !isempty(mm_signature_matches)
            # Find best partial match manually
            best_partial = mm_signature_matches[1]
            for match in mm_signature_matches
                if match.confidence > best_partial.confidence
                    best_partial = match
                end
            end
            println("  Best partial match: Mode $(best_partial.mode_index) ($(best_partial.dominant_process))")
        end
    end
    
    return mm_signature_matches
end

"""
    analyze_mm_timescales(flow_modes, selected_states, species_names)

Analyze different timescales in MM reaction (binding, catalysis, release).
"""
function analyze_mm_timescales(flow_modes, selected_states, species_names)
    if length(species_names) != 4 || isempty(flow_modes)
        return nothing
    end
    
    println("\n=== MM Timescale Analysis ===")
    
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
    println("  Fast modes (τ < 5): $(length(fast_modes)) - typically SE binding/unbinding")
    println("  Medium modes (5 ≤ τ < 20): $(length(medium_modes)) - typically catalytic conversion")
    println("  Slow modes (τ ≥ 20): $(length(slow_modes)) - typically substrate depletion")
    
    # Analyze dominant process for each timescale
    timescale_processes = Dict()
    
    for (timescale, mode_list) in [("Fast", fast_modes), ("Medium", medium_modes), ("Slow", slow_modes)]
        if !isempty(mode_list)
            # Get the mode with highest flow magnitude manually
            dominant_mode = mode_list[1]
            max_flow = maximum(dominant_mode.flow_magnitude)
            
            for mode in mode_list
                mode_max_flow = maximum(mode.flow_magnitude)
                if mode_max_flow > max_flow
                    dominant_mode = mode
                    max_flow = mode_max_flow
                end
            end
            
            # Analyze this mode's MM processes
            mm_analysis = categorize_mm_processes([dominant_mode], selected_states, species_names)
            
            if mm_analysis !== nothing && !isempty(mm_analysis)
                result = mm_analysis[1]
                timescale_processes[timescale] = result
                
                println("\n$timescale Timescale (Mode $(dominant_mode.mode_index)):")
                println("  Decay time: $(round(dominant_mode.decay_time, digits=2)) time units")
                println("  Dominant process: $(result.dominant_process)")
                println("  Biological role: $(result.interpretation)")
            end
        end
    end
    
    # MM kinetic interpretation
    println("\n=== MM Kinetic Interpretation ===")
    
    if haskey(timescale_processes, "Fast") && haskey(timescale_processes, "Slow")
        fast_proc = timescale_processes["Fast"].dominant_process
        slow_proc = timescale_processes["Slow"].dominant_process
        
        if fast_proc == "complex_dissociation" && slow_proc == "product_formation"
            println("✓ Classic MM kinetics detected:")
            println("  Fast: SE ⇌ S + E equilibration")
            println("  Slow: SE → E + P conversion (rate-limiting)")
            println("  This matches the pre-equilibrium approximation!")
            
        elseif fast_proc == "substrate_binding" && slow_proc == "substrate_depletion"
            println("✓ Encounter-limited MM kinetics detected:")
            println("  Fast: S + E → SE binding")
            println("  Slow: Substrate consumption")
            println("  This suggests binding-limited regime!")
            
        else
            println("⚠ Non-standard MM kinetics:")
            println("  Fast process: $fast_proc")
            println("  Slow process: $slow_proc")
            println("  This may indicate complex kinetic coupling")
        end
    end
    
    return timescale_processes
end

"""
    complete_mm_analysis(results, species_names)

Run complete MM-specific analysis pipeline.
"""
function complete_mm_analysis(results, species_names)
    println("\n" * "="^60)
    println("COMPLETE MICHAELIS-MENTEN ANALYSIS")
    println("="^60)
    
    # First run general flow analysis
    flow_results = basic_flow_analysis(results, species_names)
    
    if flow_results === nothing
        println("No flow modes available for MM analysis")
        return nothing
    end
    
    flow_modes = flow_results["flow_modes"]
    
    # MM-specific analyses
    println("\n" * "="^40)
    println("MM-SPECIFIC ANALYSES")
    println("="^40)
    
    # 1. Process categorization
    mm_processes = categorize_mm_processes(flow_modes, results["selected_states"], species_names)
    
    # 2. Mechanism signature detection
    mm_signature = detect_mm_mechanism_signature(flow_modes, results["selected_states"], species_names)
    
    # 3. Timescale analysis
    timescale_analysis = analyze_mm_timescales(flow_modes, results["selected_states"], species_names)
    
    # Combine results
    complete_results = merge(flow_results, Dict(
        "mm_processes" => mm_processes,
        "mm_signature" => mm_signature,
        "mm_timescales" => timescale_analysis
    ))
    
    println("\n" * "="^60)
    println("MM ANALYSIS COMPLETE")
    println("="^60)
    
    return complete_results
end

"""
    run_with_mm_analysis(n_trajs=500, max_states=500)

Run complete MM inference with MM-specific flow analysis.
"""
function run_with_mm_analysis(n_trajs=500, max_states=500)
    # Run basic inference
    results = run_mm_inference(n_trajs, max_states)
    
    # Add MM-specific analysis
    mm_results = complete_mm_analysis(results, results["species_names"])
    
    if mm_results !== nothing
        results["mm_flow_analysis"] = mm_results
    end
    
    return results
end

println("MM-Specific Analysis Module Loaded! 🧪")
println("Usage: run_with_mm_analysis(n_trajs, max_states)")
