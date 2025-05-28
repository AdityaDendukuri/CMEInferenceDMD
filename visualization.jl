# visualization.jl - FLOW FIELD VISUALIZATION FOR PRESENTATIONS
# Professional visualizations for method explanation and results

using Plots
using Statistics
using LinearAlgebra
using Colors
using ColorSchemes
using LaTeXStrings

"""
    plot_reachability_matrix(R, selected_states, species_names; save_path=nothing)

Visualize the reachability matrix structure.
"""
function plot_reachability_matrix(R, selected_states, species_names; save_path=nothing)
    n_display = min(50, size(R, 1))  # Show subset for clarity
    R_display = R[1:n_display, 1:n_display]
    
    p = heatmap(R_display, 
                xlabel="From State Index",
                ylabel="To State Index", 
                title="Reachability Matrix Structure\n(Confidence-Weighted)",
                colorbar_title="Transition Confidence",
                aspect_ratio=:equal,
                size=(600, 600),
                dpi=300)
    
    # Add annotations for interpretation  
    annotate!(n_display*0.7, n_display*0.1, 
             text("Red: High Confidence\nYellow: Medium\nBlue: Low/Blocked", 10, :black))
    
    if save_path !== nothing
        savefig(p, save_path)
        println("Saved reachability matrix to $save_path")
    end
    
    return p
end

"""
    plot_flow_field_2d_projection(flow_modes, selected_states, species_names, 
                                 mode_idx=1, projection=[1,3]; save_path=nothing)

Create 2D projection of flow field for a specific mode.
"""
function plot_flow_field_2d_projection(flow_modes, selected_states, species_names, 
                                      mode_idx=1, projection=[1,3]; save_path=nothing)
    
    if mode_idx > length(flow_modes)
        error("Mode index $mode_idx exceeds available modes ($(length(flow_modes)))")
    end
    
    mode = flow_modes[mode_idx]
    
    # Extract coordinates for projection
    x_coords = [max(0, state[projection[1]]-1) for state in selected_states]  # Molecular counts
    y_coords = [max(0, state[projection[2]]-1) for state in selected_states]
    flow_magnitude = mode.flow_magnitude
    
    # Create size and color based on flow magnitude
    max_flow = maximum(flow_magnitude)
    if max_flow > 0
        normalized_flow = flow_magnitude ./ max_flow
        sizes = 2 .+ 8 .* normalized_flow  # Size 2-10 pixels
        colors = [get(ColorSchemes.plasma, f) for f in normalized_flow]
    else
        sizes = fill(3, length(flow_magnitude))
        colors = fill(colorant"blue", length(flow_magnitude))
    end
    
    # Create scatter plot
    p = scatter(x_coords, y_coords,
                markersize=sizes,
                markercolor=colors,
                markerstrokewidth=0.5,
                markerstrokecolor=:black,
                xlabel=L"%$(species_names[projection[1]]) \text{ Count}",
                ylabel=L"%$(species_names[projection[2]]) \text{ Count}",
                title="Flow Field Mode $mode_idx: $(mode.mode_type)\n" * 
                      L"\lambda = %$(round(mode.eigenvalue, digits=3)), \tau = %$(round(mode.decay_time, digits=1))",
                legend=false,
                grid=true,
                gridwidth=1,
                gridcolor=:lightgray,
                background_color=:white,
                size=(700, 600),
                dpi=300)
    
    # Add flow vectors for high-magnitude states
    top_flow_indices = sortperm(flow_magnitude, rev=true)[1:min(20, length(flow_magnitude))]
    
    for idx in top_flow_indices
        if flow_magnitude[idx] > 0.3 * max_flow && mode.eigenvalue isa Number
            x, y = x_coords[idx], y_coords[idx]
            
            # Flow direction based on eigenvalue phase
            if isa(mode.eigenvalue, Complex)
                angle = Base.angle(mode.eigenvalue)
                dx = 2 * cos(angle) * normalized_flow[idx]
                dy = 2 * sin(angle) * normalized_flow[idx]
            else
                # Real eigenvalue - radial flow
                center_x, center_y = mean(x_coords), mean(y_coords)
                dx = 0.5 * sign(real(mode.eigenvalue)) * (x - center_x) / max(1, abs(x - center_x))
                dy = 0.5 * sign(real(mode.eigenvalue)) * (y - center_y) / max(1, abs(y - center_y))
            end
            
            # Add arrow
            plot!([x, x+dx], [y, y+dy], 
                  arrow=true, arrowsize=0.5, 
                  color=:red, linewidth=2, alpha=0.7)
        end
    end
    
    # Add interpretation text
    interpretation = get_flow_interpretation(mode)
    annotate!(maximum(x_coords)*0.02, maximum(y_coords)*0.95, 
             text(interpretation, 9, :darkblue, :left))
    
    if save_path !== nothing
        savefig(p, save_path)
        println("Saved flow field projection to $save_path")
    end
    
    return p
end

"""
    plot_eigenvalue_spectrum_complex(eigenvalues; save_path=nothing)

Plot eigenvalue spectrum in complex plane with flow interpretation.
"""
function plot_eigenvalue_spectrum_complex(eigenvalues; save_path=nothing)
    real_parts = real.(eigenvalues)
    imag_parts = imag.(eigenvalues)
    magnitudes = abs.(eigenvalues)
    
    # Color by magnitude
    colors = [get(ColorSchemes.viridis, m/maximum(magnitudes)) for m in magnitudes]
    sizes = 4 .+ 8 .* magnitudes ./ maximum(magnitudes)
    
    p = scatter(real_parts, imag_parts,
                markersize=sizes,
                markercolor=colors,
                markerstrokecolor=:black,
                markerstrokewidth=0.5,
                xlabel=L"\text{Re}(\lambda)",
                ylabel=L"\text{Im}(\lambda)",
                title="DMD Eigenvalue Spectrum\n(Flow Field Analysis)",
                grid=true,
                aspect_ratio=:equal,
                size=(700, 700),
                dpi=300)
    
    # Add unit circle for reference
    θ = range(0, 2π, length=100)
    plot!(cos.(θ), sin.(θ), 
          linestyle=:dash, 
          linecolor=:gray, 
          linewidth=2,
          label="Unit Circle",
          alpha=0.7)
    
    # Add stability regions
    plot!([-2, 0], [0, 0], linecolor=:red, linewidth=3, alpha=0.7, label="Decay")
    plot!([0, 2], [0, 0], linecolor=:orange, linewidth=3, alpha=0.7, label="Growth")
    
    # Annotate important eigenvalues
    important_indices = sortperm(magnitudes, rev=true)[1:min(5, length(eigenvalues))]
    for (i, idx) in enumerate(important_indices)
        if i <= 3
            annotate!(real_parts[idx], imag_parts[idx], 
                     text(L"\lambda_%$i", 12, :red, :center))
        end
    end
    
    # Add interpretation regions
    annotate!(-1.5, 1.0, text("Stable Decay\n(Relaxation)", 10, :darkgreen, :center))
    annotate!(1.5, 1.0, text("Unstable Growth\n(Instability)", 10, :darkred, :center))
    annotate!(0, -1.5, text("Oscillatory\n(Complex Dynamics)", 10, :darkblue, :center))
    
    if save_path !== nothing
        savefig(p, save_path)
        println("Saved eigenvalue spectrum to $save_path")
    end
    
    return p
end

"""
    plot_flow_magnitude_evolution(flow_modes, selected_states; save_path=nothing)

Plot how flow magnitude evolves across different modes (timescales).
"""
function plot_flow_magnitude_evolution(flow_modes, selected_states; save_path=nothing)
    n_modes = min(6, length(flow_modes))
    n_states_display = min(100, length(selected_states))
    
    # Create matrix of flow magnitudes
    flow_matrix = zeros(n_states_display, n_modes)
    mode_labels = String[]
    
    for (i, mode) in enumerate(flow_modes[1:n_modes])
        flow_matrix[:, i] = mode.flow_magnitude[1:n_states_display]
        τ = mode.decay_time < Inf ? "τ=$(round(mode.decay_time, digits=1))" : "τ=∞"
        push!(mode_labels, "Mode $i\n$τ")
    end
    
    p = heatmap(mode_labels, 1:n_states_display, flow_matrix',
                xlabel="DMD Modes (by timescale)",
                ylabel="State Index",
                title="Flow Magnitude Evolution\nAcross Timescales",
                colorbar_title="Flow Magnitude |λφ|",
                color=:hot,
                size=(800, 600),
                dpi=300)
    
    # Add timescale interpretation
    annotate!(1, n_states_display*0.9, 
             text("Fastest\nDynamics", 9, :white, :center))
    annotate!(n_modes, n_states_display*0.9, 
             text("Slowest\nDynamics", 9, :white, :center))
    
    if save_path !== nothing
        savefig(p, save_path)
        println("Saved flow evolution to $save_path")
    end
    
    return p
end

"""
    plot_mm_mechanism_signature(mm_analysis; save_path=nothing)

Visualize MM mechanism signature from flow analysis.
"""
function plot_mm_mechanism_signature(mm_analysis; save_path=nothing)
    if mm_analysis === nothing || isempty(mm_analysis)
        println("No MM analysis data available")
        return nothing
    end
    
    # Extract data from first (dominant) mode
    result = mm_analysis[1]
    
    processes = ["Substrate\nBinding", "Complex\nDissociation", 
                "Product\nFormation", "Enzyme\nRecycling", "Substrate\nDepletion"]
    
    percentages = [
        result.percentages["substrate_binding"],
        result.percentages["complex_dissociation"], 
        result.percentages["product_formation"],
        result.percentages["enzyme_recycling"],
        result.percentages["substrate_depletion"]
    ]
    
    # Create bar chart with MM color scheme
    colors = [:red, :orange, :green, :blue, :purple]
    
    p = bar(processes, percentages,
            color=colors,
            xlabel="MM Process Type",
            ylabel="Flow Percentage (%)",
            title="Michaelis-Menten Mechanism Signature\n" *
                  "Mode $(result.mode_index): $(result.mode_type)",
            legend=false,
            size=(900, 600),
            dpi=300,
            xrotation=45)
    
    # Add percentage labels on bars
    for (i, pct) in enumerate(percentages)
        if pct > 5  # Only label significant percentages
            annotate!(i, pct + 2, text("$(round(pct, digits=1))%", 10, :black, :center))
        end
    end
    
    # Highlight dominant process
    dominant_idx = argmax(percentages)
    bar!([dominant_idx], [percentages[dominant_idx]], 
         color=:gold, 
         linewidth=3, 
         linecolor=:black)
    
    # Add interpretation
    annotate!(length(processes)*0.7, maximum(percentages)*0.8,
             text("Dominant: $(result.dominant_process)\n" *
                  "Interpretation: $(result.interpretation)", 
                  10, :darkblue, :left))
    
    if save_path !== nothing
        savefig(p, save_path)
        println("Saved MM signature to $save_path")
    end
    
    return p
end

"""
    plot_spurious_elimination_comparison(unconstrained_reactions, constrained_reactions, species_names; save_path=nothing)

Compare unconstrained vs constrained DMD results.
"""
function plot_spurious_elimination_comparison(unconstrained_reactions, constrained_reactions, species_names; save_path=nothing)
    # Define spurious and valid reaction patterns
    spurious_patterns = [
        ([-1, 0, 0, 1], "S → P"),
        ([1, 0, 0, -1], "P → S"),
        ([0, -1, 1, -1], "E + P → SE")
    ]
    
    valid_patterns = [
        ([-1, -1, 1, 0], "S + E → SE"),
        ([1, 1, -1, 0], "SE → S + E"),
        ([0, 1, -1, 1], "SE → E + P")
    ]
    
    # Count occurrences
    unconstrained_spurious = 0
    unconstrained_valid = 0
    constrained_spurious = 0
    constrained_valid = 0
    
    for (stoich, name) in spurious_patterns
        if tuple(stoich...) in unconstrained_reactions
            unconstrained_spurious += 1
        end
        if tuple(stoich...) in constrained_reactions
            constrained_spurious += 1
        end
    end
    
    for (stoich, name) in valid_patterns
        if tuple(stoich...) in unconstrained_reactions
            unconstrained_valid += 1
        end
        if tuple(stoich...) in constrained_reactions
            constrained_valid += 1
        end
    end
    
    # Create comparison plot
    methods = ["Unconstrained\nDMD", "Reachability-\nConstrained DMD"]
    valid_counts = [unconstrained_valid, constrained_valid]
    spurious_counts = [unconstrained_spurious, constrained_spurious]
    
    p = groupedbar([valid_counts spurious_counts],
                   bar_position=:dodge,
                   labels=["Valid MM Reactions" "Spurious Reactions"],
                   color=[:green :red],
                   xlabel="Method",
                   ylabel="Number of Reactions",
                   title="Spurious Reaction Elimination\nReachability Masking Effectiveness",
                   xticks=(1:2, methods),
                   size=(700, 500),
                   dpi=300)
    
    # Add value labels
    for i in 1:2
        annotate!(i-0.2, valid_counts[i] + 0.1, text("$(valid_counts[i])", 10, :black, :center))
        annotate!(i+0.2, spurious_counts[i] + 0.1, text("$(spurious_counts[i])", 10, :black, :center))
    end
    
    if save_path !== nothing
        savefig(p, save_path)
        println("Saved comparison plot to $save_path")
    end
    
    return p
end

"""
    get_flow_interpretation(mode)

Get biological interpretation of flow mode.
"""
function get_flow_interpretation(mode)
    if mode.decay_time < 5
        return "Fast dynamics:\nRapid equilibration"
    elseif mode.decay_time < 20
        return "Medium dynamics:\nCatalytic conversion"
    else
        return "Slow dynamics:\nSubstrate depletion"
    end
end

"""
    create_presentation_plots(results; output_dir="presentation_plots/")

Generate all plots needed for presentation.
"""
function create_presentation_plots(results; output_dir="presentation_plots/")
    mkpath(output_dir)
    
    println("Creating presentation plots in $output_dir...")
    
    plots_created = []
    
    # 1. Reachability matrix visualization
    if haskey(results, "reachability_info") && haskey(results["reachability_info"], "reachability_matrix")
        R = results["reachability_info"]["reachability_matrix"]
        if R !== nothing
            plot_reachability_matrix(R, results["selected_states"], results["species_names"],
                                   save_path=joinpath(output_dir, "reachability_matrix.png"))
            push!(plots_created, "reachability_matrix.png")
        end
    end
    
    # 2. Flow field projections
    if haskey(results, "flow_modes") && !isempty(results["flow_modes"])
        flow_modes = results["flow_modes"]
        
        # S vs SE projection (main reaction coordinate)
        plot_flow_field_2d_projection(flow_modes, results["selected_states"], results["species_names"],
                                     1, [1,3], save_path=joinpath(output_dir, "flow_field_S_SE.png"))
        push!(plots_created, "flow_field_S_SE.png")
        
        # E vs P projection (enzyme-product coordinate)
        plot_flow_field_2d_projection(flow_modes, results["selected_states"], results["species_names"],
                                     1, [2,4], save_path=joinpath(output_dir, "flow_field_E_P.png"))
        push!(plots_created, "flow_field_E_P.png")
        
        # Flow magnitude evolution
        plot_flow_magnitude_evolution(flow_modes, results["selected_states"],
                                    save_path=joinpath(output_dir, "flow_evolution.png"))
        push!(plots_created, "flow_evolution.png")
    end
    
    # 3. Eigenvalue spectrum
    if haskey(results, "eigenvalues")
        plot_eigenvalue_spectrum_complex(results["eigenvalues"],
                                       save_path=joinpath(output_dir, "eigenvalue_spectrum.png"))
        push!(plots_created, "eigenvalue_spectrum.png")
    end
    
    # 4. MM mechanism signature
    if haskey(results, "mm_flow_analysis") && results["mm_flow_analysis"] !== nothing
        plot_mm_mechanism_signature(results["mm_flow_analysis"],
                                   save_path=joinpath(output_dir, "mm_signature.png"))
        push!(plots_created, "mm_signature.png")
    end
    
    println("Created $(length(plots_created)) presentation plots:")
    for plot_name in plots_created
        println("  ✓ $plot_name")
    end
    
    return plots_created
end

println("Presentation Visualization Module Loaded! 🎨")
println("Functions:")
println("  plot_reachability_matrix(R, states, species)")
println("  plot_flow_field_2d_projection(modes, states, species, mode_idx, projection)")
println("  plot_eigenvalue_spectrum_complex(eigenvalues)")
println("  plot_mm_mechanism_signature(mm_analysis)")
println("  create_presentation_plots(results)")
println()
println("Usage for your results:")
println("  create_presentation_plots(results)")
println("This will generate all plots needed for the Beamer presentation!")
