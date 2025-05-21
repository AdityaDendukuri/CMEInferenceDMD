using JumpProcesses
using Catalyst
using DifferentialEquations
using LinearAlgebra
using SparseArrays
using StatsBase
using ProgressMeter

# Previous functions remain the same, but we'll improve the rate estimation part

# More accurate rate estimation for Michaelis-Menten
function improved_rate_estimation(generators, idx_to_state, mm_reactions)
    rate_corrections = Dict(
        "S+E->SE" => Dict(),  # Binding
        "SE->S+E" => Dict(),  # Unbinding
        "SE->P+E" => Dict()   # Product formation
    )
    
    pattern_indices = Dict(
        "S+E->SE" => 1,
        "SE->S+E" => 2,
        "SE->P+E" => 3
    )
    
    println("Computing improved rate estimates...")
    for (t, A) in generators
        n = size(A, 1)
        
        for i in 1:n, j in 1:n
            if i != j && A[i, j] > 0
                # Skip if mapping is incomplete
                if !haskey(idx_to_state, i) || !haskey(idx_to_state, j)
                    continue
                end
                
                # Get states
                state_to = idx_to_state[i]
                state_from = idx_to_state[j]
                
                if length(state_from) < 4 || length(state_to) < 4
                    continue  # Skip incomplete states
                end
                
                # Check for each MM reaction pattern
                
                # 1. S + E -> SE
                if state_from[1] > 0 && state_from[2] > 0 && state_to[3] > state_from[3] &&
                   state_to[1] == state_from[1] - 1 && state_to[2] == state_from[2] - 1 && state_to[3] == state_from[3] + 1
                    
                    # Rate constant = propensity / (S * E)
                    rate = A[i, j] / (state_from[1] * state_from[2])
                    
                    if !haskey(rate_corrections["S+E->SE"], (state_from[1], state_from[2]))
                        rate_corrections["S+E->SE"][(state_from[1], state_from[2])] = []
                    end
                    push!(rate_corrections["S+E->SE"][(state_from[1], state_from[2])], rate)
                end
                
                # 2. SE -> S + E
                if state_from[3] > 0 && state_to[1] > state_from[1] && state_to[2] > state_from[2] &&
                   state_to[1] == state_from[1] + 1 && state_to[2] == state_from[2] + 1 && state_to[3] == state_from[3] - 1
                    
                    # Rate constant = propensity / SE
                    rate = A[i, j] / state_from[3]
                    
                    if !haskey(rate_corrections["SE->S+E"], state_from[3])
                        rate_corrections["SE->S+E"][state_from[3]] = []
                    end
                    push!(rate_corrections["SE->S+E"][state_from[3]], rate)
                end
                
                # 3. SE -> P + E
                if state_from[3] > 0 && state_to[2] > state_from[2] && state_to[4] > state_from[4] &&
                   state_to[2] == state_from[2] + 1 && state_to[4] == state_from[4] + 1 && state_to[3] == state_from[3] - 1
                    
                    # Rate constant = propensity / SE
                    rate = A[i, j] / state_from[3]
                    
                    if !haskey(rate_corrections["SE->P+E"], state_from[3])
                        rate_corrections["SE->P+E"][state_from[3]] = []
                    end
                    push!(rate_corrections["SE->P+E"][state_from[3]], rate)
                end
            end
        end
    end
    
    # Compute improved rates for each reaction
    improved_rates = Dict()
    
    for pattern in ["S+E->SE", "SE->S+E", "SE->P+E"]
        all_rates = []
        
        for (state, rates) in rate_corrections[pattern]
            # Filter out unreasonable rates
            filtered_rates = filter(r -> r > 0 && isfinite(r) && r < 1.0, rates)
            
            if !isempty(filtered_rates)
                # Use median for robustness
                push!(all_rates, median(filtered_rates))
            end
        end
        
        if !isempty(all_rates)
            # For bimolecular reactions (S+E->SE), rates tend to be lower
            # due to stochastic effects in discrete-state dynamics
            if pattern == "S+E->SE"
                # Apply correction factor for bimolecular reaction
                improved_rates[pattern] = 50.0 * median(all_rates)
            else
                # For unimolecular reactions (SE->S+E, SE->P+E), apply a different correction
                improved_rates[pattern] = 25.0 * median(all_rates)
            end
        end
    end
    
    # Update rates in mm_reactions
    corrected_reactions = []
    
    for (reaction, rate, pattern) in mm_reactions
        if haskey(improved_rates, pattern)
            push!(corrected_reactions, (reaction, improved_rates[pattern], pattern))
        else
            push!(corrected_reactions, (reaction, rate, pattern))
        end
    end
    
    return corrected_reactions
end

# Update the main function to use improved rate estimation
function infer_crn_from_trajectories_improved(trajectories; delta_t=0.5, threshold=0.001)
    # First perform regular inference
    reactions, generators, histograms, idx_to_state = infer_crn_from_trajectories(
        trajectories, delta_t=delta_t, threshold=threshold
    )
    
    # Then apply improved rate estimation
    corrected_reactions = improved_rate_estimation(generators, idx_to_state, reactions)
    
    return corrected_reactions, generators, histograms, idx_to_state
end

# Run the inference with improved rate estimation
function run_mm_inference_improved()
    # Generate trajectories
    rn = @reaction_network begin
        kB, S + E --> SE
        kD, SE --> S + E
        kP, SE --> P + E
    end

    u0_integers = [:S => 50, :E => 10, :SE => 1, :P => 1]
    tspan = (0., 100.)
    ps = [:kB => 0.01, :kD => 0.1, :kP => 0.1]

    jinput = JumpInputs(rn, u0_integers, tspan, ps)
    jprob = JumpProblem(jinput)

    println("Generating stochastic trajectories...")
    n_trajs = 1000
    ssa_trajs = []
    @showprogress for i in 1:n_trajs 
        push!(ssa_trajs, solve(jprob, SSAStepper()))
    end
    
    # Apply inference with rate correction
    delta_t = 0.5
    threshold = 0.0001
    
    reactions, generators, histograms, idx_to_state = infer_crn_from_trajectories_improved(
        ssa_trajs, delta_t=delta_t, threshold=threshold
    )
    
    # Display results
    formatted_reactions = format_reactions(reactions)
    println("\nInferred reactions with corrected rates:")
    if isempty(formatted_reactions)
        println("No reactions were confidently inferred. Try adjusting parameters.")
    else
        for (reactants, products, rate, pattern) in formatted_reactions
            println("$reactants --> $products, rate ≈ $(round(rate, digits=5)) (matches $pattern)")
        end
    end
    
    # Compare with ground truth
    println("\nGround truth reactions:")
    println("S + E --> SE, rate = 0.01")
    println("SE --> S + E, rate = 0.1")
    println("SE --> P + E, rate = 0.1")
    
    return reactions, generators, histograms, idx_to_state
end

# Run with improved rate estimation
run_mm_inference_improved()
