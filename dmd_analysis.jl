# dmd_analysis.jl
# Core implementation of Dynamic Mode Decomposition (DMD) for CRN inference

using LinearAlgebra
using SparseArrays

"""
    apply_dmd(reduced_data, dt; svd_rank_threshold=1e-10)

Apply Dynamic Mode Decomposition to reduced probability data to extract
the underlying generator matrix and its spectral properties.

# Arguments
- `reduced_data`: Matrix of reduced probability data (states × timepoints)
- `dt`: Time step between snapshots
- `svd_rank_threshold`: Threshold for singular value truncation (relative to largest sv)

# Returns
- G: Generator matrix
- λ: Eigenvalues
- Φ: Eigenvectors (DMD modes)
- A: DMD operator
- r: Effective rank used
"""
function apply_dmd(reduced_data, dt; svd_rank_threshold=1e-10)
    # Form data matrices X and X'
    X = reduced_data[:, 1:end-1]
    Xp = reduced_data[:, 2:end]
    
    println("Data matrix dimensions: X$(size(X)), Xp$(size(Xp))")
    
    # Apply DMD via SVD
    println("Computing SVD for DMD...")
    U, Σ, V = svd(X)
    
    # Print singular value decay for diagnostics
    println("Singular value decay: ")
    for i in 1:min(10, length(Σ))
        println("σ$i = $(Σ[i]), ratio to σ1: $(Σ[i]/Σ[1])")
    end
    
    # Determine rank based on singular value threshold
    r = sum(Σ ./ Σ[1] .> svd_rank_threshold)
    r = min(r, size(X, 2))  # Cannot exceed matrix dimensions
    
    println("Using DMD rank: $r (out of $(length(Σ)) singular values)")
    
    # Truncate SVD to keep r components
    U_r = U[:, 1:r]
    Σ_r = Diagonal(Σ[1:r])
    V_r = V[:, 1:r]
    
    # Compute DMD operator with improved numerical stability
    A_tilde = U_r' * Xp * V_r * inv(Σ_r)
    
    # Compute eigendecomposition of A_tilde
    λ, W = eigen(A_tilde)
    
    # DMD modes
    Φ = U_r * W
    
    # Full DMD operator
    A = U_r * A_tilde * U_r'
    
    # CME generator (approximation)
    G = (A - I) / dt
    
    return G, λ, Φ, A, r
end

"""
    calculate_spectral_distance(G1, G2)

Calculate the distance between two generator matrices based on their
dominant eigenvalues and eigenvectors.

# Arguments
- `G1`: First generator matrix
- `G2`: Second generator matrix

# Returns
- Spectral distance measure
"""
function calculate_spectral_distance(G1, G2)
    # Compute eigendecomposition of both matrices
    λ1, V1 = eigen(Matrix(G1))
    λ2, V2 = eigen(Matrix(G2))
    
    # Sort eigenvalues by absolute value (largest first)
    idx1 = sortperm(abs.(λ1), rev=true)
    idx2 = sortperm(abs.(λ2), rev=true)
    
    # Select top k eigenvalues (excluding steady state)
    k = min(10, length(λ1), length(λ2))
    
    # Compute eigenvalue distance (skip the first which should be ~0)
    eig_distance = sum(abs.(λ1[idx1[2:k]] - λ2[idx2[2:k]])) / (k-1)
    
    # Add a small penalty for differences in matrix structure
    frobenius_norm = norm(G1 - G2) / sqrt(length(G1))
    
    # Combined distance metric (weighted sum)
    distance = 0.7 * eig_distance + 0.3 * frobenius_norm
    
    return distance
end

"""
    identify_conservation_laws(G, species_names; tol=1e-8)

Identify conservation laws from the generator matrix by finding 
left eigenvectors corresponding to zero eigenvalues.

# Arguments
- `G`: Generator matrix
- `species_names`: Names of species
- `tol`: Tolerance for identifying zero eigenvalues

# Returns
- Array of conservation law vectors
- Array of conservation law descriptions
"""
function identify_conservation_laws(G, species_names; tol=1e-8)
    # Compute eigendecomposition of G
    λ, V_right = eigen(Matrix(G))
    V_left = inv(V_right)  # Left eigenvectors (rows of V_left)
    
    # Find eigenvalues close to zero
    zero_indices = findall(abs.(λ) .< tol)
    
    if isempty(zero_indices)
        println("No conservation laws found (no eigenvalues close to zero)")
        return [], []
    end
    
    # Extract left eigenvectors corresponding to zero eigenvalues
    conservation_laws = []
    law_descriptions = []
    
    for idx in zero_indices
        # Get the left eigenvector (row of V_left)
        left_ev = V_left[idx, :]
        
        # Normalize the coefficients
        max_coef = maximum(abs.(left_ev))
        if max_coef > 0
            left_ev = left_ev ./ max_coef
        end
        
        # Round small values to zero for clarity
        left_ev[abs.(left_ev) .< 1e-10] .= 0.0
        
        # Create a description of the conservation law
        description = "Conservation law: "
        terms = []
        for (i, coef) in enumerate(left_ev)
            if abs(coef) > 1e-10
                if length(species_names) >= i
                    species = species_names[i]
                    push!(terms, "$(round(coef, digits=3)) × $species")
                else
                    push!(terms, "$(round(coef, digits=3)) × species$i")
                end
            end
        end
        description *= join(terms, " + ") * " = constant"
        
        push!(conservation_laws, left_ev)
        push!(law_descriptions, description)
    end
    
    return conservation_laws, law_descriptions
end

"""
    create_spectral_reconstruction(G, selected_stoich, grouped_reactions, selected_states)

Create a reconstructed generator matrix using only selected reaction stoichiometries.

# Arguments
- `G`: Original generator matrix
- `selected_stoich`: List of selected reaction stoichiometries
- `grouped_reactions`: Dictionary mapping stoichiometries to reactions
- `selected_states`: List of states in the reduced space

# Returns
- Reconstructed generator matrix
"""
function create_spectral_reconstruction(G, selected_stoich, grouped_reactions, selected_states)
    # Initialize reconstruction with zeros
    G_recon = zeros(size(G))
    
    # Populate reconstruction with selected reactions
    for stoich in selected_stoich
        if haskey(grouped_reactions, stoich)
            rxns = grouped_reactions[stoich]
            
            for r in rxns
                # Get from and to indices
                from_idx = findfirst(s -> all(s .== r.from_state), selected_states)
                to_idx = findfirst(s -> all(s .== r.to_state), selected_states)
                
                if from_idx !== nothing && to_idx !== nothing
                    # Copy the corresponding rate from original generator
                    G_recon[to_idx, from_idx] = G[to_idx, from_idx]
                end
            end
        end
    end
    
    # Fix diagonal elements to ensure proper generator structure
    for i in 1:size(G_recon, 1)
        G_recon[i, i] = -sum(G_recon[:, i])
    end
    
    return G_recon
end
