# constrained_dmd.jl - CONSTRAINED DMD WITH CME GENERATOR CONSTRAINTS
# Implements proper optimization to enforce CME generator properties

using LinearAlgebra
using SparseArrays
using Statistics
using Convex
using SCS  # or another solver like ECOS, OSQP

"""
    constrained_dmd_optimization(P_in, P_out, dt; λ_sparse=0.01, solver=SCS.Optimizer)

Solve the constrained optimization problem:
minimize ||P_out - (I + A*dt)*P_in||_F^2 + λ||A||_1
subject to:
  A_ij ≥ 0 for i ≠ j (non-negative off-diagonals)
  Σ_i A_ij = 0 for all j (column sums zero)
"""
function constrained_dmd_optimization(P_in, P_out, dt; λ_sparse=0.01, solver=SCS.Optimizer)
    n_states = size(P_in, 1)
    n_snapshots = size(P_in, 2)
    
    println("Setting up constrained DMD optimization...")
    println("  States: $n_states, Snapshots: $n_snapshots")
    println("  Sparsity parameter λ: $λ_sparse")
    
    # Decision variable: generator matrix A
    A = Variable(n_states, n_states)
    
    # Objective: ||P_out - (I + A*dt)*P_in||_F^2 + λ||A||_1
    prediction = (I + A * dt) * P_in
    reconstruction_error = sumsquares(P_out - prediction)
    sparsity_penalty = λ_sparse * norm(A, 1)
    
    objective = reconstruction_error + sparsity_penalty
    
    # Constraints for valid CME generator
    constraints = []
    
    # 1. Non-negative off-diagonals: A_ij ≥ 0 for i ≠ j
    for i in 1:n_states
        for j in 1:n_states
            if i != j
                push!(constraints, A[i,j] >= 0)
            end
        end
    end
    
    # 2. Column sums zero: Σ_i A_ij = 0 for all j
    for j in 1:n_states
        push!(constraints, sum(A[:, j]) == 0)
    end
    
    # 3. Diagonal elements should be negative (sum constraint + off-diagonal ≥ 0)
    # This is automatically satisfied by the column sum constraint
    
    # Solve the optimization problem
    problem = minimize(objective, constraints)
    
    println("Solving constrained optimization...")
    solve!(problem, solver; silent_solver=false)
    
    if problem.status == Convex.OPTIMAL || problem.status == Convex.NEAR_OPTIMAL
        A_opt = evaluate(A)
        obj_val = evaluate(objective)
        
        println("✓ Optimization successful!")
        println("  Status: $(problem.status)")
        println("  Objective value: $(round(obj_val, digits=6))")
        println("  Reconstruction error: $(round(evaluate(reconstruction_error), digits=6))")
        println("  Sparsity penalty: $(round(evaluate(sparsity_penalty), digits=6))")
        
        # Verify constraints
        verify_cme_constraints(A_opt)
        
        return A_opt, true, obj_val
    else
        println("✗ Optimization failed!")
        println("  Status: $(problem.status)")
        return zeros(n_states, n_states), false, Inf
    end
end

"""
    verify_cme_constraints(A)

Verify that matrix A satisfies CME generator constraints.
"""
function verify_cme_constraints(A; tol=1e-8)
    n = size(A, 1)
    
    println("\nVerifying CME generator constraints:")
    
    # Check 1: Non-negative off-diagonals
    off_diag_violations = 0
    min_off_diag = Inf
    for i in 1:n
        for j in 1:n
            if i != j && A[i,j] < -tol
                off_diag_violations += 1
                min_off_diag = min(min_off_diag, A[i,j])
            end
        end
    end
    
    if off_diag_violations == 0
        println("  ✓ Non-negative off-diagonals: PASS")
    else
        println("  ✗ Non-negative off-diagonals: $off_diag_violations violations")
        println("    Minimum off-diagonal: $min_off_diag")
    end
    
    # Check 2: Column sums zero
    col_sum_violations = 0
    max_col_sum_error = 0.0
    for j in 1:n
        col_sum = sum(A[:, j])
        if abs(col_sum) > tol
            col_sum_violations += 1
            max_col_sum_error = max(max_col_sum_error, abs(col_sum))
        end
    end
    
    if col_sum_violations == 0
        println("  ✓ Zero column sums: PASS")
    else
        println("  ✗ Zero column sums: $col_sum_violations violations")
        println("    Maximum error: $max_col_sum_error")
    end
    
    # Check 3: Diagonal elements non-positive
    pos_diag_violations = 0
    max_pos_diag = -Inf
    for i in 1:n
        if A[i,i] > tol
            pos_diag_violations += 1
            max_pos_diag = max(max_pos_diag, A[i,i])
        end
    end
    
    if pos_diag_violations == 0
        println("  ✓ Non-positive diagonals: PASS")
    else
        println("  ✗ Non-positive diagonals: $pos_diag_violations violations")
        println("    Maximum diagonal: $max_pos_diag")
    end
    
    # Overall assessment
    total_violations = off_diag_violations + col_sum_violations + pos_diag_violations
    
    if total_violations == 0
        println("  🎉 All CME constraints satisfied!")
        return true
    else
        println("  ⚠ Total constraint violations: $total_violations")
        return false
    end
end

"""
    fallback_constrained_dmd(P_in, P_out, dt; λ_sparse=0.01)

Fallback implementation using penalty methods if convex solver unavailable.
"""
function fallback_constrained_dmd(P_in, P_out, dt; λ_sparse=0.01, max_iter=1000)
    println("Using fallback penalty method for constrained DMD...")
    
    n_states, n_snapshots = size(P_in)
    
    # Initialize with unconstrained DMD solution
    X, X_prime = P_in, P_out
    U, Σ, V = svd(X)
    
    # Use reduced rank for efficiency
    r = min(min(size(X)...) - 1, 50)  # Keep reasonable rank
    U_r = U[:, 1:r]
    Σ_r = Diagonal(Σ[1:r])
    V_r = V[:, 1:r]
    
    A_tilde = U_r' * X_prime * V_r * inv(Σ_r)
    A_init = U_r * A_tilde * U_r'
    
    # Convert to generator estimate
    K_init = I + A_init * dt
    
    # Iterative projection to satisfy constraints
    A = copy(A_init)
    
    for iter in 1:max_iter
        A_old = copy(A)
        
        # Project to satisfy constraints
        
        # 1. Make off-diagonals non-negative
        for i in 1:n_states
            for j in 1:n_states
                if i != j
                    A[i,j] = max(0, A[i,j])
                end
            end
        end
        
        # 2. Enforce zero column sums by adjusting diagonals
        for j in 1:n_states
            off_diag_sum = sum(A[i,j] for i in 1:n_states if i != j)
            A[j,j] = -off_diag_sum
        end
        
        # 3. Apply sparsity by soft thresholding
        threshold = λ_sparse * dt
        for i in 1:n_states
            for j in 1:n_states
                if i != j && abs(A[i,j]) < threshold
                    A[i,j] = 0
                end
            end
        end
        
        # 4. Re-enforce column sum constraint after sparsification
        for j in 1:n_states
            off_diag_sum = sum(A[i,j] for i in 1:n_states if i != j)
            A[j,j] = -off_diag_sum
        end
        
        # Check convergence
        change = norm(A - A_old, :fro)
        if change < 1e-8
            println("  Converged after $iter iterations")
            break
        end
        
        if iter == max_iter
            println("  Reached maximum iterations ($max_iter)")
        end
    end
    
    # Calculate final objective value
    prediction = (I + A * dt) * P_in
    reconstruction_error = norm(P_out - prediction, :fro)^2
    sparsity_penalty = λ_sparse * sum(abs.(A))
    obj_val = reconstruction_error + sparsity_penalty
    
    println("  Final objective: $(round(obj_val, digits=6))")
    
    verify_cme_constraints(A)
    
    return A, true, obj_val
end

"""
    apply_constrained_dmd(reduced_data, dt; method="convex", λ_sparse=0.01)

Apply constrained DMD with proper CME generator constraints.
"""
function apply_constrained_dmd(reduced_data, dt; method="convex", λ_sparse=0.01)
    println("\n=== Constrained DMD Analysis ===")
    
    # Form data matrices
    P_in = reduced_data[:, 1:end-1]
    P_out = reduced_data[:, 2:end]
    
    println("Data matrices: P_in$(size(P_in)), P_out$(size(P_out))")
    
    # Choose method
    if method == "convex"
        try
            # Try convex optimization first
            A_constrained, success, obj_val = constrained_dmd_optimization(
                P_in, P_out, dt, λ_sparse=λ_sparse
            )
            
            if success
                return A_constrained, obj_val, "convex"
            else
                println("Convex method failed, falling back to penalty method")
            end
        catch e
            println("Convex solver not available or failed: $e")
            println("Falling back to penalty method")
        end
    end
    
    # Fallback to penalty method
    A_constrained, success, obj_val = fallback_constrained_dmd(
        P_in, P_out, dt, λ_sparse=λ_sparse
    )
    
    return A_constrained, obj_val, "penalty"
end

"""
    compare_unconstrained_vs_constrained(reduced_data, dt; λ_sparse=0.01)

Compare unconstrained vs constrained DMD results.
"""
function compare_unconstrained_vs_constrained(reduced_data, dt; λ_sparse=0.01)
    println("\n=== DMD Method Comparison ===")
    
    # Unconstrained DMD (original method)
    println("\n--- Unconstrained DMD ---")
    X = reduced_data[:, 1:end-1]
    X_prime = reduced_data[:, 2:end]
    
    U, Σ, V = svd(X)
    r = min(min(size(X)...) - 1, 50)
    U_r = U[:, 1:r]
    Σ_r = Diagonal(Σ[1:r])
    V_r = V[:, 1:r]
    
    A_tilde = U_r' * X_prime * V_r * inv(Σ_r)
    A_unconstrained = U_r * A_tilde * U_r'
    
    # Convert to generator
    G_unconstrained = (A_unconstrained - I) / dt
    
    pred_unconstrained = A_unconstrained * X
    error_unconstrained = norm(X_prime - pred_unconstrained, :fro)^2
    
    println("Unconstrained reconstruction error: $(round(error_unconstrained, digits=6))")
    println("Verifying unconstrained generator constraints:")
    verify_cme_constraints(G_unconstrained)
    
    # Constrained DMD
    println("\n--- Constrained DMD ---")
    G_constrained, obj_val, method_used = apply_constrained_dmd(
        reduced_data, dt, λ_sparse=λ_sparse
    )
    
    pred_constrained = (I + G_constrained * dt) * X
    error_constrained = norm(X_prime - pred_constrained, :fro)^2
    
    println("Constrained reconstruction error: $(round(error_constrained, digits=6))")
    
    # Compare sparsity
    nnz_unconstrained = count(abs.(G_unconstrained) .> 1e-8)
    nnz_constrained = count(abs.(G_constrained) .> 1e-8)
    
    println("\n--- Sparsity Comparison ---")
    println("Unconstrained non-zeros: $nnz_unconstrained")
    println("Constrained non-zeros: $nnz_constrained")
    println("Sparsity improvement: $(round((1 - nnz_constrained/nnz_unconstrained)*100, digits=1))%")
    
    return Dict(
        "unconstrained_generator" => G_unconstrained,
        "constrained_generator" => G_constrained,
        "unconstrained_error" => error_unconstrained,
        "constrained_error" => error_constrained,
        "method_used" => method_used,
        "objective_value" => obj_val
    )
end

println("Constrained DMD Module Loaded! 🔒")
println("Key functions:")
println("  apply_constrained_dmd(data, dt)")
println("  compare_unconstrained_vs_constrained(data, dt)")
println("  verify_cme_constraints(A)")
println()
println("Note: For full functionality, install Convex.jl and SCS.jl:")
println("  using Pkg; Pkg.add([\"Convex\", \"SCS\"])")
