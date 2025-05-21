# data_generation.jl
# Functions for generating synthetic chemical reaction data

using Random
using ProgressMeter
using JumpProcesses
using Catalyst
using DifferentialEquations

"""
    generate_mm_trajectories(n_trajs=1000)

Generate trajectory data for a Michaelis-Menten enzyme kinetics system using 
Catalyst and JumpProcesses.

# Arguments
- `n_trajs::Int`: Number of trajectories to generate

# Returns
- Array of trajectory solutions
- Reaction network object
"""
function generate_mm_trajectories(n_trajs=1000)
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Define the Michaelis-Menten reaction network
    rn = @reaction_network begin
        kB, S + E --> SE
        kD, SE --> S + E
        kP, SE --> P + E
    end
    
    # Initial conditions and parameters
    u0_integers = [:S => 50, :E => 10, :SE => 1, :P => 1]
    tspan = (0.0, 200.0)
    ps = [:kB => 0.01, :kD => 0.1, :kP => 0.1]
    
    # Create the jump problem
    jinput = JumpInputs(rn, u0_integers, tspan, ps)
    jprob = JumpProblem(jinput)
    
    # Generate trajectories
    ssa_trajs = []
    @showprogress desc="Generating trajectories..." for i in 1:n_trajs 
        push!(ssa_trajs, solve(jprob, SSAStepper()))
    end
    
    return ssa_trajs, rn
end

"""
    generate_toggle_switch_trajectories(n_trajs=1000)

Generate trajectory data for a genetic toggle switch system.

# Arguments
- `n_trajs::Int`: Number of trajectories to generate

# Returns
- Array of trajectory solutions
- Reaction network object
"""
function generate_toggle_switch_trajectories(n_trajs=1000)
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Define toggle switch reaction network
    rn = @reaction_network begin
        α₁/(1 + (P₂/K₁)^n), ∅ --> P₁
        γ₁, P₁ --> ∅
        α₂/(1 + (P₁/K₂)^n), ∅ --> P₂
        γ₂, P₂ --> ∅
    end
    
    # Initial conditions and parameters
    u0_integers = [:P₁ => 5, :P₂ => 15]
    tspan = (0.0, 300.0)
    ps = [:α₁ => 50.0, :K₁ => 20.0, :n => 3.0, 
          :α₂ => 50.0, :K₂ => 20.0, :γ₁ => 1.0, :γ₂ => 1.0]
    
    # Create the jump problem
    jinput = JumpInputs(rn, u0_integers, tspan, ps)
    jprob = JumpProblem(jinput)
    
    # Generate trajectories
    ssa_trajs = []
    @showprogress desc="Generating trajectories..." for i in 1:n_trajs 
        push!(ssa_trajs, solve(jprob, SSAStepper()))
    end
    
    return ssa_trajs, rn
end

"""
    generate_lotka_volterra_trajectories(n_trajs=1000)

Generate trajectory data for a stochastic Lotka-Volterra (predator-prey) system.

# Arguments
- `n_trajs::Int`: Number of trajectories to generate

# Returns
- Array of trajectory solutions
- Reaction network object
"""
function generate_lotka_volterra_trajectories(n_trajs=1000)
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Define Lotka-Volterra reaction network
    rn = @reaction_network begin
        a, X --> X + X    # prey reproduction
        b, X + Y --> Y + Y  # predator reproduction through prey consumption
        c, Y --> ∅        # predator death
    end
    
    # Initial conditions and parameters
    u0_integers = [:X => 50, :Y => 100]  # prey, predator
    tspan = (0.0, 100.0)
    ps = [:a => 0.1, :b => 0.005, :c => 0.6]
    
    # Create the jump problem
    jinput = JumpInputs(rn, u0_integers, tspan, ps)
    jprob = JumpProblem(jinput)
    
    # Generate trajectories
    ssa_trajs = []
    @showprogress desc="Generating trajectories..." for i in 1:n_trajs 
        push!(ssa_trajs, solve(jprob, SSAStepper()))
    end
    
    return ssa_trajs, rn
end
