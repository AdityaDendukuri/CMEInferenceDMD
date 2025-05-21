# Chemical Reaction Network Inference via DMD

A computational framework for inferring chemical reaction networks from trajectory data using Dynamic Mode Decomposition.

## Overview

This package provides tools to infer the underlying chemical reaction network (CRN) from stochastic trajectory data. By applying Dynamic Mode Decomposition (DMD) to empirical probability distributions, we can reconstruct the Chemical Master Equation (CME) generator matrix and extract reaction mechanisms and rate constants.

## Features

- Generate synthetic trajectory data for benchmark systems (Michaelis-Menten, toggle switch, Lotka-Volterra)
- Process and reduce trajectory data with intelligent state selection
- Apply DMD to extract dynamics from probability distributions
- Identify reactions and their propensities from the generator matrix
- Analyze reaction kinetics and estimate rate constants
- Perform spectral analysis to identify dynamically important reactions
- Visualize results with Unicode plots

## Requirements

- Julia 1.6+
- Required packages:
  - LinearAlgebra
  - SparseArrays
  - Statistics
  - Random
  - ProgressMeter
  - JumpProcesses
  - Catalyst
  - DifferentialEquations
  - UnicodePlots

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/CRNInferenceDMD.jl.git
cd CRNInferenceDMD.jl
```

2. Install dependencies in Julia:
```julia
using Pkg
Pkg.add(["LinearAlgebra", "SparseArrays", "Statistics", "Random", "ProgressMeter", 
         "JumpProcesses", "Catalyst", "DifferentialEquations", "UnicodePlots"])
```

## Usage

### Basic Example

Run the main script to see a demonstration on a benchmark system:

```bash
julia main.jl
```

This will prompt you to choose a benchmark system, generate trajectory data, run the inference pipeline, and visualize the results.

### Programmatic Usage

```julia
# Include the main script
include("main.jl")

# Run Michaelis-Menten example
results = run_example("mm")

# Run toggle switch example
results = run_example("ts")

# Run Lotka-Volterra example
results = run_example("lv")

# Customize parameters
results = run_example("mm", 1000, 2000, true)  # 1000 trajectories, 2000 max states, with visualization
```

### Using Your Own Data

```julia
# Define species names
species_names = ["S", "E", "SE", "P"]

# Assuming you have trajectory data in ssa_trajs
results = infer_crn_from_trajectories(ssa_trajs, species_names)

# Visualize results
visualize_all_results(results, species_names)
```

## File Structure

- `main.jl` - Main script that brings everything together
- `data_generation.jl` - Functions for generating synthetic data
- `data_processing.jl` - Functions for processing trajectory data
- `dmd_analysis.jl` - Core DMD algorithm implementation
- `reaction_extraction.jl` - Functions for extracting and validating reactions
- `kinetics_analysis.jl` - Functions for analyzing reaction kinetics
- `spectral_analysis.jl` - Functions for spectral analysis of the generator matrix
- `visualization.jl` - All visualization functions

## Benchmark Systems

### Michaelis-Menten Enzyme Kinetics

A fundamental model in biochemistry where an enzyme (E) catalyzes the conversion of a substrate (S) to a product (P) through an enzyme-substrate complex (SE):

```
S + E → SE
SE → S + E
SE → P + E
```

### Genetic Toggle Switch

A bistable system of two mutually repressing genes, producing a switch-like behavior:

```
∅ → P₁  (repressed by P₂)
P₁ → ∅
∅ → P₂  (repressed by P₁)
P₂ → ∅
```

### Lotka-Volterra (Predator-Prey)

A classic ecological model of predator-prey interactions:

```
X → X + X  (prey reproduction)
X + Y → Y + Y  (predator reproduction through prey consumption)
Y → ∅  (predator death)
```

## Method Details

Our approach consists of the following key steps:

1. Convert trajectory data to empirical probability distributions via histogram binning
2. Apply intelligent state selection to reduce dimensionality
3. Use DMD to approximate the generator matrix of the Chemical Master Equation
4. Extract reaction stoichiometry and rates from the generator structure
5. Validate reactions using conservation principles and spectral analysis
6. Apply corrections to estimate true microscopic rate constants

## License

This project is licensed under the MIT License - see the LICENSE file for details.
