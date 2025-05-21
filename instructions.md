# Quick Start Guide

This guide provides a quick introduction to getting started with the Chemical Reaction Network Inference via DMD framework.

## Installation

1. First, make sure you have Julia 1.6 or newer installed.

2. Clone this repository:
```bash
git clone https://github.com/yourusername/CRNInferenceDMD.jl.git
cd CRNInferenceDMD.jl
```

3. Install the required dependencies:
```julia
using Pkg
Pkg.add(["LinearAlgebra", "SparseArrays", "Statistics", "Random", "ProgressMeter", 
         "JumpProcesses", "Catalyst", "DifferentialEquations", "UnicodePlots"])
```

## Running Your First Example

1. **Interactive Example**

The simplest way to get started is to run the main script, which provides an interactive interface:

```bash
julia main.jl
```

This will prompt you to choose between Michaelis-Menten, Toggle Switch, or Lotka-Volterra example systems, then run the inference pipeline on your choice.

2. **From within Julia**

Alternatively, you can use the framework from within Julia:

```julia
# Start Julia and include the main script
include("main.jl")

# Run the Michaelis-Menten example
results = run_example("mm")
```

3. **Simple Example Script**

For a more guided example, try running the example script:

```bash
julia example.jl
```

This will run the Michaelis-Menten and Lotka-Volterra examples in sequence, showing key results from each.

## Understanding the Results

The inference pipeline returns a dictionary containing various results:

- `significant_stoichiometries`: List of identified reaction stoichiometries
- `grouped_reactions`: Detailed information about each reaction
- `stoich_stats`: Statistics for each stoichiometry (count, rate, variance)
- `generator`: Reconstructed generator matrix
- `eigenvalues`: Eigenvalues of the generator matrix
- `spectral_selected_reactions`: Reactions selected by spectral analysis
- `kinetics_results`: Detailed kinetics analysis results

You can access these as follows:

```julia
# Extract specific results
significant_stoich = results["significant_stoichiometries"]
stoich_stats = results["stoich_stats"]

# Print top reactions
species_names = ["S", "E", "SE", "P"]  # for Michaelis-Menten
for stoich in significant_stoich[1:3]
    reaction_str = format_reaction(stoich, species_names)
    rate = stoich_stats[stoich].total_rate
    println("$reaction_str: $rate")
end
```

## Visualizing Results

The framework includes visualization functions:

```julia
# Generate all visualizations for the results
visualize_all_results(results, species_names)

# Or individual visualizations
visualize_eigenvalues(results["eigenvalues"])
visualize_reaction_scores(significant_stoich, spectral_scores, species_names)
```

## Using Your Own Data

To use the framework with your own trajectory data:

1. Format your trajectory data similarly to the output of `generate_mm_trajectories()` or other generator functions
2. Define species names for your system
3. Call the inference function:

```julia
# Assuming you have trajectory data in ssa_trajs
species_names = ["A", "B", "C"]  # your species names
results = infer_crn_from_trajectories(ssa_trajs, species_names)
```

## Common Issues

- **Memory limitations**: If you encounter memory issues, try reducing `max_dim` (e.g., `run_example("mm", 500, 300)` to use 300 max states instead of default 1000)
- **Slow performance**: For faster results with lower accuracy, reduce the number of trajectories (e.g., `run_example("mm", 100)` for 100 trajectories)
- **Missing reactions**: If expected reactions are not detected, try increasing sensitivity with a lower threshold (modify `threshold` parameter in `extract_reactions_from_generator`)

## Next Steps

Once you're comfortable with the basic usage, you can:

1. Experiment with different system parameters in `data_generation.jl`
2. Modify the spectral analysis methods in `spectral_analysis.jl` to improve reaction selection
3. Add your own reaction validation rules in `reaction_extraction.jl`
4. Add new visualization techniques in `visualization.jl`

For more detailed information, refer to the main README.md file.
