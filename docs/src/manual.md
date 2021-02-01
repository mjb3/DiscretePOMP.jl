# Package manual
```@contents
Pages = ["manual.md"]
Depth = 3
```

## Types

### Model types
```@docs
DPOMPModel
Particle
Event
Observation
ARQModel
```

### Results
```@docs
SimResults
ImportanceSample
RejectionSample
MCMCSample
ARQMCMCSample
```

## Functions

### Models
```@docs
generate_model
generate_custom_model
partial_gaussian_obs_model
```

### Simulation
```@docs
gillespie_sim
```

### Bayesian inference

```@docs
run_mcmc_analysis
run_ibis_analysis
run_arq_mcmc_analysis
```

### Bayesian model analysis

```@docs
run_model_comparison_analysis
```

### Utilities
```@docs
get_observations
tabulate_results
print_results
get_particle_filter_lpdf
```

### Visualisation

```@docs
plot_trajectory
plot_parameter_trace
plot_parameter_marginal
plot_parameter_heatmap
plot_model_comparison
```

### Advanced features.
This section covers functionality for customising the algorithms themselves.

```@docs
run_custom_mcmc_analysis
generate_custom_particle
```

## Index
```@index
```
