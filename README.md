# DiscretePOMP.jl
**Bayesian inference for Discrete-state-space Partially Observed Markov Processes in Julia**

![Documentation](https://github.com/mjb3/DiscretePOMP.jl/workflows/Documentation/badge.svg)

This package contains tools for Bayesian inference and simulation of DPOMP models. See the [docs][docs].

## Features

- Simulation and
- Bayesian parameter inference for,
- Discrete-state-space Partially Observed Markov Processes, in Julia.
- Includes automated tools for convergence diagnosis and analysis.

### Algorithms

The package implements several different customisable algorithms for Bayesian parameter inference, including:
- Data-augmented MCMC
- Particle filters
- Iterative-batch-importance sampling

## Installation

The package is not registered and must be added via the package manager Pkg.
From the Julia REPL type `]` to enter the Pkg mode, and run:

```
pkg> add https://github.com/mjb3/DiscretePOMP.jl
```

## Usage

The [package documentation][docs] has more information and examples.

[docs]: https://mjb3.github.io/DiscretePOMP.jl/stable
