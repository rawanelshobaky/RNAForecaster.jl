# Dependencies and Installation Guide

This document outlines the dependencies required to run the RNAForecaster code and how to set up your Julia environment.

## Required Packages

Below are the Julia packages required to run the code:

### Core Data Processing and Utilities
- `CSV` - For handling CSV file reading and writing.
- `DataFrames` - For manipulating tabular data.
- `Random` - For handling random number generation.
- `IterTools` - For advanced iteration utilities.

### Machine Learning and Neural ODEs
- `Flux` - A Julia machine learning library.
- `DifferentialEquations` - For solving differential equations.
- `DiffEqFlux` - For integrating differential equations with Flux.
- `Lux` - For building neural networks.
- `Optimisers` - For optimization algorithms.
- `Zygote` - For automatic differentiation.
- `OrdinaryDiffEq` - Solvers for ordinary differential equations.
- `ComponentArrays` - For flexible arrays with names for dimensions.

### Statistics
- `Statistics` - For basic statistical operations (e.g., mean).

## Installation Instructions

### Step 1: Install Julia
Download and install Julia from the [official website](https://julialang.org/downloads/).

### Step 2: Set Up Dependencies

1. Open the Julia REPL by typing `julia` in your terminal.
2. Press `]` to enter the package manager mode.
3. Add the required packages by running the following commands:
   ```julia
   add CSV
   add DataFrames
   add Random
   add IterTools
   add Flux
   add DifferentialEquations
   add DiffEqFlux
   add Lux
   add Optimisers
   add Zygote
   add OrdinaryDiffEq
   add ComponentArrays
   add Statistics
   ```

### Step 3: Verify Installation
Run the following in the Julia REPL to ensure all packages are installed and loadable:
   ```julia
   using Distributed, CSV, DataFrames, Random, IterTools, Flux, DifferentialEquations
   using DiffEqFlux, Lux, Optimisers, Zygote, OrdinaryDiffEq, ComponentArrays, Statistics
   println("All dependencies are correctly installed!")
   ```

### Step 4: Run the file
train_demo.jl uses trainRNAForecaster.jl 
   ```julia
   julia train_demo.jl
   ```

## Notes
The following was not tested:
   - using GPU, so I'm not sure about the anticipated bugs if any.
   - using the function `createEnsembleForecaster` since it was not called. If you want to use it, you should download the package `Distributed`.

## Author & Date
   - Rawan Elshobaky
   - 12/12/24