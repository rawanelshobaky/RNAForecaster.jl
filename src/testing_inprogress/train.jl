using Distributed
using CSV
using DataFrames

# for RNAForecaster
using Random
using IterTools
using Flux
using DifferentialEquations
# using DiffEqFlux

println("Starting program")

include("./trainRNAForecaster.jl") # to train the data

println("Reading expression data")

# Read the expression data
t0_matrix = CSV.read("t0_matrix.csv", DataFrame)
t1_matrix = CSV.read("t1_matrix.csv", DataFrame)

println("Converting to matrices")
# Convert DataFrames to Matrices
t0_matrix = convert(Matrix{Float32}, Matrix(t0_matrix))
t1_matrix = convert(Matrix{Float32}, Matrix(t1_matrix))

println("Entering trainRNAForecaster")
# Run the RNA Forecaster model using the subset of genes
testForecaster = trainRNAForecaster(t0_matrix, t1_matrix)
