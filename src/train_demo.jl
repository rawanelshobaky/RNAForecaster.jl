using Distributed
using CSV
using DataFrames

# for RNAForecaster
using Random
using IterTools
using Flux
using DifferentialEquations

println("Starting program")

include("./trainRNAForecaster.jl") # to train the data

#=
testT0 = log1p.(Float32.(abs.(randn(5,500))))
testT1 = log1p.(0.5f0 .* testT0)

# Save t0_matrix and t1_matrix without headers
CSV.write("demo0_matrix.csv", DataFrame(testT0, :auto); header=false)
CSV.write("demo1_matrix.csv", DataFrame(testT1, :auto); header=false)
=#

println("Reading expression data")

# Read CSV files back as Float32 matrices
t0_matrix = CSV.read("demo0_matrix.csv", DataFrame) |> Matrix{Float32}
t1_matrix = CSV.read("demo1_matrix.csv", DataFrame) |> Matrix{Float32}

# Optional verification of shapes and types
println("t0_matrix type: ", typeof(t0_matrix), ", size: ", size(t0_matrix))
println("t1_matrix type: ", typeof(t1_matrix), ", size: ", size(t1_matrix))

testForecaster = trainRNAForecaster(t0_matrix, t1_matrix);