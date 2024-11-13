# Load required packages
using DifferentialEquations
using Flux
using CSV
using DataFrames
using Random
using BSON

# Include RNAForecaster
include("./trainRNAForecaster.jl")
#=
println("Setting up simulation parameters and initial conditions...")

# Step 1: Define random parameters for transcription and degradation rates for 5 genes
n_genes = 5
transcription_rates = rand(Float32, n_genes)  # Random transcription rates for each gene
degradation_rates = rand(Float32, n_genes)    # Random degradation rates for each gene
parameters = [transcription_rates; degradation_rates]  # Combine for use in simulation

# Step 2: Define initial expression state for each gene
initial_expression_state = rand(Float32, n_genes)  # Random initial mRNA levels

# Print known parameters for reference
println("Known Transcription Rates: ", transcription_rates)
println("Known Degradation Rates: ", degradation_rates)

# Step 3: Define the differential equations model
function differential_equations_model(y, p, t)
    α = p[1:n_genes]              # Transcription rates
    β = p[n_genes+1:2*n_genes]    # Degradation rates
    dydt = α .- β .* y            # Transcription - degradation dynamics
    return dydt
end

# Step 4: Generate simulated expression data using known model parameters
function simulate_expression(parameters, initial_state, tspan)
    prob = ODEProblem((y, p, t) -> differential_equations_model(y, p, t), initial_state, tspan, parameters)
    sol = solve(prob, Tsit5())
    return sol
end

# Set time span for simulation (e.g., simulate from time 0 to 1)
tspan = (0.0f0, 1.0f0)  # Use Float32 literals for the time span
n_time_points = 500
time_points = range(tspan[1], tspan[2], length=n_time_points)  # 500 equally spaced points

# Simulate expression data with 500 fixed time points
function simulate_expression(parameters, initial_state, tspan, time_points)
    prob = ODEProblem((y, p, t) -> differential_equations_model(y, p, t), initial_state, tspan, parameters)
    sol = solve(prob, Tsit5(); saveat=time_points)  # Use saveat to fix the time points
    return sol
end

# Run the simulation with 500 fixed time points
simulated_data = simulate_expression(parameters, initial_expression_state, tspan, time_points)

# Step 5: Convert simulated data to matrix format for RNAForecaster
# Extract gene expression levels at the start and end of the time interval
t0_matrix = hcat(simulated_data.u[1:end-1]...)  # Expression levels at initial time points
t1_matrix = hcat(simulated_data.u[2:end]...)    # Expression levels at subsequent time points

# Save to CSV to match RNAForecaster's expected input
# Save t0_matrix and t1_matrix without headers
CSV.write("test0_matrix.csv", DataFrame(t0_matrix, :auto); header=false)
CSV.write("test1_matrix.csv", DataFrame(t1_matrix, :auto); header=false)

println("Starting RNAForecaster training...")

# Step 6: Run RNAForecaster on the simulated data
t0_matrix = CSV.read("test0_matrix.csv", DataFrame)
t1_matrix = CSV.read("test1_matrix.csv", DataFrame)

# Convert DataFrames to Matrices
t0_matrix = convert(Matrix{Float32}, Matrix(t0_matrix))
t1_matrix = convert(Matrix{Float32}, Matrix(t1_matrix))

# Load the trained model and parameters from trainRNAForecaster
trained_model, _ = trainRNAForecaster(t0_matrix, t1_matrix)  # Assuming only the model and losses are returned

# Save the model
file_path = "trained_model.bson"
BSON.@save file_path trained_model
=#

# Load the model
BSON.@load file_path trained_model

println("RNAForecaster training completed.")
println("Extracting and comparing parameters...")

# Step 8: Predict using trained model and evaluate accuracy
function predict_expression(trained_model, t0_matrix)
    # Use the trained model to predict t1 expression levels from t0 input
    # Assuming trained_model is directly callable with input data
    predicted_t1 = [trained_model(t0_matrix[:, i])[1] for i in 1:size(t0_matrix, 2)]
    return hcat(predicted_t1...)  # Convert list of predictions to a matrix
end

# Predict t1 expressions using t0_matrix
predicted_t1_matrix = predict_expression(trained_model, t0_matrix)

# Calculate prediction error (mean squared error)
function prediction_error(actual, predicted)
    return mean((actual .- predicted) .^ 2)
end

# Check that predicted matrix matches the actual matrix dimensions
if size(predicted_t1_matrix) == size(t1_matrix)
    error = prediction_error(t1_matrix, predicted_t1_matrix)
    println("Prediction error (MSE): ", error)
else
    println("Error: Predicted matrix dimensions do not match actual matrix dimensions.")
    println("Predicted dimensions: ", size(predicted_t1_matrix))
    println("Actual dimensions: ", size(t1_matrix))
end
