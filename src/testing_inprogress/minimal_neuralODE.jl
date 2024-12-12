using Flux, DifferentialEquations, DiffEqFlux

# Define a simple neural network using Flux
nn = Flux.Chain(
    Flux.Dense(2, 50, relu),
    Flux.Dense(50, 2)
)

# Construct NeuralODE
tspan = (0.0f0, 1.0f0)
model = NeuralODE(nn, tspan, Tsit5(), reltol=1e-3, abstol=1e-3)

# Define loss function
function loss(x, y)
    y_pred = model(x, tspan)[1]
    return Flux.Losses.mse(y_pred, y)
end

# Sample data
x = rand(Float32, 2, 100)  # Input
y = rand(Float32, 2, 100)  # Target

# Train the model
ps = Flux.params(nn)
gs = gradient(() -> loss(x, y), ps)
