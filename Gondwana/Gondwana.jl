using Pkg
Pkg.activate(dirname(Base.current_project()))
using Revise

using LinearAlgebra, Statistics, Distributions
using SciMLSensitivity
using OrdinaryDiffEqCore, OrdinaryDiffEqTsit5
using Lux
using JLD2

# using Infiltrator, ReverseDiff

using SphereUDE

# Random seed
using Random
rng = Random.default_rng()
Random.seed!(rng, 613)

using DataFrames, CSV

df = CSV.read("Gondwana/data/Torsvik-etal-2012_dataset.csv", DataFrame, delim=",")

# Filter the plates that were once part of the supercontinent Gondwana
Gondwana = [
    "Amazonia", "Parana", "Colorado", "Southern_Africa",
    "East_Antarctica", "Madagascar", "Patagonia", "Northeast_Africa",
    "Northwest_Africa", "Somalia", "Arabia", "East_Gondwana"
    ]

df = filter(row -> row.Plate ∈ Gondwana, df)
df.Times = df.Age

df = sort(df, :Times)
times = df.Times

# Fill missing values
df.RLat .= coalesce.(df.RLat, df.Lat)
df.RLon .= coalesce.(df.RLon, df.Lon)

X = sph2cart(Matrix(df[:,["RLat","RLon"]])'; radians=false)

# Retrieve uncertanties from poles and convert α95 into κ
kappas = (140.0 ./ df.a95).^2

data = SphereData(times=times, directions=X, kappas=kappas, L=nothing)

# Training

# Expected maximum angular deviation in one unit of time (degrees)
Δω₀ = 2.0
# Angular velocity
ωmax = Δω₀ * π / 180.0
# Time span of the simulation
# tspan = [times[begin], times[end]]
tspan = [0.0, times[end]]

params = SphereParameters(
    tmin = tspan[1],
    tmax = tspan[2],
    # reg = [Regularization(order = 1, power = 2.0, λ = 10.0^(6.2), diff_mode = FiniteDiff())],
    reg = [Regularization(order = 1, power = 2.0, λ = 10.0^(6.2), diff_mode = LuxNestedAD())],
    # reg = nothing,
    pretrain = true,
    u0 = [0.0, 0.0, -1.0],
    ωmax = ωmax,
    reltol = 1e-6,
    abstol = 1e-6, # Set to this from 1e-6 for fourier features
    weighted = false,
    ADAM_learning_rate = 0.001,
    niter_ADAM = 400,
    # niter_LBFGS = 400,
    niter_LBFGS = 200,
    quadrature = 200,
    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
    # sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
    out_of_place = false,
    verbose_step = 10
    )


n_fourier_features = 4
U = Lux.Chain(
    # Scale function to bring input to [-1.0, 1.0]
    Lux.WrappedFunction(x -> scale_input(x; xmin = params.tmin, xmax = params.tmax)),
    # Fourier feautues
    Lux.WrappedFunction(x -> fourier_feature(x; n = n_fourier_features)),
    Lux.Dense(2 * n_fourier_features, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 3, tanh),
    # Output function to scale output to have norm less than ωmax
    Lux.WrappedFunction(x -> scale_norm(params.ωmax * x; scale = params.ωmax))
)

# results = train(data, params, rng, nothing, U)

using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.samples = 1
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

trial = @benchmark train(data, params, rng, nothing, U)
# results_dict = convert2dict(data, results)

# JLD2.@save "Gondwana/results/results_dict.jld2" results_dict

# plot_sphere(data, results, -30., 0., saveas="Gondwana/plots/Gondwana_sphere.pdf", title="Gondwana")
# plot_L(data, results, saveas="Gondwana/plots/Gondwana_L_speed.pdf", title="Gondwana")
