using Pkg
Pkg.activate(dirname(Base.current_project()))

using LinearAlgebra, Statistics, Distributions
using SciMLSensitivity
using OrdinaryDiffEqCore, OrdinaryDiffEqTsit5
# using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Lux
using JLD2

using SphereUDE

##############################################################
###############  Simulation of Simple Example ################
##############################################################

# Random seed
using Random
rng = Random.default_rng()
Random.seed!(rng, 666)

function run(;
    kappa = 50.,
    regs = regs,
    title="plot"
    )

    # Total time simulation
    tspan = [0, 160.0]
    # Number of sample points
    N_samples = 100
    # Times where we sample points
    times_samples = sort(rand(sampler(Uniform(tspan[1], tspan[2])), N_samples))

    # Expected maximum angular deviation in one unit of time (degrees)
    Δω₀ = 1.0
    # Angular velocity
    ω₀ = Δω₀ * π / 180.0
    # Change point
    τ₀ = 70.0
    # Angular momentum
    L0 = ω₀    .* [1.0, 0.0, 0.0]
    L1 = 0.6ω₀ .* [0.0, 1/sqrt(2), 1/sqrt(2)]

    function L_true(t::Float64; τ₀=τ₀, p=[L0, L1])
        if t < τ₀
            return p[1]
        else
            return p[2]
        end
    end

    function true_rotation!(du, u, p, t)
        L = L_true(t; τ₀ = τ₀, p = p)
        du .= cross(L, u)
    end

    prob = ODEProblem(true_rotation!, [0.0, 0.0, -1.0], tspan, [L0, L1])
    true_sol  = solve(prob, Tsit5(), reltol = 1e-12, abstol = 1e-12, saveat = times_samples)

    # Add Fisher noise to true solution
    X_noiseless = Array(true_sol)
    X_true = X_noiseless + FisherNoise(kappa=kappa)

    ##############################################################
    #######################  Training  ###########################
    ##############################################################

    data = SphereData(times=times_samples, directions=X_true, kappas=nothing, L=L_true)

    params = SphereParameters(
        tmin = tspan[1], tmax = tspan[2],
        reg = regs,
        train_initial_condition = false,
        multiple_shooting = false,
        pretrain = true,
        quadrature = 200,
        u0 = [0.0, 0.0, -1.0],
        ωmax = ω₀,
        reltol = 1e-6,
        abstol = 1e-6,
        niter_ADAM = 500, # 3000
        niter_LBFGS = 500, # 2000
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
        )

    # init_bias(rng, in_dims) = LinRange(tspan[1], tspan[2], in_dims)
    # init_weight(rng, out_dims, in_dims) = 0.1 * ones(out_dims, in_dims)

    # # Testing sigmoids to see how it works
    # U = Lux.Chain(
    #     Lux.Dense(1, 200, rbf, init_bias=init_bias, init_weight=init_weight, use_bias=true),
    # #     Lux.Dense(200,10, relu),
    #       Lux.Dense(200,50, SphereUDE.sigmoid),
    #       Lux.Dense(50,10, SphereUDE.sigmoid),
    #     Lux.Dense(10, 3, Base.Fix2(sigmoid_cap, params.ωmax), use_bias=false)
    # )
    function scale_norm(x; scale = 1.0)
        return (scale * tanh(norm(x) / scale)) .* (x ./ norm(x))
    end

    function fourier_feature(v, n::Integer = 10; σ = 2.0)
        a₁ = ones(n)
        b₁ = ones(n)
        # W = rand(Normal(0, σ), (n, length(v)))
        # W = rand(Uniform(-σ, σ), (n, length(v)))
        W = 1.0:1.0:n |> collect
        return [a₁ .* sin.(π .* W .* v); b₁ .* cos.(π .* W .* v)]
    end

    n_fourier_features = 4
    U = Lux.Chain(
        # Scale function to bring input to [-1.0, 1.0]
        Lux.WrappedFunction(t -> 2.0 .* (t .- params.tmin) ./ (params.tmax .- params.tmin) .- 1.0),
        Lux.WrappedFunction(x -> fourier_feature(x, n_fourier_features)),
        Lux.Dense(2 * n_fourier_features, 10, Lux.sigmoid),
        Lux.Dense(10, 10, Lux.sigmoid),
        Lux.Dense(10, 10, Lux.sigmoid),
        Lux.Dense(10, 3, Lux.sigmoid),
        # Output function to scale output to have norm less than ωmax
        Lux.WrappedFunction(x -> scale_norm(params.ωmax * x; scale = params.ωmax))
    )

    results = train(data, params, rng, nothing, U)

    ##############################################################
    ######################  PyCall Plots #########################
    ##############################################################

    plot_sphere(data, results, -20., 125., saveas="double-rotation/" * title * "sphere.pdf", title="Double rotation") # , matplotlib_rcParams=Dict("font.size"=> 50))
    plot_L(data, results, saveas="double-rotation/" * title * "L.pdf", title="Double rotation")

    return data, results
end # run

# Run different experiments

data_50, results_50 = run(;
    kappa = 50.0,
    regs = [Regularization(order=1, power=1.0, λ=4e-1, diff_mode=FiniteDiff(ϵ=1.0)),
            Regularization(order=0, power=2.0, λ=1e-1)],
    title = "plots/_AD_plot_50"
    )
results_dict_50 = convert2dict(data_50, results_50)
JLD2.@save "double-rotation/results/custom_results_dict_50.jld2" results_dict_50

# data_200, results_200 = run(;
#     kappa = 200.0,
#     regs = [Regularization(order=1, power=10.0, λ=1.0, diff_mode=FiniteDiff()),
#             Regularization(order=0, power=2.0, λ=0.1)],
#     title = "plots/_AD_plot_200"
#     )
# results_dict_200 = convert2dict(data_200, results_200)
# JLD2.@save "double-rotation/results/custom_results_dict_200.jld2" results_dict_200

data_1000, results_1000 = run(;
    kappa = 1000.0,
    regs = [Regularization(order=1, power=1.0, λ=4e-1, diff_mode=FiniteDiff()),
            Regularization(order=0, power=2.0, λ=1e-1)],
    title = "plots/_AD_plot_1000"
    )
results_dict_1000 = convert2dict(data_1000, results_1000)
JLD2.@save "double-rotation/results/custom_results_dict_1000.jld2" results_dict_1000
