### correspondes to:
# https://mjb3.github.io/DiscretePOMP.jl/dev/models/
using DiscretePOMP      # simulation / inference for epidemiological models
import Distributions    # priors

## for testing
function test_model(m, lbl::String, theta = [0.003, 0.1])
    println(lbl)
    x = gillespie_sim(m, theta)     # run simulation
    println(plot_trajectory(x))		# plot (optional)
end

## Predefined models
model = generate_model("SIS", [100,1])
test_model(model, "test one")

## Customising predefined models
model.prior = Distributions.Product(Distributions.Uniform.(zeros(2), [0.01, 0.5]))

## Custom models from scratch
# rate function
function sis_rf!(output, parameters::Array{Float64, 1}, population::Array{Int64, 1})
    output[1] = parameters[1] * population[1] * population[2]
    output[2] = parameters[2] * population[2]
end
# define obs function
function obs_fn(y::Observation, population::Array{Int64, 1}, theta::Array{Float64,1})
    y.val .= population
end
# prior
prior = Distributions.Product(Distributions.Uniform.(zeros(2), [0.01, 0.5]))
# obs model
function si_gaussian(y::Observation, population::Array{Int64, 1}, theta::Array{Float64,1})
    obs_err = 2
    tmp1 = log(1 / (sqrt(2 * pi) * obs_err))
    tmp2 = 2 * obs_err * obs_err
    obs_diff = y.val[2] - population[2]
    return tmp1 - ((obs_diff * obs_diff) / tmp2)
end
tm = [-1 1; 1 -1] # transition matrix
# define model
model = DPOMPModel("SIS", sis_rf!, [100, 1], tm, obs_fn, si_gaussian, prior, 0)
test_model(model, "test two")
