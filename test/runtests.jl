# dummy test result
using DPOMPs
# using DPOMPs.ARQMCMC
import Random
import Distributions
# import Test

Random.seed!(1)

## getting started
# data_fp = "https://raw.githubusercontent.com/mjb3/DiscretePOMP.jl/master/data/pooley.csv"
data_fp = "data/pooley.csv"
y = get_observations(data_fp) # val_seq=2:3
model = generate_model("SIS", [100,1])
# mcmc = run_met_hastings_mcmc(model, y, [0.0025, 0.12])
## simulation # NB. first define the SIS 'model' variable, per above
theta = [0.003, 0.1]
x = gillespie_sim(model, theta)	    # run simulation
println(plot_trajectory(x))			# plot (optional)

## ARQMCMC
sample_interval = [0.0005, 0.02]
rs = run_arq_mcmc_analysis(model, y, sample_interval)
tabulate_results(rs)

## DA MCMC
model.prior = Distributions.Product(Distributions.Uniform.(zeros(2), [0.01, 0.5]))
rs = run_mcmc_analysis(model, y)
tabulate_results(rs)
# println(plot_parameter_trace(rs, 1))  # trace plot of contact parameter (optional)

## SMC^2
results = run_ibis_analysis(model, y)
tabulate_results(results)

## model comparison
# define model to compare against
# seis_model = generate_model("SEIS", [100,0,1])
# seis_model.prior = Distributions.Product(Distributions.Uniform.(zeros(3), [0.1,0.5,0.5]))
# seis_model.obs_model = partial_gaussian_obs_model(2.0, seq = 3, y_seq = 2)
#
# # run comparison
# models = [model, seis_model]
# mcomp = run_model_comparison_analysis(models, y)
# tabulate_results(mcomp; null_index = 1)
# println(plot_model_comparison(mcomp))

## custom models
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
prior = Distributions.Product(Distributions.Uniform.(zeros(2), [0.1, 0.5]))
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
x = gillespie_sim(model, theta)	# run simulation and plot
# println(plot_trajectory(x))

true
