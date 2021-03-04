### ph
# - see pkg tests for now
using DiscretePOMP
import Random
import Distributions

Random.seed!(1)

theta = [0.003, 0.1]
# data_fp = "https://raw.githubusercontent.com/mjb3/DiscretePOMP.jl/main/data/pooley.csv"
data_fp = "data/pooley.csv"

## getting started
y = get_observations(data_fp) # val_seq=2:3
model = generate_model("SIS", [100,1])

## simulation # NB. first define the SIS 'model' variable, per above
x = gillespie_sim(model, theta)	    # run simulation
println(plot_trajectory(x))			# plot (optional)

## ARQMCMC
model.prior = Distributions.Product(Distributions.Uniform.(zeros(2), [0.01, 0.5]))
sample_interval = [0.0005, 0.02]
rs = run_arq_mcmc_analysis(model, y, sample_interval)
tabulate_results(rs)
println(plot_parameter_trace(rs, 1))

## MBP IBIS
results = run_ibis_analysis(model, y; algorithm="MBPI")
tabulate_results(results)

## model comparison
# define model to compare against
seis_model = generate_model("SEIS", [100,0,1])
seis_model.prior = Distributions.Product(Distributions.Uniform.(zeros(3), [0.1,0.5,0.5]))
seis_model.obs_model = partial_gaussian_obs_model(2.0, seq = 3, y_seq = 2)

# run comparison
models = [model, seis_model]
mcomp = run_model_comparison_analysis(models, y)
tabulate_results(mcomp; null_index = 1)
println(plot_model_comparison(mcomp))
