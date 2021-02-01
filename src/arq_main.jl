#### ARQ-MCMC ####
## module
module ARQMCMC

## ARQ defaults
const C_DF_ARQ_SL = 1       # sample limit
const C_DF_ARQ_SR = 50      # inital sample distribution
const C_DF_ARQ_MC = 5       # chains
const C_DF_ARQ_AR = 0.33    # targeted AR
const C_DF_ARQ_JT = 0.0     # jitter

import Distributions
import StatsBase
import Random
import Statistics
import PrettyTables

## constants
const C_ALG_STD = "ARQ"
const C_ALG_DAUG  = "DAQ"
# const C_ALG_AD  = "ADARQ"

##
export ARQModel, ARQMCMCSample#, run_arq_mcmc_analysis, GridPoint
export tabulate_results
export plot_parameter_trace, plot_parameter_marginal, plot_parameter_heatmap

## structs
include("cmn_structs.jl")
include("arq_structs.jl")
# length(x::ARQMCMCSample) = 1

## common functions, macro
include("cmn.jl")
include("arq_alg_cmn.jl")
## standard ARQ MCMC algorithm
include("arq_alg_std.jl")
## delayed acceptance ARQ MCMC algorithm
# include("arq_alg_da.jl")
## data augmented ARQ MCMC algorithm
include("arq_alg_daug.jl")
## common functions, printing, etc
include("arq_utils.jl")
## visualisation tools
include("arq_visualisation_uc.jl")

## for internal use (called by public functions)
function run_inner_mcmc_analysis(mdl::LikelihoodModel, da::Bool, steps::Int64, burnin::Int64, chains::Int64, tgt_ar::Float64, grid::Dict{Array{Int64, 1}, GridPoint})
    start_time = time_ns()
    ## designate inner MCMC function and results array
    mcmc_fn = da ? daarq_met_hastings! : arq_met_hastings!
    # is_mu_fn = da ? compute_da_is_mean : compute_is_mean
    ## initialise
    n_theta = length(mdl.sample_interval)
    samples = zeros(n_theta, steps, chains)
    is_uc = 0.0
    fx = zeros(Int64, chains)
    for i in 1:chains   # run N chains using designated function
        # theta_init = rand(1:mdl.sample_dispersal, n_theta)     # choose initial theta coords TEST PRIOR HERE *****
        print(" chain ", i, " initialised")
        mcmc = arq_met_hastings!(samples, i, grid, mdl, steps, burnin, tgt_ar)
        fx[i] = mcmc[1]
        println(" - complete (calls to f(θ) := ", mcmc[1], "; AAR := ", round(mcmc[3] * 100, digits = 1), "%)")
    end
    ## compute scale reduction factor est.
    rejs = handle_rej_samples(samples, burnin)      # shared HMM functions
    sre = gelman_diagnostic(samples, burnin).sre
    ## get importance sample
    theta_w = collect_theta_weight(grid, n_theta)
    is_mu = zeros(n_theta)
    cv = zeros(length(is_mu),length(is_mu))
    # shared HMM fn:
    compute_is_mu_covar!(is_mu, cv, theta_w[1], theta_w[2])
    # grsp = mdl.sample_dispersal ^ n_theta
    is_output = ImportanceSample(is_mu, cv, theta_w[1], theta_w[2], 0, [-log(sum(theta_w[2]) / length(theta_w[2])), -log(sum(theta_w[2]) / (length(theta_w[2]) ^ (1 / n_theta)))])
    ## return results
    output = ARQMCMCSample(is_output, rejs, mdl.sample_interval, mdl.sample_limit, mdl.sample_dispersal, burnin, sre, time_ns() - start_time, fx, grid)
    println("- finished in ", print_runtime(output.run_time), ". (Iμ = ", round.(is_output.mu; sigdigits = C_PR_SIGDIG), "; Rμ = ", round.(rejs.mu; sigdigits = C_PR_SIGDIG), "; BME = ", round.(output.imp_sample.bme[1]; sigdigits = C_PR_SIGDIG), ")")
    return output
end

## run standard ARQMCMC analysis
"""
    run_arq_mcmc_analysis(model::ARQModel, priors = []; ... )

Run ARQMCMC analysis with `chains` Markov chains, where `n_chains > 1` the Gelman-Rubin convergence diagnostic is also run.

**Parameters**
- `model`               -- `ARQModel` (see docs.)
- `priors`              -- optional: prior distributions or density function. I.e. `Array` of `Function` or `Distributions.Distribution` types.
**Named parameters**
- `sample_dispersal`   -- the dispersal of intial samples.
- `sample_limit`        -- sample limit, should be increased when the variance of `model.pdf` is high (default: 1.)
- `n_chains`            -- number of Markov chains (default: 3.)
- `steps`               -- number of iterations.
- `burnin`              -- number of discarded samples.
- `tgt_ar`              -- acceptance rate (default: 0.33.)
- `jitter`              --  (default: 0.0.)
- `sample_cache`        -- the underlying model likelihood cache - can be retained and reused for future analyses.
"""
function run_arq_mcmc_analysis(model::ARQModel, priors::Array{Function,1};
    sample_dispersal::Int64 = C_DF_ARQ_SR, sample_limit::Int64 = C_DF_ARQ_SL, steps::Int64 = C_DF_MCMC_STEPS
    , burnin::Int64 = df_adapt_period(steps), n_chains::Int64 = C_DF_ARQ_MC, tgt_ar::Float64 = C_DF_ARQ_AR
    , jitter::Float64 = C_DF_ARQ_JT, sample_cache = Dict{Array{Int64, 1}, GridPoint}())

    output = []
    for i in eachindex(priors)
        println("Running: ARQMCMC analysis ",  length(priors) == 1 ? "" : string(i, " / ", length(priors)," -"), " (", n_chains, " x " , steps, " steps):")
        mdl = LikelihoodModel(model.pdf, model.sample_interval, model.sample_offset, sample_limit, sample_dispersal, jitter, priors[i])
        push!(output, run_inner_mcmc_analysis(mdl, false, steps, burnin, n_chains, tgt_ar, sample_cache))
    end
    length(priors) == 1 && (return output[1])
    return output
end

function run_arq_mcmc_analysis(model::ARQModel, prior::Function = get_df_arq_prior(); sample_dispersal::Int64 = C_DF_ARQ_SR, sample_limit::Int64 = C_DF_ARQ_SL, steps::Int64 = C_DF_MCMC_STEPS, burnin::Int64 = df_adapt_period(steps), n_chains::Int64 = C_DF_ARQ_MC, tgt_ar::Float64 = C_DF_ARQ_AR, jitter::Float64 = C_DF_ARQ_JT, sample_cache = Dict{Array{Int64, 1}, GridPoint}())
    prs = Array{Function,1}(undef, 1)
    prs[1] = prior
    return run_arq_mcmc_analysis(model, prs; sample_dispersal = sample_dispersal, sample_limit = sample_limit, steps = steps, burnin = burnin, n_chains = n_chains, tgt_ar = tgt_ar, jitter = jitter, sample_cache = sample_cache)
end

function run_arq_mcmc_analysis(model::ARQModel, priors::Array{Distributions.Distribution,1};
    sample_dispersal::Int64 = C_DF_ARQ_SR, sample_limit::Int64 = C_DF_ARQ_SL, steps::Int64 = C_DF_MCMC_STEPS
    , burnin::Int64 = df_adapt_period(steps), n_chains::Int64 = C_DF_ARQ_MC, tgt_ar::Float64 = C_DF_ARQ_AR
    , jitter::Float64 = C_DF_ARQ_JT, sample_cache = Dict{Array{Int64, 1}, GridPoint}())

    pfn = Array{Function,1}(undef, length(priors))
    for i in eachindex(priors)
        pfn[i] = get_arq_prior(priors[i])
    end
    return run_arq_mcmc_analysis(model, pfn; sample_dispersal = sample_dispersal, sample_limit = sample_limit, steps = steps, burnin = burnin, n_chains = n_chains, tgt_ar = tgt_ar, jitter = jitter, sample_cache = sample_cache)
end

function run_arq_mcmc_analysis(model::ARQModel, prior::Distributions.Distribution;
    sample_dispersal::Int64 = C_DF_ARQ_SR, sample_limit::Int64 = C_DF_ARQ_SL, steps::Int64 = C_DF_MCMC_STEPS
    , burnin::Int64 = df_adapt_period(steps), n_chains::Int64 = C_DF_ARQ_MC, tgt_ar::Float64 = C_DF_ARQ_AR
    , jitter::Float64 = C_DF_ARQ_JT, sample_cache = Dict{Array{Int64, 1}, GridPoint}())

    return run_arq_mcmc_analysis(model, [prior]; sample_dispersal = sample_dispersal, sample_limit = sample_limit, steps = steps, burnin = burnin, n_chains = n_chains, tgt_ar = tgt_ar, jitter = jitter, sample_cache = sample_cache)
end



end ## end of module

# ## run delayed acceptance ARQMCMC analysis
# """
#     run_daarq_mcmc_analysis(model, steps, adapt_period, chains::Int64 = 3)
#
# **Parameters**
# - `model`               -- `ARQModel` (see docs).
# - `sample_dispersal`   -- i.e. the length of each dimension in the importance sample.
# - `steps`               -- number of iterations.
# - `burnin`              -- number of discarded samples.
# - `chains`              -- number of Markov chains (default: 3).
# **Named parameters**
# - `jitter`              -- add random noise to samples (0.0 to 0.5).
# - `da_limit`            -- delayed acceptance 'limit', i.e. threshold (default: 1).
# - `tgt_ar`              -- acceptance rate (default: 0.33).
# - `retain_samples`      -- for evaluation only, can be safely ignored (default: true).
#
# Run delayed acceptance ARQMCMC analysis with `chains` Markov chains. Where `chains > 1` the Gelman-Rubin convergence diagnostic is also run.
# """
# function run_daarq_mcmc_analysis(model::ARQModel, sample_dispersal::Int64, steps::Int64, burnin::Int64, chains::Int64 = 3; jitter::Float64 = 0.25, da_limit::Int64 = 1, tgt_ar::Float64 = 0.33, retain_samples::Bool = true)
#     mdl = LikelihoodModel(model.pdf, model.parameter_range, sample_dispersal, da_limit, jitter)
#     println("running DAARQ MCMC analysis (", chains, " x " , steps, " steps):")
#     return run_inner_mcmc_analysis(mdl, true, steps, burnin, chains, tgt_ar, retain_samples)
# end

# ## augmented data ARQ MCMC
# """
#     run_adarq_mcmc_analysis(model, steps, adapt_period, chains::Int64 = 3)
#
# **Parameters**
# - `model`               -- `ADARQModel` (see docs).
# - `sample_dispersal`   -- i.e. the length of each dimension in the importance sample.
# - `steps`               -- number of iterations.
# - `burnin`              -- number of discarded samples.
# - `chains`              -- number of Markov chains (default: 3).
# **Named parameters**
# - `jitter`              -- add random noise to samples (0.0 to 0.5).
# - `da_limit`            -- delayed acceptance 'limit', i.e. threshold (default: steps).
# - `tgt_ar`              -- acceptance rate (default: 0.33).
# - `retain_samples`      -- for evaluation only, can be safely ignored (default: true).
#
# Run augmented data ARQ MCMC analysis with optional delayed acceptance (set `da_limit`). The Gelman-Rubin convergence diagnostic is run automatically.
# """
# function run_daq_mcmc_analysis(model::DAQModel, sample_dispersal::Int64, steps::Int64, burnin::Int64, chains::Int64 = 3, jitter::Float64 = 0.25, da_limit::Int64 = steps, tgt_ar::Float64 = 0.33, retain_samples::Bool = true)
#     ## for performance evaluation
#     start_time = time_ns()
#     # MERGE SOME OF THIS STUFF? ***
#     mdl = LikelihoodModel(model.pdf, model.parameter_range, sample_dispersal, da_limit, jitter)
#     println("running data augmented QMCMC analysis (", chains, " x " , steps, " steps):")
#     # return run_inner_mcmc_analysis(mdl, true, steps, burnin, chains, tgt_ar, retain_samples)
#     mcmc = Array{MCMCResults, 1}(undef, chains)
#     ## draw grid
#     # NEED TO ADD NEW STRUCTS
#     grid = Dict() # {Array{Int64, 1}, GridX}
#     is_mu = zeros(size(mdl.grid_range, 1))
#     ## run N chains
#     for i in eachindex(mcmc)
#         ## retain_samples && (grid = mcmc[i].grid)
#         retain_samples || (grid = Dict())
#         ## choose initial theta coords
#         theta_init = rand(1:mdl.sample_dispersal, length(is_mu))    #length(DF_THETA)
#         theta_i = get_theta_val(mdl, theta_init)
#         print(" initialising chain ", i, ": θ = ", round.(theta_i; sigdigits = C_PR_SIGDIG + 1))
#         ## initialise x0
#         x0 = model.generate_x0(theta_i)
#         ## run (delayed acceptance) augmented data ARQ MCMC
#         mcmc[i] = adarq_met_hastings!(grid, mdl, x0, steps, burnin, theta_init, tgt_ar)   #, hsd, log_ir_y, prop_type
#         retain_samples || (is_mu .+= compute_da_is_mean(grid, length(is_mu)))
#         println(" - complete (AR: ", round(sum(mcmc[i].mc_accepted[burnin:steps]) / (steps - burnin) * 100, digits = 1), "%).")
#     end
#     ## compute scale reduction factor est.
#     gmn = gelman_diagnostic(mcmc, length(is_mu), steps - burnin)
#     ## compute importance sample mean
#     if retain_samples
#         is_mu .= compute_da_is_mean(grid, length(is_mu))[1]
#     else
#         is_mu ./= length(mcmc)
#     end
#     ## return results
#     output = ARQMCMCResults(C_ALG_DAUG, time_ns() - start_time, mdl.sample_dispersal, mdl.sample_limit, mdl.jitter, steps, burnin, gmn[1], gmn[2], gmn[3], mdl.grid_range, grid, is_mu, 0.0, gmn[4], gmn[5], gmn[6], mcmc) #, prop_type
#     println("finished (sample μ = ", round.(output.mu; sigdigits = C_PR_SIGDIG), ").")
#     return output
# end
#
# function run_daq_mcmc_analysis(model::HiddenMarkovModel, theta_range::Array{Float64,2}, theta_resolution::Int64, steps::Int64, burnin::Int64, chains::Int64 = 3; jitter::Float64 = 0.25, da_limit::Int64 = steps, tgt_ar::Float64 = 0.33, retain_samples::Bool = true)
#     ## generate initial augmented data
#     function gen_x0(theta::Array{Float64,1})
#         x0::Particle = generate_x0(model, theta)
#         return AugDataResult(x0.log_like[2], x0)
#     end
#     ## evaluate pdf
#     function compute_density(x_i::AugDataResult, theta_f::Array{Float64,1})
#         ## make model based proposal
#         # current state
#         # pf = Discuit.ParameterProposal(theta_f, 1)  # C_DSCT_MODEL.prior_density(theta_f) - NOT USED CURRENTLY
#         mbp::Particle = model_based_proposal(model, theta_f, x_i.aug_data_var)
#         ## return as ARQMCMC type
#         return AugDataResult(mbp.log_like[2], mbp)
#     end
#     ## model
#     mdl = DAQModel(compute_density, gen_x0, theta_range)
#     return run_daq_mcmc_analysis(mdl, theta_resolution, steps, burnin, chains, jitter, da_limit, tgt_ar, retain_samples)
# end
