#### data augmented MCMC ####

## TO DO
# - add custom example for Robs

## global const
const C_INITIAL = 0.1                    # proposal scalar

## proposal step
macro initialise_mcmc()
    esc(quote
    steps::Int64 = size(theta, 2) #GET RID
    ADAPT_INTERVAL = adapt_period / C_MCMC_ADAPT_INTERVALS   # interval between adaptation steps
    xi = x0
    ## covar matrix
    covar = zeros(length(xi.theta), length(xi.theta))
    for i in eachindex(xi.theta)
        covar[i,i] = xi.theta[i] == 0.0 ? 1 : (xi.theta[i]^2)
    end
    propd = Distributions.MvNormal(covar)
    c = C_INITIAL                           # autotune
    theta[:,1,mc] .= xi.theta               # initial sample
    a_cnt = zeros(Int64, 2)
    a_cnt[1] = 1
    end)
end

## end of adaption period
macro mcmc_adapt_period()
    esc(quote
    if i % ADAPT_INTERVAL == 0
        # i == ADAPT_INTERVAL && println("first adapt:", i)
        covar = Distributions.cov(transpose(theta[:,1:i,mc]))
        propd = get_prop_density(covar, propd)
        # if sum(covar) == 0
        #     println("warning: low acceptance rate detected in adaptation period")
        # else
        #     propd = Distributions.MvNormal(covar)
        # end
    end
    end)
end

## adaptation
macro met_hastings_adapt()
    esc(quote
    ## adaptation
    if (!fin_adapt || i < adapt_period)
        c *= (accepted ? 1.002 : 0.999)
        @mcmc_adapt_period      # end of adaption period
    end
    end)
end

## adaptation
macro gibbs_adapt()
    esc(quote
    ## adaptation
    if (!fin_adapt || i < adapt_period)
        pp && (c *= (accepted ? 1.002 : 0.999))
        @mcmc_adapt_period      # end of adaption period
    end
    end)
end

## acceptance handling (also used by GS)
macro mcmc_handle_mh_step()
    esc(quote
    if accepted
        xi = xf
        a_cnt[i > adapt_period ? 2 : 1] += 1
    end
    theta[:,i,mc] .= xi.theta
    end)
end

## used by gibbs
function compute_full_log_like!(model::HiddenMarkovModel, p::Particle)
    t = model.t0_index > 0 ? p.theta[model.t0_index] : 0.0
    lambda = zeros(model.n_events)
    if length(p.trajectory) > 0 && p.trajectory[1].time < t
        p.log_like[1] = -Inf                    # void sequence
    else
        p.log_like[1] = 0.0                     # reset and initialise
        p.final_condition .= p.initial_condition
        evt_i = 1
        for obs_i in eachindex(model.obs_data)  # handle trajectory segments
            while evt_i <= length(p.trajectory)
                p.trajectory[evt_i].time > model.obs_data[obs_i].time && break
                model.rate_function(lambda, p.theta, p.final_condition)
                try
                    p.log_like[1] += log(lambda[p.trajectory[evt_i].event_type]) - (sum(lambda) * (p.trajectory[evt_i].time - t))
                catch
                    C_DEBUG && println("ERROR:\n theta := ", p.theta, "; pop := ", p.final_condition, "; r := ", lambda)
                    p.log_like[1] = -Inf
                    return
                end
                p.final_condition .+= model.fn_transition(p.trajectory[evt_i].event_type)
                if any(x->x<0, p.final_condition)
                    p.log_like[1] = -Inf
                    return
                else
                    t = p.trajectory[evt_i].time
                    evt_i += 1
                end
            end
            model.rate_function(lambda, p.theta, p.final_condition)
            p.log_like[1] += model.obs_model(model.obs_data[obs_i], p.final_condition, p.theta)
            p.log_like[1] -= sum(lambda) * (model.obs_data[obs_i].time - t)
            p.log_like[1] == -Inf && return
            t = model.obs_data[obs_i].time
        end
    end
end

## Single particle adaptive Metropolis Hastings algorithm
function met_hastings_alg!(theta::Array{Float64,3}, mc::Int64, model::HiddenMarkovModel, adapt_period::Int64, x0::Particle, proposal_alg::Function, fin_adapt::Bool)
    @initialise_mcmc
    for i in 2:steps            # met_hastings_step
        xf = proposal_alg(model, get_mv_param(propd, c, theta[:,i-1,mc]), xi)
        if (xf.prior == -Inf || xf.log_like[1] == -Inf)
            accepted = false    # reject automatically
        else                    # accept or reject
            # NB: [2] == full g(x) log like
            mh_prob::Float64 = exp(xf.prior - xi.prior) * exp(xf.log_like[1] - xi.log_like[1])
            accepted = (mh_prob > 1 || mh_prob > rand())
        end
        @mcmc_handle_mh_step    # handle accepted proposals
        # accepted && (xi = xf)
        # theta[:,i,mc] .= xi.theta
        @met_hastings_adapt     # adaptation
    end ## end of MCMC loop
    C_DEBUG && print(" - Xn := ", length(xi.trajectory), " events; ll := ", xi.log_like, " - ")
    return a_cnt
end

## Single particle adaptive Gibbs sampler - TO BE FINISHED ****
function gibbs_mh_alg!(theta::Array{Float64,3}, mc::Int64, model::HiddenMarkovModel, adapt_period::Int64, x0::Particle, proposal_alg::Function, fin_adapt::Bool, ppp::Float64, adapt_prop_alg::Function)
    @initialise_mcmc
    prop_fn = adapt_prop_alg
    for i in 2:steps            # Gibbs
        pp = rand() < ppp
        if pp                   # parameter proposal
            theta_f = get_mv_param(propd, c, theta[:,i-1,mc])
            xf = Particle(theta_f, xi.initial_condition, xi.final_condition, xi.trajectory, Distributions.logpdf(model.prior, theta_f), zeros(2))
        else                    # trajectory proposal
            xf = prop_fn(xi)
        end
        (xf.prior == -Inf || xf.log_like[2] == -Inf) || compute_full_log_like!(model, xf)
        if (xf.prior == -Inf || sum(xf.log_like) == -Inf)
            accepted = false    # reject automatically
        else                    # accept or reject
            # NB: [3] == proposal log like
            mh_prob::Float64 = exp(xf.prior - xi.prior) * exp(sum(xf.log_like[1:2]) - xi.log_like[1])
            accepted = (mh_prob > 1 || mh_prob > rand())
        end
        @mcmc_handle_mh_step    # handle accepted proposals
        @gibbs_adapt            # adaptation
        i == Int64(floor(adapt_period * 0.2)) && (prop_fn = proposal_alg)
    end ## end of MCMC loop
    C_DEBUG && print(" - Xn := ", length(xi.trajectory), " events; ll := ", xi.log_like, " - ")
    return a_cnt
end

## generic met hastings
# function generic_mcmc!(theta::Array{Float64,3}, mc::Int64, model::HiddenMarkovModel, adapt_period::Int64, theta0::Array{Float64,1}, target_log_density::Function) #, joint_proposals::Bool, ppp::Float64
#     steps::Int64 = size(theta,2) #GET RID
#     ADAPT_INTERVAL = adapt_period / 10   # interval between adaptation steps
#     ll_i = target_log_density(theta0)
#     # covar matrix
#     covar = zeros(length(theta0), length(theta0))
#     for i in eachindex(theta0)
#         covar[i,i] = 0.1 * (theta0[i] == 0.0 ? 1 : theta0[i]^2)
#     end
#     propd = Distributions.MvNormal(covar)
#     c = C_INITIAL
#     ## results
#     # theta = Array{Float64, 2}(undef, steps, length(theta0))
#     theta[:,mc,1,:] .= theta0 # FIX *
#     for i in 2:steps
#         theta[mc,i,:] = get_mv_param(propd, c, theta[mc,i-1,:])
#         ll_f = target_log_density(theta[mc,i,:])
#         if ll_f == -Inf
#             # reject automatically
#             accepted = false
#         else
#             # accept or reject
#             mh_prob::Float64 = exp(ll_f - ll_i)
#             accepted = (mh_prob > 1 || mh_prob > rand())
#         end
#         if accepted
#             ll_i = ll_f
#         else
#             theta[mc,i,:] .= theta[mc,i-1,:]
#         end
#         ## adaptation
#         if i < adapt_period
#             c *= (accepted ? 1.002 : 0.999)
#             # end of adaption period
#             if i % ADAPT_INTERVAL == 0
#                 covar = Distributions.cov(theta[:,1:i,mc])
#                 # output.cv .= Statistics.cov(reshape(theta[:,(ap+1):size(theta,3),:], d, size(theta,1)))
#                 if sum(covar) == 0
#                     println("warning: low acceptance rate detected in adaptation period")
#                 else
#                     propd = Distributions.MvNormal(covar)
#                 end
#             end
#         end
#     end ## end of MCMC loop
# end




#### public functions (custom framework) ####

sample_space(theta_init::Array{Float64,2}, steps::Int64) = zeros(size(theta_init,1), steps, size(theta_init,2))

## analyse results
macro mcmc_tidy_up()
    esc(quote
    rejs = handle_rej_samples(samples, adapt_period)
    gd = gelman_diagnostic(samples, adapt_period)         # run convergence diagnostic
    output = MCMCSample(rejs, adapt_period, gd.sre, time_ns() - start_time)
    println("- finished in ", print_runtime(output.run_time), ". E(x) := ", output.samples.mu)
    end)
end

## ADD GENERIC GIBBS RUN FN

## Std MCMC, i.e. gelman diagnostic
function run_std_mcmc(model::HiddenMarkovModel, theta_init::Array{Float64,2}, steps::Int64, adapt_period::Int64, fin_adapt::Bool, ppp::Float64, mvp::Int64)
    function x0_prop(theta::Array{Float64,1})
        x0 = generate_x0(model, theta)         # simulate initial particle
        compute_full_log_like!(model, x0)           # NB. sim initialises with OM ll only
        return x0
    end
    trajectory_prop =  get_std_mcmc_proposal_fn(model, mvp)
    adapt_tp = get_std_mcmc_proposal_fn(model, 2)
    println("Running: ", size(theta_init, 2) ,"-chain ", steps, "-sample ", fin_adapt ? "finite-" : "", "adaptive DA-MCMC analysis (model: ", model.model_name, ")")
    start_time = time_ns()
    samples = sample_space(theta_init, steps)
    for i in 1:size(theta_init,2)
        print(" initialising chain ", i)
        x0 = x0_prop(theta_init[:,i])
        ## run inference
        C_DEBUG && print(" with x0 := ", x0.theta, " (", length(x0.trajectory), " events)")
        a_cnt = gibbs_mh_alg!(samples, i, model, adapt_period, x0, trajectory_prop, fin_adapt, ppp, adapt_tp)
        println(" - complete (AAR := ", round(100 * a_cnt[2] / (steps - adapt_period), digits = 1), "%)")
    end
    @mcmc_tidy_up
    return output
end

## Std MCMC, i.e. gelman diagnostic
function run_custom_gibbs_mcmc(model::HiddenMarkovModel, theta_init::Array{Float64,2}, steps::Int64, adapt_period::Int64, fin_adapt::Bool, ppp::Float64, trajectory_prop::Function, x0_prop::Function)
    println("Running: ", size(theta_init, 2) ,"-chain ", steps, "-sample ", fin_adapt ? "finite-" : "", "adaptive custom DA-MCMC analysis (model: ", model.model_name, ")")
    start_time = time_ns()
    samples = sample_space(theta_init, steps)
    for i in 1:size(theta_init,2)
        print(" initialising chain ", i)
        x0 = x0_prop(theta_init[:,i])    # generate initial particle
        ## run inference
        C_DEBUG && print(" with x0 := ", x0.theta, " (", length(x0.trajectory), " events)")
        a_cnt = gibbs_mh_alg!(samples, i, model, adapt_period, x0, trajectory_prop, fin_adapt, ppp, trajectory_prop)
        println(" - complete (AAR := ", round(100 * a_cnt[2] / (steps - adapt_period), digits = 1), "%)")
    end
    @mcmc_tidy_up
    return output
end

## ADD DOCS
function generate_custom_particle(model::HiddenMarkovModel, trajectory::Array{Event,1}; theta::Array{Float64,1} = rand(model.prior), initial_condition::Array{Int64,1} = model.fn_initial_condition())
    p = Particle(theta, initial_condition, copy(initial_condition), trajectory, Distributions.logpdf(model.prior, theta), zeros(2))
    sort!(p.trajectory)
    compute_full_log_like!(model, p)
    return p
end

"""
    generate_custom_particle(model, obs_data, trajectory_prop, [x0_prop], ... )

For use with `run_custom_mcmc_analysis()`. Initialises a Particle based on an array of type Event. Also evaluates the likelihood function.

**Parameters**
- `model`               -- `DPOMPModel` (see [DCTMPs.jl models]@ref).
- `obs_data`            -- `Observations` data.
- `trajectory`          -- Array of Event types.

**Optional**
- `theta`               -- model parameters, sampled from prior unless otherwise specified.

"""
function generate_custom_particle(model::DPOMPModel, obs_data::Array{Observation,1}, trajectory::Array{Event,1}; theta::Array{Float64,1} = rand(model.prior))#, initial_condition::Array{Int64,1} = model.initial_condition
    hmm = get_private_model(model, obs_data)
    return generate_custom_particle(hmm, trajectory; theta = theta, initial_condition = model.initial_condition)
end

"""
    run_custom_mcmc_analysis(model, obs_data, trajectory_prop, [x0_prop], ... )

Run an `n_chains` data-augmented MCMC analysis, based on the Gibbs sampler with a user defined proposal function.

A function for conditionally sampling the initial trajectory variable can optionally be specified, use the Doob-Gillespie algorithm is used by default.

Elsewise, this function is equivalent to calling run_mcmc_analysis with `mbp = false`, which invokes the standard Gibbs sampler.

**Parameters**
- `model`               -- `DPOMPModel` (see [DCTMPs.jl models]@ref).
- `obs_data`            -- `Observations` data.
- `trajectory_prop`     -- .
- `x0_prop`             -- Initial state variable sampler, DGA by default.

**Optional**
- `n_chains`            -- number of Markov chains (optional, default: 3.)
- `initial_parameters`  -- 2d array of initial model parameters. Each column vector correspondes to a single model parameter.
- `steps`               -- number of iterations.
- `adapt_period`        -- number of discarded samples.
- `ppp`                 -- the proportion of parameter (vs. trajectory) proposals in Gibbs sampler. Default: 0.3, or 30%.
- `fin_adapt`           -- finite adaptive algorithm. The default is `false`, i.e. [fully] adaptive.

"""
function run_custom_mcmc_analysis(model::DPOMPModel, obs_data::Array{Observation,1}, trajectory_prop::Function, x0_prop::Function; n_chains::Int64 = 3, initial_parameters = rand(model.prior, n_chains), steps::Int64 = C_DF_MCMC_STEPS, adapt_period::Int64 = df_adapt_period(steps), fin_adapt::Bool = false, ppp::Float64 = 0.3)
    hmm = get_private_model(model, obs_data)
    return run_custom_gibbs_mcmc(hmm, initial_parameters, steps, adapt_period, fin_adapt, ppp, trajectory_prop, x0_prop)
end

## MBP MCMC
function run_mbp_mcmc(model::HiddenMarkovModel, theta_init::Array{Float64,2}, steps::Int64, adapt_period::Int64, fin_adapt::Bool)
    start_time = time_ns()
    # samples = zeros(size(theta_init,1), steps, size(theta_init,2))
    samples = sample_space(theta_init, steps)
    println("Running: ", size(theta_init, 2) ,"-chain ", steps, "-sample ", fin_adapt ? "finite-" : "", "adaptive MBP-MCMC analysis (model: ", model.model_name, ")")
    for i in 1:size(theta_init,2)
        print(" initialising chain ", i)
        x0 = generate_x0(model, theta_init[:,i])    # simulate initial particle
        ## run inference
        C_DEBUG && print( " with x0 := ", x0.theta, " (", length(x0.trajectory), " events)")
        a_cnt = met_hastings_alg!(samples, i, model, adapt_period, x0, model_based_proposal, fin_adapt)
        println(" - complete (AAR := ", round(100 * a_cnt[2] / (steps - adapt_period), digits = 1), "%)")
    end
    @mcmc_tidy_up
    return output
end

## particle MCMC
# - NEED TO FIX THIS FOR NEW THETA LAYOUT
function run_pmcmc(model::HiddenMarkovModel, theta_init::Array{Float64,2}, steps::Int64 = 50000, adapt_period::Int64 = 10000, p::Int64 = 200)
    start_time = time_ns()
    samples = sample_space(theta_init, steps)
    println("Running PMCMC analysis: " , size(theta_init, 2), " x ", steps, " samples")
    ## target density
    ps = length(model.fn_initial_condition())
    function comp_log_pdf(theta::Array{Float64, 1})
        return model.fn_log_prior(theta) + estimate_likelihood(model, theta, p, ps, rsp_systematic)
    end
    ## run inference
    for i in 1:size(theta_init,1)
        generic_mcmc!(samples, i, model, adapt_period, theta_init[:,i], comp_log_pdf)
        println(" chain ", i, " complete.")
    end
    @mcmc_tidy_up
    return output
end
