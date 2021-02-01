
## does what it says on the tin
# function read_obs_from_file(fp::String) #, col
#     df = CSV.read(fp)
#     nc = size(df, 2)
#     return Observations(df[1], df[2:nc])
# end

## get observations data from DataFrame or file location
## NEED TO UPDATE FOR SIM PROP ********
# NB. OLD OBS FILES WILL THEN BE VOID *
"""
    get_observations(source)

Return an array of type `Observation`, based on a two-dimensional array, `DataFrame` or file location (i.e. `String`.)

Note that a observation times must be in the first column of the input variable.
"""
function get_observations(df::DataFrames.DataFrame; time_col=1, type_col=0, val_seq=2:size(df,2))
    obs = Observation[]
    for i in 1:size(df,1)
        obs_type = type_col==0 ? 1 : df[i,type_col]
        push!(obs, Observation(df[i,time_col], obs_type, 1.0, df[i,val_seq]))
    end
    sort!(obs)
    return obs
end
function get_observations(fpath::String)
    df = CSV.read(fpath, DataFrames.DataFrame)
    return get_observations(df)
end

## save simulation results to file
function print_results(results::SimResults, dpath::String)
    # check dir
    isdir(dpath) || mkpath(dpath)
    # print sequence
    open(string(dpath, "sim.csv"), "w") do f
        # print headers
        write(f, "time, event")
        for p in 1:size(results.population, 2)
            write(f, ",$p")
        end
        # print event sequence
        for i in eachindex(results.particle.trajectory)
            tm = results.particle.trajectory[i].time
            tp = results.particle.trajectory[i].event_type
            write(f, "\n$tm,$tp")
            for p in 1:size(results.population, 2)
                write(f, ",$(results.population[i,p])")
            end
        end
    end # end of print sequence
    # print observation
    open(string(dpath, "obs.csv"), "w") do f
        # print headers
        write(f, "time,id")
        for p in eachindex(results.observations[1].val)
            # c = model.model_name[p]
            write(f, ",$p")
        end
        # print event sequence
        for i in eachindex(results.observations)
            write(f, "\n$(results.observations[i].time),$(results.observations[i].obs_id)")
            tp = results.observations[i].val
            for p in eachindex(tp)
                write(f, ",$(tp[p])")
            end
        end
    end # end of print observations
end

#### print samples ####

## print theta summary (internal use)
function print_sample_summary(results, dpath::String)
    # print theta summary
    open(string(dpath, "summary.csv"), "w") do f
        # print headers
        write(f, "theta,mu,sigma")
        # print data
        for p in eachindex(results.mu)
            # c = model.model_name[p]
            write(f, "\n$p,$(results.mu[p]),$(sqrt(results.cv[p,p]))")
        end
    end
end

## print rejection sample (just the samples, summary and sre)
function print_rej_sample(samples::RejectionSample, dpath::String, gelman::Array{Float64,2})
    # print rejection/re samples
    open(string(dpath, "samples.csv"), "w") do f
        # print headers
        write(f, "mc,iter")
        for i in 1:size(samples.theta, 1)
            write(f, ",$i")
        end
        # print data
        for mc in 1:size(samples.theta, 3)
            for i in 1:size(samples.theta, 2)
                write(f, "\n$(mc),$i")
                for p in 1:size(samples.theta, 1)
                    write(f, ",$(samples.theta[p,i,mc])")
                end
            end
        end
    end
    # print theta summary
    print_sample_summary(samples, string(dpath, "rj_"))
    # print gelman results (make optional?)
    open(string(dpath, "gelman.csv"), "w") do f
        # print headers
        write(f, "theta,sre_ll,sre,sre_ul")
        # print data
        for p in eachindex(samples.mu)
            write(f, "\n$p,$(gelman[p,1]),$(gelman[p,2]),$(gelman[p,3])")
        end
    end
end

## print importance sample (just the weighed sample and summary)
function print_imp_sample(results::ImportanceSample, dpath::String)
    # check dir
    isdir(dpath) || mkpath(dpath)
    # print importance samples
    open(string(dpath, "theta.csv"), "w") do f
        # print headers
        write(f, "1")
        for i in 2:length(results.mu)
            write(f, ",$i")
        end
        # print data
        for i in 1:size(results.theta, 2)
            write(f, "\n$(results.theta[1,i])")
            for p in 2:length(results.mu)
                write(f, ",$(results.theta[p,i])")
            end
        end
    end
    # print weights
    open(string(dpath, "weight.csv"), "w") do f
        # print headers
        write(f, "i,w")
        for i in eachindex(results.weight)
            write(f, "\n$i,$(results.weight[i])")
        end
    end
    # print theta summary
    print_sample_summary(results, string(dpath, "is_"))
end

## print importance sample results
"""
    print_results

Print the results of a parameter inference analysis to file.

**Parameters**
- `samples`     -- a data structure of type `SimResults`, `ImportanceSample`, `MCMCSample` or `ARQMCMCSample`.
- `dpath`       -- the directory where the results will be saved.

"""
function print_results(results::ImportanceSample, dpath::String)
    # check dir
    isdir(dpath) || mkpath(dpath)
    # print metadata
    open(string(dpath, "metadata.csv"), "w") do f
        write(f, "alg,np,run_time,bme\nibis,$(length(results.mu)),$(results.run_time),$(results.bme[1])")
    end
    ##
    print_imp_sample(results, dpath)
end



## print MCMC sample
function print_results(results::MCMCSample, dpath::String)
    # check dir
    isdir(dpath) || mkpath(dpath)
    # print metadata
    open(string(dpath, "metadata.csv"), "w") do f
        write(f, "alg,np,adapt_period,run_time\nmcmc,")
        write(f, "$(length(results.samples.mu)),$(results.adapt_period),$(results.run_time)")
    end
    # print rejection/re samples
    print_rej_sample(results.samples, dpath, results.sre)
end

#### tabulate stuff ####

## results summary
#- `proposals`   -- display proposal analysis (MCMC only).
"""
    tabulate_results

Display the results of an inference analysis.

The main parameter is `results` -- a data structure of type `MCMCSample`, `ImportanceSample`, `ARQMCMCSample` or `ModelComparisonResults`. When invoking the latter, the named parameter `null_index = 1` by default but can be overridden. This determines the 'null' model, used to compute the Bayes factor.
"""
function tabulate_results(results::MCMCSample)
    ## proposals
    # proposals && tabulate_proposals(results)
    ## samples
    # println("MCMC results:")
    h = ["θ", "E[θ]", ":σ", "SRE", "SRE975"]
    d = Matrix(undef, length(results.samples.mu), 5)
    sd = compute_sigma(results.samples.cv)
    d[:,1] .= 1:length(results.samples.mu)
    d[:,2] .= round.(results.samples.mu; sigdigits = C_PR_SIGDIG)
    d[:,3] .= round.(sd; sigdigits = C_PR_SIGDIG)
    d[:,4] .= round.(results.sre[:,2]; sigdigits = C_PR_SIGDIG + 1)
    d[:,5] .= round.(results.sre[:,3]; sigdigits = C_PR_SIGDIG + 1)
    PrettyTables.pretty_table(d, h)
end

## importance sample:
function tabulate_results(results::ImportanceSample)
    ## samples
    # println("IBIS results:")
    h = ["θ", "E[θ]", ":σ", C_LBL_BME]
    d = Matrix(undef, length(results.mu), 4)
    sd = compute_sigma(results.cv)
    d[:,1] .= 1:length(results.mu)
    d[:,2] .= round.(results.mu; sigdigits = C_PR_SIGDIG)
    d[:,3] .= round.(sd; sigdigits = C_PR_SIGDIG)
    d[:,4] .= 0
    bme_seq = C_DEBUG ? (1:2) : (1:1)
    d[bme_seq, 4] = round.(results.bme[bme_seq]; digits = 1)
    PrettyTables.pretty_table(d, h)
end

function tabulate_results(results::ARQMCMCSample)
    ARQMCMC.tabulate_results(results)
end


## IS resampler - artifical RejectionSamples
function resample_is(sample::ImportanceSample; n = 10000)
    rsi = StatsBase.sample(collect(1:length(sample.weight)), StatsBase.Weights(sample.weight), n)
    resamples = zeros(length(sample.mu), n, 1)
    for i in eachindex(rsi)
        resamples[:,i,1] .= sample.theta[:,rsi[i]]
    end
    return RejectionSample(resamples, sample.mu, sample.cv)
end

function compute_bayes_factor(ml::Array{Float64,1}, null_index::Int64)
    output = exp.(-ml)
    output ./= output[1]
    return output
end

## model evidence comparison
function tabulate_results(results::ModelComparisonResults; null_index = 1)
    h = ["Model", string("ln E[p(y)]"), ":σ", "BF"]   # ADD THETA ******************
    d = Matrix(undef, length(results.mu), length(h))
    d[:,1] .= results.names
    d[:,2] .= round.(results.mu; digits = 1)
    d[:,3] .= round.(results.sigma; sigdigits = C_PR_SIGDIG)
    d[:,4] .= round.(compute_bayes_factor(results.mu, null_index); digits = 1)
    PrettyTables.pretty_table(d, h)
end

## pf interface
"""
    get_particle_filter_lpdf(model, obs_data; ... )

Generate a SMC [log] likelihood estimating function for a `DPOMPModel` -- a.k.a. a particle filter.

**Parameters**
- `model`               -- `DPOMPModel` (see [DCTMPs.jl models]@ref).
- `obs_data`            -- `Observations` data.

**Named parameters**
- `np`                  -- number of particles.
- `rs_type`             -- resampling method: 2 for stratified, 3 for multinomial, or 1 for systematic (the default.)
- `ess_rs_crit`         -- effective-sample-size resampling criteria.
```
"""
function get_particle_filter_lpdf(model::DPOMPModel, obs_data::Array{Observation,1}; np::Int64 = C_DF_PF_P, rs_type::Int64 = 1, essc::Float64 = C_DF_ESS_CRIT)
    mdl = get_private_model(model, obs_data)
    return get_log_pdf_fn(mdl, np, rs_type; essc = essc)
end
