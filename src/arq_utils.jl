### for internal use:

### compute importance sample mean

## collect theta and weights
function collect_theta_weight(grid::Dict{Array{Int64, 1}, GridPoint}, np::Int64)
    theta = zeros(np, length(grid))
    w = zeros(length(grid))
    for (index, value) in enumerate(grid)
        theta[:,index] .= value.second.sample
        w[index] = exp(value.second.log_likelihood)
    end
    return (theta, w) # ADD LENGTH OF GRID ***********
end

## compute autocorrelation for a single lag (C samples given mu and wcv)
function compute_autocorrelation(samples::Array{Float64, 2}, mu_var::Array{Float64, 2}, lag::Int64)
    output = zeros(size(mu_var, 1))
    # for each param:
    for j in eachindex(output)
        for i in 1:(size(samples, 1) - lag)
            output[j] += (samples[i,j] - mu_var[j,1]) * (samples[i + lag, j] - mu_var[j,1])
        end
        output[j] /= (size(samples, 1) - lag) * mu_var[j,2]
    end
    return output
end

## see Pooley 2018 DIC comparison
# function model_evidence_ic(p_y::Float64)
#     return -2 * log(p_y)
# end

## arq mcmc analysis:
function tabulate_results(results::ARQMCMCSample; display::Bool = true)
    d = Matrix(undef, length(results.imp_sample.mu), 7)
    rj_sd = compute_sigma(results.samples.cv)
    is_sd = compute_sigma(results.imp_sample.cv)
    d[:,1] .= 1:length(results.imp_sample.mu)
    d[:,2] .= round.(results.samples.mu; sigdigits = C_PR_SIGDIG)
    d[:,3] .= round.(rj_sd; sigdigits = C_PR_SIGDIG)
    d[:,4] .= round.(results.imp_sample.mu; sigdigits = C_PR_SIGDIG)
    d[:,5] .= round.(is_sd; sigdigits = C_PR_SIGDIG)
    d[:,6] .= round.(results.sre[:,2]; sigdigits = C_PR_SIGDIG)
    d[:,7] .= round.(results.sre[:,3]; sigdigits = C_PR_SIGDIG)
    # d[:,8] .= 0
    # bme_seq = C_DEBUG ? (1:2) : (1:1)
    # d[bme_seq,8] = round.(results.imp_sample.bme[bme_seq]; digits = 1)
    if display
        h = ["θ", "E[θ]", ":σ", "E[f(θ)]", ":σ", "SRE", "SRE975"]
        PrettyTables.pretty_table(d, h)
    else
        h = ["θ", "e_x", "sd_x", "e_fx", "sd_fx", "SRE", "SRE975"]
        return DataFrames.DataFrame(d, h)
    end
end

## print autocorrelation
"""
    print_autocorrelation(autocorrelation, fpath)

**Parameters**
- `autocorrelation` -- the results of a call to `compute_autocorrelation`.
- `fpath`           -- the file path of the destination file.

Save the results from a call to `compute_autocorrelation` to the file `fpath`, e.g. "./out/ac.csv".
"""
function print_autocorrelation(acr::AutocorrelationResults, fpath::String)
    open(fpath, "w") do f
        # print headers
        write(f, "lag")
        for j in 1:size(acr.autocorrelation, 2)
            write(f, ", x$j")
        end
        # print autocorr
        for i in 1:size(acr.autocorrelation, 1)
            # write(f, "\n$((i - 1) * AC_LAG_INT)")
            write(f, "\n$(acr.lag[i])")
            for j in 1:size(acr.autocorrelation, 2)
                write(f, ",$(acr.autocorrelation[i,j])")
            end
        end
    end
end

## print arq mcmc results
function print_results(results::ARQMCMCSample, dpath::String)
    isdir(dpath) || mkpath(dpath)                       # check dir
    open(string(dpath, "metadata.csv"), "w") do f       # print metadata
        write(f, "alg,np,adapt_period,sample_limit,sample_dispersal,run_time,fx,bme\narq,")
        write(f, "$(length(results.imp_sample.mu)),$(results.adapt_period),$(results.sample_limit),$(results.sample_dispersal),$(results.run_time),$(sum(results.fx)),$(results.imp_sample.bme[1])")
    end
    open(string(dpath, "sinterval.csv"), "w") do f      # print grid range
        write(f, "h")
        for i in eachindex(results.sample_interval)
            write(f, "\n$(results.sample_interval[i])")
        end
    end
    open(string(dpath, "fx.csv"), "w") do f      # print grid range
        write(f, "mc,fx")
        for i in eachindex(results.fx)
            write(f, "\n$i,$(results.fx[i])")
        end
    end
    print_imp_sample(results.imp_sample, dpath)         # print importance sample
    print_rej_sample(results.samples, dpath, results.sre)   # print MCMC resamples
end
