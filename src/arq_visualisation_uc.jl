
## autocorrelation R
"""
    plot_autocorrelation(autocorrelation)

**Parameters**
- `autocorrelation`     -- The results from a call to `compute_autocorrelation`.

Plot autocorrelation for an MCMC analysis.
"""
function plot_autocorrelation(autocorrelation::AutocorrelationResults)
    # build y
    for i in eachindex(autocorrelation.autocorrelation)
        autocorrelation.autocorrelation[i] = max(autocorrelation.autocorrelation[i], 0)
    end
    # plot
    p = UnicodePlots.lineplot(autocorrelation.lag, autocorrelation.autocorrelation[:,1], title = string("θ autocorrelation"))
    for i in 2:size(autocorrelation.autocorrelation, 2)
        UnicodePlots.lineplot!(p, autocorrelation.lag, autocorrelation.autocorrelation[:,i])
    end
    UnicodePlots.xlabel!(p, "lag")
    UnicodePlots.ylabel!(p, "R")
    return p
end

function plot_parameter_trace(sample::ARQMCMCSample, parameter::Int64)
    return plot_parameter_trace(sample.samples, parameter)
end

## all parameters
function plot_parameter_trace(sample::ARQMCMCSample)
    return plot_parameter_trace.([sample], [i for i in eachindex(sample.sample_interval)])
end

## Multiple analyses
# function plot_parameter_trace(sample::Array{ARQMCMCSample,1}, parameter::Int64)
#     return plot_parameter_trace.(sample[for i in eachindex(sample)].samples, i)
# end

## ARQ
function plot_parameter_heatmap(sample::ARQMCMCSample, x_parameter::Int64, y_parameter::Int64; use_is::Bool = false)
    use_is && (return plot_parameter_heatmap(sample.imp_sample, x_parameter, y_parameter))
    return plot_parameter_heatmap(sample.samples, x_parameter, y_parameter, sample.adapt_period)
end

function plot_parameter_marginal(sample::ARQMCMCSample, parameter::Int64; use_is::Bool = false)
    x = use_is ? resample_is(sample.imp_sample) : sample.samples
    xx =  use_is ? x.theta[parameter,:,:] : x.theta[parameter, (sample.adapt_period+1):size(x.theta, 2), :]
    nbins = Int(round((maximum(xx) - minimum(xx)) / sample.sample_interval[parameter]))
    return plot_parameter_marginal(x, parameter, use_is ? 0 : sample.adapt_period, nbins)
end

## Multi
function plot_parameter_marginal(sample::ARQMCMCSample; use_is::Bool = false)
    return plot_parameter_marginal.([sample], [i for i in eachindex(sample.sample_interval)], use_is = use_is)
end
