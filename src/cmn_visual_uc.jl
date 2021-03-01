##
"""
    plot_parameter_trace(mcmc, [parameter::Int64])

Produce a trace plot of samples using [UnicodePlots.jl](https://github.com/Evizero/UnicodePlots.jl).

The `mcmc` input is of type `MCMCSample`, `ARQMCMCSample` or `RejectionSample`. The `parameter` index can be optionally specified, else all parameters are plotted and returned as an `Array` of unicode plots.
"""
function plot_parameter_trace(sample::RejectionSample, parameter::Int64)
    x = 1:size(sample.theta, 2)
    yl = [floor(minimum(sample.theta[parameter,:,:]), sigdigits = 2), ceil(maximum(sample.theta[parameter,:,:]), sigdigits = 2)]
    p = UnicodePlots.lineplot(x, sample.theta[parameter,:,1], title = string("θ", Char(8320 + parameter), " traceplot."), ylim = yl)
    for i in 2:size(sample.theta, 3)
        UnicodePlots.lineplot!(p, sample.theta[parameter,:,i])
    end
    UnicodePlots.xlabel!(p, "sample")
    UnicodePlots.ylabel!(p, string("θ", Char(8320 + parameter)))
    return p
end
