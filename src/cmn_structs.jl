## generic rejection sample
"""
    RejectionSample

Essentially, the main results of an MCMC analysis, consisting of samples, mean, and covariance matrix.

**Fields**
- `samples`         -- three dimensional array of samples, e.g. parameter; iteration; Markov chain.
- `mu`              -- sample mean.
- `cv`              -- sample covariance matrix.

"""
struct RejectionSample
    theta::Array{Float64,3}         # dims: theta index; chain; sample
    mu::Array{Float64,1}
    cv::Array{Float64,2}
end

## IBIS sample
"""
    ImportanceSample

The results of an importance sampling analysis, such as iterative batch importance sampling algorithms.

**Fields**
- `mu`              -- weighted sample mean.
- `cv`              -- weighted covariance.
- `theta`           -- two dimensional array of samples, e.g. parameter; iteration.
- `weight`          -- sample weights.
- `run_time`        -- application run time.
- `bme`             -- Estimate (or approximation) of the Bayesian model evidence.

"""
struct ImportanceSample
    mu::Array{Float64,1}
    cv::Array{Float64,2}
    theta::Array{Float64,2}
    weight::Array{Float64,1}
    run_time::UInt64
    bme::Array{Float64,1}
end
