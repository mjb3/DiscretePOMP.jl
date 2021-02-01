## model
#(see [Discuit.jl models](@ref) for further details)
"""
    ARQModel

Contains the PDF (or estimate, or approximation of the target density - a function) and parameter density, which together specify the model.

**Fields**
- `pdf`         -- prior density function.
- `sample_interval`     -- An array specifying the (fixed or fuzzy) interval between samples.
"""
struct ARQModel{PFT<:Function}
    pdf::PFT
    # parameter_range::Array{Float64, 2}
    sample_interval::Array{Float64,1}
    sample_offset::Array{Float64, 1}
end

## augmented data model
# """
#     DAQModel
#
# **Fields**
# - `pdf`         -- prior density function.
# - `grid_range`  -- matrix representing the upper and lower limits of the parameter space.
#
# Like `ARQModel` but for an augmented data model. The `pdf` has a signature like `pdf(xi::AugDataResult, theta::Array{Float64})` and must also return an `AugDataResult` (see the docs for further information).
# """
# struct DAQModel{PFT<:Function, XFT<:Function}
#     pdf::PFT
#     generate_x0::XFT
#     parameter_range::Array{Float64, 2}
# end

## for internal use only
struct LikelihoodModel{PFT<:Function, PRT<:Function}
    pdf::PFT
    sample_interval::Array{Float64, 1}
    sample_offset::Array{Float64, 1}
    sample_limit::Int64
    sample_dispersal::Int64
    jitter::Float64
    prior::PRT
end

## DA grid sample
# struct GridSample
#     sample::Array{Float64, 1}   # i.e. theta
#     log_likelihood::Float64
# end

## DA grid 'value'
# struct GridSet
#     anchor::Array{Float64, 1}   # i.e. theta REQ'D? ***
#     samples::Array{GridSample, 1}
#     # log_likelihood::Float64     # i.e weighted density
# end

## DA grid request
# struct DAGridRequest
#     set::GridSet
#     result::GridSample
#     delayed::Bool
# end

## grid point (internal)
struct GridPoint
    sample::Array{Float64, 1}   # i.e. theta
    log_likelihood::Float64
    visited::Int64  # i.e. # times PDF computed (max: N)
    sampled::Int64  # number of adapted samples
end

## results for a density estimate (internal)
struct GridRequest
    result::GridPoint
    prior::Float64
    process_run::Bool
end

## autocorrelation results
"""
    AutocorrelationResults

**Fields**
- `lag`             -- autocorrelation lag.
- `autocorrelation` -- autocorrelation statistics.

Results of a call to `compute_autocorrelation`.
"""
struct AutocorrelationResults
    lag::Array{Int64,1}
    autocorrelation::Array{Float64,2}
end

## ARQMCMC
"""
    ARQMCMCSample

The results of an ARQ MCMC analysis including the ImportanceSample and resampled RejectionSample.

The `sre` scale factor reduction estimates relate the rejection (re)samples to the underlying importance sample.

**Fields**
- `imp_sample`          -- main results, i.e. ImportanceSample.
- `samples`             -- resamples, of type RejectionSample.
- `adapt_period`        -- adaptation (i.e. 'burn in') period.
- `sample_dispersal`    -- number of distinct [possible] sample values along each dimension in the unit cube.
- `sample_limit`        -- maximum number of samples per theta tupple.
- `grid_range`          -- bounds of the parameter space.
- `sre`                 -- scale reduction factor estimate, i.e. Gelman diagnostic. NB. *only valid for resamples*.
- `run_time`            -- application run time.
- `sample_cache`        -- a link to the underlying likelihood cache - can be reused.
"""
struct ARQMCMCSample
    imp_sample::ImportanceSample
    samples::RejectionSample
    sample_interval::Array{Float64,1}
    sample_limit::Int64
    sample_dispersal::Int64
    adapt_period::Int64
    #     jitter::Float64
    sre::Array{Float64,2}
    run_time::UInt64
    fx::Array{Int64, 1}
    sample_cache::Dict{Array{Int64, 1}, GridPoint}
end
