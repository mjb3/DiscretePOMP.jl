## event
"""
    Event

Requires no explanation.

**Fields**
- `time`        -- the time of the event.
- `event_type`  -- indexes the rate function and transition matrix.

"""
struct Event
    time::Float64
    event_type::Int64
end

## observation tuple
"""
    Observation

A single observation. Note that by default `val` has the same size as the model state space. However that is not necessary - it need only be compatible with the observation model.

**Fields**
- `time`        -- similar to `Event.time`, the time of the observation.
- `obs_id`      -- <1 if not a resampling step.
- `prop`        -- optional information for the observation model.
- `val`         -- the observation value.

"""
struct Observation
    time::Float64
    obs_id::Int64   # <1 if not a resampling step
    prop::Float64 #df: 1.0
    val::Array{Int64,1}
end

## a single realisation of the model
"""
    Particle

E.g. the main results of a simulation including the initial and final conditions, but not the full state trajectory.

**Fields**
- `theta`               -- e.g. simulation parameters.
- `initial_condition`   -- initial system state.
- `final_condition`     -- final system state.
- `trajectory`          -- the event history.
- `log_like`            -- trajectory log likelihood, mainly for internal use.

"""
struct Particle
    theta::Array{Float64,1}
    initial_condition::Array{Int64}
    final_condition::Array{Int64}
    trajectory::Array{Event,1}
    prior::Float64              # log prior;
    log_like::Array{Float64,1}  # full log like g(x); [latest] marginal g(x) / proposal likelihood (SMC / MCMC)
end

# - for dependent f/g
struct DFGParticle
    theta::Array{Float64,1}
    initial_condition::Array{Int64}
    final_condition::Array{Int64}
    trajectory::Array{Event,1}
    log_like::Array{Float64,1}  # prior, g(x)
    g_trans::Array{Int64,2}
end

## results of gillespie sim
"""
    SimResults

The results of a simulation, including the full state trajectory.

**Fields**
- `model_name`      -- string, e,g, `"SIR"`.
- `particle`        -- the 'trajectory' variable, of type `Particle`.
- `population`      -- records the final system state.
- `observations`    -- simulated observations data (an `Array` of `Observation` types.)

"""
struct SimResults
    model_name::String
    particle::Particle
    population::Array{Array{Int64},1}
    observations::Array{Observation,1}
end

## public model
"""
    DPOMPModel

A `mutable struct` which represents a DSSCT model (see [Models](@ref) for further details).

**Fields**
- `model_name`          -- string, e,g, `"SIR"`.
- `rate_function`       -- event rate function.
- `initial_condition`   -- initial condition.
- `m_transition`        -- transition matrix.
- `obs_function         -- observation function, use this to add 'noise' to simulated observations.
- `obs_model`           -- observation model likelihood function.
- `prior`               -- prior [multivariate] Distributions.Distribution.
- `t0_index`            -- index of the parameter that represents the initial time. `0` if fixed at `0.0`.

"""
mutable struct DPOMPModel
    model_name::String                  # model name
    rate_function::Function             # computes event rates (in place)
    initial_condition::Array{Int64,1}   # sets (e.g. draw a sample from some known density) initial condition
    m_transition::Array{Int64,2}        # i.e adjusts the population according to event type
    obs_function::Function              # observation function (sim only) - TO BE REMOVED?
    obs_model::Function                 # observation model (log likelihood)
    prior::Distributions.Distribution   # prior distribution
    t0_index::Int64                     # == 0 if initial time known
end

## DAC private model
struct HiddenMarkovModel{RFT<:Function, ICT<:Function, TFT<:Function, OFT<:Function, OMT<:Function, PFT<:Distributions.Distribution}
    model_name::String                  # model name
    n_events::Int64                     # number of event types
    rate_function::RFT                  # computes event rates (in place)
    fn_initial_condition::ICT           # sets (e.g. draw a sample from some known density) initial condition
    fn_transition::TFT                  # i.e adjusts the population according to event type
    obs_function::OFT                   # observation function (sim only) - TO BE REMOVED
    obs_model::OMT                      # observation model (log likelihood)
    obs_data::Array{Observation,1}      # obs data
    prior::PFT   # prior distribution
    t0_index::Int64                     # == 0 if initial time known
end



## MBP MCMC, PMCMC
"""
    MCMCSample

The results of an MCMC analysis, mainly consisting of a `RejectionSample`.

**Fields**
- `samples`         -- samples of type `RejectionSample`.
- `adapt_period`    -- adaptation (i.e. 'burn in') period.
- `sre`             -- scale reduction factor estimate, i.e. Gelman diagnostic.
- `run_time`        -- application run time.

"""
struct MCMCSample
    samples::RejectionSample
    adapt_period::Int64
    sre::Array{Float64,2}
    run_time::UInt64
end

"""
    ModelComparisonResults

The results of a model comparison, based on the Bayesian model evidence (BME.)

**Fields**
- `names`       -- model names.
- `bme`         -- matrix of BME estimates.
- `mu`          -- vector of BME estimate means (by model.)
- `sigma`       -- vector of BME estimate standard deviations.
- `n_runs`      -- number of independent analyses for each model.
- `run_time`    -- total application run time.

"""
struct ModelComparisonResults
    names::Array{String, 1}
    bme::Array{Float64,2}
    mu::Array{Float64,1}
    sigma::Array{Float64,1}
    n_runs::Int64
    run_time::UInt64
    theta_mu::Array{Array{Float64,1}, 2}
end
