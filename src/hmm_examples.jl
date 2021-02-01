# ## generalised HMM pf
# include("hmm_particle_filter.jl")
# import .HMMParticleFilter

## dummy obs function - MAKE PUBLIC
function dmy_obs_fn(y::Observation, population::Array{Int64,1}, parameters::Array{Float64,1})
    y.val .= population
end

## gen transition fn - MAKE PUBLIC
function generate_trans_fn(tm::Array{Int64})
    function fnt(et::Int64)
        return tm[et, :]
    end
    return fnt
end

## generate private model
function get_private_model(m::DPOMPModel, y::Array{Observation,1})
    fnic() = m.initial_condition
    return HiddenMarkovModel(m.model_name, size(m.m_transition,1), m.rate_function, fnic, generate_trans_fn(m.m_transition), m.obs_function, m.obs_model, y, m.prior, m.t0_index)
end

## uninformative prior distribution generator
"""
    generate_weak_prior(n)

**Parameters**

- `n`   -- the number of parameters in the model.
- `b`   -- the upper bound of the [Uniform] distribution.

# Examples

    generate_weak_prior(1)

Generate a "weak" prior distribution, Uniform multivariate ~ U(0, max) for dim = n,  where `n` is the number of parameters in the model.
"""
function generate_weak_prior(n::Int, b::Float64 = 1.0)
    return Distributions.Product(Distributions.Uniform.(zeros(n), b))
end

## gaussian observation likelihood model
"""
    partial_gaussian_obs_model(σ = 2.0; seq = 2, y_seq = seq)

Generate a simple Gaussian observation model. So-called because the accuracy of observations is 'known' and [assumed to be] normally distributed according to~N(0, σ), where observation error `σ` can be specified by the user.

**Parameters**
- `σ`       -- observation error.
- `seq`     -- the indexing sequence of the observed state, e.g. `2` for that state only, `3:4` for the third and fourth, etc.
- `y_seq`   -- as above, the corresponding [indexing] values for the observations data, `seq` unless otherwise specified.

test latex eqn:

```math
\frac{n!}{k!(n - k)!} = \binom{n}{k}
```

# Examples

    p = partial_gaussian_obs_model(1.0, seq = 2)

"""

## generic Gaussian obs model generator
function partial_gaussian_obs_model(σ::Float64 = 2.0; seq = 2:2, y_seq = seq)
    # do some preliminary computation
    tmp1 = log(1 / (sqrt(2 * pi) * σ))
    tmp2 = 2 * σ * σ
    function gom2(y::Observation, population::Array{Int64,1}, theta::Array{Float64,1})
        return tmp1 - (( (sum(y.val[y_seq]) - sum(population[seq])) ^ 2 ) / tmp2)
    end
    return gom2
end

## PUBLIC FUNCTION
"""
    generate_model(model_name, initial_condition; freq_dep = false, obs_error = 2.0)

Generates an `DPOMPModel` instance. Observation models are generated using the `partial_gaussian_obs_model` function, with ``σ = obs_error` (see that functions entry for further details.)

**Parameters**
- `model_name`          -- the model, e.g. "SI"; "SIR"; "SEIR"; etc
- `initial_condition`   -- initial condition.

**Optional parameters**
- `freq_dep`            -- epidemiological models only, set to `true` for frequency-dependent contact rates.
- `obs_error`           -- average observation error (default = 2.)

`model_name` **options**
- `"SI"`
- `"SIR"`
- `"SIS"`
- `"SEI"`
- `"SEIR"`
- `"SEIS"`
- `"SEIRS"`
- `"PREDPREY"`
- `"ROSSMAC"`

# Examples

    generate_model("SIS", [100,1])

"""
function generate_model(model_name::String, initial_condition::Array{Int64, 1}; freq_dep = false, obs_error = 2.0)
    ### density dependent rate functions ###

    ## SI rate function
    function si_rf(output, parameters::Array{Float64, 1}, population::Array{Int64, 1})
        output[1] = parameters[1] * population[1] * population[2]
    end
    ## SIR/SIS rate function
    function sir_rf(output, parameters::Array{Float64, 1}, population::Array{Int64, 1})
        output[1] = parameters[1] * population[1] * population[2]
        output[2] = parameters[2] * population[2]
    end
    ## SEI rate function
    function sei_rf(output, parameters::Array{Float64, 1}, population::Array{Int64, 1})
        output[1] = parameters[1] * population[1] * population[3]
        output[2] = parameters[2] * population[2]
    end
    ## SEIR rate function
    function seir_rf(output, parameters::Array{Float64, 1}, population::Array{Int64, 1})
        output[1] = parameters[1] * population[1] * population[3]
        output[2] = parameters[2] * population[2]
        output[3] = parameters[3] * population[3]
    end

    ### frequency dependent ###

    ## SI rate function
    function si_rf_fd(output, parameters::Array{Float64, 1}, population::Array{Int64, 1})
        output[1] = parameters[1] * population[1] * population[2] / sum(population)
    end
    ## SIR/SIS rate function
    function sir_rf_fd(output, parameters::Array{Float64, 1}, population::Array{Int64, 1})
        output[1] = parameters[1] * population[1] * population[2] / sum(population)
        output[2] = parameters[2] * population[2]
    end
    ## SEI rate function
    function sei_rf_fd(output, parameters::Array{Float64, 1}, population::Array{Int64, 1})
        output[1] = parameters[1] * population[1] * population[3] / sum(population)
        output[2] = parameters[2] * population[2]
    end
    ## SEIR/SEIS rate function
    function seir_rf_fd(output, parameters::Array{Float64, 1}, population::Array{Int64, 1})
        output[1] = parameters[1] * population[1] * population[3] / sum(population)
        output[2] = parameters[2] * population[2]
        output[3] = parameters[3] * population[3]
    end

    ### OTHER ###

    ## Lotka-Volterra
    function lotka_rf(output, parameters::Array{Float64, 1}, population::Array{Int64, 1})
        # prey; predator reproduction; predator death
        output[1] = parameters[1] * population[2]
        output[2] = parameters[2] * population[1] * population[2]
        output[3] = parameters[3] * population[1]
    end

    ## rate function
    # mosquito lifespan ^-1
    # MOS_DEATH = 0.02
    function rossmac_rf(output, parameters::Array{Float64, 1}, population::Array{Int64, 1})
        # mosquito population dynamics (birth; deaths)
        output[1] = parameters[1] * (population[3] + population[4])
        output[2] = parameters[1] * population[3]
        output[3] = parameters[1] * population[4]
        # disease dynamics (human, mosquito infection; human recovery)
        output[4] = parameters[2] * (population[1] * population[4] / (population[1] + population[2]))
        output[5] = parameters[3] * (population[2] * population[3] / (population[3] + population[4]))
        output[6] = parameters[4] * population[2]
    end

    ### MODEL
    if model_name == "SI"
        rate_fn = freq_dep ? si_rf_fd : si_rf
        m_transition = [-1 1;]
        obs_model = partial_gaussian_obs_model(obs_error)
    elseif model_name == "SIR"
        rate_fn = freq_dep ? sir_rf_fd : sir_rf
        m_transition = [-1 1 0; 0 -1 1]
        obs_model = partial_gaussian_obs_model(obs_error)
    elseif model_name == "SIS"
        rate_fn = freq_dep ? sir_rf_fd : sir_rf
        m_transition = [-1 1; 1 -1]
        obs_model = partial_gaussian_obs_model(obs_error)
    elseif model_name == "SEI"
        rate_fn = freq_dep ? sei_rf_fd : sei_rf
        m_transition = [-1 1 0; 0 -1 1]
        obs_model = partial_gaussian_obs_model(obs_error; seq = 3)
    elseif model_name == "SEIR"
        rate_fn = freq_dep ? seir_rf_fd : seir_rf
        m_transition = [-1 1 0 0; 0 -1 1 0; 0 0 -1 1]
        obs_model = partial_gaussian_obs_model(obs_error; seq = 3)
    elseif model_name == "SEIS"
        rate_fn = freq_dep ? seir_rf_fd : seir_rf
        m_transition = [-1 1 0; 0 -1 1; 1 0 -1]
        obs_model = partial_gaussian_obs_model(obs_error; seq = 3)
    elseif model_name == "LOTKA"
        model_name = "PN"
        rate_fn = lotka_rf
        m_transition = [0 1; 1 -1; -1 0]
        obs_model = partial_gaussian_obs_model(obs_error)
    elseif model_name == "ROSSMAC"
        model_name = "SIAB"
        rate_fn = rossmac_rf
        m_transition = [0 0 1 0; 0 0 -1 0; 0 0 0 -1; -1 1 0 0; 0 0 -1 1; 1 -1 0 0]
        obs_model = partial_gaussian_obs_model(obs_error)
    else
        println(" - SORRY: model name '", model_name, "' not recognised.")
        return  # handle this better? ***
    end
    prior = generate_weak_prior(size(m_transition, 1))
    return DPOMPModel(model_name, rate_fn, initial_condition, m_transition, dmy_obs_fn, obs_model, prior, 0)
end


"""
    generate_custom_model(model_name, rate_function, initial_condition, m_transition; ... )

Generates an `DPOMPModel` instance. Observation models are generated using the `partial_gaussian_obs_model` function, with ``σ = obs_error` (see that functions entry for further details.)

**Parameters**
- `model_name`          -- the model, e.g. "SIR", "SEIR-custom", etc.
- `rate_function`       -- event rate function.
- `initial_condition`   -- initial condition
- `m_transition`        -- transition matrix.

**Optional parameters**
- `observation_function -- observation function, use this to add 'noise' to simulated observations.
- `obs_error`           -- average observation error (default = 2.)
- `obs_model`           -- use this option to manuallu specify the observation model likelihood function.
- `prior_density`       -- prior density function.
- `t0_index`            -- index of the parameter that represents the initial time. `0` if fixed at `0.0`.

# Examples

    generate_custom_model("SIS", [100,1])

"""
function generate_custom_model(model_name::String, rate_function::Function, initial_condition::Array{Int64, 1}, m_transition::Array{Int64,2}; obs_function::Function = dmy_obs_fn, obs_error = 2.0, obs_model::Function = partial_gaussian_obs_model(obs_error; n = length(initial_condition)), prior = generate_weak_prior(size(m_transition, 1)), t0_index::Int64 = 0)
    return DPOMPModel(model_name, rate_function, initial_condition, m_transition, obs_function, obs_model, prior, t0_index)
end
