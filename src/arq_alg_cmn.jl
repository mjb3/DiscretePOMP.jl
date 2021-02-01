### common algorithm stuffs

## new default prior: partially unbounded
function get_df_arq_prior()
    function arq_prior(theta::Array{Float64,1})
        for i in eachindex(theta)
            theta[i] < 0.0 && return -Inf
        #     theta[i] > theta_range[i, 2] && return -Inf
        end
        return 0.0
    end
    return arq_prior
end

## new prior functions
function get_arq_prior(priord::Distributions.Distribution)
    function arq_prior(theta::Array{Float64,1})
        return Distributions.logpdf(priord, theta)
    end
    return arq_prior
end

## realise theta index as sample value
function get_theta_val(model::LikelihoodModel, theta::Array{Int64, 1})
    output = model.sample_offset + (theta .* model.sample_interval)
    # output = zeros(Float64, length(theta))
    for i in eachindex(output)
        # output[i] = (theta[i] - 0.5) * model.sample_interval[i]
        model.jitter > 0.0 && (output[i] += (((rand() * 2) - 1) * model.jitter * model.sample_interval[i]))
    end
    return output
end


## propose new theta coords
function get_theta_f(theta_i::Array{Int64, 1}, j_w::StatsBase.ProbabilityWeights, max_dist::Int64, min_dist::Int64)
    output = zeros(Int64, length(theta_i))
    d = (min_dist == max_dist) ? max_dist : rand(min_dist:max_dist)
    while sum(abs.(output)) != d# determine
        p::Int64 = StatsBase.sample(j_w)
        output[p] += Random.bitrand()[1] ? 1 : -1
    end
    output .+= theta_i          # move
    return output
end

## inner alg constants:
const Q_JUMP = 0.1              # initial = Q_JUMP * grid res * NP
const Q_J_MIN = 2
const N_ADAPT_PERIODS = 100     # adaptive mh mcmc (parameterise?)
const C_DF_ARQ_CJ = 10          # contingency jumps

## adapts jump weights
function adapt_jw!(j_w::StatsBase.ProbabilityWeights, lar_j::Int64, j::Int64, mc_accepted::BitArray{1}, a_h::Int64, i::Int64, tgt_ar::Float64, mc_idx::Array{Int64,2})
    # if (j == Q_J_MIN && sum(mc_accepted[(i + 1 - a_h):i]) == 0)
    if (j == Q_J_MIN && sum(mc_accepted[(i + 1 - a_h):i]) == 0)
        if sum(mc_accepted[1:i]) == 1
            C_DEBUG && print(" *LAR - contingency j invoked*")
            j = round(C_DF_ARQ_CJ * (i / a_h))
        else
            C_DEBUG && print(" *LAR*")
            j = lar_j
        end
    else    # adjust max jump size based on acceptance rate
        j = round(j * ((sum(mc_accepted[(i + 1 - a_h):i]) / a_h) / tgt_ar))
        j = max(j, Q_J_MIN)
    end
    ## tune var
    sd = Statistics.std(mc_idx[:, 1:i], dims = 2)[:,1]
    if sum(sd) == 0.0       # adjust for zeros
        C_DEBUG && print(" *ZAR*")
        sd .= 1.0
    else
        msd = minimum(sd[sd .> 0.0])
        for s in eachindex(sd)
            sd[s] == 0.0 && (sd[s] = msd)
        end
    end
    j_w .= sd               # update weights and return j
    return j
end

## compute mean and covar matrix for a single chain
# pass mean?
function compute_chain_mean_covar(samples::Array{Float64, 3}, mc::Int64, adapt_period::Int64, steps::Int64)
    # C_DEBUG && println(" - SS := ", size(samples))
    adapted = (adapt_period + 1):steps
    mc_bar = zeros(size(samples, 1))
    for i in eachindex(mc_bar)
        mc_bar[i] = Statistics.mean(samples[i, adapted, mc])
    end
    scv = Statistics.cov(transpose(samples[:, adapted, mc]))
    return (mc_bar, scv)
end


## get intial parameter index and 'grid' sample - returns tuple(index, GridRequest)
# NB. update for offsets ********
function get_initial_sample(mdl::LikelihoodModel, grid::Dict{Array{Int64, 1}, GridPoint}, mc_fx::Array{Int64, 1}, sample_dispersal::Int64 = mdl.sample_dispersal)
    theta_i = rand(1:sample_dispersal, length(mdl.sample_interval))     # choose initial theta coords
    x0 = get_grid_point!(grid, theta_i, mdl, true)
    x0.prior == -Inf && (return get_initial_sample(mdl, grid, mc_fx, sample_dispersal + 1))
    x0.process_run && (mc_fx[1] += 1)
    (C_DEBUG && sample_dispersal > mdl.sample_dispersal) && print(" *ISA: ", sample_dispersal - mdl.sample_dispersal, "*")
    return (theta_i, x0)
end

## initialise inner ARQ MCMC
macro init_inner_mcmc()
      esc(quote
      mc_fx::Array{Int64, 1} = zeros(Int64, 3)   # process run
      x0 = get_initial_sample(model, grid, mc_fx)
      theta_i::Array{Int64, 1} = x0[1]
      xi = x0[2]
      C_DEBUG && print(": θ ~ ", round.(xi.result.sample; sigdigits = C_PR_SIGDIG + 1))
      ## adaptive stuff
      C_LAR_J_MP = 0.2                                # low AR contingency values
      lar_j::Int64 = round(C_LAR_J_MP * model.sample_dispersal * length(theta_i))
      a_h::Int64 = max(steps / N_ADAPT_PERIODS, 100)    # interval
      j::Int64 = round(Q_JUMP * model.sample_dispersal * length(theta_i))
      j_w = StatsBase.ProbabilityWeights(ones(length(theta_i)))

      ## declare results
      mc_idx = Array{Int64, 2}(undef, length(theta_i), steps)
      mc_accepted = falses(steps)
      # mc_fx = zeros(Int64, 3)   # process run
      ## estimate x0
      # xi = get_grid_point!(grid, theta_i, model, true)
      # xi.process_run && (mc_fx[1] += 1)

      ## write first sample and run the Markov chain:
      samples[:,1,mc] .= xi.result.sample
      mc_idx[:,1] .= theta_i
      mc_accepted[1] = true
      end)
end
