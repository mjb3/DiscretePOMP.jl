
#### NEED TO REDO THESE WITH MACROS *************** ####

## gillespie sim iteration
# - NEED TO REDO WITH MACROS ***************
function iterate_particle!(p::Particle, model::HiddenMarkovModel, time::Float64, y::Observation) #tmax::Float64
    cum_rates = Array{Float64, 1}(undef, model.n_events)
    while true
        model.rate_function(cum_rates, p.theta, p.final_condition)
        cumsum!(cum_rates, cum_rates)
        cum_rates[end] == 0.0 && break          # 0 rate test
        time -= log(rand()) / cum_rates[end]
        time > y.time && break                  # break if max time exceeded
        et = choose_event(cum_rates)            # else choose event type (init as final event)
        p.final_condition .+= model.fn_transition(et)  # update population
        push!(p.trajectory, Event(time, et))    # add event to sequence
        if length(p.trajectory) > MAX_TRAJ      # HACK
            p.log_like[1] = -Inf
            return p.log_like[1]
        end
    end
    output = model.obs_model(y, p.final_condition, p.theta)
    y.obs_id > 0 && (p.log_like[1] += output)
    return output
end

## DFG version
# - this could be tidier... ***
# function iterate_dfg_particle!(p::DFGParticle, model::HiddenMarkovModel, time::Float64, y::Observation) #tmax::Float64
#     cum_rates = Array{Float64, 1}(undef, model.n_events)
#     while true
#         model.rate_function(cum_rates, p.theta, p.final_condition)
#         cumsum!(cum_rates, cum_rates)
#         cum_rates[end] == 0.0 && break          # 0 rate test
#         time -= log(rand()) / cum_rates[end]
#         time > y.time && break                  # break if max time exceeded
#         et = choose_event(cum_rates)            # else choose event type (init as final event)
#         p.final_condition .+= model.fn_transition(et)  # update population
#         push!(p.trajectory, Event(time, et))    # add event to sequence
#         if length(p.trajectory) > MAX_TRAJ      # HACK
#             p.log_like[1] = -Inf
#             return p.log_like[1]
#         end
#     end
#     output = zeros(Int64, length(p.final_condition))    # MESS CLEAR THIS UP
#     if y.obs_id > 0
#         output = model.obs_model(y, p.final_condition, p.theta)
#         p.log_like[2] = output[1]
#         p.log_like[1] += p.log_like[2]
#     end
#     return output[2]
# end

## gillespie sim iteration (full state vectors)
function iterate_particle!(p::Particle, pop_v::Array{Array{Int64},1}, model::HiddenMarkovModel, time::Float64, y::Observation, cmpt_ll::Bool = true) # GET RID *
    cum_rates = Array{Float64, 1}(undef, model.n_events)
    while true
        model.rate_function(cum_rates, p.theta, p.final_condition)
        # C_DEBUG && println(" r := ", cum_rates)
        cumsum!(cum_rates, cum_rates)
        cum_rates[end] == 0.0 && break          # 0 rate test
        time -= log(rand()) / cum_rates[end]
        time > y.time && break                  # break if max time exceeded
        et = choose_event(cum_rates)            # else choose event type (init as final event)
        p.final_condition .+= model.fn_transition(et)  # update population
        push!(p.trajectory, Event(time, et))    # add event to sequence
        push!(pop_v, copy(p.final_condition))
    end
    cmpt_ll && (p.log_like[1] += model.obs_model(y, p.final_condition, p.theta))
end

## generate 'blank' observations for sim
# HACK: need to replace C_DEFAULT_OBS_PROP with [optional] random number at some point... **
C_DEFAULT_OBS_PROP = 1.0
function generate_observations(tmax::Float64, num_obs::Int64, n_states::Int64)
    obs = Observation[]
    t = collect(tmax / num_obs : tmax / num_obs : tmax)
    for i in eachindex(t)
        push!(obs, Observation(t[i], 1, C_DEFAULT_OBS_PROP, zeros(Int64, n_states)))
    end
    return obs
end

## run sim and return trajectory (full state var)
# reconstruct with macros? ***
function gillespie_sim(model::HiddenMarkovModel, theta::Array{Float64, 1}, observe::Bool) #, y::Observations
    # initialise some things
    y = deepcopy(model.obs_data)
    ic = model.fn_initial_condition()
    p = Particle(theta, ic, copy(ic), Event[], Distributions.logpdf(model.prior, theta), zeros(2))
    pop_v = Array{Int64}[]
    t = model.t0_index == 0 ? 0.0 : theta[model.t0_index]
    # run
    for i in eachindex(y)
        iterate_particle!(p, pop_v, model, t, y[i])
        observe && (y[i].val .= model.obs_function(y[i], p.final_condition, theta))
        t = y[i].time
    end
    # return sequence
    return SimResults(model.model_name, p, pop_v, y)
end

#### BTB testing scenario code ####
## get next observation
# nb. seed initial in sim
# ditto with IFN
function get_next_obs(obs::Array{Observation,1})
    C_INT_SI = 60
    C_INT_FU = 180
    C_INT_RH = 360
    if obs[end].val[1] > 0                     ## SI
        return Observation(obs[end].time + C_INT_SI, 2, C_DEFAULT_OBS_PROP, zeros(Int64,1))
    else
        if obs[end].obs_id > 1              ## breakdown in progress
            if obs[length(obs) - 1].val[1] > 0 ## SI
                return Observation(obs[end].time + C_INT_SI, 2, C_DEFAULT_OBS_PROP, zeros(Int64,1))
            else                            ## cleared - follow up
                return Observation(obs[end].time + C_INT_FU, 1, C_DEFAULT_OBS_PROP, zeros(Int64,1))
            end
        else                                ## schedule RHT
            return Observation(obs[end].time + C_INT_RH, 1, C_DEFAULT_OBS_PROP, zeros(Int64,1))
        end
    end
end
##
function init_obs()
    y = Observation[]
    push!(y, Observation(0.0, 1, C_DEFAULT_OBS_PROP, zeros(Int64,1)))
    return y
end
## TO BE MOVED
# NB. where is this called from? ***********
function gillespie_scenario(model::HiddenMarkovModel, theta::Array{Float64, 1}; tmax::Float64 = 720.0, ifn_y::Int64 = 0)
    ## initialise some things
    y = init_obs()
    ic = model.fn_initial_condition()
    p = Particle(theta, ic, copy(ic), Event[], [model.fn_log_prior(theta), 0.0])
    pop_v = Array{Int64}[]
    t = model.t0_index == 0 ? 0.0 : theta[model.t0_index]
    ## run
    while y[end].time < tmax
        iterate_particle!(p, pop_v, model, t, y[end], false)
        y[end].val .= model.obs_function(y[end], p.final_condition, theta)
        t = y[end].time
        if ifn_y == length(y) ## WHAT?*
            ## IFN
            push!(y, Observation(t + 1, 3, C_DEFAULT_OBS_PROP, zeros(Int64,1)))
        else
            ## schedule next test
            push!(y, get_next_obs(y))
        end
    end
    ## return sequence
    return SimResults(model.model_name, p, pop_v, y)
end
#### #### #### #### #### #### ####

## for inference
function generate_x0(model::HiddenMarkovModel, theta::Array{Float64, 1}, ntries = 10000)
    for i in 1:ntries
        x0 = gillespie_sim(model, theta, false).particle
        x0.log_like[1] != -Inf && return x0
    end
    ## ADD PROPER ERR HANDLING ***
    println("WARNING: having an issue generating a valid trajectory for ", theta)
    return generate_x0(model, theta, false)
end
