
#### particle filter ####

## iterate particle and evaluate cumulative w = g(x)
function iterate_particles!(pop::Array{Int64,2}, cum_weight::Array{Float64,1}, model::HiddenMarkovModel, obs_i::Int64, parameters::Array{Float64,1}, t::Float64, tmax::Float64)
    cum_rates = Array{Float64}(undef, model.n_events)
    ptemp = Array{Int64}(undef, size(pop,2))
    # iterate each particle
    total_weight = 0.0
    for p in eachindex(cum_weight)
        # initialise some things
        time = t
        ptemp .= pop[p,:]   # - HACK ***************
        # sim
        while true
            model.rate_function(cum_rates, parameters, ptemp)
            cumsum!(cum_rates, cum_rates)
            cum_rates[end] == 0.0 && break          # 0 rate test
            time -= log(rand()) / cum_rates[end]
            time > tmax && break                # break if max time exceeded
            # else choose event type and update population
            ptemp .+= model.fn_transition(choose_event(cum_rates))
        end
        # update likelihood
        total_weight += exp(model.obs_model(model.obs_data[obs_i], ptemp, parameters))
        cum_weight[p] = total_weight
        pop[p,:] .= ptemp
    end
end

## update state variable and return partial LOG likelihood
# ADD FUNCTION FOR NEXT VIA LOOP?

# e.g. for 1st y ymin = 1, ymax = 1
function partial_log_likelihood!(pop::Array{Int64,2}, model::HiddenMarkovModel, parameters::Array{Float64, 1}, fn_rs::Function, essc::Float64, ymin::Int64, ymax::Int64)
    ## initialise population matrix
    if ymin == 1
        for i in 1:size(pop, 1)
            pop[i,:] .= model.fn_initial_condition()
        end
        t_prev = model.t0_index == 0 ? 0.0 : parameters[model.t0_index]
    else
        t_prev = model.obs_data[ymin - 1].time
    end
    # - for use by resampler
    old_p = copy(pop)
    ## for each observation
    output = 0.0    # log likelihood
    cum_weight = Array{Float64}(undef, size(pop,1)) # declare array for use by loop
    for obs_i in ymin:ymax
        # iterate each particle
        iterate_particles!(pop, cum_weight, model, obs_i, parameters, t_prev, model.obs_data[obs_i].time)
        # cumsum!(cum_weight, cum_weight)
        if model.obs_data[obs_i].obs_id > 0
            # update avg ll
            output += log(cum_weight[end] / size(pop, 1))
            # resample particles (if < n, and rs = true)
            if obs_i < length(model.obs_data)
                ## ADD ESS CRITERIA **********
                # update old p and resample
                old_p .= pop
                fn_rs(pop, old_p, cum_weight)
            end
        end
        # reset sim parameters
        t_prev = model.obs_data[obs_i].time
    end
    # return unbiased estimator of LOG likelihood
    return output
end

## estimate full LOG likelihood
function estimate_likelihood(model::HiddenMarkovModel, parameters::Array{Float64,1}, particles::Int64, pop_size::Int64, fn_rs::Function, ess_crit::Float64)
    # , t0_index::Int64
    ## initialise population matrix and iterate PF over full system trajectory
    pop = zeros(Int64, particles, pop_size)
    return partial_log_likelihood!(pop, model, parameters, fn_rs, ess_crit, 1, length(model.obs_data))
end

## generate pdf function
function get_log_pdf_fn(mdl::HiddenMarkovModel, p::Int64 = C_DF_PF_P, rs_type::Int64 = 1; essc::Float64 = C_DF_ESS_CRIT)
    if rs_type == 2                 ## resampler
        fn_rs = rsp_stratified      # Kitagawa (1996)
    elseif rs_type == 3
        fn_rs = rsp_multinomial     # inverse CDF multinomial
    else
        fn_rs = rsp_systematic      # Carpenter (1999) - default
    end
    ps = length(mdl.fn_initial_condition()) ## population size
    ## generate function and return
    function comp_log_pdf(parameters::Array{Float64, 1})
        return estimate_likelihood(mdl, parameters, p, ps, fn_rs, essc)
    end
    return comp_log_pdf
end
