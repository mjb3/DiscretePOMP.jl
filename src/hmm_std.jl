## standard DA-MCMC proposals

## insert event at appropriate index (std)
function add_event!(xf_trajectory::Array{Event,1}, evt_tp::Int64, evt_tm::Float64)
    if (length(xf_trajectory) == 0 || evt_tm > xf_trajectory[end].time)
        push!(xf_trajectory, Event(evt_tm, evt_tp))
    else
        for i in eachindex(xf_trajectory)
            if xf_trajectory[i].time > evt_tm
                insert!(xf_trajectory, i, Event(evt_tm, evt_tp))
                break
            end
        end
    end
end

## trajectory proposal
function get_std_mcmc_proposal_fn(model::HiddenMarkovModel, mvp::Int64)
    mvp += 2
    function std_mcmc_proposal(xi::Particle) #, theta_f::Array{Float64,1}
        ## NB. RETRIEVE PRIOR
        prop_type = rand(1:mvp)   # make proposal
        t0 = (model.t0_index == 0 ? 0.0 : xi.theta[model.t0_index])
        xf_trajectory = deepcopy(xi.trajectory)
        if prop_type > 2       # move
            length(xi.trajectory) == 0 && (return Particle(xi.theta, xi.initial_condition, xi.final_condition, xf_trajectory, xi.prior, [-Inf, -Inf]))
            # choose event and define new one
            evt_i = rand(1:length(xi.trajectory))
            evt_tp = xi.trajectory[evt_i].event_type
            # remove old one
            splice!(xf_trajectory, evt_i)
            # add new one at random time
            add_event!(xf_trajectory, evt_tp, (rand() * (model.obs_data[end].time - t0)) + t0)
            # compute ln g(x)
            prop_lk = 0.0
        else
            if prop_type == 1   # insert
                ## choose type:
                tp = rand(1:model.n_events)
                ## insert at randomly chosen time
                add_event!(xf_trajectory, tp, (rand() * (model.obs_data[end].time - t0)) + t0)
                ## compute ln g(x)
                prop_lk = log((model.n_events * (model.obs_data[end].time - t0)) / length(xf_trajectory))
            else                # delete
                # println(" deleting... tp:", tp, " - ec: ", ec)
                length(xi.trajectory) == 0 && (return Particle(xi.theta, xi.initial_condition, xi.final_condition, xf_trajectory, xi.prior, [-Inf, -Inf]))
                # choose event index (repeat if != tp)
                evt_i = rand(1:length(xi.trajectory))
                # remove
                splice!(xf_trajectory, evt_i)
                prop_lk = log(length(xi.trajectory) / ((model.obs_data[end].time - t0) * model.n_events))
            end
            # return (Particle, prop_ll)
        end
        return Particle(xi.theta, xi.initial_condition, copy(xi.final_condition), xf_trajectory, xi.prior, [0.0, prop_lk])
    end
    return std_mcmc_proposal
end



## new standard proposal function
# NEED TO BENCHMARK AGAINST OLD ***
# function standard_proposal(model::PrivateDiscuitModel, xi::MarkovState, xf_parameters::ParameterProposal)
#     ## choose proposal type
#     prop_type = rand(1:3)
#     # trajectory proposal
#     xf_trajectory = deepcopy(xi.trajectory)
#     t0 = (model.t0_index == 0) ? 0.0 : xf_parameters.value[model.t0_index]
#     if prop_type == 3
#         ## move
#         length(xi.trajectory.time) == 0 && (return MarkovState(xf_parameters, xi.trajectory, NULL_LOG_LIKE, DF_PROP_LIKE, prop_type))
#         # - IS THERE A MORE EFFICIENT WAY TO DO THIS? I.E. ROTATE using circshift or something?
#         # choose event and define new one
#         evt_i = rand(1:length(xi.trajectory.time))
#         # evt_tm = (rand() * (model.obs_data.time[end] - t0)) + t0 #, xi.trajectory.event_type[evt_i])
#         evt_tp = xi.trajectory.event_type[evt_i]
#         # remove old one
#         splice!(xf_trajectory.time, evt_i)
#         splice!(xf_trajectory.event_type, evt_i)
#         # add new one at random time
#         add_event!(xf_trajectory, evt_tp, (rand() * (model.obs_data.time[end] - t0)) + t0)
#         # compute ln g(x)
#         prop_lk = 1.0
#     else
#         ## insert / delete
#         if prop_type == 1
#             ## choose type:
#             tp = rand(1:size(model.m_transition, 1))
#             ## insert at randomly chosen time
#             add_event!(xf_trajectory, tp, (rand() * (model.obs_data.time[end] - t0)) + t0)
#             ## compute ln g(x)
#             prop_lk = (size(model.m_transition, 1) * (model.obs_data.time[end] - t0)) / length(xf_trajectory.time)
#         else
#             ## delete
#             # println(" deleting... tp:", tp, " - ec: ", ec)
#             length(xi.trajectory.time) == 0 && (return MarkovState(xi.parameters, xf_trajectory, NULL_LOG_LIKE, DF_PROP_LIKE, prop_type))
#             # choose event index (repeat if != tp)
#             evt_i = rand(1:length(xi.trajectory.time))
#             # remove
#             splice!(xf_trajectory.time, evt_i)
#             splice!(xf_trajectory.event_type, evt_i)
#             # compute ln g(x)
#             prop_lk = length(xi.trajectory.time) / ((model.obs_data.time[end] - t0) * size(model.m_transition, 1))
#         end # end of insert/delete
#     end
#     ## evaluate full likelihood for trajectory proposal and return
#     return MarkovState(xi.parameters, xf_trajectory, compute_full_log_like(model, xi.parameters.value, xf_trajectory), prop_lk, prop_type)
# end # end of std proposal function
