# ### data augmented ARQMCMC algorithm
#
# ## augmented PDF result
# # i.e. result of a call to model.pdf
# # NEED TO BENCHMARK OPTIONS ********
# struct AugDataResult
#     # parameters::Array{Float64}
#     log_like::Float64
#     aug_data_var
# end
#
# ## AD grid request
# struct ADGridRequest
#     set::GridSet    ## NECESSARY? ******
#     result::GridSample # LOG LIKE DUPLICATED HERE ****
#     delayed::Bool
#     aug_data::AugDataResult
# end
#
# ## augmented data (delayed acceptance) grid request
# # called during proposal
# # AugDataResult
# function get_ad_grid_point!(grid, theta_i::Array{Int64, 1}, xi::AugDataResult, model::LikelihoodModel)
#     theta_val = get_theta_val(model, theta_i)
#     ## check hashmap
#     exists = haskey(grid, theta_i)
#     if exists
#         # return delayed acceptance (if limit reached)
#         x = grid[theta_i]
#         if length(x.samples) > model.sample_limit
#             # estimate likelihood:
#             like = 0.0
#             for s in eachindex(x.samples)
#                 like += exp(x.samples[s].log_likelihood)
#             end
#             # return
#             return ADGridRequest(x, GridSample(theta_val, log(like/length(x.samples))), true, xi)
#         else
#             # sample anyway
#             xx = model.pdf(xi, theta_val)
#             s = GridSample(theta_val, xx.log_like)
#             push!(x.samples, s)
#             return ADGridRequest(x, s, false, xx)
#         end
#     else
#         ## new sample set:
#         xx = model.pdf(xi, theta_val)
#         s = GridSample(theta_val, xx.log_like)
#         ss = []
#         push!(ss, s)
#         # anchor, store and return
#         anchor = get_theta_val(model, theta_i, 0.0)
#         x = GridSet(anchor, ss)
#         grid[theta_i] = x
#         # println("theta fs: ", x.samples[1].sample)
#         return ADGridRequest(x, s, false, xx)
#     end
# end
#
# ## run augmented data inner MCMC procedure and return results
# # TBF **********************
# function adarq_met_hastings!(grid::Dict, model::LikelihoodModel, x0::AugDataResult, steps::Int64, adapt_period::Int64, theta_init::Array{Int64, 1}, tgt_ar::Float64)
#     C_MIN_J = 0
#     ## initialise inner MCMC
#     @init_inner_mcmc
#     mc_fx = ones(Int16, steps)
#
#     ## get initial grid point
#     xi = get_ad_grid_point!(grid, theta_init, x0, model)
#
#     ## initialise some things and run the Markov chain:
#     mc_idx[1,:] .= theta_init
#     mc[1,:] .= xi.result.sample
#     mcf[1,:] .= mc[1,:]
#     mc_log_like[1] = x0.log_like
#     mc_accepted[1] = true
#     mc_fx[1] = 1
#     mc_time[1] = time_ns() - start_time
#     # st_time = time_ns()
#     for i in 2:steps
#         ## propose new theta
#         theta_f = get_theta_f(theta_i, j_w, j, C_MIN_J)
#         ## validate
#         if validate_theta(theta_f, model.sample_dispersal)
#             ## get log likelihood
#             xf = get_ad_grid_point!(grid, theta_f, xi.aug_data, model)
#             mcf[i,:] .= xf.result.sample
#             mc_log_like[i] = xf.result.log_likelihood
#             ## mh step
#             # mh_prob = exp(mc_log_like[i] - mc_log_like[i - 1])
#             aurn = rand()
#             # accept or reject
#             exp(mc_log_like[i] - mc_log_like[i - 1]) > aurn && (mc_accepted[i] = true)
#             ## delay handling:
#             if xf.delayed
#                 if mc_accepted[i]
#                     ## sample
#                     xx = model.pdf(xi.aug_data, xf.result.sample)
#                     # s = GridSample(xf.result.sample, xx.log_like)
#                     mc_log_like[i] = xx.log_like
#                     push!(xf.set.samples, GridSample(xf.result.sample, mc_log_like[i]))
#                     ## reapply MH step
#                     exp(mc_log_like[i] - mc_log_like[i - 1]) > aurn || (mc_accepted[i] = false)
#                 else
#                     ## 'delayed rejection'
#                     mc_fx[i] = 0
#                 end
#             end
#         else
#             ## reject automatically
#             mcf[i,:] .= mc[i - 1,:] # fix: backfill with theta i-1
#         end
#
#         ### acceptance handling
#         if mc_accepted[i]
#             mc[i,:] .= xf.result.sample
#             mc_idx[i,:] .= theta_f
#             theta_i = theta_f
#             mc_log_like[i] = xf.result.log_likelihood
#             ## only need to do for aug_data, do this more concisely somehow? **********
#             xi = xf
#         else
#             ## reject
#             mc[i,:] .= mc[i - 1,:]
#             mc_idx[i,:] .= mc_idx[i - 1,:]
#             mc_log_like[i] = mc_log_like[i-1]
#         end ## end of acceptance handling
#         mc_time[i] = time_ns() - start_time
#         ## ADAPTATION
#         i % a_h == 0 && (j = adapt_jw!(j_w, Q_J_MIN * 2, j, mc_accepted, a_h, i, tgt_ar, mc_idx))
#         # end of adaption period
#     end ## end of Markov chain for loop
#     ## compute mean/var and return results
#     mc_m_cv = compute_chain_mean_covar(mc, adapt_period, steps)
#     ## compute SMC runs and return results
#     return MCMCResults(sum(mc_fx), mc_idx, mc, mc_accepted, mc_fx, mc_m_cv[1], mc_m_cv[2], mcf, mc_log_like, mc_time)
# end
