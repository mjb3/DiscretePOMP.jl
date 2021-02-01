### standard ARQ MCMC algorithm

## standard grid request
function get_grid_point!(grid, theta_i::Array{Int64, 1}, model::LikelihoodModel, burn_in::Bool)  #, herd_size_dist::EmpiricalDist, log_irw_y::Array{Float64, 2}
    Q_BI_SAMPLE_LIM = 1
    ## check hashmap
    exists = haskey(grid, theta_i)
    if exists
        x = grid[theta_i]
        visited = x.visited
        sampled = x.sampled
        theta_val = x.sample
    else
        visited = 0
        sampled = 0
        theta_val = get_theta_val(model, theta_i)
    end
    pr =  model.prior(theta_val)
    if pr == -Inf
        output = GridPoint(theta_val, -Inf, visited, sampled)
        return GridRequest(output, pr, false)
    else
        ## VISITED: COMPUTED
        if visited < (burn_in ? Q_BI_SAMPLE_LIM : model.sample_limit)
            ## update density
            log_like = model.pdf(theta_val)
            exists && (log_like = log(exp(x.log_likelihood) + ((exp(log_like) - exp(x.log_likelihood)) / visited)))
            visited += 1
            comp = true
        else
            log_like = x.log_likelihood
            comp = false
        end
        burn_in || (sampled += 1)
        ## update hashmap
        output = GridPoint(theta_val, log_like, visited, sampled)
        grid[theta_i] = output
        ## return updated sample
        return GridRequest(output, pr, comp)
    end
end

## run standard inner MCMC procedure and return results
function arq_met_hastings!(samples::Array{Float64,3}, mc::Int64, grid::Dict, model::LikelihoodModel, steps::Int64, adapt_period::Int64, tgt_ar::Float64)
    @init_inner_mcmc                                # initialise inner MCMC
    C_DEBUG && print("- mc", mc, " initialised ")
    for i in 2:steps
        theta_f = get_theta_f(theta_i, j_w, j, 1)   # propose new theta
        ## get log likelihood
        xf = get_grid_point!(grid, theta_f, model, i < a_h) # limit sample (n=1) for first interval only
        xf.process_run && (mc_fx[2] += 1)
        ## mh step
        mh_prob = exp(xf.prior - xi.prior + xf.result.log_likelihood - xi.result.log_likelihood)
        if mh_prob > 1.0            # accept or reject
            mc_accepted[i] = true
        else
            mh_prob > rand() && (mc_accepted[i] = true)
        end
        if mc_accepted[i]           # acceptance handling
            samples[:,i,mc] .= xf.result.sample
            mc_idx[:,i] .= theta_f
            theta_i = theta_f
            xi = xf
        else                        # rejected
            samples[:,i,mc] .= samples[:,i-1,mc]
            mc_idx[:,i] .= mc_idx[:,i - 1]
            Q_REJECT_TRIGGER = 100  # TRIGGER REFRESH if rejected N times
            if i > Q_REJECT_TRIGGER
                if sum(mc_accepted[(i - Q_REJECT_TRIGGER):i]) == 0
                    # C_DEBUG && println(" warning: refresh triggered on chain #", mc)
                    xi = get_grid_point!(grid, theta_i, model, false)
                    xi.process_run && (mc_fx[3] += 1)
                end
            end
        end     ## end of acceptance handling
        ## ADAPTATION
        i % a_h == 0 && (j = adapt_jw!(j_w, lar_j, j, mc_accepted, a_h, i, tgt_ar, mc_idx))
        ## end of adaption period
    end ## end of Markov chain for loop
    if C_DEBUG
        print("- mc", mc, " processing -> ")
        ## compute mean/var and return results
        mc_m_cv = compute_chain_mean_covar(samples, mc, adapt_period, steps)
        C_DEBUG && print(" mcv := ", mc_m_cv, "; fx := ", mc_fx, " ")
    end
    ## compute SMC runs and return metadata
    return (sum(mc_fx), sum(mc_accepted) / steps, sum(mc_accepted[(adapt_period+1):steps]) / (steps - adapt_period))
end
