### global constants
const C_DEBUG = false
const C_RT_UNITS = 1000000000
const C_PR_SIGDIG = 3


## compute mean and covar matrix for a rejection sample
function handle_rej_samples(theta::Array{Float64,3}, ap::Int64 = 0)
    # x.mu .= zeros(size(samples,1))
    output = RejectionSample(theta, zeros(size(theta,1)), zeros(size(theta,1),size(theta,1)))
    for p in 1:size(theta,1)
         output.mu[p] = Statistics.mean(theta[p,(ap+1):size(theta,2),:])
    end
    d::Int64 = size(theta,3) * (size(theta,2) - ap)
    output.cv .= Statistics.cov(transpose(reshape(theta[:,(ap+1):size(theta,2),:], size(theta,1), d)))
    return output
end

## gelman diagnostic (internal)
function gelman_diagnostic(samples::Array{Float64,3}, discard::Int64)
    np = size(samples,1)
    niter::Int64  = size(samples,2)
    nmc = size(samples,3)
    fsmpl = discard + 1
    nsmpl = niter - discard
    ## compute W; B; V
    # collect means and variances
    mce = zeros(nmc, np)
    mcv = zeros(nmc, np)
    for i in 1:nmc
        for j in 1:np
            mce[i,j] = Statistics.mean(samples[j,fsmpl:end,i])
            mcv[i,j] = Statistics.cov(samples[j,fsmpl:end,i])
        end
    end
    # compute W, B
    b = zeros(np)
    w = zeros(np)
    mu = zeros(np)
    co = zeros(np)
    v = zeros(np)
    for j in 1:np
        b[j] = nsmpl * Statistics.cov(mce[:,j])
        w[j] = Statistics.mean(mcv[:,j])
        # mean of means and var of vars (used later)
        mu[j] = Statistics.mean(mce[:,j])
        co[j] = Statistics.cov(mcv[:,j])
        # compute pooled variance
        v[j] = w[j] * ((nsmpl - 1) / nsmpl) + b[j] * ((np + 1) / (np * nsmpl))
    end
    #
    vv_w = zeros(np)   # var of vars (theta_ex, i.e. W)
    vv_b = zeros(np)   # var of vars (B)
    mce2 = mce.^2                                          # ev(theta)^2
    cv_wb = zeros(np)   # wb covar
    for j in 1:np
        vv_w[j] = co[j] / nmc
        vv_b[j] = (2 * b[j] * b[j]) / (nmc - 1)
        cv_wb[j] = (nsmpl / nmc) * (Statistics.cov(mcv[:,j], mce2[:,j]) - (2 * mu[j] * Statistics.cov(mcv[:,j], mce[:,j])))
    end
    # compute d; d_adj (var.V)
    d = zeros(np)
    dd = zeros(np)
    atmp = nsmpl - 1
    btmp = 1 + (1 / nmc)
    for j in 1:np
        tmp = ((vv_w[j] * atmp * atmp) + (vv_b[j] * btmp * btmp) + (cv_wb[j] * 2 * atmp * btmp)) / (nsmpl * nsmpl)
        d[j] = (2 * v[j] * v[j]) / tmp
        dd[j] = (d[j] + 3) / (d[j] + 1)
    end
    # compute scale reduction estimate
    sre = zeros(np,3)
    try
        for j in 1:np
            rr = btmp * (1 / nsmpl)  * (b[j] / w[j]) ## NB. btmp ***
            sre[j,2] = sqrt(dd[j] * ((atmp / nsmpl) + rr)) ## atmp
            # F dist(nu1, nu2)
            fdst = Distributions.FDist(nmc - 1, 2 * w[j] * w[j] / vv_w[j])
            sre[j,1] = sqrt(dd[j] * ((atmp / nsmpl) + Statistics.quantile(fdst, 0.025) * rr))
            sre[j,3] = sqrt(dd[j] * ((atmp / nsmpl) + Statistics.quantile(fdst, 0.975) * rr))
        end
        # return GelmanResults
        return (mu = mu, wcv = sqrt.(w), sre = sre)
    catch gmn_err
        println("GELMAN ERROR: ", gmn_err)
        return (mu = mu, wcv = sqrt.(w), sre = sre) # return zeros
    end
end

## compute is mu var
function compute_is_mu_covar!(mu::Array{Float64,1}, cv::Array{Float64,2}, theta::Array{Float64,2}, w::Array{Float64,1})
    for i in eachindex(mu)
        mu[i] = sum(w .* theta[i,:]) / sum(w)                       # compute mu
        cv[i,i] = sum(w .* ((theta[i,:] .- mu[i]).^2)) / sum(w)     # variance
        for j in 1:(i-1)                                            # covar
            cv[i,j] = cv[j,i] = sum(w .* (theta[i,:] .- mu[i]) .* (theta[j,:] .- mu[j])) / sum(w)
        end
    end
end

## write time nicely
function print_runtime(rt)
    s = Int64(round(rt / C_RT_UNITS))
    s > 600 && (return string(round(s / 60; digits = 1), " minutes"))
    s > 120 && (return string(round(s / 60; digits = 2), " minutes"))
    return string(s, " seconds")
end

## compute sd (tabulate)
function compute_sigma(cv::Array{Float64,2})
    sd = zeros(size(cv,1))
    for i in eachindex(sd)
        sd[i] = sqrt(cv[i,i])
    end
    return sd
end
