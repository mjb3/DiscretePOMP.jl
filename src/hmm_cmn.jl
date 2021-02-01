#### common functions ####

## choose event type
function choose_event(cum_rates::Array{Float64,1})
    etc = rand() * cum_rates[end]
    for i in 1:(length(cum_rates) - 1)
        cum_rates[i] > etc && return i
    end
    return length(cum_rates)
end

## Gaussian mv parameter proposal
function get_mv_param(propd::Distributions.MvNormal, sclr::Float64, theta_i::Array{Float64, 1})
    output = rand(propd)
    output .*= sclr
    output .+= theta_i
    return output
end

# ## compute mean and covar matrix for a rejection sample
# function handle_rej_samples(theta::Array{Float64,3}, ap::Int64 = 0)
#     # x.mu .= zeros(size(samples,1))
#     output = RejectionSample(theta, zeros(size(theta,1)), zeros(size(theta,1),size(theta,1)))
#     for p in 1:size(theta,1)
#          output.mu[p] = Statistics.mean(theta[p,(ap+1):size(theta,2),:])
#     end
#     d::Int64 = size(theta,3) * (size(theta,2) - ap)
#     output.cv .= Statistics.cov(transpose(reshape(theta[:,(ap+1):size(theta,2),:], size(theta,1), d)))
#     return output
# end

##
function get_prop_density(cv::Array{Float64,2}, old)
    ## update proposal density
    tmp = LinearAlgebra.Hermitian(cv)
    if LinearAlgebra.isposdef(tmp)
        return Distributions.MvNormal(Matrix(tmp))
    else
        C_DEBUG && println(" warning: particle degeneracy problem\n  covariance matrix: ", cv)
        return old
    end
end
