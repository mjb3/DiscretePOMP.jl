### particle filter: resample populations ###
# GIVEN CUMULATIVE WEIGHTS

## basic multinomial resampler (given cumulative weights, i.e. inverse CDF)
function rsp_multinomial(m_pop::Array{Int64,2}, old_p::Array{Int64,2}, cw::Array{Float64,1})
    # update old p
    # old_p .= m_pop
    # choose new p
    for p in eachindex(weights)
        new_p = length(weights)
        chs_p = rand() * weights[end]
        for p2 in 1:(length(weights) - 1)
            if chs_p < weights[p2]
                new_p = p2
                break
            end
        end
        m_pop[p,:] .= old_p[new_p,:]
    end
end

## systematic (samples single seed u(0,1/N])
# Carpenter (1999)
function rsp_systematic(m_pop::Array{Int64,2}, old_p::Array{Int64,2}, cw::Array{Float64,1})
    # output = Array{Int64,1}(undef, length(cw))
    u = Array{Float64,1}(undef, length(cw))
    u[1] = rand() / length(cw) # sample ~ U(0,1/N]
    for i in 2:length(cw)
        u[i] = u[1] + ((i - 1) / length(cw))
    end
    u .*= cw[end]
    # set output = new index
    j = 1
    for i in 1:size(m_pop, 1)
        while u[i] > cw[j]
            j = j + 1
        end
        m_pop[i,:] .= old_p[j,:]
        # output[i] = j
    end
    # return output
end

## stratified (i.e. jittered)
# Kitagawa (1996)
function rsp_stratified(m_pop::Array{Int64,2}, old_p::Array{Int64,2}, cw::Array{Float64,1})
    # output = Array{Int64,1}(undef, length(cw))
    u = rand(length(cw)) / length(cw)
    for i in eachindex(u)
        u[i] += ((i - 1) / length(cw))
    end
    u .*= cw[end]
    # set output = new index
    j = 1
    for i in eachindex(output)
        while u[i] > cw[j]
            j = j + 1
        end
        # output[i] = j
        m_pop[i,:] .= old_p[j,:]
    end
    # return output
end
