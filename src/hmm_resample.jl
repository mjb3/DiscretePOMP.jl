## NB. these aren't used for anything atm
# NB. also note that parameter 'w' is typically aletered

## basic multinomial resampler (inverse CDF method)
function rs_multinomial(w::Array{Float64,1}, n::Int64 = length(w))
    cumsum!(w, w)
    output = Array{Int64,1}(undef, n)
    # choose new p
    for p in eachindex(output)
        output[p] = length(w)
        chs_p = rand() * w[end]
        for p2 in 1:(length(w) - 1)
            if chs_p < w[p2]
                output[p] = p2
                break   # next p
            end
        end
        # m_pop[p,:] .= old_p[new_p,:]
    end
    return output
end

## residual multinomial
# add option for normalising weights? **
# function rs_residual_mn(w::Array{Float64,1})
#     w ./= sum(w)    # normalise
#     nt = Int64.(floor.(length(w) .* w)) # N~
#     output = zeros(Int64, length(w))
#     i = 1
#     for j in eachindex(nt)
#         for k in 1:nt[j]
#             output[i] = nt[j]
#         end
#     end
#     wb = w .- (nt .* (1 / length(w)))
#     cumsum!(wb, wb)
#     rs = rs_multinomial(wb, length(w) - sum(nt))
#     output[(sum(nt)+1):end] .= rs
#     return output
# end
## NB. W.I.P.

## systematic (samples single seed u(0,1/N])
# Carpenter (1999)
function rs_systematic(w::Array{Float64,1})
    cw = cumsum(w)
    output = Array{Int64,1}(undef, length(w))
    u = Array{Float64,1}(undef, length(w))
    u[1] = rand() / length(w) # sample ~ U(0,1/N]
    for i in 2:length(cw)
        u[i] = u[1] + ((i - 1) / length(w))
    end
    u .*= cw[end]
    # set output = new index
    j = 1
    for i in eachindex(output)
        while u[i] > cw[j]
            j += 1
        end
        output[i] = j
    end
    return output
end

## stratified (i.e. jittered)
# Kitagawa (1996)
function rs_stratified(w::Array{Float64,1})
    cumsum!(w, w)
    output = Array{Int64,1}(undef, length(w))
    u = rand(length(w)) / length(w)
    for i in eachindex(u)
        u[i] += ((i - 1) / length(w))
    end
    u .*= w[end]
    # set output = new index
    j = 1
    for i in eachindex(output)
        while u[i] > w[j]
            j = j + 1
        end
        output[i] = j
    end
    return output
end

### CHOPTHIN GOES HERE ***

## test stuff
function rs_dist(x)
    output = zeros(maximum(x))
    for i in eachindex(x)
        output[x[i]] += 1 / length(x)
    end
    return output
end

## test avg error
function test_rs(c_rs::Int64, c_p::Int64, alg::Int64 = 1, show_detail = false)
    if alg == 1
        fn_rs = rs_multinomial
        println("testing multinomial resampler:")
    elseif alg == 2
        fn_rs = rs_systematic
        println("testing systematic resampler:")
    elseif alg == 3
        println("testing stratified resampler:")
        fn_rs = rs_stratified
    elseif alg == 4
        println("testing residual multinomial resampler:")
        fn_rs = rs_residual_mn
    else
        fn_rs = rs_multinomial
        println("ERR: unknown resampler ", alg, ", using multinomial")
    end
    x = zeros(Int64, c_rs, c_p)
    px = rand(c_p)
    nwx = copy(px)
    nwx ./= sum(px)
    show_detail && println("p = ", px)
    show_detail && println("nw = ", nwx)
    # optionally accumulate (residual takes raw weights atm)
    # alg < 4 && cumsum!(px, px)
    for i in 1:c_rs
        pxx = copy(px)
        x[i,:] .= fn_rs(copy(pxx))
    end
    println("samples: ", x)
    rs = rs_dist(x)
    show_detail && println("rs: ", rs)
    println(length(rs))
    # rs .-= nwx
    # show_detail && println("discrepancy: ", rs)
    # println("ns = ", c_rs, "; p = ", c_p, ". avg error = ", sum(abs.(rs)) / c_p)
end

## run tests
# test_rs(10, 100, 1, false)
# test_rs(1000, 100, 2)
# test_rs(1000, 100, 3)
# test_rs(10, 10, 4, true)
#
# ## benchmark
# import BenchmarkTools
# BenchmarkTools.@btime rs_multinomial(rand(100))
# BenchmarkTools.@btime rs_systematic(rand(100))
# BenchmarkTools.@btime rs_stratified(rand(100))
# BenchmarkTools.@btime rs_residual_mn(rand(100))
# BenchmarkTools.@btime rs_chopthin(rand(100))
