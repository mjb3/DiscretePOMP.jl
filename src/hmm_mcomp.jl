## model comparison

function run_model_comparison_analysis(models::Array{HiddenMarkovModel, 1}, n_runs::Int64, fn_algorithm::Function)
    ## run analysis
    bme = zeros(n_runs, length(models))
    theta_mu = Array{Array{Float64,1}, 2}(undef, n_runs, length(models))
    mnames = String[]
    start_time = time_ns()
    for m in eachindex(models)
        println(" processing model m", m, ": ", models[m].model_name)
        for n in 1:n_runs
            print("  analysis ", n, " ")
            rs = fn_algorithm(models[m])
            bme[n, m] = rs.bme[1]
            theta_mu[n, m] = rs.mu
        end
        push!(mnames, models[m].model_name)
    end
    ## process results
    output = ModelComparisonResults(mnames, bme, -log.(vec(Statistics.mean(exp.(-bme); dims = 1))), vec(Statistics.std(bme; dims = 1)), n_runs, time_ns() - start_time, theta_mu)
    println("Analysis complete (total runtime := ", Int64(round(output.run_time / C_RT_UNITS)), "s)")
    return output
end

"""
    run_model_comparison_analysis(model, obs_data; ... )

Run `n_runs` independent analyses for each `DPOMPModel` element in `models`, and compute [estimate] the **Bayesian model evidence** (BME.)

Returns an object of type `ModelComparisonResults`, which includes the mean and standard deviation of the estimates obtained.

**Parameters**
- `models`          -- An `Array` of `DPOMPModel`.
- `obs_data`        -- An array of type `Observation`.

**Optional parameters**
- `n_runs`          -- number of independent analyses used to estimate the BME for each model.
- `algorithm`       -- `String` representing the inference method used for the analysis, `"SMC2"` for **SMC^2** (default); or `"MBPI"` for *MBP-IBIS*.

**Optional inference algorithm parameters**
- `np`              -- number of [outer, i.e. theta] particles used in IBIS procedures (doesn't apply to ARQ-MCMC.)
- `ess_rs_crit`     -- Effective sample size (ESS) resampling criteria.
- `npf`             -- number of particles used in particle filter (doesn't apply to MBP-IBIS.)
- `n_props`         -- see the docs for `run_mbp_ibis_analysis`.

**Example**
```@repl
# NB. first define some models to compare, e.g. as m1, m2, etc
models = [m1, m2, m3]
results = run_model_comparison_analysis(models, y; n_runs = 10)
tabulate_results(results)   # show the results (optional)
```

"""
function run_model_comparison_analysis(models::Array{DPOMPModel, 1}, y::Array{Observation, 1}; n_runs = 3, algorithm = C_ALG_NM_SMC2
    , np::Int64 = algorithm == C_ALG_NM_SMC2 ? C_DF_SMC2_P : C_DF_MBPI_P
    , ess_rs_crit::Float64 = algorithm == C_ALG_NM_SMC2 ? C_DF_ESS_CRIT : C_DF_MBPI_ESS_CRIT
    , npf::Int64 = C_DF_PF_P, n_props::Int64 = C_DF_MBPI_MUT)

    println("Running: ", n_runs, "-run ", length(models), "-model Bayesian evidence analysis (algorithm := ", algorithm, ")\n - please note: this may take a while...")
    ## set up inference algorithm
    function alg_smc2(mdl::HiddenMarkovModel)
        theta_init = rand(mdl.prior, np)
        return run_pibis(mdl, theta_init, ess_rs_crit, true, C_ACCEPTANCE_ALPHA, npf)
    end
    function alg_mibis(mdl::HiddenMarkovModel)
        theta_init = rand(mdl.prior, np)
        return run_mbp_ibis(mdl, theta_init, ess_rs_crit, n_props, false, C_ACCEPTANCE_ALPHA, false)
    end

    ##
    if algorithm == C_ALG_NM_SMC2
        inf_alg = alg_smc2
    elseif (SubString(algorithm, 1, 4) == C_ALG_NM_MBPI || algorithm == "MIBIS")
        inf_alg = alg_mibis
    else
        println(" WARNING - algorithm unknown: ", algorithm, "\n - defaulting to SMC2")
        inf_alg = alg_smc2
    end
    # initialise models
    hmm = HiddenMarkovModel[]
    for m in eachindex(models)
        mdl = get_private_model(models[m], y)
        push!(hmm, mdl)
    end
    ## run analysis
    return run_model_comparison_analysis(hmm, n_runs, inf_alg)
end
