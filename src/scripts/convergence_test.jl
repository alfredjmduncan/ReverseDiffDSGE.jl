using Distributed
using Turing
using Dates
addprocs(4)


@everywhere using ReverseDiffDSGE
@everywhere using Random
@everywhere using Zygote
@everywhere using Turing
@everywhere using Distributions
@everywhere using ComponentArrays
@everywhere using SharedArrays
using DataFrames
using CSV

# Estimation parameters
nperiods = 100
B = [1,5,6]  # observables
σ_obs = 0.01 # sd of measurement error shocks

## Likelihood related functions
@everywhere logll(θ,ys) = kalman_ll(blanchardkahn(HerbstSchorfheide(θ)),ys,B,σ_obs)


## Generate fake data
θ = rand.(𝒫ₕₛ);
println("True parameter values:")
println("θ = $(round.(θ,digits=2))")

ts(θ,B,σ_obs,nperiods) = ReverseDiffDSGE.timeseries(blanchardkahn(HerbstSchorfheide(θ)),nperiods)[:,B] +
        σ_obs*rand(Normal(),nperiods,length(B)); # ad hoc measurement error shocks
ys = ts(θ,B,σ_obs,nperiods)

θ_axes = getaxes(θ);

## Turing Model
@everywhere @model function m(y)

        τ⁻¹ ~ InverseGamma(8,8)
        κ   ~ Uniform(0.0,1.0)
        ψ_1 ~ Gamma(4,1/8)
        ψ_2 ~ Gamma(4,1/8)
        ρ_R ~ Uniform(0.5,0.9)
        ρ_g ~ Uniform(0.90,0.99)
        ρ_z ~ Uniform(0.90,0.99)
        σ_R ~ InverseGamma(4,0.32)
        σ_g ~ InverseGamma(4,2.0)
        σ_z ~ InverseGamma(4,0.5)

        ρ = ComponentArray([θ.rA,τ⁻¹,κ,ψ_1,ψ_2,ρ_R,ρ_g,ρ_z,σ_R,σ_g,σ_z],θ_axes)

        @Turing.addlogprob! logll(ρ,y)

end

_nuts  = (alg = NUTS{Turing.ZygoteAD}(0.65;max_depth=5,Δ_max=500.0,init_ϵ=0.1),iters = (500,1_000))
_mh    = (alg = MH(),iters = (250_000,500_000,1_000_000))

algs = (_nuts,_mh)

println("# Precompilation")
for alg in algs
        sample(m(ys), alg.alg, MCMCDistributed(), 1, 4)
end

# Dataframe to collect results
df = DataFrame(
        values = [],
        algorithm = [],
        iterations = [],
        elapsed = [],
        gelmanrubin = [],
        ess = [],
        rhat = [],
)

function test_function(alg,iters,θ,ys)
        ch = sample(m(ys), alg, MCMCDistributed(), iters, 4)
        results = (
                values = round.(θ,digits=2),
                algorithm = alg,
                iterations = iters,
                elapsed = maximum(ch.info.stop_time - ch.info.start_time),
                gelmanrubin = maximum(gelmandiag(ch).nt.psrf),
                ess = minimum(ess(ch).nt.ess),
                rhat = maximum(ess(ch).nt.rhat)
        )
        println("Estimates")
        display(describe(ch)[1])
        return results
end


for θ_i in 1:20

        global θ = rand.(𝒫ₕₛ);
        global ys = ts(θ,B,σ_obs,nperiods);
        println("-----------------------------------------------------------------------")
        println("True values")
        display(DataFrame([NamedTuple(round.(θ;digits=2))]))
        println("Prior means")
        display(DataFrame([NamedTuple(round.(mean.(𝒫ₕₛ);digits=2))]))
        for alg in algs
                println("-----------------------------------------------------------------------")
                println("Algorithm:    $(alg.alg)")
                println("Iterations:   $(alg.iters)")
                println("Current time: $(Dates.format(now(),"HH:MM"))")
                for mcmc_iters in alg.iters
                        try
                                results = test_function(alg.alg,mcmc_iters,θ,ys)
                                push!(df,results)
                                println("Results so far:")
                                display(df[:,[:iterations,:elapsed,:gelmanrubin,:ess,:rhat]])
                                CSV.write("results/benchmarktests.csv",df;delim=";")
                        catch
                                println("-----------------------------------------------------------------------")
                                println("Error. Trying new estimator.")
                                println("-----------------------------------------------------------------------")
                        end
                end
        end

end
