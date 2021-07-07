
using Turing
using Dates

using ReverseDiffDSGE
using Random
using Zygote
using Distributions
using ComponentArrays
using SharedArrays
using DataFrames
using CSV

using DimensionalData
using PGFPlots
using DataFrames
using KernelDensity
using Statistics
using StatsBase
using LinearAlgebra

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
@model function m(y)

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


ch_nuts = sample(m(ys), NUTS{Turing.ZygoteAD}(0.65;max_depth=5,Δ_max=500.0,init_ϵ=0.01), 500)
ch_mh = sample(m(ys), MH(),200_000)

params = (:τ⁻¹,:κ,:ψ_1,:ψ_2,:ρ_R,:ρ_g,:ρ_z,:σ_R,:σ_g,:σ_z)

# irf_true  = impulse_response(blanchardkahn(HerbstSchorfheide(θ)),80)

@dim Θ
@dim ε
@dim σ

function true_irf(θ,periods)
        vars = (:R,:g,:z,:y,:Δy,:π)
        shocks = (:ε_z,:ε_g,:ε_R)
        bkhs = blanchardkahn(HerbstSchorfheide(θ))
        irf_table = zeros(
                        Θ(Val(vars)),
                        Ti(Val(Tuple(1:periods))),
                        ε(Val(shocks)),
                        )
        for k in 1:length(shocks)
                irf_table[:,:,k] = impulse_response(bkhs,periods,k)'
        end
        return irf_table
end

function prior_irf(𝒫,samples,periods)
        vars = (:R,:g,:z,:y,:Δy,:π)
        shocks = (:ε_z,:ε_g,:ε_R)
        irf_table_all = zeros(
                        Θ(Val(vars)),
                        Ti(Val(Tuple(1:periods))),
                        X(Val(Tuple(1:samples))),
                        ε(Val(shocks)),
                        )
        for i in 1:samples
                θ_ = rand.(𝒫)
                bkhs = blanchardkahn(HerbstSchorfheide(θ_))
                for j in 1:length(shocks)
                        irf_table_all[:,:,i,j] = impulse_response(bkhs,periods,j)'
                end
        end
        stats = (:mean,:mode,:median,:p5,:p10,:p90,:p95)
        irf_table = zeros(
                        Θ(Val(vars)),
                        Ti(Val(Tuple(1:periods))),
                        σ(Val(stats)),
                        ε(Val(shocks)),
                        )
        for i in vars, j in 1:periods, k in shocks
                irf_table[i,j,:mean,k] = mean(irf_table_all[i,j,:,k])
                irf_table[i,j,:mode,k] = StatsBase.mode(irf_table_all[i,j,:,k])
                irf_table[i,j,:median,k] = StatsBase.median(irf_table_all[i,j,:,k])
                irf_table[i,j,:p5,k] = percentile(irf_table_all[i,j,:,k],5)
                irf_table[i,j,:p10,k] = percentile(irf_table_all[i,j,:,k],10)
                irf_table[i,j,:p90,k] = percentile(irf_table_all[i,j,:,k],90)
                irf_table[i,j,:p95,k] = percentile(irf_table_all[i,j,:,k],95)
        end
        return irf_table
end

function posterior_irf(chain,samples,periods)
        ndraws,nparams,nchains = size(chain)
        vars = (:R,:g,:z,:y,:Δy,:π)
        shocks = (:ε_z,:ε_g,:ε_R)
        irf_table_all = zeros(
                        Θ(Val(vars)),
                        Ti(Val(Tuple(1:periods))),
                        X(Val(Tuple(1:samples))),
                        ε(Val(shocks)),
                        )
        for i in 1:samples
                θi = (rand(1:ndraws),rand(1:nchains))
                θ_ = ComponentArray([[θ.rA];[chain.value[θi[1],p,θi[2]] for p in params]],θ_axes)
                bkhs = blanchardkahn(HerbstSchorfheide(θ_))
                for j in 1:length(shocks)
                        irf_table_all[:,:,i,j] = impulse_response(bkhs,periods,j)'
                end
        end
        stats = (:mean,:mode,:median,:p5,:p10,:p90,:p95)
        irf_table = zeros(
                        Θ(Val(vars)),
                        Ti(Val(Tuple(1:periods))),
                        σ(Val(stats)),
                        ε(Val(shocks)),
                        )
        for i in vars, j in 1:periods, k in shocks
                irf_table[i,j,:mean,k] = mean(irf_table_all[i,j,:,k])
                irf_table[i,j,:mode,k] = StatsBase.mode(irf_table_all[i,j,:,k])
                irf_table[i,j,:median,k] = StatsBase.median(irf_table_all[i,j,:,k])
                irf_table[i,j,:p5,k] = percentile(irf_table_all[i,j,:,k],5)
                irf_table[i,j,:p10,k] = percentile(irf_table_all[i,j,:,k],10)
                irf_table[i,j,:p90,k] = percentile(irf_table_all[i,j,:,k],90)
                irf_table[i,j,:p95,k] = percentile(irf_table_all[i,j,:,k],95)
        end
        return irf_table
end

nperiods = 40
irf_true      = true_irf(θ,nperiods)
irf_prior     = prior_irf(𝒫ₕₛ,500,nperiods)
irf_posterior_nuts = posterior_irf(ch_nuts,500,nperiods)
irf_posterior_mh = posterior_irf(ch_mh,500,nperiods)

x_labels = (
        y = L"y",
        R = L"R",
        π = L"\pi",
)

ε_labels = (
        ε_z = L"\varepsilon_z",
        ε_g = L"\varepsilon_g",
        ε_R = L"\varepsilon_R",
)

g = GroupPlot(3,3, groupStyle = "horizontal sep = 1.75cm, vertical sep = 1.5cm")



for ε in (:ε_z,:ε_g,:ε_R)
        for x in (:y,:R,:π)
            push!(g,
                PGFPlots.Axis([
                    Plots.Linear([collect(1:nperiods); collect(nperiods:-1:1)],[irf_prior[x,:,:p5,ε];reverse(irf_prior[x,:,:p95,ε])], style="olive!0, fill=black!100, fill opacity=0.1, no marks", closedCycle=true),
                    Plots.Linear(collect(1:nperiods),irf_posterior_nuts[x,:,:p5,ε], style="blue!100, very thick, no marks"),
                    Plots.Linear(collect(1:nperiods),irf_posterior_nuts[x,:,:p95,ε], style="blue!100, very thick, no marks"),
                    Plots.Linear(collect(1:nperiods),irf_posterior_mh[x,:,:p5,ε], style="red!100, dotted, very thick, no marks"),
                    Plots.Linear(collect(1:nperiods),irf_posterior_mh[x,:,:p95,ε], style="red!100, dotted, very thick, no marks"),
                    Plots.Linear(irf_true[x,:,ε], style="black, dashed, very thick, no marks"),
                ],
                title=L"%$(x_labels[x]) response to %$(ε_labels[ε])",
                ymin=minimum([irf_posterior_nuts[x,:,:p5,ε];irf_posterior_mh[x,:,:p5,ε];irf_true[x,:,ε]]),
                ymax=maximum([irf_posterior_nuts[x,:,:p95,ε];irf_posterior_mh[x,:,:p95,ε];irf_true[x,:,ε]]),
                xmin=0,
                xmax=(ε == :ε_R ? 8 : 40),
                width="5cm",
                height="5cm"))
        end
end

save("src/images/irfs_standalone.tex",g)
save("src/images/irfs.tex",g, include_preamble=false)



post_nuts = (;zip(params,(kde(ch_nuts[:,p,1].data) for p in params))...)
post_mh = (;zip(params,(kde(ch_mh[:,p,1].data) for p in params))...)

labels = (
        τ⁻¹ = L"\tau^{-1}",
        κ   = L"\kappa",
        ψ_1 = L"\psi_1",
        ψ_2 = L"\psi_2",
        ρ_R = L"\rho_R",
        ρ_g = L"\rho_G",
        ρ_z = L"\rho_z",
        σ_R = L"\sigma_R",
        σ_g = L"\sigma_g",
        σ_z = L"\sigma_z",
)

function support(var,percentile)
    upper = Distributions.percentile(𝒫ₕₛ[var],50 + percentile/2)
    lower = Distributions.percentile(𝒫ₕₛ[var],50 - percentile/2)
    return (lower,upper)
end

g = GroupPlot(3,4, groupStyle = "horizontal sep = 1.75cm, vertical sep = 1.5cm")

for v in params
    push!(g,
        PGFPlots.Axis([
            Plots.Linear(x -> pdf(𝒫ₕₛ[v],x), support(v,99.5), style="black!10, fill=black!10", closedCycle=true),
            Plots.Linear(x -> pdf(post_nuts[v],x),support(v,99.9), style="blue, thick"),
            Plots.Linear(x -> pdf(post_mh[v],x),support(v,99.9), style="red, dotted, thick"),
            Plots.Linear([θ[v],θ[v]],[0.0,1.5*maximum(post_nuts[v].density)], style="black, no marks, very thick"),
        ],ymin=0,title=labels[v],width="5cm", height="5cm"))
end

save("src/images/posteriors_standalone.tex",g)
save("src/images/posteriors.tex",g, include_preamble=false)
