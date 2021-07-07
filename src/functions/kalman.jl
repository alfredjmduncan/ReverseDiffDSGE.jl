
"""
kalman_ll(bk,y,B)

Derive the log-likelihood (up to constant) of time series y given
DSGE solution bk. B is a vector containing the indexes of the
observed variables.

Based on the MXNet function LDS_forward, which can be found here:
https://gluon.mxnet.io/chapter12_time-series/lds-scratch.html
"""


function kalman_ll(bk,y,B,σ_obs)

    Zygote.ignore() do
        global nx,nε = size(bk.D)
        global n     = nx+nε
        global Σₕ    = Diagonal([zeros(nx);ones(nε)]) # shocks
        global Bm    = I(nx+nε)[B,:] # observation equation
        global T,nobs = size(y)      # num of observations
        global Σᵥ     = σ_obs        # observation errors
    end

    A  = [[bk.C  bk.D]; zeros(nε,nx+nε)]

    # Priors (updated throughout the loop)
    μₕ  = zeros(n)
    Σₕₕ = Matrix(I(n))

    # initialise variables
    ll  = 0.0

    for t in 1:T

        Σᵥᵥ = update_Σᵥᵥ(Σₕₕ,B,Σᵥ)
        Σch = cholesky(Σᵥᵥ;check=false)
        if Σch.info != 0
            Zygote.ignore() do
                @warn "Cholesky decomposition failed. Covariance matrix not positive definite."
            end
            ll = -Inf
            break
        end
        ΣL = Σch.L

        K   = update_K(Σₕₕ,B,ΣL)
        δ   = update_δ(μₕ,y,t,B)
        μₕ  = update_μ(A,K,μₕ,δ)
        Σₕₕ = update_Σ(A,K,Σₕₕ,Bm,Σᵥ,Σₕ)
        ll  = update_ll(ll,ΣL,δ)

    end

  return ll

end


"""
# Matrix rules with pullbacks to avoid repeat calculations.
"""

"""
sumlogdiag(A::AbstractMatrix{T})

Take logs of the diagonal values of A, then sum the log values.
"""

function sumlogdiag(A::AbstractMatrix{T}) where T
    return sum(log.(diag(A)))
end

function ChainRulesCore.rrule(::typeof(sumlogdiag), A)
    Σ = sumlogdiag(A)
    function sumlogdiag_pullback(ΔΣ)
        ∂A =  Diagonal(ΔΣ ./diag(A))
        return NoTangent(), ∂A
    end
    return Σ, sumlogdiag_pullback
end

diag(F::CHOLMOD.FactorComponent) = diag(CHOLMOD.sparse(F))

function update_K(Σₕₕ,B,ΣL)
    K   = (Σₕₕ[:,B]/(ΣL'))/ΣL
    return K
end

# function ChainRulesCore.rrule(::typeof(update_K),Σₕₕ,B,ΣL)
#     A₁  = Σₕₕ[:,B]
#     A₂  = A₁/(ΣL')
#     K   = A₂/ΣL
#     function update_K_pullback(ΔK)
#         ∂A₂   = ΔK/(ΣL')
#         ∂A₁   = ∂A₂/ΣL
#         ∂ΣL   = - tril(K'*∂A₂ + ∂A₁'*K)
#         ∂Σₕₕ  = zeros(size(Σₕₕ))
#         ∂Σₕₕ[:,B]  += ∂A₁
#         return NoTangent(),∂Σₕₕ,NoTangent(),∂ΣL
#     end
#     return K, update_K_pullback
# end

function update_Σ(A,K,Σₕₕ,Bm,Σᵥ,Σₕ)
    ImKB = I - K*Bm
    X = A*(ImKB*Σₕₕ*ImKB' + sym_square(Σᵥ,K))*A' + Σₕ
    return X
end


function ChainRulesCore.rrule(::typeof(update_Σ),A,K,Σₕₕ,Bm,Σᵥ,Σₕ)
    ImKB = I - K*Bm
    B₁ = ImKB*Σₕₕ*ImKB'
    B₂ = sym_square(Σᵥ,K)
    B₃ = B₁ + B₂
    B₄ = A*B₃*A'
    X  = B₄ + Σₕ
    function update_Σ_pullback(ΔX)
        ∂B₄   = @thunk(ΔX)
        ∂B₃   = @thunk(A'*∂B₄*A)
        ∂A    = @thunk((∂B₄+∂B₄')*A*B₃)
        ∂B₂   = @thunk(∂B₃)
        ∂B₁   = @thunk(∂B₃)
        ∂Σₕₕ  = @thunk(ImKB'*∂B₁*ImKB)
        ∂ImKB = @thunk((∂B₁+∂B₁')*ImKB*Σₕₕ)
        ∂K    = @thunk(Σᵥ*(∂B₂+∂B₂')*K - ∂ImKB * Bm')
        return NoTangent(),∂A,∂K,∂Σₕₕ,NoTangent(),NoTangent(),NoTangent()
    end
    return X, update_Σ_pullback
end

function update_μ(A,K,μₕ,δ)
    μₕ_ = A*(μₕ + K*δ)
    return μₕ_
end

function ChainRulesCore.rrule(::typeof(update_μ),A,K,μₕ,δ)
    M   = (μₕ + K*δ)
    μₕ_ = A*M
    function update_μ_pullback(Δμₕ_)
        ∂A    = @thunk(Δμₕ_*M')
        ∂M    = A'*Δμₕ_
        ∂μₕ   = @thunk(∂M)
        ∂K    = @thunk(∂M*δ')
        ∂δ    = @thunk(K'*∂M)
        return (NoTangent(),∂A,∂K,∂μₕ,∂δ)
    end
    return μₕ_, update_μ_pullback
end

function update_ll(ll,Σᵥᵥ_chol,δ)
    Z   = Σᵥᵥ_chol\ δ
    ll_ = ll -0.5*Z'*Z - sumlogdiag(Σᵥᵥ_chol)
    return ll_
end


function ChainRulesCore.rrule(::typeof(update_ll),ll,Σᵥᵥ_chol,δ)
    Z   = Σᵥᵥ_chol\δ
    ll_ = ll - 0.5*dot(Z,Z) - sumlogdiag(Σᵥᵥ_chol)
    function update_ll_pullback(Δll_)
        ∂ll       =  @thunk(Δll_)
        ∂Z        = -Δll_*Z
        ∂δ        =  Σᵥᵥ_chol' \ ∂Z
        ∂Σᵥᵥ_chol =  @thunk(Diagonal(-Δll_ ./diag(Σᵥᵥ_chol)) - ∂δ *Z')
        return (NoTangent(),∂ll,∂Σᵥᵥ_chol,∂δ)
    end
    return ll_, update_ll_pullback
end


function update_δ(μₕ,ys,t,B)
    δ = ys[t,:] - μₕ[B]
    return δ
end

function ChainRulesCore.rrule(::typeof(update_δ),μₕ,ys,t,B)
    δ = ys[t,:] - μₕ[B]
    function update_δ_pullback(Δδ)
        ∂μₕ     =  zeros(size(μₕ))
        ∂μₕ[B] += -Δδ
        return (NoTangent(),∂μₕ,NoTangent(),NoTangent(),NoTangent())
    end
    return δ, update_δ_pullback
end

function update_Σᵥᵥ(Σₕₕ,B,Σᵥ)
    Σᵥᵥ  = Symmetric(Σₕₕ[B,B],:U) + Σᵥ*I
    return Σᵥᵥ
end

function ChainRulesCore.rrule(::typeof(update_Σᵥᵥ),Σₕₕ,B,Σᵥ)
    Σᵥᵥ = update_Σᵥᵥ(Σₕₕ,B,Σᵥ)
    function update_Σᵥᵥ_pullback(ΔΣᵥᵥ)
        ∂Σₕₕ       =  zeros(size(Σₕₕ))
        ∂Σₕₕ[B,B] +=  ΔΣᵥᵥ
        return (NoTangent(),∂Σₕₕ,NoTangent(),NoTangent())
    end
    return Σᵥᵥ, update_Σᵥᵥ_pullback
end



"""
sym_square(A,B)

Implements the following product:
    A*B*A'
where B is symmetric.
"""

function sym_square(A::AbstractMatrix{Float64},B::Symmetric{Float64})
    return Symmetric(A*B*A', LinearAlgebra.sym_uplo(B.uplo))
end

function sym_square(A::AbstractMatrix{Float64},B::Diagonal{Float64})
    return Symmetric(A*B*A', :U)
end


function ChainRulesCore.rrule(::typeof(sym_square),A,B)
    X = A*B*A'
    function sym_square_pullback(ΔX)
        ∂A = @thunk((ΔX+ΔX')*A*B)
        ∂B = @thunk(A'*ΔX*A)
        return NoTangent(), ∂A, ∂B
    end
    return X, sym_square_pullback
end

function sym_square(α::Float64,K::AbstractMatrix)
    return Symmetric(BLAS.syrk('U','N',α,K))
end

function ChainRulesCore.rrule(::typeof(sym_square),α::Float64,K::AbstractMatrix)
   X = sym_square(α,K)
   function sym_square_pullback(ΔX)
       ∂K = @thunk(α*(ΔX+ΔX')*K)
       return NoTangent(), NoTangent(),∂K
   end
   return X, sym_square_pullback
end
