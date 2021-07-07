
# default parameter values



𝒫ₕₛ = ComponentVector{Any}(
        # Steady state related
        rA = (0.5,),
        # Endogenous propagation
        τ⁻¹ = InverseGamma(8,8),
        κ   = Uniform(0.0,1.0),
        ψ_1 = Gamma(4,1/8),
        ψ_2 = Gamma(4,1/8),
        ρ_R = Uniform(0.5,0.9),
        # Exogenous shock parameters
        ρ_g = Uniform(0.90,0.99),
        ρ_z = Uniform(0.90,0.99),
        σ_R = InverseGamma(4,0.32),
        σ_g = InverseGamma(4,2.0),
        σ_z = InverseGamma(4,0.5),
)

"""
HerbstSchorfheide()

Small scale New Keynesian model from Herbst and Schorfheide,
"Bayesian Estimation of DSGE Models" (2016, Ch 2.1), in
Blanchard-Kahn canonical form.

"""

function HerbstSchorfheide(θ=Distributions.mode.(𝒫ₕₛ); output_indexes=false)

        eqn= (:mr,:exog_z,:exog_g,:y₋₁,:Δy,:π_obs,:euler,:phillips)  # equations (equations with expectations last)
        x  = (:R,:g,:z,:y₋₁,:Δy,:π_obs)                         # state vars
        y  = (:y,:π)                 # jump vars
        ε  = (:ε_z,:ε_g,:ε_R)        # exogenous shocks

        neqn,nx,ny,nε = length.((eqn,x,y,ε))
        ieqn = (; zip(eqn,(1:neqn...,))...)
        ix   = (; zip(x,(1:nx...,))...)
        iy   = (; zip(y,nx .+ (1:ny...,))...)
        iε   = (; zip(ε,(1:nε...,))...)

        β  = 1/(1+θ.rA/400)

        B = zeros(neqn,nx+ny)
        A = zeros(neqn,nx+ny)
        G = zeros(neqn,nε)

        B[ieqn[:euler],ix[:R]] =  θ.τ⁻¹
        B[ieqn[:euler],ix[:z]] = -θ.ρ_z*θ.τ⁻¹
        B[ieqn[:euler],ix[:g]] = -(1-θ.ρ_g)
        B[ieqn[:euler],iy[:y]] = -1
        B[ieqn[:euler],iy[:π]] = -θ.τ⁻¹
        A[ieqn[:euler],iy[:y]] = -1

        B[ieqn[:y₋₁],ix[:y₋₁]] = 1
        A[ieqn[:y₋₁],iy[:y]]   = 1

        B[ieqn[:Δy],ix[:Δy]]  =  1
        A[ieqn[:Δy],ix[:y₋₁]] = -1
        A[ieqn[:Δy],iy[:y]]   =  1

        B[ieqn[:π_obs],ix[:π_obs]] = 1
        A[ieqn[:π_obs],iy[:π]]     = 1

        B[ieqn[:phillips],iy[:π]] = -β
        B[ieqn[:phillips],ix[:g]] =  θ.κ
        A[ieqn[:phillips],iy[:y]] =  θ.κ
        A[ieqn[:phillips],iy[:π]] = -1

        B[ieqn[:mr],ix[:R]]   =  1
        B[ieqn[:mr],ix[:g]]   = (1-θ.ρ_R)*θ.ψ_2
        A[ieqn[:mr],ix[:R]]   =  θ.ρ_R
        A[ieqn[:mr],iy[:y]]   = (1-θ.ρ_R)*θ.ψ_2
        A[ieqn[:mr],iy[:π]]   = (1-θ.ρ_R)*(1+θ.ψ_1)
        G[ieqn[:mr],iε[:ε_R]] = -θ.σ_R

        B[ieqn[:exog_z],ix[:z]]   = 1
        A[ieqn[:exog_z],ix[:z]]   = θ.ρ_z
        G[ieqn[:exog_z],iε[:ε_z]] = θ.σ_z

        B[ieqn[:exog_g],ix[:g]]   = 1
        A[ieqn[:exog_g],ix[:g]]   = θ.ρ_g
        G[ieqn[:exog_g],iε[:ε_g]] = θ.σ_g


        B₁₁=vec(B[1:nx,1:nx])
        B₁₂=vec(B[1:nx,nx+1:neqn])
        B₂₁=vec(B[nx+1:neqn,1:nx])
        B₂₂=vec(B[nx+1:neqn,nx+1:neqn])
        A₁₁=vec(A[1:nx,1:nx])
        A₁₂=vec(A[1:nx,nx+1:neqn])
        A₂₁=vec(A[nx+1:neqn,1:nx])
        A₂₂=vec(A[nx+1:neqn,nx+1:neqn])
        G₁=vec(G[1:nx,:])
        G₂=vec(G[nx+1:neqn,:])


        Γ_vec = [B₁₁;B₁₂;B₂₁;B₂₂;A₁₁;A₁₂;A₂₁;A₂₂;G₁;G₂]


        axes = (Axis(
                B₁₁=ViewAxis(1:nx^2, ShapedAxis((nx, nx), NamedTuple())),
                B₁₂=ViewAxis(nx^2+1:nx^2+nx*ny, ShapedAxis((nx, ny), NamedTuple())),
                B₂₁=ViewAxis(nx^2+nx*ny+1:nx^2+2*nx*ny, ShapedAxis((ny, nx), NamedTuple())),
                B₂₂=ViewAxis(nx^2+2*nx*ny+1:neqn^2, ShapedAxis((ny, ny), NamedTuple())),
                A₁₁=ViewAxis(neqn^2+1:neqn^2+nx^2, ShapedAxis((nx, nx), NamedTuple())),
                A₁₂=ViewAxis(neqn^2+nx^2+1:neqn^2+nx^2+nx*ny, ShapedAxis((nx, ny), NamedTuple())),
                A₂₁=ViewAxis(neqn^2+nx^2+nx*ny+1:neqn^2+nx^2+2*nx*ny, ShapedAxis((ny, nx), NamedTuple())),
                A₂₂=ViewAxis(neqn^2+nx^2+2*nx*ny+1:2*neqn^2, ShapedAxis((ny, ny), NamedTuple())),
                G₁ =ViewAxis(2*neqn^2+1:2*neqn^2+nx*nε, ShapedAxis((nx, nε), NamedTuple())),
                G₂ =ViewAxis(2*neqn^2+nx*nε+1:2*neqn^2+neqn*nε, ShapedAxis((ny, nε), NamedTuple()))
                ),)

        Γ = ComponentArray(Γ_vec,axes)

        if output_indexes
                return Γ,neqn,nx,ny,nε,ieqn,ix,iy,iε
        else
                return Γ
        end
end

function ChainRulesCore.rrule(::typeof(HerbstSchorfheide), θ)

    Γ,neqn,nx,ny,nε,ieqn,ix,iy,iε = HerbstSchorfheide(θ; output_indexes=true)

    function HerbstSchorfheide_pullback(ΔΓ)

        CAΔΓ = ComponentArray(ΔΓ,getaxes(Γ))

        Ḡ = [reshape(CAΔΓ.G₁,nx,nε) ; reshape(CAΔΓ.G₂,ny,nε)]
        B̄ = [reshape(CAΔΓ.B₁₁,nx,nx) reshape(CAΔΓ.B₁₂,nx,ny) ;
             reshape(CAΔΓ.B₂₁,ny,nx) reshape(CAΔΓ.B₂₂,ny,ny)]
        Ā = [reshape(CAΔΓ.A₁₁,nx,nx) reshape(CAΔΓ.A₁₂,nx,ny) ;
             reshape(CAΔΓ.A₂₁,ny,nx) reshape(CAΔΓ.A₂₂,ny,ny)]

        β  = 1/(1+θ.rA/400)

        ∂σ_g =  Ḡ[ieqn[:exog_g],iε[:ε_g]]
        ∂ρ_g =  Ā[ieqn[:exog_g],ix[:g]] + B̄[ieqn[:euler],ix[:g]]
        ∂σ_z =  Ḡ[ieqn[:exog_z],iε[:ε_z]]
        ∂ρ_z =  Ā[ieqn[:exog_z],ix[:z]] - B̄[ieqn[:euler],ix[:z]]*θ.τ⁻¹
        ∂σ_R = -Ḡ[ieqn[:mr],iε[:ε_R]]
        ∂κ   =  B̄[ieqn[:phillips],ix[:g]] + Ā[ieqn[:phillips],iy[:y]]
        ∂β   = -B̄[ieqn[:phillips],iy[:π]]
        ∂rA  = -∂β*(1/400)*β^2
        ∂ρ_R = -(B̄[ieqn[:mr],ix[:g]]+Ā[ieqn[:mr],iy[:y]])*θ.ψ_2 +
                 Ā[ieqn[:mr],ix[:R]] - Ā[ieqn[:mr],iy[:π]]*(1+θ.ψ_1)
        ∂τ⁻¹ =  B̄[ieqn[:euler],ix[:R]] - θ.ρ_z*B̄[ieqn[:euler],ix[:z]] - B̄[ieqn[:euler],iy[:π]]

        ∂ψ_1 =  Ā[ieqn[:mr],iy[:π]]*(1-θ.ρ_R)
        ∂ψ_2 =  (B̄[ieqn[:mr],ix[:g]] + Ā[ieqn[:mr],iy[:y]])*(1-θ.ρ_R)

        ∂θ = ComponentArray(
                [∂rA,∂τ⁻¹,∂κ,∂ψ_1,∂ψ_2,∂ρ_R,∂ρ_g,∂ρ_z,∂σ_R,∂σ_g,∂σ_z],
                getaxes(θ)
             )

        return NO_FIELDS, ∂θ
    end
    return Γ, HerbstSchorfheide_pullback
end
