
"""
blanchardkahn()

Blanchard-Kahn method for solving linear rational
expectations models.

"""

function blanchardkahn(Γ::AbstractArray{T};output_all=false) where T

    A = [Γ.A₁₁ Γ.A₁₂ ; Γ.A₂₁ Γ.A₂₂]
    B = [Γ.B₁₁ Γ.B₁₂ ; Γ.B₂₁ Γ.B₂₂]

    F_ = schur(A,B)
    F  = ordschur(F_,abs.(F_.α./F_.β) .< 1) # push large eigs to the bottom

    nx,ny = size(Γ.A₁₂)
    S₁₁,S₁₂,S₂₁,S₂₂ = blocks(F.S,nx)
    T₁₁,T₁₂,T₂₁,T₂₂ = blocks(F.T,nx)
    Qᵀ₁₁,Qᵀ₁₂,Qᵀ₂₁,Qᵀ₂₂ = blocks(F.Q',nx)
    Zᵀ₁₁,Zᵀ₁₂,Zᵀ₂₁,Zᵀ₂₂ = blocks(F.Z',nx)

    V = inv(Zᵀ₂₂)
    N = V*Zᵀ₂₁
    L = V*(S₂₂\(Qᵀ₂₁*Γ.G₁ + Qᵀ₂₂*Γ.G₂))

    β₁⁻¹ = inv(Γ.B₁₁ - Γ.B₁₂*N)
    C  = β₁⁻¹*(Γ.A₁₁ - Γ.A₁₂*N)
    D  = β₁⁻¹*(Γ.G₁ - Γ.A₁₂*L)

    axes = (Axis(
            C=ViewAxis((1:length(C)), ShapedAxis(size(C), NamedTuple())),
            D=ViewAxis(length(C) .+ (1:length(D)), ShapedAxis(size(D), NamedTuple())),
            ),)
    Φ = ComponentArray([vec(C);vec(D)],axes)

    if output_all
        return Φ,N,L,β₁⁻¹,nx,ny
    else
        return Φ
    end
end

function ChainRulesCore.rrule(::typeof(blanchardkahn), Γ)

    Φ,N,L,β₁⁻¹,nx,ny = blanchardkahn(Γ;output_all=true)

    β₁⁻ᵗ = β₁⁻¹'
    ηᵗ   = β₁⁻ᵗ*(Γ.B₂₁' - N'*Γ.B₂₂')
    nε   = size(Φ.D)[2]

    function blanchardkahn_pullback(ΔΦ)

        # If you use or extend this code, please cite
        # Duncan, Alfred (2021) "Reverse Mode Differentiation for DSGE Models,"
        # University of Kent Discussion Paper Series.

        CAΔΦ = ComponentArray(ΔΦ,getaxes(Φ))

        α₂  = Γ.A₁₂'*ηᵗ-Γ.A₂₂'
        β₂  = Γ.B₂₂'-Γ.B₁₂'*ηᵗ
        Ξ   = kron(Φ.C,β₂) + kron(I(nx),α₂)

        β₁⁻ᵗΔC = β₁⁻ᵗ*CAΔΦ.C
        β₁⁻ᵗΔD = β₁⁻ᵗ*CAΔΦ.D
        βCD = β₁⁻ᵗΔC*Φ.C'+β₁⁻ᵗΔD*Φ.D'

        Q   = inv(α₂)*Γ.A₁₂'*β₁⁻ᵗΔD

        w   = -Ξ\vec(Γ.B₁₂'*βCD - Γ.A₁₂'*β₁⁻ᵗΔC + β₂*Q*Φ.D')
        W   =  reshape(w,ny,nx)

        ∂Γ = similar(Γ)
        ∂Γ.A₁₁ = -ηᵗ*W + β₁⁻ᵗΔC
        ∂Γ.A₂₁ =  W
        ∂Γ.A₂₂ = -W*N' - Q*L'
        ∂Γ.A₁₂ = -ηᵗ*∂Γ.A₂₂ - β₁⁻ᵗΔC*N' - β₁⁻ᵗΔD*L'
        ∂Γ.B₂₁ = -W*Φ.C' - Q*Φ.D'
        ∂Γ.B₁₁ = -ηᵗ*∂Γ.B₂₁ - βCD
        ∂Γ.B₁₂ = -∂Γ.B₁₁*N'
        ∂Γ.B₂₂ = -∂Γ.B₂₁*N'
        ∂Γ.G₁  = -ηᵗ*Q+β₁⁻ᵗΔD
        ∂Γ.G₂  =  Q

        return NO_FIELDS, ∂Γ
    end
    return Φ, blanchardkahn_pullback
end


blocks(A::AbstractArray,nx) =
    A[1:nx,1:nx],A[1:nx,nx+1:size(A)[1]],
    A[nx+1:size(A)[1],1:nx],A[nx+1:size(A)[1],nx+1:size(A)[1]]
