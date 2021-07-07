

function step!(x,bkhs)
       nε = size(bkhs.D)[2]
       ε  = rand(Normal(),nε)
       x .= bkhs.C*x + bkhs.D*ε
       return x
end

function timeseries(bkhs,n)
       m  = size(bkhs.C)[1] # number of equations
       x  = zeros(m)
       ts = zeros(n,m)
       for τ in 1:n
           ts[τ,:] = step!(x,bkhs)
       end
       return ts
end

function impulse_response(bkhs,n,iε)
       m  = size(bkhs.C)[1] # number of equations
       x  = zeros(m)
       ts = zeros(n,m)
       nε = size(bkhs.D)[2]
       ε  = I(nε)[:,iε]
       x .= bkhs.C*x + bkhs.D*ε
       for τ in 1:n
           ts[τ,:] = x
           x      .= bkhs.C*x
       end
       return ts
end
