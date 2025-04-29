function [B,par] = prox_nuclear_norm(Binput,rho,flag)
[m,q] = size(Binput);
if flag == 1
    [U,S,V] = svd(Binput);
    s = max(diag(S)-rho,0);
    Snew = [diag(s),zeros(m,q-m)];
    B = U*Snew*V';
    if nargout > 1
        par.U = U;
        par.V = V;
        par.dS = diag(S);
        par.alpha1 = (diag(S)>rho);
        par.alpha2 = (diag(S)==rho);
        par.alpha3 = (diag(S)<rho);
    end
elseif flag == 2
    [U,S,V] = svd(Binput);
    s = max(diag(S)-rho,0);
    Snew = diag(s);
    B = U*Snew*V';
    if nargout > 1
        par.U = U;
        par.V = V;
        par.dS = diag(S);
        par.alpha1 = (diag(S)>rho);
        par.alpha2 = (diag(S)==rho);
        par.alpha3 = (diag(S)<rho);
    end
else
    [U,S,V] = svd(Binput');
    s = max(diag(S)-rho,0);
    Snew = [diag(s),zeros(q,m-q)];
    B = V*Snew'*U';
    if nargout > 1
        par.U = U;
        par.V = V;
        par.dS = diag(S);
        par.alpha1 = (diag(S)>rho);
        par.alpha2 = (diag(S)==rho);
        par.alpha3 = (diag(S)<rho);
    end
end
end
