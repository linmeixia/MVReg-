function y = mat_ppdna_linear_vecmat_vec(xi,par,nzcol)
para = par.para;
Xmap = par.Xmap;
Xtmap = par.Xtmap;
m = par.m;
q = par.q;


y=par.sigma/(par.tau+par.sigma)*xi + par.sigma*(par.XkXkt*xi);

if strcmp(para.vec_regularizer,'lasso')
    y = y + par.sigma*(par.ZkZkt*xi);
elseif strcmp(para.vec_regularizer,'fused lasso')
    y = y + par.sigma*matFL(xi,par,par.numblk1);
elseif strcmp(para.vec_regularizer,'clustered lasso')
    y = y + par.sigma*matClu(xi,par,par.numblk1);
elseif strcmp(para.vec_regularizer,'sparse group lasso')
    if ~par.id_yes 
        y = y + par.D*(xi'*par.D)';
    end
elseif strcmp(para.vec_regularizer,'slope')
    y = y + par.sigma*matslope(xi,par,par.numblk1);
elseif strcmp(para.vec_regularizer,'exclusive lasso')
    y = y + par.sigma*matexclusive(xi,par,[]);
end
end