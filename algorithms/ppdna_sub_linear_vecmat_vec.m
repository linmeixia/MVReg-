function [U,beta,w,TtXtxi,Ztxi,xi,par,runhist,info] = ppdna_sub_linear_vecmat_vec(y,Ainput,B,gamma,XTB,Zgamma,TtXtxi,Ztxi,xi,para,par,options)
printsub = 1;
breakyes = 0;
maxitersub = 50;
tol = 1e-6;
maxitpsqmr =500;
if isfield(options,'printsub'); printsub = options.printsub; end
if isfield(options,'maxitersub'); maxitersub = options.maxitersub; end
if isfield(options,'tol'); tol = min(tol,options.tol); end
%% parameter
rho = para.rho;
p = par.p;
if strcmp(para.vec_regularizer,'lasso')
    lambda = para.lambda;
elseif strcmp(para.vec_regularizer,'fused lasso')
    lambda1 = para.lambda1;
    lambda2 = para.lambda2;
    Binput = para.Binput;
elseif strcmp(para.vec_regularizer,'clustered lasso')
    lambda1 = para.lambda1;
    lambda2 = para.lambda2;
    wvec = linspace(p-1,1-p,p);
elseif strcmp(para.vec_regularizer,'sparse group lasso')
    c = para.c_lambda;
    P = Def_P(p,para.G,para.ind);
elseif strcmp(para.vec_regularizer,'slope')
    c_lambda = para.c_lambda;
elseif strcmp(para.vec_regularizer,'exclusive lasso')
    lambda1 = para.lambda1;
    group_info = para.group_info;
end
%%
sig = par.sigma;
sigdtau = par.sigdtau;
[m,q] = size(B);
XTBpZgamma = XTB+Zgamma;
%% preperation
flag = Ainput.flag;
Xmap = @(x) Ainput.Xmap(x);
Xtmap = @(x) Ainput.Xtmap(x);
Zmap = @(x) Ainput.Zmap(x);
Ztmap = @(x) Ainput.Ztmap(x);
par.flag = flag;
par.Xmap = Xmap;
par.Xtmap = Xtmap;
par.rho = rho;
par.para = para;
par.y = y;

par.m = m;
par.q = q;
%%
Uinput  = B - sig*TtXtxi;
[Up,info_U] = proxL1(Uinput(:),sig*rho);
par.info_U = info_U;
Up = reshape(Up,m,q);
U = (Uinput-Up)/sig;

betainput = gamma - sig*Ztxi;
if strcmp(para.vec_regularizer,'lasso')
    [betap,info_beta] = proxL1(betainput,sig*lambda);
    par.info_beta = info_beta;
elseif strcmp(para.vec_regularizer,'fused lasso')
    [betap,info_beta] = proxFL(Binput,betainput,sig*lambda1,sig*lambda2);
    par.info_beta = info_beta;
elseif strcmp(para.vec_regularizer,'clustered lasso')
    [betap,info_beta] = proxClu(betainput, sig*lambda1, sig*lambda2);
    par.info_beta = info_beta;
elseif strcmp(para.vec_regularizer,'sparse group lasso')
    betap = proxsgl(betainput,sig*c,P);
    par.input = betainput/sig;
    par.c = c;
    par.P = P;
elseif strcmp(para.vec_regularizer,'slope')
    [betap,info_beta] = proxSortedL1(betainput,sig*c_lambda);
    par.info_beta = info_beta;
elseif strcmp(para.vec_regularizer,'exclusive lasso')
    [betap,info_beta] = prox_exclusive(betainput,p,sig*lambda1,group_info);
    par.info_beta = info_beta;
    par.sl1 = sig*lambda1;
end
beta = (betainput-betap)/sig;

winput = XTBpZgamma + sigdtau*xi;
wp = (winput + sigdtau*y)/(1+sigdtau);
w = (winput-wp)/sig;

Lxi = - norm(Uinput,'fro')^2/(2*sig) + (sig/2)*norm(U,'fro')^2 - norm(betainput)^2/(2*sig) + (sig/2)*norm(beta)^2 - norm(winput)^2/(2*sigdtau) + (sigdtau/2)*norm(w)^2;
if strcmp(para.vec_regularizer,'lasso')
    Lxi = Lxi + rho*sum(abs(Up(:))) + lambda*sum(abs(betap)) + 0.5*norm(wp-y)^2;
elseif strcmp(para.vec_regularizer,'fused lasso')
    Bbetap = Binput.Bmap(betap);
    Lxi = Lxi + rho*sum(abs(Up(:))) + lambda1*sum(abs(betap)) + lambda2*sum(abs(Bbetap)) + 0.5*norm(wp-y)^2;
elseif strcmp(para.vec_regularizer,'clustered lasso')
    betapsort = sort(betap,'descend');
    Lxi = Lxi + rho*sum(abs(Up(:))) + lambda1*sum(abs(betap)) + lambda2*(wvec*betapsort) + 0.5*norm(wp-y)^2;
elseif strcmp(para.vec_regularizer,'sparse group lasso')
    Pbetap = P.matrix*(betap);
    Lxi = Lxi + rho*sum(abs(Up(:))) + c(1)*sum(abs(betap)) + c(2)*P.Lasso_fz(Pbetap) + 0.5*norm(wp-y)^2;
elseif strcmp(para.vec_regularizer,'slope')
    Lxi = Lxi + rho*sum(abs(Up(:))) + c_lambda'*sort(abs(betap),'descend') + 0.5*norm(wp-y)^2;
elseif strcmp(para.vec_regularizer,'exclusive lasso')
    Lxi = Lxi + rho*sum(abs(Up(:))) + lambda1*xgroupnorm(betap,group_info) + 0.5*norm(wp-y)^2;
end
runhist.psqmr(1) = 0;
runhist.findstep(1) = 0;

const_Lxi = norm(B,'fro')^2/(2*sig) + norm(gamma)^2/(2*sig) + norm(XTBpZgamma)^2/(2*sigdtau);
%% main Newton iteration
for itersub = 1:maxitersub
    Rd1 = TtXtxi + U;
    Rd2 = Ztxi + beta;
    Rd3 = w - xi;
    XUp = Xmap(Up(:));
    Zbetap = Zmap(betap);
    XUppZbetap = XUp + Zbetap;
    GradLxi = -(wp - XUppZbetap);
    normxi = norm(xi);
    normGradLxi = norm(GradLxi)/(1+normxi);
    priminf_sub = normGradLxi;
    normRd1 = norm(Rd1,'fro');
    normRd2 = norm(Rd2);
    normRd3 = norm(Rd3);
    dualinf_sub = max([normRd1/(1+norm(U,'fro')),normRd2/(1+norm(beta)),normRd3/(1+normxi)]);
    if max(priminf_sub,dualinf_sub) < tol
        tolsubconst = 0.01; %0.1;
    else
        tolsubconst = 0.005; %0.05;
    end
    tolsub = max(min(1,par.tolconst*dualinf_sub),tolsubconst*tol);
    runhist.priminf(itersub) = priminf_sub;
    runhist.dualinf(itersub) = dualinf_sub;
    runhist.Lxi(itersub)      = Lxi;
    if (printsub)
        fprintf('\n      %2.0d  %- 11.10e [%3.2e %3.2e]',itersub,Lxi,priminf_sub,dualinf_sub);
    end
    psix = 0.5*norm(XUppZbetap-y)^2 + rho*sum(abs(Up(:))) + norm(Up-B,'fro')^2/(2*sig) + norm(betap-gamma)^2/(2*sig) + norm(XUppZbetap-XTBpZgamma)^2/(2*sigdtau);
    if strcmp(para.vec_regularizer,'lasso')
        psix = psix + lambda*sum(abs(betap));
    elseif strcmp(para.vec_regularizer,'fused lasso')
        Bbetap = Binput.Bmap(betap);
        psix = psix + lambda1*sum(abs(betap)) + lambda2*sum(abs(Bbetap));
    elseif strcmp(para.vec_regularizer,'clustered lasso')
        betapsort = sort(betap,'descend');
        psix = psix + lambda1*sum(abs(betap)) + lambda2*(wvec*betapsort);
    elseif strcmp(para.vec_regularizer,'sparse group lasso')
        Pbetap = P.matrix*(betap);
        psix = psix + c(1)*sum(abs(betap)) + c(2)*P.Lasso_fz(Pbetap);
    elseif strcmp(para.vec_regularizer,'slope')
        psix = psix + c_lambda'*sort(abs(betap),'descend');
    elseif strcmp(para.vec_regularizer,'exclusive lasso')
        psix = psix + lambda1*xgroupnorm(betap,group_info);
    end

    if abs(psix-Lxi-const_Lxi) < tolsub && itersub > 1
        msg = 'good termination in subproblem:';
        if printsub
            fprintf('\n       %s  ',msg);
            fprintf(' normRd1=%3.2e, normRd2=%3.2e, normRd3=%3.2e, gap_sub=%3.2e, gradLxi = %3.2e, tolsub=%3.2e',...
                normRd1,normRd2,normRd3,abs(psix-Lxi-const_Lxi),normGradLxi,tolsub);
        end
        breakyes = -1;
        XTUp = Xmap(Up(:));
        Zbetap = Zmap(betap);
        break;
    end
    %% Compute Newton direction
    if (dualinf_sub > 1e-3) || (itersub <= 5)
        maxitpsqmr = max(maxitpsqmr,200);
    elseif (dualinf_sub > 1e-4)
        maxitpsqmr = max(maxitpsqmr,300);
    elseif (dualinf_sub > 1e-5)
        maxitpsqmr = max(maxitpsqmr,400);
    elseif (dualinf_sub > 5e-6)
        maxitpsqmr = max(maxitpsqmr,500);
    end
    if (itersub > 1)
        prim_ratio = priminf_sub/runhist.priminf(itersub-1);
        dual_ratio = dualinf_sub/runhist.dualinf(itersub-1);
    else
        prim_ratio = 0; dual_ratio = 0;
    end
    rhs = GradLxi;
    tolpsqmr = min([5e-3, 0.1*norm(rhs)]);
    const2 = 1;
    if itersub > 1 && (prim_ratio > 0.5 || priminf_sub > 0.1*runhist.priminf(1))
        const2 = 0.5*const2;
    end
    if (dual_ratio > 1.1); const2 = 0.5*const2; end
    tolpsqmr = const2*tolpsqmr;
    par.tol = tolpsqmr; par.maxit = 2*maxitpsqmr;

    [dxi,resnrm,solve_ok,par] = ppdna_Netwonsolve_linear_vecmat_vec(Ainput,rhs,par);

    TtXtdxi = reshape(Xtmap(dxi),m,q);
    Ztdxi = Ztmap(dxi);
    iterpsqmr = length(resnrm)-1;

    if (printsub)
        fprintf('| [%3.1e %3.1e %3.1d]',par.tol,resnrm(end),iterpsqmr);
    end
    if (itersub <= 3) && (dualinf_sub > 1e-4) || (itersub <3)
        stepop = 1;
    else
        stepop = 2;
    end
    steptol = 1e-5;
    step_op.stepop = stepop;
    [par,Lxi,xi,TtXtxi,Ztxi,U,Up,beta,betap,w,wp,alp,iterstep] = findstep(par,y,para,Lxi,xi,TtXtxi,Ztxi,U,Up,beta,betap,w,wp,dxi,TtXtdxi,Ztdxi,steptol,step_op);
    runhist.solve_ok(itersub) = solve_ok;
    runhist.psqmr(itersub)    = iterpsqmr;
    runhist.findstep(itersub) = iterstep;
    Ly_ratio = 1;
    if (itersub > 1)
        Ly_ratio = (Lxi-runhist.Lxi(itersub-1))/(abs(Lxi)+eps);
    end
    if (printsub)
        fprintf(' | %3.2e %2.0f',alp,iterstep);
        if (Ly_ratio < 0); fprintf('-'); end
    end
    %% check for stagnation
    if (itersub > 4)
        idx = max(1,itersub-3):itersub;
        tmp = runhist.priminf(idx);
        ratio = min(tmp)/max(tmp);
        if (all(runhist.solve_ok(idx) <= -1)) && (ratio > 0.9) ...
                && (min(runhist.psqmr(idx)) == max(runhist.psqmr(idx))) ...
                && (max(tmp) < 5*tol)
            fprintf('#')
            breakyes = 1;
        end
        const3 = 0.7;
        priminf_1half  = min(runhist.priminf(1:ceil(itersub*const3)));
        priminf_2half  = min(runhist.priminf(ceil(itersub*const3)+1:itersub));
        priminf_best   = min(runhist.priminf(1:itersub-1));
        priminf_ratio  = runhist.priminf(itersub)/runhist.priminf(itersub-1);
        stagnate_idx   = find(runhist.solve_ok(1:itersub) <= -1);
        stagnate_count = length(stagnate_idx);
        idx2 = max(1,itersub-7):itersub;
        if (itersub >= 10) && all(runhist.solve_ok(idx2) == -1) ...
                && (priminf_best < 1e-2) && (dualinf_sub < 1e-3)
            tmp = runhist.priminf(idx2);
            ratio = min(tmp)/max(tmp);
            if (ratio > 0.5)
                if (printsub); fprintf('##'); end
                breakyes = 2;
            end
        end
        if (itersub >= 15) && (priminf_1half < min(2e-3,priminf_2half)) ...
                && (dualinf_sub < 0.8*runhist.dualinf(1)) && (dualinf_sub < 1e-3) ...
                && (stagnate_count >= 3)
            if (printsub); fprintf('###'); end
            breakyes = 3;
        end
        if (itersub >= 15) && (priminf_ratio < 0.1) ...
                && (priminf_sub < 0.8*priminf_1half) ...
                && (dualinf_sub < min(1e-3,2*priminf_sub)) ...
                && ((priminf_sub < 2e-3) || (dualinf_sub < 1e-5 && priminf_sub < 5e-3)) ...
                && (stagnate_count >= 3)
            if (printsub); fprintf(' $$'); end
            breakyes = 4;
        end
        if (itersub >=10) && (dualinf_sub > 5*min(runhist.dualinf)) ...
                && (priminf_sub > 2*min(runhist.priminf)) %% add: 08-Apr-2008
            if (printsub); fprintf('$$$'); end
            breakyes = 5;
        end
        if (itersub >= 20)
            dualinf_ratioall = runhist.dualinf(2:itersub)./runhist.dualinf(1:itersub-1);
            idx = find(dualinf_ratioall > 1);
            if (length(idx) >= 3)
                dualinf_increment = mean(dualinf_ratioall(idx));
                if (dualinf_increment > 1.25)
                    if (printsub); fprintf('^^'); end
                    breakyes = 6;
                end
            end
        end
        if breakyes > 0
            XTUp = Xmap(Up(:));
            Zbetap = Zmap(betap);
            break;
        end
    end
    if itersub == maxitersub
        XTUp = Xmap(Up(:));
        Zbetap = Zmap(betap);
    end
end
info.breakyes = breakyes;
info.Up = Up;
info.betap = betap;
info.wp = wp;
info.XTUp = XTUp;
info.Zbetap = Zbetap;
end

function [par,Lxi,xi,TtXtxi,Ztxi,U,Up,beta,betap,w,wp,alp,iter] = findstep(par,y,para,Lxi0,xi0,TtXtxi0,Ztxi0,U0,Up0,beta0,betap0,w0,wp0,dxi,TtXtdxi,Ztdxi,tol,options)
printlevel = 0;
if isfield(options,'stepop'); stepop = options.stepop; end
if isfield(options,'printlevel'); printlevel = options.printlevel; end
maxit = ceil(log(1/(tol+eps))/log(2));
c1 = 1e-4; c2 = 0.9;
sig = par.sigma;
sigdtau = par.sigdtau;
flag = par.flag;
m = par.m;
q = par.q;
%% parameter
rho = para.rho;
p = par.p;
if strcmp(para.vec_regularizer,'lasso')
    lambda = para.lambda;
elseif strcmp(para.vec_regularizer,'fused lasso')
    lambda1 = para.lambda1;
    lambda2 = para.lambda2;
    Binput = para.Binput;
elseif strcmp(para.vec_regularizer,'clustered lasso')
    lambda1 = para.lambda1;
    lambda2 = para.lambda2;
    wvec = linspace(p-1,1-p,p);
elseif strcmp(para.vec_regularizer,'sparse group lasso')
    c = para.c_lambda;
    P = Def_P(p,para.G,para.ind);
elseif strcmp(para.vec_regularizer,'slope')
    c_lambda = para.c_lambda;
elseif strcmp(para.vec_regularizer,'exclusive lasso')
    lambda1 = para.lambda1;
    group_info = para.group_info;
end
%%
g0  = dxi'* (- wp0) + sum(sum(TtXtdxi.*Up0)) + Ztdxi'*betap0;
if  (g0 <= 0)
    alp = 0; iter = 0;
    if (printlevel)
        fprintf('\n Need an ascent direction, %2.1e  ',g0);
    end
    xi = xi0;
    Lxi = Lxi0;
    TtXtxi = TtXtxi0;
    Ztxi = Ztxi0;
    U = U0;
    Up = Up0;
    beta = beta0;
    betap = betap0;
    w = w0;
    wp = wp0;
    return;
end
%%
alp = 1; alpconst = 0.5;
for iter = 1:maxit
    if (iter == 1)
        alp = 1; LB = 0; UB = 1;
    else
        alp = alpconst*(LB+UB);
    end
    xi = xi0 + alp*dxi;

    Uinput = Up0 + sig*U0 - sig*alp*TtXtdxi;
    [Up,info_U] = proxL1(Uinput(:),sig*rho);
    par.info_U = info_U;
    Up=reshape(Up,[m,q]);
    U = (Uinput - Up)/sig;

    betainput = betap0 + sig*beta0 - sig*alp*Ztdxi;
    if strcmp(para.vec_regularizer,'lasso')
        [betap,info_beta] = proxL1(betainput,sig*lambda);
        par.info_beta = info_beta;
    elseif strcmp(para.vec_regularizer,'fused lasso')
        [betap,info_beta] = proxFL(Binput,betainput,sig*lambda1,sig*lambda2);
        par.info_beta = info_beta;
    elseif strcmp(para.vec_regularizer,'clustered lasso')
        [betap,info_beta] = proxClu(betainput, sig*lambda1, sig*lambda2);
        par.info_beta = info_beta;
    elseif strcmp(para.vec_regularizer,'sparse group lasso')
        betap = proxsgl(betainput,sig*c,P);
        par.input = betainput/sig;
        par.c = c;
        par.P = P;
    elseif strcmp(para.vec_regularizer,'slope')
        [betap,info_beta] = proxSortedL1(betainput,sig*c_lambda);
        par.info_beta = info_beta;
    elseif strcmp(para.vec_regularizer,'exclusive lasso')
        [betap,info_beta] = prox_exclusive(betainput,p,sig*lambda1,group_info);
        par.info_beta = info_beta;
        par.sl1 = sig*lambda1;
    end

    beta = (betainput - betap)/sig;

    winput = sigdtau*w0 + wp0 + sigdtau*alp*dxi;
    wp = (winput + sigdtau*y)/(1+sigdtau);
    w = (winput-wp)/sigdtau;

    galp = dxi'*(- wp) + sum(sum(TtXtdxi.*Up)) + Ztdxi'*betap;
    Lxi = - norm(Uinput,'fro')^2/(2*sig) + (sig/2)*norm(U,'fro')^2 - norm(betainput)^2/(2*sig) + (sig/2)*norm(beta)^2 - norm(winput)^2/(2*sigdtau) + (sigdtau/2)*norm(w)^2;
    if strcmp(para.vec_regularizer,'lasso')
        Lxi = Lxi + rho*sum(abs(Up(:))) + lambda*sum(abs(betap)) + 0.5*norm(wp-y)^2;
    elseif strcmp(para.vec_regularizer,'fused lasso')
        Bbetap = Binput.Bmap(betap);
        Lxi = Lxi + rho*sum(abs(Up(:))) + lambda1*sum(abs(betap)) + lambda2*sum(abs(Bbetap)) + 0.5*norm(wp-y)^2;
    elseif strcmp(para.vec_regularizer,'clustered lasso')
        betapsort = sort(betap,'descend');
        Lxi = Lxi + rho*sum(abs(Up(:))) + lambda1*sum(abs(betap)) + lambda2*(wvec*betapsort) + 0.5*norm(wp-y)^2;
    elseif strcmp(para.vec_regularizer,'sparse group lasso')
        Pbetap = P.matrix*(betap);
        Lxi = Lxi + rho*sum(abs(Up(:))) + c(1)*sum(abs(betap)) + c(2)*P.Lasso_fz(Pbetap) + 0.5*norm(wp-y)^2;
    elseif strcmp(para.vec_regularizer,'slope')
        Lxi = Lxi + rho*sum(abs(Up(:))) + c_lambda'*sort(abs(betap),'descend') + 0.5*norm(wp-y)^2;
    elseif strcmp(para.vec_regularizer,'exclusive lasso')
        Lxi = Lxi + rho*sum(abs(Up(:))) + lambda1*xgroupnorm(betap,group_info) + 0.5*norm(wp-y)^2;
    end
    if printlevel
        fprintf('\n ------------------------------------- \n');
        fprintf('\n alp = %4.3f, LQ = %11.10e, LQ0 = %11.10e',alp,Lxi,Lxi0);
        fprintf('\n galp = %4.3f, g0 = %4.3f',galp,g0);
        fprintf('\n ------------------------------------- \n');
    end
    if (iter==1)
        gLB = g0; gUB = galp;
        if (sign(gLB)*sign(gUB) > 0)
            if (printlevel); fprintf('|'); end
            TtXtxi = TtXtxi0+alp*TtXtdxi;
            Ztxi = Ztxi0+alp*Ztdxi;
            return;
        end
    end
    if ((abs(galp) < c2*abs(g0))) && (Lxi-Lxi0-c1*alp*g0 > -1e-12/max(1,abs(Lxi0)))
        if (stepop==1) || ((stepop == 2) && (abs(galp) < tol))
            if (printlevel); fprintf(':'); end
            TtXtxi = TtXtxi0+alp*TtXtdxi;
            Ztxi = Ztxi0+alp*Ztdxi;
            return
        end
    end
    if (sign(galp)*sign(gUB) < 0)
        LB = alp; gLB = galp;
    elseif (sign(galp)*sign(gLB) < 0)
        UB = alp; gUB = galp;
    end
end
if iter == maxit
    TtXtxi = TtXtxi0+alp*TtXtdxi;
    Ztxi = Ztxi0+alp*Ztdxi;
end
if (printlevel); fprintf('m'); end
end