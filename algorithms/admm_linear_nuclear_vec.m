function [obj,B,gamma,s,w,U,beta,xi,info,runhist] = admm_linear_nuclear_vec(Xinput,Zinput,y,m,q,p,para,options,B0,gamma0,xi0,U0,beta0)
maxiter = 20000;
sigma   = 1;
paragamma   = 1.618;
stoptol = 1e-6;
printyes = 1;
printminoryes = 1;
sig_fix = 0;
phase2 = 0;
Asolver = 'direct';
Obj=-Inf;
objTol=0;

if isfield(options,'sigma'); sigma = options.sigma; end
if isfield(options,'gamma'); paragamma = options.gamma; end
if isfield(options,'stoptol'); stoptol = options.stoptol; end
if isfield(options,'printyes'); printyes = options.printyes; end
if isfield(options,'maxiter');  maxiter = options.maxiter; end
if isfield(options,'printminoryes'); printminoryes = options.printminoryes; end
if isfield(options,'sig_fix'); sig_fix = options.sig_fix; end
if isfield(options,'phase2'); phase2 = options.phase2; end
if isfield(options,'Asolver'); Asolver = options.Asolver; end

if isfield(options,'Obj'); Obj = options.Obj; end
if isfield(options,'objTol'); objTol = options.objTol; end
%% Xinput and Zinput
tstart = clock;
tstart_cpu = cputime;
n = length(y); 
mq = m*q;
if m < q
    flag = 1;
elseif m == q
    flag = 2;
else
    flag = 3;
end
if (n<=4000) && (mq<=50000)
    if isstruct(Xinput) 
        if isfield(Xinput,'X'); X0 = Xinput.X; end
        if isfield(Xinput,'Xmap'); Xmap0 = Xinput.Xmap; end
        if isfield(Xinput,'Xtmap'); Xtmap0 = Xinput.Xtmap; end
        XXt0 = X0*X0';
        XXtmap0 = @(x) XXt0*x;
    else
        X0 = Xinput;
        Xmap0 = @(x) X0*x;
        Xtmap0 = @(y) X0'*y;
        XXt0 = X0*X0';
        XXtmap0 = @(x) XXt0*x;
    end
    XXt = XXt0;
    if isstruct(Zinput) 
        if isfield(Zinput,'Z'); Z0 = Zinput.Z; end
        if isfield(Zinput,'Zmap'); Zmap0 = Zinput.Zmap; end
        if isfield(Zinput,'Ztmap'); Ztmap0 = Zinput.Ztmap; end
        ZZt0 = Z0*Z0';
        ZZtmap0 = @(x) ZZt0*x;
    else
        Z0 = Zinput;
        Zmap0 = @(x) Z0*x;
        Ztmap0 = @(y) Z0'*y;
        ZZt0 = Z0*Z0';
        ZZtmap0 = @(x) ZZt0*x;
    end
    ZZt = ZZt0;
else
    if isstruct(Xinput)
        if isfield(Xinput,'Xmap'); Xmap0 = Xinput.Xmap; end
        if isfield(Xinput,'Xtmap'); Xtmap0 = Xinput.Xtmap; end
        XXtmap0 = @(x) Xmap0(Xtmap0(x));
    else
        X0 = Xinput;
        Xmap0 = @(x) X0*x;
        Xtmap0 = @(y) X0'*y;
        XXtmap0 = @(x) Xmap0(Xtmap0(x));
    end
    if isstruct(Zinput)
        if isfield(Zinput,'Zmap'); Zmap0 = Zinput.Zmap; end
        if isfield(Zinput,'Ztmap'); Ztmap0 = Zinput.Ztmap; end
        ZZtmap0 = @(x) Zmap0(Ztmap0(x));
    else
        Z0 = Zinput;
        Zmap0 = @(x) Z0*x;
        Ztmap0 = @(y) Z0'*y;
        ZZtmap0 = @(x) Zmap0(Ztmap0(x));
    end
    if strcmp(Asolver,'direct')
        Asolver = 'cg';
    end
end
Xmap  = Xmap0;
Xtmap = Xtmap0;
XXtmap = XXtmap0;
Zmap  = Zmap0;
Ztmap = Ztmap0;
ZZtmap = ZZtmap0;
Tmap = @(x) x(:);
Ttmap = @(x) reshape(x,m,q);
mapop.XXtmap = XXtmap;
mapop.ZZtmap = ZZtmap;
%% parameter
rho = para.rho;
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
%% initiallization
normy = 1 + norm(y);
if ~exist('B0','var') || ~exist('gamma0','var') || ~exist('xi0','var') || ~exist('U0','var') || ~exist('beta0','var')
    B = zeros(m,q);
    gamma = zeros(p,1);
    xi = zeros(n,1);
    U = zeros(m,q);
    beta = zeros(p,1);
else
    B = B0; gamma = gamma0; xi = xi0; U = U0; beta = beta0;
end
XTB = Xmap(Tmap(B));
Zgamma = Zmap(gamma);
s = XTB + Zgamma;
w = xi;
if printyes
    fprintf('\n *******************************************************');
    fprintf('******************************************');
    if strcmp(Asolver,'cg')
        fprintf('\n \t\t   admm (cg)  for solving matrix regression with linear loss and paragamma = %6.3f', paragamma);
        fprintf('\n ******************************************************');
        fprintf('*******************************************\n');
        if printminoryes
            fprintf('\n problem size: n = %3.0f, m = %3.0f, q = %3.0f',n, m, q);
            if strcmp(para.vec_regularizer,'lasso')
                fprintf('\n regularizers: nuclear norm with rho = %g, lasso with lambda = %g', rho, lambda);
            elseif strcmp(para.vec_regularizer,'fused lasso')
                fprintf('\n regularizers: nuclear norm with rho = %g, fused lasso with lambda1 = %g, lambda2 = %g', rho, lambda1, lambda2);
            elseif strcmp(para.vec_regularizer,'clustered lasso')
                fprintf('\n regularizers: nuclear norm with rho = %g, clustered lasso with lambda1 = %g, lambda2 = %g', rho, lambda1, lambda2);
            elseif strcmp(para.vec_regularizer,'sparse group lasso')
                fprintf('\n regularizers: nuclear norm with rho = %g, sparse group lasso with c1 = %g, c2 = %g', rho, c(1), c(2));
            elseif strcmp(para.vec_regularizer,'slope')
                fprintf('\n regularizers: nuclear norm with rho = %g, slope with weights', rho);
            elseif strcmp(para.vec_regularizer,'exclusive lasso')
                fprintf('\n regularizers: nuclear norm with rho = %g, exclusive lasso with lambda1 = %g', rho, lambda1);
            end
            fprintf('\n ---------------------------------------------------');
        end
        fprintf('\n  iter| [pinfeas  dinfeas  complement]   eta   |      pobj          dobj         relgap |');
        fprintf(' time |  sigma  | [  cg  ]');
    else
        if strcmp(Asolver,'direct')
            fprintf('\n \t\t   admm (direct) for solving matrix regression with linear loss and paragamma = %6.3f', paragamma);
        end
        fprintf('\n ******************************************************');
        fprintf('*******************************************\n');
        if printminoryes
            fprintf('\n problem size: n = %3.0f, m = %3.0f, q = %3.0f',n, m, q);
            if strcmp(para.vec_regularizer,'lasso')
                fprintf('\n regularizers: nuclear norm with rho = %g, lasso with lambda = %g', rho, lambda);
            elseif strcmp(para.vec_regularizer,'fused lasso')
                fprintf('\n regularizers: nuclear norm with rho = %g, fused lasso with lambda1 = %g, lambda2 = %g', rho, lambda1, lambda2);
            elseif strcmp(para.vec_regularizer,'clustered lasso')
                fprintf('\n regularizers: nuclear norm with rho = %g, clustered lasso with lambda1 = %g, lambda2 = %g', rho, lambda1, lambda2);
            elseif strcmp(para.vec_regularizer,'sparse group lasso')
                fprintf('\n regularizers: nuclear norm with rho = %g, sparse group lasso with c1 = %g, c2 = %g', rho, c(1), c(2));
            elseif strcmp(para.vec_regularizer,'slope')
                fprintf('\n regularizers: nuclear norm with rho = %g, slope with weights', rho);
            elseif strcmp(para.vec_regularizer,'exclusive lasso')
                fprintf('\n regularizers: nuclear norm with rho = %g, exclusive lasso with lambda1 = %g', rho, lambda1);
            end
            fprintf('\n ---------------------------------------------------');
        end
        fprintf('\n  iter| [pinfeas  dinfeas  complement]   eta   |      pobj          dobj         relgap |');
        fprintf(' time |  sigma  |');
    end
end
%%
const = 100;
parxi.const = const;
TtXtxi = Ttmap(Xtmap(xi));
Ztxi = Ztmap(xi);
if strcmp(Asolver,'cg')
    XTTtXtxi = Xmap(Tmap(TtXtxi));
    ZZtxi = Zmap(Ztxi);
    IpsigAAtxi = const*xi + XTTtXtxi + ZZtxi;
elseif strcmp(Asolver,'direct')
    Lxi = mychol(const*eye(n) + XXt + ZZt,n);
end
%%
Rd1 = TtXtxi + U;
Rd2 = Ztxi + beta;
Rd3 = w - xi;
Rp2 = XTB + Zgamma - s;
normRp2 = norm(Rp2);
normRd1 = norm(Rd1,'fro');
normRd2 = norm(Rd2);
normRd3 = norm(Rd3);
normU = norm(U,'fro');
normbeta = norm(beta);
normxi = norm(xi);
primfeas = normRp2/(1+norm(s));
dualfeas = max([normRd1/(1+normU),normRd2/(1+normbeta),normRd3/(1+normxi)]);
maxfeas = max(primfeas,dualfeas);
%% main Loop
breakyes = 0;
prim_win = 0;
dual_win = 0;
msg = [];
for iter = 1:maxiter
    Bold = B; gammaold = gamma; sold = s;
    %% compute xi
    rhsxi = Xmap(Tmap(B/sigma-U))+Zmap(gamma/sigma-beta)+w*const-s/sigma;
    if strcmp(Asolver,'cg')
        parxi.tol = max(0.9*stoptol,min(1/iter^1.1,0.9*maxfeas));
        parxi.sigma = sigma;
        [xi,IpsigAAtxi,resnrmxi,solve_okxi] = psqmry('matvecxi',mapop,rhsxi,parxi,xi,IpsigAAtxi);
    elseif strcmp(Asolver,'direct')
        xi = mylinsysolve(Lxi,rhsxi);
    end
    TtXtxi = Ttmap(Xtmap(xi));
    Ztxi = Ztmap(xi);
    %% compute w, U, beta   
    winput = (const*sigma)*xi + s;
    wp = (winput+const*sigma*y)/(1+const*sigma);
    w = (winput-wp)/(const*sigma);

    Uinput = B - sigma*TtXtxi;
    Up = prox_nuclear_norm(Uinput,sigma*rho,flag);
    U = (Uinput -Up)/sigma;

    betainput = gamma - sigma*Ztxi;
    if strcmp(para.vec_regularizer,'lasso')
        betap = proxL1(betainput,sigma*lambda);
    elseif strcmp(para.vec_regularizer,'fused lasso')
        betap = proxFL(Binput,betainput,sigma*lambda1,sigma*lambda2);
    elseif strcmp(para.vec_regularizer,'clustered lasso')
        betap = proxClu(betainput, sigma*lambda1, sigma*lambda2);
    elseif strcmp(para.vec_regularizer,'sparse group lasso')
        betap = proxsgl(betainput,sigma*c,P);
    elseif strcmp(para.vec_regularizer,'slope')
        betap = proxSortedL1(betainput,sigma*c_lambda); 
    elseif strcmp(para.vec_regularizer,'exclusive lasso')
        betap = prox_exclusive(betainput,p,sigma*lambda1,group_info);
    end
    beta = (betainput-betap)/sigma;
    %% update mutilpliers B, gamma, s
    Rd1 = TtXtxi + U;
    B = Bold - paragamma*sigma*Rd1;
    Rd2 = Ztxi + beta;
    gamma = gammaold - paragamma*sigma*Rd2;
    Rd3 = w - xi;
    s = sold - paragamma*(const*sigma)*Rd3;
    %%
    Rp1 = s-y-w;
    XTB = Xmap(Tmap(B));
    Zgamma = Zmap(gamma);
    Rp2 = XTB + Zgamma - s;
    normRp2 = norm(Rp2);
    normRd1 = norm(Rd1,'fro');
    normRd2 = norm(Rd2);
    normRd3 = norm(Rd3);
    normU = norm(U,'fro');
    normbeta = norm(beta);
    normxi = norm(xi);
    primfeas = max(normRp2/(1+norm(s)),norm(Rp1)/normy);   
    dualfeas = max([normRd1/(1+normU),normRd2/(1+normbeta),normRd3/(1+normxi)]);
    maxfeas = max(primfeas,dualfeas);
    proxpBpU = prox_nuclear_norm(B+U,rho,flag);
    if strcmp(para.vec_regularizer,'lasso')
        proxqgammapbeta = proxL1(gamma+beta,lambda);
    elseif strcmp(para.vec_regularizer,'fused lasso')
        proxqgammapbeta = proxFL(Binput,gamma+beta,lambda1,lambda2);
    elseif strcmp(para.vec_regularizer,'clustered lasso')
        proxqgammapbeta = proxClu(gamma+beta,lambda1,lambda2);
    elseif strcmp(para.vec_regularizer,'sparse group lasso')
        proxqgammapbeta = proxsgl(gamma+beta,c,P);
    elseif strcmp(para.vec_regularizer,'slope')
        proxqgammapbeta = proxSortedL1(gamma+beta,c_lambda);
    elseif strcmp(para.vec_regularizer,'exclusive lasso')
        proxqgammapbeta = prox_exclusive(gamma+beta,p,lambda1,group_info);
    end
    complement = max(norm(proxpBpU-B,'fro')/(1+norm(B,'fro')),norm(proxqgammapbeta-gamma)/(1+norm(gamma)));
    %%
    if strcmp(para.vec_regularizer,'lasso')
        primobj = 0.5*norm(XTB+Zgamma-y)^2 + rho*norm(svd(B),1) + lambda*sum(abs(gamma));
    elseif strcmp(para.vec_regularizer,'fused lasso')
        Bgamma = Binput.Bmap(gamma);
        primobj = 0.5*norm(XTB+Zgamma-y)^2 + rho*norm(svd(B),1) + lambda1*sum(abs(gamma)) + lambda2*sum(abs(Bgamma)); 
    elseif strcmp(para.vec_regularizer,'clustered lasso')
        gammasort = sort(gamma,'descend');
        primobj = 0.5*norm(XTB+Zgamma-y)^2 + rho*norm(svd(B),1) + lambda1*sum(abs(gamma)) + lambda2*(wvec*gammasort); 
    elseif strcmp(para.vec_regularizer,'sparse group lasso')
        Pgamma = P.matrix*(gamma);
        primobj = 0.5*norm(XTB+Zgamma-y)^2 + rho*norm(svd(B),1) + c(1)*sum(abs(gamma)) + c(2)*P.Lasso_fz(Pgamma);  
    elseif strcmp(para.vec_regularizer,'slope')
        primobj = 0.5*norm(XTB+Zgamma-y)^2 + rho*norm(svd(B),1) + c_lambda'*sort(abs(gamma),'descend');
    elseif strcmp(para.vec_regularizer,'exclusive lasso')
        primobj = 0.5*norm(XTB+Zgamma-y)^2 + rho*norm(svd(B),1) + lambda1*xgroupnorm(gamma,group_info);
    end
    dualobj = -y'*xi - 0.5*norm(xi)^2;
    relgap = (primobj-dualobj)/( 1+abs(primobj)+abs(dualobj));
    ttime = etime(clock,tstart);
    %%-------------------------------------------------------
    %% record history
    runhist.dualfeas(iter) = dualfeas;
    runhist.primfeas(iter) = primfeas;
    runhist.maxfeas(iter) = maxfeas;
    runhist.sigma(iter) = sigma;
    if strcmp(Asolver,'cg'); runhist.psqmrxiiter(iter) = length(resnrmxi) - 1; end
    %% check for termination
    eta = max([maxfeas,complement]);
    if ((abs(primobj-Obj)/(1+abs(Obj))<objTol) || primobj<=Obj)
        breakyes = 1;
        msg = 'converge to objective ';
    end
    if eta < stoptol
        breakyes = 1;
        msg = 'converged';
    end
    if etime(clock, tstart) > 3600
        breakyes = 777;
        msg = 'time out';
    end
    
    runhist.primobj(iter)   = primobj;
    runhist.dualobj(iter)   = dualobj;
    runhist.time(iter)      = ttime;
    runhist.obj(iter)=primobj;
    runhist.time1(iter)=ttime;
    runhist.relgap(iter)    = relgap;
    runhist.eta(iter) = eta;
    %% print results
    if (iter <= 200)
        print_iter = 20;
    elseif (iter <= 2000)
        print_iter = 100;
    else
        print_iter = 200;
    end
    if (rem(iter,print_iter)==1 || iter==maxiter) || (breakyes)
        if (printyes)
            fprintf('\n %5.0d| [%3.2e %3.2e %3.2e]  %3.2e | %- 10.7e %- 10.7e %- 3.2e|',...
                iter,primfeas,dualfeas,complement,eta,primobj,dualobj,relgap);
            fprintf(' %5.1f| %3.2e|',ttime, sigma);
            fprintf('%2.3f|',paragamma);
            if strcmp(Asolver,'cg')
                fprintf('[%3.0d %3.0d]', length(resnrmxi)-1, solve_okxi);
            end
        end
    end
    if (breakyes > 0)
        fprintf('\n  breakyes = %3.1f, %s',breakyes,msg);
        break;
    end
    %% update sigma
    feasratio = primfeas/dualfeas;
    runhist.feasratio(iter) = feasratio;
    
    if (feasratio < 1/5)
        prim_win = prim_win+1;
    elseif (feasratio > 5)
        dual_win = dual_win+1;
    end
    
    sigma_update_iter = sigma_fun(iter);
    sigmascale = 1.1; %1.25;
    if (~sig_fix) && (rem(iter,sigma_update_iter)==0)
        sigmamax = 1e6; sigmamin = 1e-8;
        if (iter <= 1*2500) 
           if (prim_win > max(1,1.2*dual_win))
                prim_win = 0;
                sigma = min(sigmamax,sigma*sigmascale);
            elseif (dual_win > max(1,1.2*prim_win))
                dual_win = 0;
                sigma = max(sigmamin,sigma/sigmascale);
            end
        else
            feasratiosub = runhist.feasratio(max(1,iter-19):iter);
            meanfeasratiosub = mean(feasratiosub);
            if meanfeasratiosub < 0.1 || meanfeasratiosub > 1/0.1
                sigmascale = 1.4;
            elseif meanfeasratiosub < 0.2 || meanfeasratiosub > 1/0.2
                sigmascale = 1.35;
            elseif meanfeasratiosub < 0.3 || meanfeasratiosub > 1/0.3
                sigmascale = 1.32;
            elseif meanfeasratiosub < 0.4 || meanfeasratiosub > 1/0.4
                sigmascale = 1.28;
            elseif meanfeasratiosub < 0.5 || meanfeasratiosub > 1/0.5
                sigmascale = 1.26;
            end
            primidx = find(feasratiosub <= 1);
            dualidx = find(feasratiosub >  1);
            if (length(primidx) >= 12)
                sigma = min(sigmamax,sigma*sigmascale);
            end
            if (length(dualidx) >= 12)
                sigma = max(sigmamin,sigma/sigmascale);
            end
        end
    end
    const_fix = 1;
    if (~const_fix) && (rem(iter,100)==0) 
        tmp1 = max(normRd1/(1+normU),normRd2/(1+normbeta));
        tmp2 = normRd3/(1+normxi);
        if tmp1 > 10*tmp2
            const = const/5;
            parxi.const = const;
        end
    end
end
%% recover orignal variables
if (iter == maxiter)
    msg = ' maximum iteration reached';
end

obj = [primobj, dualobj];
ttime = etime(clock,tstart);
ttime_cpu = cputime - tstart_cpu;
if strcmp(Asolver,'cg'); ttCG = sum(runhist.psqmrxiiter); end
[hh,mm,ss] = changetime(ttime);
info.m = m;
info.q = q;
info.p = p;
info.n = n;
info.relgap = relgap;
if strcmp(Asolver,'cg'); info.ttCG = ttCG; end
info.iter = iter;
info.time = ttime;
info.time_cpu = ttime_cpu;
info.eta = eta;
info.obj = obj;

if phase2 == 1
    info.XTB = XTB;
    info.Zgamma = Zgamma;
    info.TtXtxi = TtXtxi;
    info.Ztxi = Ztxi;
end

if (printminoryes)
    if ~isempty(msg); fprintf('\n %s',msg); end
    fprintf('\n--------------------------------------------------------------');
    fprintf('------------------');
    fprintf('\n  number iter = %2.0d',iter);
    fprintf('\n  time = %3.2f,  (%d:%d:%d)',ttime,hh,mm,ss);
    if iter >= 1; fprintf('\n  time per iter = %5.4f',ttime/iter); end
    fprintf('\n  cputime = %3.2f', ttime_cpu);
    fprintf('\n     primobj = %10.9e, dualobj = %10.9e, relgap = %3.2e',primobj,dualobj, relgap);
    if (iter >= 1) && (strcmp(Asolver,'cg'))
        fprintf('\n  Total CG number = %3.0d, CG per iter = %3.1f', ttCG, ttCG/iter);
    end
    fprintf('\n  eta = %3.2e', eta);
    fprintf('\n--------------------------------------------------------------');
    fprintf('------------------\n');
end
end
%%**********************************************************************
function sigma_update_iter = sigma_fun(iter)
if (iter < 30)
    sigma_update_iter = 3;
elseif (iter < 60)
    sigma_update_iter = 6;
elseif (iter < 120)
    sigma_update_iter = 12;
elseif (iter < 250)
    sigma_update_iter = 25;
elseif (iter < 500)
    sigma_update_iter = 50;
elseif (iter < inf)  %% better than (iter < 1000)
    sigma_update_iter = 100;
end
end
%%**********************************************************************

% To change the format of time
function [h,m,s] = changetime(t)
t = round(t);
h = floor(t/3600);
m = floor(rem(t,3600)/60);
s = rem(rem(t,60),60);
end
