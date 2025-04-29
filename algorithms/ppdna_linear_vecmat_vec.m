function [obj,B,gamma,s,w,U,beta,xi,info,runhist] = ppdna_linear_vecmat_vec(Xinput,Zinput,y,m,q,p,para,options)
maxiter = 50;
stoptol = 1e-6;
precond = 0;
printyes = 1; % print the iterations
printminoryes = 1; % print the results
printsub = 1; % print the sub iterations
Obj=-Inf;
objTol=0;

runphaseI = 0;
phaseI_stoptol = 1e-3;
phaseI_maxiter = 100;
startfun = @admm_linear_vecmat_vec;
admm.iter = 0;
admm.time = 0;
admm.timecpu = 0;

if isfield(options,'maxiter');  maxiter  = options.maxiter; end
if isfield(options,'stoptol');  stoptol  = options.stoptol; end
if isfield(options,'printyes'); printyes = options.printyes; end
if isfield(options,'printminoryes'); printminoryes = options.printminoryes; end
if isfield(options,'precond'); precond = options.precond; end
if isfield(options,'printsub'); printsub = options.printsub; end
if isfield(options,'runphaseI'); runphaseI = options.runphaseI; end
if isfield(options,'phaseI_stoptol'); phaseI_stoptol = options.phaseI_stoptol; end
if isfield(options,'phaseI_maxiter'); phaseI_maxiter = options.phaseI_maxiter; end
if isfield(options,'Obj'); Obj = options.Obj; end
if isfield(options,'objTol'); objTol = options.objTol; end


if isfield(options,'B0'); B0 = options.B0; end
if isfield(options,'gamma0'); gamma0 = options.gamma0; end
if isfield(options,'xi0'); xi0 = options.xi0; end
if isfield(options,'U0'); U0 = options.U0; end
if isfield(options,'beta0'); beta0 = options.beta0; end

%% Xinput and Zinput
tstart = clock;
tstart_cpu = cputime;
n = length(y);
if m < q
    flag = 1;
elseif m == q
    flag = 2;
else
    flag = 3;
end
if isstruct(Xinput)
    if isfield(Xinput,'X'); X = Xinput.X; end
    if isfield(Xinput,'Xmap'); Xmap = Xinput.Xmap; end
    if isfield(Xinput,'Xtmap'); Xtmap = Xinput.Xtmap; end
else
    X = Xinput;
    Xmap = @(x) X*x;
    Xtmap = @(y) X'*y;
end
if isstruct(Zinput)
    if isfield(Zinput,'Z'); Z = Xinput.Z; end
    if isfield(Zinput,'Zmap'); Zmap = Zinput.Zmap; end
    if isfield(Zinput,'Ztmap'); Ztmap = Zinput.Ztmap; end
else
    Z = Zinput;
    Zmap =@(x) Z*x;
    Ztmap = @(y) Z'*y;
end
%%
if exist('X','var')
    diagXXt = sum(X.*X,2);
else
    diagXXt = diagXXt_fun(Xinput,n);
end
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
%%
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
%% phase I
admm.op.stoptol =  phaseI_stoptol;
admm.op.maxiter = phaseI_maxiter;
admm.op.sigma =  1;
admm.op.phase2 = 1;
admm.op.Asolver = 'cg';
admm.op.printminoryes = 0;

admm.op.Obj=Obj;
admm.op.objTol=objTol;
if admm.op.maxiter > 0 && runphaseI
    fprintf('\n Phase I: ADMM cg (dual approach)');
    [obj,B,gamma,s,w,U,beta,xi,info_admm,runhist_admm] = startfun(Xinput,Zinput,y,m,q,p,para,admm.op,B,gamma,xi,U,beta);
    XTB = info_admm.XTB;
    Zgamma = info_admm.Zgamma;
    TtXtxi = info_admm.TtXtxi;
    Ztxi = info_admm.Ztxi;
    admm.iter = admm.iter + info_admm.iter;
    admm.time = admm.time + info_admm.time;
    admm.timecpu = admm.timecpu + info_admm.time_cpu;
    if (info_admm.eta < stoptol)
        fprintf('\n Problem solved in Phase I \n');
        info = info_admm;
        info.m = m;
        info.q = q;
        info.p = p;
        info.n = n;
        info.relgap = info_admm.relgap;
        info.iter = 0;
        info.time = admm.time;
        info.time_cpu = admm.timecpu;
        info.admmtime = admm.time;
        info.admmtime_cpu = admm.timecpu;
        info.admmiter = admm.iter;
        info.eta = info_admm.eta;
        info.obj = obj;
        runhist = runhist_admm;
        return;
    end
else
    XTB = Xmap(B(:));
    Zgamma = Zmap(gamma);
    s = XTB + Zgamma;
    if strcmp(para.vec_regularizer,'lasso')
        obj(1) = 0.5*norm(s-y)^2 + rho*sum(abs(B(:))) + lambda*sum(abs(gamma));
    elseif strcmp(para.vec_regularizer,'fused lasso')
        Bgamma = Binput.Bmap(gamma);
        obj(1) = 0.5*norm(s-y)^2 + rho*sum(abs(B(:))) + lambda1*sum(abs(gamma)) + lambda2*sum(abs(Bgamma));
    elseif strcmp(para.vec_regularizer,'clustered lasso')
        gammasort = sort(gamma,'descend');
        obj(1) = 0.5*norm(s-y)^2 + rho*sum(abs(B(:))) + lambda1*sum(abs(gamma)) + lambda2*(wvec*gammasort);
    elseif strcmp(para.vec_regularizer,'sparse group lasso')
        Pgamma = P.matrix*(gamma);
        obj(1) = 0.5*norm(s-y)^2 + rho*sum(abs(B(:))) + c(1)*sum(abs(gamma)) + c(2)*P.Lasso_fz(Pgamma);
    elseif strcmp(para.vec_regularizer,'slope')
        obj(1) = 0.5*norm(s-y)^2 + rho*sum(abs(B(:))) + c_lambda'*sort(abs(gamma),'descend');
    elseif strcmp(para.vec_regularizer,'exclusive lasso')
        obj(1) = 0.5*norm(s-y)^2 + rho*sum(abs(B(:))) + lambda1*xgroupnorm(gamma,group_info);
    end
    obj(2) = -y'*xi - 0.5*norm(xi)^2;
    TtXtxi = reshape(Xtmap(xi),m,q);
    Ztxi = Ztmap(xi);
end
%%
Ainput_nal.flag = flag;
Ainput_nal.X = X;
Ainput_nal.Xmap = Xmap;
Ainput_nal.Xtmap = Xtmap;
Ainput_nal.diagXXt = diagXXt;
Ainput_nal.Z = Z;
Ainput_nal.Zmap = Zmap;
Ainput_nal.Ztmap = Ztmap;
sigma = 1;
if isfield(options,'sigma'); sigma = options.sigma; end
%%
Rp1 = s-y-xi;
Rd1 = TtXtxi + U;
Rd2 = Ztxi + beta;
primfeas = norm(Rp1)/normy;
dualfeas = max(norm(Rd1,'fro')/(1+norm(U,'fro')),norm(Rd2)/(1+norm(beta)));
maxfeas = max(primfeas,dualfeas);
proxpBpU = proxL1(B+U,rho);
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
eta = max([maxfeas,complement]);
relgap = (obj(1) - obj(2))/(1+obj(1)+obj(2));
if printyes
    fprintf('\n *******************************************************');
    fprintf('******************************************');
    fprintf('\n \t\t   Phase II: PPDNA for solving matrix regression with linear loss');
    fprintf('\n ******************************************************');
    fprintf('*******************************************\n');
    if printminoryes
        fprintf('\n problem size: n = %3.0f, m = %3.0f, q = %3.0f',n, m, q);
        if strcmp(para.vec_regularizer,'lasso')
            fprintf('\n regularizers: vecL1 norm with rho = %g, lasso with lambda = %g', rho, lambda);
        elseif strcmp(para.vec_regularizer,'fused lasso')
            fprintf('\n regularizers: vecL1 norm with rho = %g, fused lasso with lambda1 = %g, lambda2 = %g', rho, lambda1, lambda2);
        elseif strcmp(para.vec_regularizer,'clustered lasso')
            fprintf('\n regularizers: vecL1 norm with rho = %g, clustered lasso with lambda1 = %g, lambda2 = %g', rho, lambda1, lambda2);
        elseif strcmp(para.vec_regularizer,'sparse group lasso')
            fprintf('\n regularizers: vecL1 norm with rho = %g, sparse group lasso with c1 = %g, c2 = %g', rho, c(1), c(2));
        elseif strcmp(para.vec_regularizer,'slope')
            fprintf('\n regularizers: vecL1 norm with rho = %g, slope with weights', rho);
        elseif strcmp(para.vec_regularizer,'exclusive lasso')
            fprintf('\n regularizers: vecL1 norm with rho = %g, exclusive lasso with lambda1 = %g', rho, lambda1);
        end
        fprintf('\n ---------------------------------------------------');
    end
    fprintf('\n  iter| [pinfeas  dinfeas  complement]   eta   |      pobj          dobj         relgap |');
    fprintf(' time |  sigma  |');
    fprintf('\n*****************************************************');
    fprintf('**************************************************************');
    fprintf('\n %5.0d| [%3.2e %3.2e %3.2e]  %3.2e | %- 10.7e %- 10.7e %- 3.2e|',...
        0,primfeas,dualfeas,complement,eta,obj(1),obj(2),relgap);
    fprintf(' %5.1f| %3.2e|',etime(clock,tstart), sigma);
end
%% ssncg
tau = 1;%0.1;
parNCG.tolconst = 0.5;
parNCG.precond = precond;
parNCG.m = m;
parNCG.q = q;
parNCG.p = p;
maxitersub = 10;
breakyes = 0;
prim_win = 0;
dual_win = 0;
ssncgop.tol = stoptol;
ssncgop.printsub = printsub;

tstart_phaseIIonly=clock;
primobj=obj(1);
dualobj=obj(2);
    
    runhist.obj(1)   = primobj;
    runhist.time_phaseIIonly(1)=0;
for iter = 1:maxiter
    parNCG.sigma = sigma;
    parNCG.sigdtau = sigma/tau;
    parNCG.tau = tau;
    if dualfeas < 1e-5
        maxitersub = max(maxitersub,30);
    elseif dualfeas < 1e-3
        maxitersub = max(maxitersub,30);
    elseif dualfeas < 1e-1
        maxitersub = max(maxitersub,20);
    end
    ssncgop.maxitersub = maxitersub;
    [U,beta,w,TtXtxi,Ztxi,xi,parNCG,~,info_NCG] = ppdna_sub_linear_vecmat_vec(y,Ainput_nal,B,gamma,XTB,Zgamma,TtXtxi,Ztxi,xi,para,parNCG,ssncgop);
    if info_NCG.breakyes < 0
        parNCG.tolconst = max(parNCG.tolconst/1.06,1e-3);
    end
    B = info_NCG.Up;
    XTB = info_NCG.XTUp;
    gamma = info_NCG.betap;
    Zgamma = info_NCG.Zbetap;
    s = info_NCG.wp;
    %%
    Rp1 = s-y-w;
    Rp2 = XTB + Zgamma - s;
    Rd1 = TtXtxi + U;
    Rd2 = Ztxi + beta;
    Rd3 = w - xi;
    primfeas = max(norm(Rp2)/(1+norm(s)),norm(Rp1)/normy);
    dualfeas = max([norm(Rd1,'fro')/(1+norm(U,'fro')),norm(Rd2)/(1+norm(beta)),norm(Rd3)/(1+norm(xi))]);
    maxfeas = max(primfeas,dualfeas);
    proxpBpU = proxL1(B+U,rho);
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
    runhist.dualfeas(iter) = dualfeas;
    runhist.primfeas(iter) = primfeas;
    runhist.maxfeas(iter)  = maxfeas;
    runhist.complement(iter) = complement;
    runhist.sigma(iter) = sigma;
    %% check for termination
    if strcmp(para.vec_regularizer,'lasso')
        primobj = 0.5*norm(XTB+Zgamma-y)^2 + rho*sum(abs(B(:))) + lambda*sum(abs(gamma));
    elseif strcmp(para.vec_regularizer,'fused lasso')
        Bgamma = Binput.Bmap(gamma);
        primobj = 0.5*norm(XTB+Zgamma-y)^2 + rho*sum(abs(B(:))) + lambda1*sum(abs(gamma)) + lambda2*sum(abs(Bgamma)); 
    elseif strcmp(para.vec_regularizer,'clustered lasso')
        gammasort = sort(gamma,'descend');
        primobj = 0.5*norm(XTB+Zgamma-y)^2 + rho*sum(abs(B(:))) + lambda1*sum(abs(gamma)) + lambda2*(wvec*gammasort); 
    elseif strcmp(para.vec_regularizer,'sparse group lasso')
        Pgamma = P.matrix*(gamma);
        primobj = 0.5*norm(XTB+Zgamma-y)^2 + rho*sum(abs(B(:))) + c(1)*sum(abs(gamma)) + c(2)*P.Lasso_fz(Pgamma);  
    elseif strcmp(para.vec_regularizer,'slope')
        primobj = 0.5*norm(XTB+Zgamma-y)^2 + rho*sum(abs(B(:))) + c_lambda'*sort(abs(gamma),'descend');
    elseif strcmp(para.vec_regularizer,'exclusive lasso')
        primobj = 0.5*norm(XTB+Zgamma-y)^2 + rho*sum(abs(B(:))) + lambda1*xgroupnorm(gamma,group_info);
    end
    dualobj = -y'*xi - 0.5*norm(xi)^2;
    relgap = (primobj-dualobj)/( 1+abs(primobj)+abs(dualobj));
    eta = max([maxfeas,complement]);
    ttime = etime(clock,tstart);
    ttime_phaseIIonly=etime(clock,tstart_phaseIIonly);
    if (abs(primobj-Obj)/(1+abs(Obj))<objTol) || primobj<=Obj
        breakyes = 1;
        msg = 'converge to objective ';
    end
    if eta < stoptol
        breakyes = 1;
        msg = 'converged';
    end
    if etime(clock, tstart) > 3*3600
        breakyes = 777;
        msg = 'time out';
    end
    if (printyes)
        fprintf('\n %5.0d| [%3.2e %3.2e %3.2e]  %3.2e | %- 10.7e %- 10.7e %- 3.2e|',...
            iter,primfeas,dualfeas,complement,eta,primobj,dualobj,relgap);
        fprintf(' %5.1f| %3.2e|',ttime, sigma);
    end
    runhist.primobj(iter)   = primobj;
    runhist.dualobj(iter)   = dualobj;

    runhist.obj(iter+1)   = primobj;
    runhist.time_phaseIIonly(iter+1)=ttime_phaseIIonly;

    runhist.time(iter)      = ttime;
    runhist.relgap(iter)    = relgap;
    runhist.eta(iter) = eta;
    if (breakyes > 0)
        if printyes; fprintf('\n  breakyes = %3.1f, %s',breakyes,msg);  end
        break;
    end
    %%
    if (primfeas < dualfeas)
        prim_win = prim_win+1;
    else
        dual_win = dual_win+1;
    end
    if (iter < 10)
        sigma_update_iter = 1;
    elseif iter < 20
        sigma_update_iter = 2;
    elseif iter < 200
        sigma_update_iter = 2;
    elseif iter < 500
        sigma_update_iter = 3;
    end
    sigmascale = 5;
    sigmamax = 1e5;
    update_sigma_options = 1;
    if update_sigma_options == 1
        if (rem(iter,sigma_update_iter)==0)
            sigmamin = 1e-4;
            if prim_win > max(1,1.2*dual_win)
                prim_win = 0;
                sigma = min(sigmamax,sigma*sigmascale);
            elseif dual_win > max(1,3*prim_win)
                dual_win = 0;
                sigma = max(sigmamin,2*sigma/sigmascale);
            end
        end
    end
end
%% recover orignal variables
if (iter == maxiter)
    msg = ' maximum iteration reached';
end
ttime = etime(clock,tstart);
obj = [primobj, dualobj];
ttime_cpu = cputime - tstart_cpu;
info.m = m;
info.q = q;
info.p = p;
info.n = n;
info.relgap = relgap;
info.iter = iter;
info.time = ttime;
[hh,mm,ss] = changetime(ttime);
info.time_cpu = ttime_cpu;
info.admmtime = admm.time;
info.admmtime_cpu = admm.timecpu;
info.admmiter = admm.iter;
info.eta = eta;
info.obj = obj;

if (printminoryes)
    if ~isempty(msg); fprintf('\n %s',msg); end
    fprintf('\n--------------------------------------------------------------');
    fprintf('------------------');
    fprintf('\n  admm iter = %3.1d, admm time = %3.1f', admm.iter, admm.time);
    fprintf('\n  number iter = %2.0d',iter);
    fprintf('\n  time = %3.2f,  (%d:%d:%d)',ttime,hh,mm,ss);
    fprintf('\n  time per iter = %5.4f',ttime/iter);
    fprintf('\n  cputime = %3.2f', ttime_cpu);
    fprintf('\n     primobj = %10.9e, dualobj = %10.9e, relgap = %3.2e',primobj,dualobj,relgap);
    fprintf('\n  eta = %3.2e', eta);
    fprintf('\n--------------------------------------------------------------');
    fprintf('------------------\n');
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


