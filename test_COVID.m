%% Run file for matrix regression on COVID-19 data set
%% @2024 by Meixia Lin, Ziyang Zeng, Yangjing Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
rng('default');
HOME = pwd;
addpath(genpath(HOME));
%% load data
load('COVID.mat');
n = size(X_mat,3);
m = size(X_mat,1);
q = size(X_mat,2);
p = size(Z,2);
%% parameter setting
alpha1 = 1e-3;
alpha2 = 0;
rho = alpha1 * max(svd(reshape(X'*y,[m,q])));
lambda = alpha2 * max(abs(Z'*y));
%% Benchmark obj* (solution with high accuracy by PPDNA)
options.printyes = 1;
options.printsub = 0;
options.printminoryes = 0;
options.maxiter = 100;
options.stoptol = 1e-10;
para.rho = rho;
para.lambda = 0;
para.vec_regularizer ='lasso';
[TargetOBJ,B,gamma] = ppdna_linear_nuclear_vec(X,Z,y,m,q,p,para,options);
r = rank(B);
sgamma = sum(gamma~=0)/p;
options.Obj = TargetOBJ(1);
options.objTol = 1e-10;
options.runphaseI = 0;
options.stoptol = -Inf;
%% FISTA
fistayes = 0;
if fistayes && alpha2 == 0
    tensor_X = tensor(X_mat);
    [~,~,info_fista,runhist_fista] = matrix_sparsereg(Z,tensor_X,y,rho,'normal','MaxIter',300000,'TolFun',1e-10,'target_obj',TargetOBJ(1));
end
%% PPDNA
[~,~,~,~,~,~,~,~,info_ppdna,runhist_ppdna] = ppdna_linear_nuclear_vec(X,Z,y,m,q,p,para,options);
%% ADMM
options.maxiter = 300000;
[~,~,~,~,~,~,~,~,info_admm,runhist_admm] = admm_linear_nuclear_vec(X,Z,y,m,q,p,para,options);
%%
if alpha2 == 0
    fprintf('\n \n alpha1=%.3e, alpha2=%.3e, rank=%d, nonsparsity(gamma)=%.3e\n',alpha1,alpha2,r,sgamma);
    if fistayes
        fprintf('       |   ADMM    |   PPDNA    |    FISTA  \n')
        fprintf('Time(s)|  %s  |  %s   |   %s    |\n', changetime(runhist_admm.time(end)), changetime(runhist_ppdna.time(end)) ,changetime(runhist_fista.time(end)))
        fprintf('Iter   |  %- 9.d|  %- 10.d|   %- 10.d\n', info_admm.iter, info_ppdna.iter,info_fista.iterations)
        fprintf('relobj | %- 10.3e|  %-9.3e |   %-9.3e \n',(info_admm.obj(1)-options.Obj)/(1+options.Obj),...
            (info_ppdna.obj(1)-options.Obj)/(1+options.Obj),(info_fista.obj(1)-options.Obj)/(1+options.Obj))
    else
        fprintf('       |   ADMM    |   PPDNA    |\n')
        fprintf('Time(s)|  %s  |  %s   |\n', changetime(runhist_admm.time(end)), changetime(runhist_ppdna.time(end)))
        fprintf('Iter   |  %- 9.d|  %- 10.d|\n', info_admm.iter, info_ppdna.iter)
        fprintf('relobj | %- 10.3e|  %-9.3e |\n',(info_admm.obj(1)-options.Obj)/(1+options.Obj),...
            (info_ppdna.obj(1)-options.Obj)/(1+options.Obj))
    end
else
    fprintf('\n \n alpha1=%.3e, alpha2=%.3e, rank=%d, nonsparsity(gamma)=%.3e\n',alpha1,alpha2,r,sgamma);
    fprintf('       |   ADMM    |   PPDNA    |\n')
    fprintf('Time(s)|  %s  |  %s   |\n', changetime(runhist_admm.time(end)), changetime(runhist_ppdna.time(end)))
    fprintf('Iter   |  %- 9.d|  %- 10.d|\n', info_admm.iter, info_ppdna.iter)
    fprintf('relobj | %- 10.3e|  %-9.3e |\n',(info_admm.obj(1)-options.Obj)/(1+options.Obj),...
        (info_ppdna.obj(1)-options.Obj)/(1+options.Obj))
end