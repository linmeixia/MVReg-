function [dxi,resnrm,solve_ok,par] = ppdna_Netwonsolve_linear_vecmat_vec(Ainput,rhs,par)
Z = Ainput.Z;
para = par.para;
if strcmp(para.vec_regularizer,'lasso')
    rr = par.info_beta;
    Zk = Z(:,rr);
    par.Zk = Zk;
    par.ZkZkt = par.Zk*par.Zk';
elseif strcmp(para.vec_regularizer,'fused lasso')
    rr1 = par.info_beta.rr1; %% for L1
    rr2 = par.info_beta.rr2; %% for fused
    [h,U] = mexFusedLassoJacobian(double(rr2));
    if ~isempty(U)
        Ut = U';
        PU = Ut(:,rr1)';
    else
        PU = [];
    end
    Ph = h(rr1);
    nzcolidx = find(sum(PU) > 0);
    numblk1 = length(nzcolidx);
    if (numblk1 > 0)
        PU1 = PU(:,nzcolidx);
    else
        PU1 = 0;
    end
    ZP = Z(:,rr1);
    par.ZP = ZP;
    par.PU1 = PU1;
    par.Ph  = Ph;
    par.numblk1 = numblk1;
elseif strcmp(para.vec_regularizer,'clustered lasso')
    rr1 = par.info_beta.rr1; %% for L1
    rr2 = par.info_beta.rr2; %% for Clustered
    [h,U] = mexFusedLassoJacobian(double(rr2));
    h(par.info_beta.idx) = h;
    U(par.info_beta.idx,:) = U;
    if ~isempty(U)
        Ut = U';
        PU = Ut(:,rr1)';
    else
        PU = [];
    end
    Ph = h(rr1);
    nzcolidx = find(sum(PU) > 0);
    numblk1 = length(nzcolidx);
    if (numblk1 > 0)
        PU1 = PU(:,nzcolidx);
    else
        PU1 = 0;
    end
    ZP = Z(:,rr1);
    par.ZP = ZP;
    par.PU1 = PU1;
    par.Ph  = Ph;
    par.numblk1 = numblk1;
elseif strcmp(para.vec_regularizer,'sparse group lasso')
    [D,~,id_yes] = matvecD(par.input,Z,par.c,par.P,par.sigma);
    par.D = D;
    par.id_yes = id_yes;
elseif strcmp(para.vec_regularizer,'slope')
    rr2 = par.info_beta.rr2;
    [h,U] = mexWK(double(rr2));
    v1idx = par.info_beta.idx(find(h>0));
    vecV1 = Z(:,v1idx);
    if ~isempty(U)
        idx_U = find(sum(U,2)>0); % nonzero row index set
        Unew  = U(idx_U,:); % dropping zeros rows from U
        iidex = find(par.info_beta.s<0);
        idx_A = par.info_beta.idx(idx_U);
        Zhat = Z(:,idx_A);
        [~,~,idA] = intersect(iidex,idx_A);
        Zhat(:,idA) = -Zhat(:,idA);
        vecV2 = Zhat*Unew;
    else
        vecV2 = [];
    end
    par.numblk1 = size(vecV2,2);
    par.vecV1 = vecV1;
    par.vecV2 = vecV2;
elseif strcmp(para.vec_regularizer,'exclusive lasso')
    ZP = Z(:,para.group_info.P);
    rr1 = ~par.info_beta.rr1;
    sl1 = par.sl1;
    D = par.info_beta.D;
    ZP1 = ZP(:,rr1);
    D1 = D(rr1);
    org_group = para.group_info.org_group;
    org_group = org_group(para.group_info.P);
    group = org_group(rr1);
    [s,~] = sort(group);
    [uniques,uidx] = unique(s);
    group_num_reduced = length(uidx);
    M_reduced = zeros(2,group_num_reduced);
    const_vec = zeros(group_num_reduced,1);
    for i = 1:group_num_reduced
        if i ~= group_num_reduced
            M_reduced(:,i) = [uidx(i);uidx(i+1)-1];
        else
            M_reduced(:,i) = [uidx(i);length(s)];
        end
        tmp = M_reduced(2,i)-M_reduced(1,i)+1;
        const_vec(i) = 2*sl1/(1+tmp*2*sl1);
    end
    par.group_num_reduced = group_num_reduced;
    if group_num_reduced == 1
        const = const_vec(1);
        ZP2 = ZP1*D1;
        par.ZP1 = ZP1;
        par.ZP2 = ZP2;
        par.const = const;
    else
        par.ZP1 = ZP1;
        par.D1 = D1;
        par.const_vec = const_vec;
        par.counts = hist(s,uniques);
        stmp = 1:group_num_reduced;
        stmp = repelem(stmp',par.counts);
        par.smat = ind2vec(stmp',group_num_reduced);
    end
end

X=Ainput.X;
rr = par.info_U;
Xk = X(:,rr);
par.Xk = Xk;
par.XkXkt = par.Xk*par.Xk';

%% iterative solver
if par.precond == 1
    diagAAt = Ainput.diagXXt + diag(par.ZkZkt);
    invdiagM = 1./full(par.sigma/(par.tau+par.sigma) + par.sigma*diagAAt);
    par.invdiagM = invdiagM;
end
[dxi,~,resnrm,solve_ok] = psqmry('mat_ppdna_linear_vecmat_vec',0,rhs,par);%numblk1
end
