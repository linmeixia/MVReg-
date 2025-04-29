function y = mat_ppdna_linear_nuclear_vec(xi,par,nzcol)
para = par.para;
Xmap = par.Xmap;
Xtmap = par.Xtmap;
m = par.m;
q = par.q;

U = par.info_U.U;
V1 = par.V1;
V2 = par.V2;
alpha1 = par.info_U.alpha1;
alpha2 = par.info_U.alpha2;
alpha3 = par.info_U.alpha3;

option = 1;
if par.flag == 1
    H = reshape(Xtmap(xi),m,q);
    H1 = U'*H*V1;
    H2 = U'*H*V2;
    a1 = sum(alpha1);
    a2 = sum(alpha2);
    a3 = m-a1-a2;
    if a1 == 0
        y = par.sigma/(par.tau+par.sigma)*xi;
    else
        if option == 1
            if a2 == 0
                if a3 == 0
                    tmp3 = (H1+H1')/2+par.w.*(H1-H1')/2;
                    tmp = U*tmp3*V1' + U(:,alpha1)*(par.mu.*H2(alpha1,:))*V2';
                else
                    hH111 = H1(alpha1,alpha1)/2;
                    hH113 = H1(alpha1,alpha3)/2;
                    hH131 = H1(alpha3,alpha1)/2;
                    tmp1 = [hH111,par.t13.*(hH113+hH131');zeros(a3,m)];
                    tmp2 = [par.w.*[hH111,(hH113-hH131')];zeros(a3,m)];
                    tmp3 = tmp1+tmp1'+tmp2-tmp2';
                    tmp = U*tmp3*V1' + U(:,alpha1)*(par.mu.*H2(alpha1,:))*V2';
                end
            else
                if a3 == 0
                    hH111 = H1(alpha1,alpha1)/2;
                    hH112 = H1(alpha1,alpha2)/2;
                    hH121 = H1(alpha2,alpha1)/2;
                    tmp1 = [hH111,hH112+hH121';zeros(a2,m)];
                    tmp2 = [par.w.*[hH111,(hH112-hH121')];zeros(a2,m)];
                    tmp3 = tmp1+tmp1'+tmp2-tmp2';
                    tmp = U*tmp3*V1' + U(:,alpha1)*(par.mu.*H2(alpha1,:))*V2';
                else
                    hH111 = H1(alpha1,alpha1)/2;
                    hH112 = H1(alpha1,alpha2)/2;
                    hH113 = H1(alpha1,alpha3)/2;
                    hH121 = H1(alpha2,alpha1)/2;
                    hH131 = H1(alpha3,alpha1)/2;
                    tmp1 = [hH111,hH112+hH121',par.t13.*(hH113+hH131');zeros(a2+a3,m)];
                    tmp2 = [par.w.*[hH111,(hH112-hH121'),(hH113-hH131')];zeros(a2+a3,m)];
                    tmp3 = tmp1+tmp1'+tmp2-tmp2';
                    tmp = U*tmp3*V1' + U(:,alpha1)*(par.mu.*H2(alpha1,:))*V2';
                end
            end
        else
            if a2 == 0
                if a3 == 0
                    tmp = U*(((H1+H1')/2 + par.w.*(H1-H1')/2)*V1'+(par.mu.*H2)*V2');
                else
                    Gaa = [ones(a1,a1),par.t13;par.t13',zeros(a3,a3)];
                    Gac = [par.w;par.w(alpha1,alpha3)',zeros(a3,a3)];
                    Gab = [repelem(par.mu, 1, q-m);zeros(a3,q-m)];
                    tmp = U*((Gaa.*(H1+H1')/2 + Gac.*(H1-H1')/2)*V1'+(Gab.*H2)*V2');
                end
            else
                if a3 == 0
                    Gaa = [ones(a1,m);ones(a2,a1),zeros(a2,a2)];
                    Gac = [par.w;par.w(alpha1,alpha2)',zeros(a2,a2)];
                    Gab = [repelem(par.mu, 1, q-m);zeros(a2,q-m)];
                    tmp = U*((Gaa.*(H1+H1')/2 + Gac.*(H1-H1')/2)*V1'+(Gab.*H2)*V2');
                else
                    Gaa = [ones(a1,a1+a2),par.t13;ones(a2,a1),zeros(a2,a2+a3);par.t13',zeros(a3,a2+a3)];
                    Gac = [par.w;par.w(alpha1,alpha2)',zeros(a2,a2+a3);par.w(alpha1,alpha3)',zeros(a3,a2+a3)];
                    Gab = [repelem(par.mu, 1, q-m);zeros(a2+a3,q-m)];
                    tmp = U*((Gaa.*(H1+H1')/2 + Gac.*(H1-H1')/2)*V1'+(Gab.*H2)*V2');
                end
            end
        end
        y = par.sigma/(par.tau+par.sigma)*xi + par.sigma*Xmap(tmp(:));
    end
elseif par.flag == 2
    H = reshape(Xtmap(xi),m,q);
    H1 = U'*H*V1;
    a1 = sum(alpha1);
    a2 = sum(alpha2);
    a3 = m-a1-a2;
    if a1 == 0
        y = par.sigma/(par.tau+par.sigma)*xi;
    else
        if option == 1
            if a2 == 0
                if a3 == 0
                    tmp3 = (H1+H1')/2+par.w.*(H1-H1')/2;
                    tmp = U*tmp3*V1';
                else
                    hH111 = H1(alpha1,alpha1)/2;
                    hH113 = H1(alpha1,alpha3)/2;
                    hH131 = H1(alpha3,alpha1)/2;
                    tmp1 = [hH111,par.t13.*(hH113+hH131');zeros(a3,m)];
                    tmp2 = [par.w.*[hH111,(hH113-hH131')];zeros(a3,m)];
                    tmp3 = tmp1+tmp1'+tmp2-tmp2';
                    tmp = U*tmp3*V1';
                end
            else
                if a3 == 0
                    hH111 = H1(alpha1,alpha1)/2;
                    hH112 = H1(alpha1,alpha2)/2;
                    hH121 = H1(alpha2,alpha1)/2;
                    tmp1 = [hH111,hH112+hH121';zeros(a2,m)];
                    tmp2 = [par.w.*[hH111,(hH112-hH121')];zeros(a2,m)];
                    tmp3 = tmp1+tmp1'+tmp2-tmp2';
                    tmp = U*tmp3*V1';
                else
                    hH111 = H1(alpha1,alpha1)/2;
                    hH112 = H1(alpha1,alpha2)/2;
                    hH113 = H1(alpha1,alpha3)/2;
                    hH121 = H1(alpha2,alpha1)/2;
                    hH131 = H1(alpha3,alpha1)/2;
                    tmp1 = [hH111,hH112+hH121',par.t13.*(hH113+hH131');zeros(a2+a3,m)];
                    tmp2 = [par.w.*[hH111,(hH112-hH121'),(hH113-hH131')];zeros(a2+a3,m)];
                    tmp3 = tmp1+tmp1'+tmp2-tmp2';
                    tmp = U*tmp3*V1';
                end
            end
        else
            if a2 == 0
                if a3 == 0
                    tmp = U*((H1+H1')/2 + par.w.*(H1-H1')/2)*V1';
                else
                    Gaa = [ones(a1,a1),par.t13;par.t13',zeros(a3,a3)];
                    Gac = [par.w;par.w(alpha1,alpha3)',zeros(a3,a3)];
                    tmp = U*(Gaa.*(H1+H1')/2 + Gac.*(H1-H1')/2)*V1';
                end
            else
                if a3 == 0
                    Gaa = [ones(a1,m);ones(a2,a1),zeros(a2,a2)];
                    Gac = [par.w;par.w(alpha1,alpha2)',zeros(a2,a2)];
                    tmp = U*(Gaa.*(H1+H1')/2 + Gac.*(H1-H1')/2)*V1';
                else
                    Gaa = [ones(a1,a1+a2),par.t13;ones(a2,a1),zeros(a2,a2+a3);par.t13',zeros(a3,a2+a3)];
                    Gac = [par.w;par.w(alpha1,alpha2)',zeros(a2,a2+a3);par.w(alpha1,alpha3)',zeros(a3,a2+a3)];
                    tmp = U*(Gaa.*(H1+H1')/2 + Gac.*(H1-H1')/2)*V1';
                end
            end
        end
        y = par.sigma/(par.tau+par.sigma)*xi + par.sigma*Xmap(tmp(:));
    end
else
    H = reshape(Xtmap(xi),m,q)';
    H1 = U'*H*V1;
    H2 = U'*H*V2;
    a1 = sum(alpha1);
    a2 = sum(alpha2);
    a3 = q-a1-a2;
    if a1 == 0
        y = par.sigma/(par.tau+par.sigma)*xi;
    else
        if option == 1
            if a2 == 0
                if a3 == 0
                    tmp3 = (H1+H1')/2+par.w.*(H1-H1')/2;
                    tmp = U*tmp3*V1' + U(:,alpha1)*(par.mu.*H2(alpha1,:))*V2';
                else
                    hH111 = H1(alpha1,alpha1)/2;
                    hH113 = H1(alpha1,alpha3)/2;
                    hH131 = H1(alpha3,alpha1)/2;
                    tmp1 = [hH111,par.t13.*(hH113+hH131');zeros(a3,q)];
                    tmp2 = [par.w.*[hH111,(hH113-hH131')];zeros(a3,q)];
                    tmp3 = tmp1+tmp1'+tmp2-tmp2';
                    tmp = U*tmp3*V1' + U(:,alpha1)*(par.mu.*H2(alpha1,:))*V2';
                end
            else
                if a3 == 0
                    hH111 = H1(alpha1,alpha1)/2;
                    hH112 = H1(alpha1,alpha2)/2;
                    hH121 = H1(alpha2,alpha1)/2;
                    tmp1 = [hH111,hH112+hH121';zeros(a2,q)];
                    tmp2 = [par.w.*[hH111,(hH112-hH121')];zeros(a2,q)];
                    tmp3 = tmp1+tmp1'+tmp2-tmp2';
                    tmp = U*tmp3*V1' + U(:,alpha1)*(par.mu.*H2(alpha1,:))*V2';
                else
                    hH111 = H1(alpha1,alpha1)/2;
                    hH112 = H1(alpha1,alpha2)/2;
                    hH113 = H1(alpha1,alpha3)/2;
                    hH121 = H1(alpha2,alpha1)/2;
                    hH131 = H1(alpha3,alpha1)/2;
                    tmp1 = [hH111,hH112+hH121',par.t13.*(hH113+hH131');zeros(a2+a3,q)];
                    tmp2 = [par.w.*[hH111,(hH112-hH121'),(hH113-hH131')];zeros(a2+a3,q)];
                    tmp3 = tmp1+tmp1'+tmp2-tmp2';
                    tmp = U*tmp3*V1' + U(:,alpha1)*(par.mu.*H2(alpha1,:))*V2';
                end
            end
        else
            if a2 == 0
                if a3 == 0
                    tmp = U*(((H1+H1')/2 + par.w.*(H1-H1')/2)*V1'+(par.mu.*H2)*V2');
                else
                    Gaa = [ones(a1,a1),par.t13;par.t13',zeros(a3,a3)];
                    Gac = [par.w;par.w(alpha1,alpha3)',zeros(a3,a3)];
                    Gab = [repelem(par.mu, 1, m-q);zeros(a3,m-q)];
                    tmp = U*((Gaa.*(H1+H1')/2 + Gac.*(H1-H1')/2)*V1'+(Gab.*H2)*V2');
                end
            else
                if a3 == 0
                    Gaa = [ones(a1,q);ones(a2,a1),zeros(a2,a2)];
                    Gac = [par.w;par.w(alpha1,alpha2)',zeros(a2,a2)];
                    Gab = [repelem(par.mu, 1, m-q);zeros(a2,m-q)];
                    tmp = U*((Gaa.*(H1+H1')/2 + Gac.*(H1-H1')/2)*V1'+(Gab.*H2)*V2');
                else
                    Gaa = [ones(a1,a1+a2),par.t13;ones(a2,a1),zeros(a2,a2+a3);par.t13',zeros(a3,a2+a3)];
                    Gac = [par.w;par.w(alpha1,alpha2)',zeros(a2,a2+a3);par.w(alpha1,alpha3)',zeros(a3,a2+a3)];
                    Gab = [repelem(par.mu, 1, m-q);zeros(a2+a3,m-q)];
                    tmp = U*((Gaa.*(H1+H1')/2 + Gac.*(H1-H1')/2)*V1'+(Gab.*H2)*V2');
                end
            end
        end
        tmp = tmp';
        y = par.sigma/(par.tau+par.sigma)*xi + par.sigma*Xmap(tmp(:));
    end
end
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