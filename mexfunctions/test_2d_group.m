% y = X*B + Z*gamma
% the 2d shape signal experiment.
% input dataset:
%       M - concatenated matrix data (input for FISTA (matrix_sparsereg))
%       Xinput - concatenate matrix data (input for PPDNA (ppdna_linear_nuclear_lasso))
%       Z - concatenate vector data
%       B - true signal (64x64)
%       gamma - true gamma
%       rho - regularize coefficient for nuclear norm
%       lambda - regularize coefficient for lasso (0 by default) 
% 
% One may use K-fold Cross Validation to choose rho and lambda. (see "cross validation" section below)


clear all;
rng default

%%
n = 300;    % sample size
p = 1000;     % dimenstion of gamma
% s = 0.3;    % non-sparsity level of gamma
% s_gamma = 0.01;    % non-sparsity level of gamma
% grpNUM=1000*s_gamma;

grpNUM=20;
% grpNUM=5;

iters=10;

pics=["square";"triangle";"circle";"heart"];
% pics=["heart"];

%%
for pic = 1:length(pics)

%% generate dataset
pic_name=pics(pic);
file_pic=append(pic_name,'.png');
shape = imread(file_pic);
shape=abs(shape-1);
shape = array_resize(shape,[32,32]); % 32-by-32
B = zeros(2*size(shape));
B((size(B,1)/4):(size(B,1)/4)+size(shape,1)-1, ...
    (size(B,2)/4):(size(B,2)/4)+size(shape,2)-1) = shape;
[p1,p2] = size(B);

gamma=zeros(p,1);

probs = 1 + 0.3 * sign(randn(grpNUM,1)) .* rand(grpNUM,1); 
probs = probs / sum(probs); 
probs = cumsum(probs);

for i = 1:grpNUM
    if i == 1
        tmp = round(probs(1)*p); tmp = max(tmp,1);
        ind(1,1) = 1; ind(2,1) = tmp; ind(3,1) = sqrt(tmp);
        gamma(1:10)=1*reshape([ones(1,5);-ones(1,5)],[10,1]);
%         gamma(1:6)=5*reshape([ones(1,3);-ones(1,2),0],[6,1]);
    else
        ind(1,i) = ind(2,i-1) + 1;
        ind(2,i) = max(round(probs(i)*p),ind(1,i));
        ind(3,i) = sqrt(ind(2,i)-ind(1,i));
        if i<=10
            gamma(ind(1,i):(ind(1,i)+9))=1*reshape([ones(1,5);-ones(1,5)],[10,1]);
%             gamma(ind(1,i):(ind(1,i)+5))=5*reshape([ones(1,3);-ones(1,2),0],[6,1]);
        end
    end
end


X=randn(n,p1*p2);
Z = randn(n,p);

mu = Z*gamma + X*B(:);
sigma = 1;  % noise level
y = mu + sigma*randn(n,1);

m = p1;
q = p2;

Xinput = X;
Zinput = Z;

%% tuning parameter

n_test = 2500;
X_test = randn(n_test,p1*p2);
Z_test = randn(n_test,p);

mu_test = Z_test*gamma + X_test*B(:);
sigma = 1;  % noise level
y_test = mu_test + sigma*randn(n_test,1);


Xinput_test = X_test;
Zinput_test = Z_test;


%% vecmat
l1_vecmat=[0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9]; 

l2_vecmat=[0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9];


options_vecmat.stoptol = 1e-4;
options_vecmat.printyes = 0;
options_vecmat.maxiter = 4000;
options_vecmat.printminoryes=0;
options_vecmat.printsub=0;

rmse_vecmat=zeros(length(l1_vecmat),length(l2_vecmat));
para_vecmat.vec_regularizer = 'lasso';
count=1;
fprintf('\nvecmat|  progress  |   rmse    |    \x03b1_1    |    \x03b1_2    |    \x03c1     |    \x03bb     \n')
for i = 1:length(l1_vecmat)
    for j = 1:length(l2_vecmat)
    para_vecmat.rho = max(abs(X'*y)) * l1_vecmat(i);
    para_vecmat.lambda = max(abs(Z'*y)) * l2_vecmat(j);
    [~,B_test,gamma_test] = admm_linear_vecmat_vec(Xinput,Zinput,y,m,q,p,para_vecmat,options_vecmat);

    y_hat=X_test*B_test(:)+Z_test*gamma_test;

    rmse_vecmat(i,j)=norm(y_test-y_hat)/sqrt(length(y_test));
    fprintf('      |   %2.d/%2.d    | %2.3e | %3.3e | %3.3e | %3.3e| %3.3e\n',count,length(l1_vecmat)*length(l2_vecmat),rmse_vecmat(i,j),l1_vecmat(i),l2_vecmat(j), para_vecmat.rho  ,para_vecmat.lambda);
    count=count+1;
    end
end

tmp = min(min(rmse_vecmat)); 
[ii_vecmat,jj_vecmat] = find(rmse_vecmat==tmp);
ii_vecmat=ii_vecmat(1);
jj_vecmat=jj_vecmat(1);

para_vecmat.rho = max(abs(X'*y)) * l1_vecmat(ii_vecmat);
para_vecmat.lambda = max(abs(Z'*y)) * l2_vecmat(jj_vecmat);

options_vecmat.stoptol = 1e-6;
options_vecmat.printyes = 1;
options_vecmat.maxiter = 5000;
% [obj_vecmat,B_vecmat,gamma_vecmat,s_vecmat,w_vecmat,U_vecmat,beta_vecmat,xi_vecmat,info_vecmat,runhist_vecmat] = admm_linear_vecmat_vec(Xinput,Zinput,y,m,q,p,para_vecmat,options_vecmat);

% subplot(1,5,1);
% imagesc(-double(B_vecmat))
% colormap(gray);
% axis equal;
% axis tight;
%% lasso
l1_lasso=[0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9];
l2_lasso=[0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9];


options_lasso.stoptol = 1e-6;
options_lasso.runphaseI = 0;
options_lasso.printyes = 0;
options_lasso.printminoryes=0;
options_lasso.printsub=0;

rmse_lasso=zeros(length(l1_lasso),length(l2_lasso));
para_lasso.vec_regularizer = 'lasso';
count=1;
fprintf('\nlasso|  progress  |   rmse    |    \x03b1_1    |    \x03b1_2    |    \x03c1     |    \x03bb     \n')
for i = 1:length(l1_lasso)
    for j = 1:length(l2_lasso)
    para_lasso.rho = max(abs(X'*y)) * l1_lasso(i);
    para_lasso.lambda = max(abs(Z'*y)) * l2_lasso(j);
    [~,B_test,gamma_test] = ppdna_linear_nuclear_vec(Xinput,Zinput,y,m,q,p,para_lasso,options_lasso);

    y_hat=X_test*B_test(:)+Z_test*gamma_test;

    rmse_lasso(i,j)=norm(y_test-y_hat)/sqrt(length(y_test));
    fprintf('     |   %2.d/%2.d    | %2.3e | %3.3e | %3.3e | %3.3e| %3.3e\n',count,length(l1_lasso)*length(l2_lasso),rmse_lasso(i,j),l1_lasso(i),l2_lasso(j), para_lasso.rho  ,para_lasso.lambda);
    count=count+1;
    end
end

tmp = min(min(rmse_lasso)); 
[ii_lasso,jj_lasso] = find(rmse_lasso==tmp);
ii_lasso=ii_lasso(1);
jj_lasso=jj_lasso(1);

para_lasso.rho = max(abs(X'*y)) * l1_lasso(ii_lasso);
para_lasso.lambda = max(abs(Z'*y)) * l2_lasso(jj_lasso);
options_lasso.printyes = 1;

% [obj_lasso,B_lasso,gamma_lasso,s_lasso,w_lasso,U_lasso,beta_lasso,xi_lasso,info_lasso,runhist_lasso] = ppdna_linear_nuclear_vec(Xinput,Zinput,y,m,q,p,para_lasso,options_lasso);
    
%% fused lasso
l1_fl=[0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9];
l2_fl=[0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9];

options_fl.stoptol = 1e-6;
options_fl.runphaseI = 0;
options_fl.printyes = 0;
options_fl.printminoryes=0;
options_fl.printsub = 0;

rmse_fl=zeros(length(l1_fl),length(l2_fl));
para_fl.vec_regularizer = 'fused lasso';
[Bmap0,BTmap0] = FLBmap(p);
Binput.Bmap = Bmap0;
Binput.BTmap = BTmap0;
para_fl.Binput = Binput;
count=1;
fprintf('\nfl|  progress  |   rmse    |    \x03b1_1    |    \x03b1_2    |    \x03c1     |    \x03bb     \n')
for i = 1:length(l1_fl)
    for j = 1:length(l2_fl)
        para_fl.rho = max(abs(X'*y)) * l1_fl(i);
        para_fl.lambda1 = max(abs(Z'*y)) * l2_fl(j);
        para_fl.lambda2 = para_fl.lambda1;
        [~,B_test,gamma_test] = ppdna_linear_nuclear_vec(Xinput,Zinput,y,m,q,p,para_fl,options_fl);
        y_hat=X_test*B_test(:)+Z_test*gamma_test;
        rmse_fl(i,j)=norm(y_test-y_hat)/sqrt(length(y_test));
        fprintf('  |   %2.d/%2.d    | %2.3e | %3.3e | %3.3e | %3.3e| %3.3e\n',count,length(l1_fl)*length(l2_fl),rmse_fl(i,j),l1_fl(i),l2_fl(j), para_fl.rho  ,para_fl.lambda1);
        count=count+1;
    end
end
tmp = min(min(rmse_fl)); 
[ii_fl,jj_fl] = find(rmse_fl==tmp);
ii_fl=ii_fl(1);
jj_fl=jj_fl(1);

para_fl.rho = max(abs(X'*y)) * l1_fl(ii_fl);
para_fl.lambda1 = max(abs(Z'*y)) * l2_fl(jj_fl);
para_fl.lambda2 = para_fl.lambda1;
options_fl.printyes = 1;
% [obj_fl,B_fl,gamma_fl,s_fl,w_fl,U_fl,beta_fl,xi_fl,info_fl,runhist_fl] = ppdna_linear_nuclear_vec(Xinput,Zinput,y,m,q,p,para_fl,options_fl);

%% sparse group lasso
l1_sgl=[0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9];
l2_sgl=[0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9];


options_sgl.stoptol = 1e-6;
options_sgl.runphaseI = 0;
options_sgl.printyes = 0;
options_sgl.printsub = 0;
options_sgl.printminoryes=0;

rmse_sgl=zeros(length(l1_sgl),length(l2_sgl));
para_sgl.vec_regularizer = 'sparse group lasso';

G=1:p;
para_sgl.G=G;
para_sgl.ind = ind;

count=1;
fprintf('\nsgl|  progress  |   rmse    |    \x03b1_1    |    \x03b1_2    |    \x03c1     |    \x03bb     \n')
for i = 1:length(l1_sgl)
    for j = 1:length(l2_sgl)
        para_sgl.rho = max(abs(X'*y)) * l1_sgl(i);
        para_sgl.c_lambda = max(abs(Z'*y)) * l2_sgl(j) * [1,1];
        [~,B_test,gamma_test] = ppdna_linear_nuclear_vec(Xinput,Zinput,y,m,q,p,para_sgl,options_sgl);
        y_hat=X_test*B_test(:)+Z_test*gamma_test;
        rmse_sgl(i,j)=norm(y_test-y_hat)/sqrt(length(y_test));
        fprintf('   |   %2.d/%2.d    | %2.3e | %3.3e | %3.3e | %3.3e| %3.3e\n',count,length(l1_sgl)*length(l2_sgl),rmse_sgl(i,j),l1_sgl(i),l2_sgl(j), para_sgl.rho  ,para_sgl.c_lambda(1));
        count=count+1;
    end
end
tmp = min(min(rmse_sgl)); 
[ii_sgl,jj_sgl] = find(rmse_sgl==tmp);
ii_sgl=ii_sgl(1);
jj_sgl=jj_sgl(1);

para_sgl.rho = max(abs(X'*y)) * l1_sgl(ii_sgl);
para_sgl.c_lambda = max(abs(Z'*y)) * l2_sgl(jj_sgl) * [1,1];

options_sgl.printyes = 1;

% [obj_sgl,B_sgl,gamma_sgl,s_sgl,w_sgl,U_sgl,beta_sgl,xi_sgl,info_sgl,runhist_sgl] = ppdna_linear_nuclear_vec(Xinput,Zinput,y,m,q,p,para_sgl,options_sgl);


%%

rmse_y_sgl=zeros(iters,1);
rmse_y_vecmat=zeros(iters,1);
rmse_y_lasso=zeros(iters,1);
rmse_y_fl=zeros(iters,1);

rmse_B_sgl=zeros(iters,1);
rmse_B_vecmat=zeros(iters,1);
rmse_B_lasso=zeros(iters,1);
rmse_B_fl=zeros(iters,1);

rmse_gamma_sgl=zeros(iters,1);
rmse_gamma_vecmat=zeros(iters,1);
rmse_gamma_lasso=zeros(iters,1);
rmse_gamma_fl=zeros(iters,1);

% save('20240116_group.mat');

rng default
for iter = 1:iters
    
X=randn(n,p1*p2);
Z = randn(n,p);

gamma=zeros(p,1);

probs = 1 + 0.3 * sign(randn(grpNUM,1)) .* rand(grpNUM,1); 
probs = probs / sum(probs); 
probs = cumsum(probs);

for i = 1:grpNUM
    if i == 1
        tmp = round(probs(1)*p); tmp = max(tmp,1);
        ind(1,1) = 1; ind(2,1) = tmp; ind(3,1) = sqrt(tmp);
        gamma(1:10)=1*reshape([ones(1,5);-ones(1,5)],[10,1]);
%         gamma(1:6)=5*reshape([ones(1,3);-ones(1,2),0],[6,1]);
    else
        ind(1,i) = ind(2,i-1) + 1;
        ind(2,i) = max(round(probs(i)*p),ind(1,i));
        ind(3,i) = sqrt(ind(2,i)-ind(1,i));
        if i<=10
            gamma(ind(1,i):(ind(1,i)+9))=1*reshape([ones(1,5);-ones(1,5)],[10,1]);
%             gamma(ind(1,i):(ind(1,i)+5))=5*reshape([ones(1,3);-ones(1,2),0],[6,1]);
        end
    end
end

G=1:p;
para_sgl.ind=ind;
para_sgl.G=G;

mu = Z*gamma + X*B(:);
sigma = 1;  % noise level
y = mu + sigma*randn(n,1);

m = p1;
q = p2;

Xinput = X;
Zinput = Z;

options_vecmat.stoptol = 1e-6;
[obj_vecmat,B_vecmat,gamma_vecmat,s_vecmat,w_vecmat,U_vecmat,beta_vecmat,xi_vecmat,info_vecmat,runhist_vecmat] = admm_linear_vecmat_vec(Xinput,Zinput,y,m,q,p,para_vecmat,options_vecmat);
[obj_lasso,B_lasso,gamma_lasso,s_lasso,w_lasso,U_lasso,beta_lasso,xi_lasso,info_lasso,runhist_lasso] = ppdna_linear_nuclear_vec(Xinput,Zinput,y,m,q,p,para_lasso,options_lasso);
[obj_fl,B_fl,gamma_fl,s_fl,w_fl,U_fl,beta_fl,xi_fl,info_fl,runhist_fl] = ppdna_linear_nuclear_vec(Xinput,Zinput,y,m,q,p,para_fl,options_fl);
[obj_sgl,B_sgl,gamma_sgl,s_sgl,w_sgl,U_sgl,beta_sgl,xi_sgl,info_sgl,runhist_sgl] = ppdna_linear_nuclear_vec(Xinput,Zinput,y,m,q,p,para_sgl,options_sgl);



y_lasso=X*B_lasso(:)+Z*gamma_lasso;
y_vecmat=X*B_vecmat(:)+Z*gamma_vecmat;
y_fl=X*B_fl(:)+Z*gamma_fl;
y_sgl=X*B_sgl(:)+Z*gamma_sgl;

rmse_y_sgl(iter)=norm(y_sgl(:)-y(:))/(sqrt(n));
rmse_y_vecmat(iter)=norm(y_vecmat(:)-y(:))/(sqrt(n));
rmse_y_lasso(iter)=norm(y_lasso(:)-y(:))/(sqrt(n));
rmse_y_fl(iter)=norm(y_fl(:)-y(:))/(sqrt(n));

rmse_B_sgl(iter)=norm(B_sgl(:)-B(:))/(sqrt(m*q));
rmse_B_vecmat(iter)=norm(B_vecmat(:)-B(:))/(sqrt(m*q));
rmse_B_lasso(iter)=norm(B_lasso(:)-B(:))/(sqrt(m*q));
rmse_B_fl(iter)=norm(B_fl(:)-B(:))/(sqrt(m*q));

rmse_gamma_sgl(iter)=norm(gamma_sgl(:)-gamma(:))/(sqrt(p));
rmse_gamma_vecmat(iter)=norm(gamma_vecmat(:)-gamma(:))/(sqrt(p));
rmse_gamma_lasso(iter)=norm(gamma_lasso(:)-gamma(:))/(sqrt(p));
rmse_gamma_fl(iter)=norm(gamma_fl(:)-gamma(:))/(sqrt(p));

end

temp.gamma=gamma;
temp.gamma_lasso=gamma_lasso;
temp.gamma_vecmat=gamma_vecmat;
temp.gamma_fl=gamma_fl;
temp.gamma_sgl=gamma_sgl;

temp.B=B;
temp.B_lasso=B_lasso;
temp.B_vecmat=B_vecmat;
temp.B_fl=B_fl;
temp.B_sgl=B_sgl;


temp.iijj_vecmat=[ii_vecmat,jj_vecmat];
temp.iijj_lasso=[ii_lasso,jj_lasso];
temp.iijj_fl=[ii_fl,jj_fl];
temp.iijj_sgl=[ii_sgl,jj_sgl];

temp.para_vecmat=para_vecmat;
temp.para_lasso=para_lasso;
temp.para_fl=para_fl;
temp.para_sgl=para_sgl;

temp.rmse_y_sgl=rmse_y_sgl;
temp.rmse_y_vecmat=rmse_y_vecmat;
temp.rmse_y_lasso=rmse_y_lasso;
temp.rmse_y_fl=rmse_y_fl;

temp.rmse_B_sgl=rmse_B_sgl;
temp.rmse_B_vecmat=rmse_B_vecmat;
temp.rmse_B_lasso=rmse_B_lasso;
temp.rmse_B_fl=rmse_B_fl;

temp.rmse_gamma_sgl=rmse_gamma_sgl;
temp.rmse_gamma_vecmat=rmse_gamma_vecmat;
temp.rmse_gamma_lasso=rmse_gamma_lasso;
temp.rmse_gamma_fl=rmse_gamma_fl;


file_name=append('/Users/howlpendragon/Desktop/20240115/20240121_2d_group_',pic_name,'.mat');
save(file_name,'temp');
end

%% draw
% set(gcf,'position',[0,100,300,300])
% 
% % a=subplot(2,5,1);
% imagesc(-double(B))
% colormap(gray);
% title('True B');
% axis equal;
% axis tight;
% saveas(gcf,'group_true_B.jpg')
% 
% % subplot(2,5,2);
% imagesc(-double(B_vecmat))
% colormap(gray);
% title('VECMAT B');
% axis equal;
% axis tight;
% saveas(gcf,'group_B_vecmat.jpg')
% 
% 
% % subplot(2,5,3);
% imagesc(-double(B_lasso))
% colormap(gray);
% title('LASSO B');
% axis equal;
% axis tight;
% saveas(gcf,'group_B_lasso.jpg')
% 
% % subplot(2,5,4);
% imagesc(-double(B_fl))
% colormap(gray);
% title('FL B');
% axis equal;
% axis tight;
% saveas(gcf,'group_B_fl.jpg')
% 
% % subplot(2,5,5);
% imagesc(-double(B_sgl))
% colormap(gray);
% title('SGL B');
% axis equal;
% axis tight;
% saveas(gcf,'group_B_sgl.jpg')
% 
% %%
% 
% set(gcf,'position',[0,100,50,500])
% % subplot(2,5,6);
% imagesc(-double(gamma))
% colormap(gray);
% title('True \gamma');
% axis tight;
% saveas(gcf,'group_true_gamma.jpg')
% 
% % subplot(2,5,7);
% imagesc(-double(gamma_vecmat))
% colormap(gray);
% title('VECMAT \gamma');
% axis tight;
% saveas(gcf,'group_vecmat_gamma.jpg')
% 
% % subplot(2,5,8);
% imagesc(-double(gamma_lasso))
% colormap(gray);
% title('LASSO \gamma');
% axis tight;
% saveas(gcf,'group_lasso_gamma.jpg')
% 
% % subplot(2,5,9);
% imagesc(-double(gamma_fl))
% colormap(gray);
% title('FL \gamma');
% % ax = gca;
% % ax.FontSize = 12;
% % ax.TickDir = 'in';
% % ax.TickLength = [0.2,0.1];
% % set(gcf,'position',[0,100,3000,1000])
% % axis equal;
% axis tight;
% saveas(gcf,'group_fl_gamma.jpg')
% 
% % subplot(2,5,10);
% imagesc(-double(gamma_sgl))
% colormap(gray);
% title('SGL \gamma');
% axis tight;
% saveas(gcf,'group_sgl_gamma.jpg')
