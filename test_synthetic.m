%% Run file for matrix regression on 2D and synthetic data set
%% @2024 by Meixia Lin, Ziyang Zeng, Yangjing Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
rng('default');
HOME = pwd;
addpath(genpath(HOME));
%% dataset size
n = 300;    % sample size
p = 1000;   % dimenstion of gamma
%% specify matrix and vector coefficients type
syntheticyes = 0;  % 0: synthetic B; else: 2D-shape B
s_gamma = 0.01;    % non-sparsity level of gamma
gamma_Scheme = 1;  % 1: S1(Sparsity); 2: S2(Local constancy); 3: S3(Group structure)
%% load matrix coefficient B
if syntheticyes % synthetic B
    s_B = 0.1; % non-sparsity level of B
    r = 4; % rank
    m = 50;
    q = 50;
    B1 = rand(m,r)<sqrt(1-(1-s_B)^(1/r));
    B2 = rand(q,r)<sqrt(1-(1-s_B)^(1/r));
    B = B1*B2';
    file_pic = "random";
else % 2D-shape B
    file_pic = "square"; %square, triangle, circle, heart
    load(file_pic)
    [m,q] = size(B);
end
%% generate vector coefficient gamma
gamma = zeros(p,1);
if gamma_Scheme == 1
    grpNUM = p*s_gamma;
    ind = zeros(3,grpNUM);
    probs = 1 + 0.3 * sign(randn(grpNUM,1)) .* rand(grpNUM,1); 
    probs = probs / sum(probs); 
    probs = cumsum(probs);
    for g = 1:grpNUM
        if g == 1
            tmp = round(probs(1)*p); tmp = max(tmp,1);
            ind(1,1) = 1; ind(2,1) = tmp; ind(3,1) = sqrt(tmp);
            idx=datasample(ind(1,g):ind(2,g),1);
            gamma(idx)=5;
        else
            ind(1,g) = ind(2,g-1) + 1;
            ind(2,g) = max(round(probs(g)*p),ind(1,g));
            ind(3,g) = sqrt(ind(2,g)-ind(1,g));
            idx=datasample(ind(1,g):ind(2,g),1);
            gamma(idx)=5;
        end
    end
elseif gamma_Scheme == 2
    grpNUM = p*s_gamma;
    ind = zeros(3,grpNUM);
    probs = 1 + 0.3 * sign(randn(grpNUM,1)) .* rand(grpNUM,1); 
    probs = probs / sum(probs); 
    probs = cumsum(probs);
    for i = 1:grpNUM
        if i == 1
            tmp = round(probs(1)*p); tmp = max(tmp,1);
            ind(1,1) = 1; ind(2,1) = tmp; ind(3,1) = sqrt(tmp);
            gamma(1:10)=ones(10,1);
        else
            ind(1,i) = ind(2,i-1) + 1;
            ind(2,i) = max(round(probs(i)*p),ind(1,i));
            ind(3,i) = sqrt(ind(2,i)-ind(1,i));
            gamma(ind(1,i):(ind(1,i)+9))=ones(10,1);
        end
    end
else
    grpNUM = 20;
    ind = zeros(3,grpNUM);
    probs = 1 + 0.3 * sign(randn(grpNUM,1)) .* rand(grpNUM,1); 
    probs = probs / sum(probs); 
    probs = cumsum(probs);
    for i = 1:grpNUM
        if i == 1
            tmp = round(probs(1)*p); tmp = max(tmp,1);
            ind(1,1) = 1; ind(2,1) = tmp; ind(3,1) = sqrt(tmp);
            gamma(1:10)=reshape([ones(1,5);-ones(1,5)],[10,1]);
        else
            ind(1,i) = ind(2,i-1) + 1;
            ind(2,i) = max(round(probs(i)*p),ind(1,i));
            ind(3,i) = sqrt(ind(2,i)-ind(1,i));
            if i<=10
                gamma(ind(1,i):(ind(1,i)+9))=reshape([ones(1,5);-ones(1,5)],[10,1]);
            end
        end
    end
end
G = 1:p;
%% generate training data
X = randn(n,m*q);
Z = randn(n,p);
mu = Z*gamma + X*B(:);
sigma = 1;  % noise level
y = mu + sigma*randn(n,1);
%% generate testing data
X_test = randn(n,m*q);
Z_test = randn(n,p);
mu = Z_test*gamma + X_test*B(:);
y_test = mu + sigma*randn(n,1);
%% 
options.stoptol = 1e-6;
options.runphaseI = 0;
options.printyes = 1;
options.printminoryes=0;
options.printsub = 0;
options.maxiter = 100;
%
rhomax_vec = max(abs(X'*y));
rhomax_mat = max(svd(reshape(X'*y,[m,q])));
lambdamax = max(abs(Z'*y));
%% VML: square loss + vectorized matrix Lasso regularizer + Lasso regularizer
para_VML.vec_regularizer = 'lasso';
para_VML.rho = 1e-3 * rhomax_vec;
para_VML.lambda = 1e-3 * lambdamax;
[obj_VML,B_VML,gamma_VML,~,~,~,~,~,info_VML,runhist_VML] = ppdna_linear_vecmat_vec(X,Z,y,m,q,p,para_VML,options);
%% NL: square loss + nuclear norm regularizer + Lasso regularizer
para_NL.vec_regularizer = 'lasso';
para_NL.rho = 1e-3 * rhomax_mat;
para_NL.lambda = 1e-3 * lambdamax;
[obj_NL,B_NL,gamma_NL,~,~,~,~,~,info_NL,runhist_NL] = ppdna_linear_nuclear_vec(X,Z,y,m,q,p,para_NL,options);
%% NFL: square loss + nuclear norm regularizer + Fused Lasso regularizer
para_NFL.vec_regularizer = 'fused lasso';
[Bmap0,BTmap0] = FLBmap(p);
Binput.Bmap = Bmap0;
Binput.BTmap = BTmap0;
para_NFL.Binput = Binput;
para_NFL.rho = 1e-3 * rhomax_mat;
para_NFL.lambda1 = 1e-3 * lambdamax;
para_NFL.lambda2 = 1e-3 * lambdamax;
[obj_NFL,B_NFL,gamma_NFL,~,~,~,~,~,info_NFL,runhist_NFL] = ppdna_linear_nuclear_vec(X,Z,y,m,q,p,para_NFL,options);
%% NSGL: square loss + nuclear norm regularizer + Sparse Group Lasso regularizer
para_NSGL.vec_regularizer = 'sparse group lasso';
para_NSGL.rho = 1e-3 * rhomax_mat;
para_NSGL.c_lambda = 1e-3 * lambdamax * [1,1];
para_NSGL.G = G;
para_NSGL.ind = ind;
[obj_NSGL,B_NSGL,gamma_NSGL,~,~,~,~,~,info_NSGL,runhist_NSGL] = ppdna_linear_nuclear_vec(X,Z,y,m,q,p,para_NSGL,options);
%% Calculate testing RMSE and Error
y_pred_NL=X_test*B_NL(:)+Z_test*gamma_NL;
y_pred_VML=X_test*B_VML(:)+Z_test*gamma_VML;
y_pred_NFL=X_test*B_NFL(:)+Z_test*gamma_NFL;
y_pred_NSGL=X_test*B_NSGL(:)+Z_test*gamma_NSGL;
%
rmse_NL = norm(y_pred_NL-y_test)/sqrt(length(y_test));
rmse_VML = norm(y_pred_VML-y_test)/sqrt(length(y_test));
rmse_NFL = norm(y_pred_NFL-y_test)/sqrt(length(y_test));
rmse_NSGL = norm(y_pred_NSGL-y_test)/sqrt(length(y_test));
%
err_B_NL = norm(B_NL(:)-B(:))/sqrt(m*q);
err_B_VML = norm(B_VML(:)-B(:))/sqrt(m*q);
err_B_NFL = norm(B_NFL(:)-B(:))/sqrt(m*q);
err_B_NSGL = norm(B_NSGL(:)-B(:))/sqrt(m*q);
%
err_gamma_NL = norm(gamma_NL(:)-gamma(:))/sqrt(p);
err_gamma_VML = norm(gamma_VML(:)-gamma(:))/sqrt(p);
err_gamma_NFL = norm(gamma_NFL(:)-gamma(:))/sqrt(p);
err_gamma_NSGL = norm(gamma_NSGL(:)-gamma(:))/sqrt(p);
%% output
fprintf("\n\n===========================================\n")
fprintf("===========================================")
fprintf("\nShape(B): %s, Scheme: S%d\n",file_pic, gamma_Scheme);
fprintf("RMSE-y: [VML, NL, NFL, NSGL] = [%.2f, %.2f, %.2f, %.2f]\n", rmse_VML, rmse_NL ,rmse_NFL ,rmse_NSGL)
fprintf("Error-B: [VML, NL, NFL, NSGL] = [%.2f, %.2f, %.2f, %.2f]\n", err_B_VML, err_B_NL ,err_B_NFL ,err_B_NSGL)
fprintf("Error-gamma: [VML, NL, NFL, NSGL] = [%.2f, %.2f, %.2f, %.2f]\n", err_gamma_VML, err_gamma_NL ,err_gamma_NFL ,err_gamma_NSGL)
%% visulization
plotyes = 0;
if plotyes
    plot_coef(B, gamma, 'True $B$', 'True $\gamma$');
    plot_coef(B_VML, gamma_VML, 'VML $B$', 'VML $\gamma$');
    plot_coef(B_NL, gamma_NL, 'NL $B$', 'NL $\gamma$');
    plot_coef(B_NFL, gamma_NFL, 'NFL $B$', 'NFL $\gamma$');
    plot_coef(B_NSGL, gamma_NSGL, 'NSGL $B$', 'NSGL $\gamma$');
end
%
function plot_coef(B, gamma, B_label, gamma_label)
    figure;
    % Plot matrix B
    subplot('Position', [0.05, 0.2, 0.7, 0.7]);
    imagesc(-double(B));
    colormap(gray);
    title(B_label, 'Interpreter', 'latex', 'FontSize', 18);
    axis equal;
    axis tight;
    
    % Plot vector gamma
    subplot('Position', [0.8, 0.2, 0.07, 0.7]);
    imagesc(-double(gamma));
    colormap(gray);
    title(gamma_label, 'Interpreter', 'latex', 'FontSize', 18);
    axis tight;
end

