% s = argmin { 0.5*||x - y||^2 + rho * sum_{i<j}|x_i-x_j| }
% idx represents the permuation s.t. newy is decreasing
% idx_inv represents the inverse

function [s,idx,rr] = prox_sClu(y,rho)
n=length(y);
w=linspace(n-1,1-n,n)';
[Py,idx] = sort(y,'descend');
tmp = Py-rho*w;
s = mexProxMonotonic(tmp);
tmp2 = s(1:n-1)-s(2:n);
rr = (abs(tmp2)<1e-12); 
s(idx) = s;
end