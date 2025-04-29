%% proximal mapping for L1 norm 
%% y = argmin{ 0.5*||y - x||^2 + lambda ||y||_1}
%%
function [y,rr] = proxL1(x,lambda)
[tmp, rr] = proj_inf(x,lambda);
y = x - tmp;
rr = ~rr;
end