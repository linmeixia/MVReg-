function [up,info] = proxClu(u, lambda1, lambda2)
[utmp,idx,rr2] = prox_sClu(u,lambda2);
if nargout > 1
   [up,rr1] = proxL1(utmp,lambda1);
   info.idx = idx;
   info.rr1 = rr1;
   info.rr2 = rr2;
else
   up = sign(utmp).*max(abs(utmp) - lambda1,0);
end
end