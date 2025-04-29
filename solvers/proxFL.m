function [up,info] = proxFL(Binput,u,lambda1,lambda2)
utmp = mexCondat(u,lambda2);
if nargout > 1
    [up,rr1] = proxL1(utmp,lambda1);
    tmp = Binput.Bmap(utmp);
    rr2 = (abs(tmp) < 1e-12);
    info.rr1 = rr1;
    info.rr2 = rr2;
    info.innerNT = 0;
    info.innerflsa = 0;
else
    up = sign(utmp).*max(abs(utmp) - lambda1,0);
end
end
