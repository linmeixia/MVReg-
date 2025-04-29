function y = matexclusive(xi,par,nzcol)
if par.group_num_reduced == 1
    ZP1 = par.ZP1;
    ZP2 = par.ZP2;
    const = par.const;
    tmp1 = (xi'*ZP1)';
    tmp2 = xi'*ZP2;
    y = ZP1*tmp1-(const*tmp2)*ZP2;
else
    ZP1 = par.ZP1;
    D1 = par.D1;
    const_vec = par.const_vec;
    counts = par.counts;
    tmp = (xi'*ZP1)';
    tmp2 = par.smat*(D1.*tmp);
    tmp2 = repelem(const_vec.*tmp2,counts);
    tmp2 = tmp-tmp2.*D1;
    y = ZP1*tmp2;
end