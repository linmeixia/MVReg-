function FLxi = matFL(xi,par,nzcol)
ZP = par.ZP;
PU1 = par.PU1;
if (size(PU1,2) > 0)
   tmp = (xi'*ZP)';  
   tmp2 = (tmp'*PU1)';
   FLxi = ZP*(par.Ph.*tmp + PU1*tmp2);
elseif (norm(par.Ph) > 0)
   tmp = (xi'*ZP)';      
   FLxi = ZP*(par.Ph.*tmp);
else
   FLxi = 0; 
end
end