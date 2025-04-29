function Cluxi = matClu(xi,par,nzcol)
ZP = par.ZP;
PU1 = par.PU1;
if (size(PU1,2) > 0)
   tmp = (xi'*ZP)';  
   tmp2 = (tmp'*PU1)';
   Cluxi = ZP*(par.Ph.*tmp + PU1*tmp2);
elseif (norm(par.Ph) > 0)
   tmp = (xi'*ZP)';      
   Cluxi = ZP*(par.Ph.*tmp);
else
   Cluxi = 0; 
end