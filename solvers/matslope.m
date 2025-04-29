function WKxi = matslope(xi,par,nzcol)
vecV1 = par.vecV1;
vecV2 = par.vecV2;

if ~isempty(vecV1)
   tmp1 = (xi'*vecV1)';
   WKxi = vecV1*tmp1;
end

if ~isempty(vecV2)
   tmp2 = (xi'*vecV2)';
   WKxi = vecV2*tmp2;
end

if isempty(vecV1) && isempty(vecV2)
    WKxi = 0;
end
end