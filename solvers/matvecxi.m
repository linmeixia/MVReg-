function IpsigAATxi = matvecxi(xi,parxi,mapop)
IpsigAATxi = parxi.const*xi + feval(mapop.XXtmap,xi) + feval(mapop.ZZtmap,xi);
end