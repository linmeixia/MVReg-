function y = diagXXt_fun(Xinput,n)
y = zeros(n,1);
for i = 1:n
    tmp = zeros(n,1);
    tmp(i) = 1;
    ytmp = Xinput.Xtmap(tmp);
    y(i) = sum(ytmp.*ytmp);
end
end