function hhmmss = changetime(t)
t = round(t);
h = floor(t/3600);
m = floor(rem(t,3600)/60);
s = rem(rem(t,60),60);
if length(char(string(m)))==1
    m=append('0',string(m));
end
if length(char(string(s)))==1
    s=append('0',string(s));
end
hhmmss=append(string(h),':',string(m),':',string(s));
 end