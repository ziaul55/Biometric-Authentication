function [hter,far,frr] = ComputeError(wolves, sheep, thrd)
%wolves and sheep are 1 dimension in column
fa = size(find (wolves >= thrd),1);
fr = size(find (sheep < thrd),1);
far = fa /size(wolves,1);
frr = fr /size(sheep,1);
hter = far + frr;
hter = hter/2;

