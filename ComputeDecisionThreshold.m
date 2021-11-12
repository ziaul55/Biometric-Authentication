function [thrd_min] = ComputeDecisionThreshold(wolves, sheep, cost)

%impostor_scores (wolves) and client_scores (sheep) must be vectors of column 1 but they can
%can have different number of rows
%
%cost(1) : cost of false acceptance, default =0.5
%cost(2) : cost of false rejection, default =0.5

% thrd_min  the decision threshold corresponding to the min error

if (nargin < 3 || isempty(cost))
  cost = [0.5 0.5];
end;

% n_hist_wolves = 0;
% n_hist_sheep = 0;

[f_I,x_I] = ecdf(wolves);
[f_C,x_C] = ecdf(sheep);

%delete non-unique abscissae
index = find(x_I(1:size(x_I,1)-1) - x_I(2:size(x_I,1)) == 0 );
f_I(index) = [];
x_I(index) = [];
index = find(x_C(1:size(x_C,1)-1) - x_C(2:size(x_C,1)) == 0 );
f_C(index) = [];
x_C(index) = [];

%curve fitting:
%sample the function
x = union(x_I, x_C);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get a linear interpolation of FAR and FRR based on sample data x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (length(x_C) == 1)
	[r,c]=size(x);
	new_f_C = zeros(size(x));
	[tmp_, ind] = min(x_C - x);
	new_f_C(ind:r) = ones(r-ind+1,1);
else
	new_f_C = interp1(x_C,f_C,x, 'linear');
end;
new_f_I = interp1(x_I,f_I,x, 'linear');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Replace NAN value with zero or one in new_f_C and new_f_I
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
index_ = isnan(new_f_C);
end_NAN = find([index_;1] - [1;index_] >= 1) - 1;
start_NAN = find([index_;1] - [1;index_] <= -1);

%sometimes the first element is not zero!
if (start_NAN == 1)
else
  new_f_C(1:start_NAN)=0;
end;
%the end element is always 1!
new_f_C(end_NAN:size(new_f_C,1))=1;

index_ = isnan(new_f_I);
end_NAN = find([index_;1] - [1;index_] >= 1) - 1;
start_NAN = find([index_;1] - [1;index_] <= -1);
if (start_NAN == 1),
else
  new_f_I(1:start_NAN)=0;
end;
new_f_I(end_NAN:size(new_f_C,1))=1;

%convert into FAR and FRR
new_f_I= 1-new_f_I;
%no change for new_f_C= new_f_C;

[min_value, min_index] = min( abs(cost(1) * new_f_I - cost(2) * new_f_C) );

thrd_min = x(min_index);
% wer_min = cost(1) * new_f_I(min_index) + cost(2) * new_f_C(min_index);
