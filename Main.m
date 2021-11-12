%% Md. Ziaul Hoque, CMVS, Faculty of ITEE, University of Oulu, Finland 

%% Multi-Modal Data fusion 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Score level fusion of  face and voice for biometric authentication   %
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all
%% Load scores (see description of variables bellow)
load scores_DCT_LFCC_GMM % dev and eva scores
%      dev (developement) scores are used for model and parameter estimation (train the fusion model) 
%      eva (evaluation) scores are used to estimate the fusion performance and report the
%      errors.
%      Both dev and eva are structures with fields : 1) sheep: corresponding to the
%      scores of authentic accesses and 2) wolves corresponding to the scores of
%      attack accesses (imposters). The scores are structured as follows :
%      dev.sheep(:,1):  developpement authentic face scores.
%      dev.wolves(:,1): developement impostor face scores.
%      eva.sheep(:,2):  evaluation authentic voice scores.
%      eva.wolves(:,2): evaluation impostor voice scores.   

%Examples of setting the decision threshold and computing the verification 
%error are given bellow

%% Compute verification error for face
    %Compute decision threshold using dev scores
    thrd=ComputeDecisionThreshold(dev.wolves(:,1), dev.sheep(:,1));

    % Compute dev error      
    [err,~,~] = ComputeError(dev.wolves(:,1), dev.sheep(:,1), thrd);
    fprintf('Face dev error: %2.2f \n',err*100);

    % Compute eva error using dev threshold
    [err_f,~,~] = ComputeError(eva.wolves(:,1), eva.sheep(:,1), thrd);
    fprintf('Face eva error: %2.2f \n',err_f*100);

%% Compute verification error for voice
    %Compute decision threshold using dev scores
    thrd = ComputeDecisionThreshold(dev.wolves(:,2), dev.sheep(:,2));
    % Compute dev error      
    [err,~,~] = ComputeError(dev.wolves(:,2), dev.sheep(:,2), thrd);
    fprintf('Voice dev error: %2.2f \n',err*100);
    
    % Computeevaerror using dev threshold
    [err_v,~,~] = ComputeError(eva.wolves(:,2), eva.sheep(:,2), thrd);
    fprintf('Voice eva error: %2.2f \n',err_v*100);

%An example of score fusion using simple sum rule is provided in the
%following

%% Face and voice fusion using simple sum rule
    % Fuse scores
    ssum_dev.wolves=(dev.wolves(:,1))+dev.wolves(:,2);
    ssum_eva.wolves=eva.wolves(:,1)+eva.wolves(:,2);

    ssum_dev.sheep=dev.sheep(:,1)+dev.sheep(:,2);
    ssum_eva.sheep=eva.sheep(:,1)+eva.sheep(:,2);
    
    % Compute simple sum fusion error
    thrd = ComputeDecisionThreshold(ssum_dev.wolves, ssum_dev.sheep);
    % Compute dev error      
    [err,~,~] = ComputeError(ssum_dev.wolves, ssum_dev.sheep, thrd);
    fprintf('Simple sum fusion dev error: %2.2f \n',err*100);
    % Computeevaerror using dev threshold
    [err,~,~] = ComputeError(ssum_dev.wolves, ssum_eva.sheep, thrd);
    fprintf('Simple sum fusion eva error: %2.2f \n',err*100);
    
% Tasks to perform for the programming homework starts here    

%%  1. TODO: Implement weighted sum fusion. Use the dev error of face and
%   voice to weight the eva scores. The weights have to be inverse to
%   the errors and sum to one. All scores of one system (face/voice) have
%   to have the same weight.
    w1=0.3;
    w2=0.7;
    ssum_dev.wolves=(dev.wolves(:,1))*w1+(dev.wolves(:,2))*w1;
    ssum_dev.wolves = (ssum_dev.wolves - mean(ssum_dev.wolves(:)))./var(ssum_dev.wolves(:));
    ssum_eva.wolves=(eva.wolves(:,1))*w2+(eva.wolves(:,2))*w2;

    ssum_dev.sheep=(dev.sheep(:,1))*w1+(dev.sheep(:,2))*w1;
    ssum_dev.sheep = (ssum_dev.sheep- mean(ssum_dev.sheep(:)))./var(ssum_dev.sheep(:));
    ssum_eva.sheep=(eva.sheep(:,1))*w2+(eva.sheep(:,2))*w2;

%%  1.1 TODO: Study the effect of score normalization for rule-based fusion
%   (sum and weighted-sum). Min-max or centered-unit variance
%   normalizations. Normalization parameters should be computed using dev
%   scores only. Apply the normalization then compute the error.

 % Compute simple sum fusion error
    thrd = ComputeDecisionThreshold(ssum_dev.wolves, ssum_dev.sheep);
    % Compute dev error      
    [err,~,~] = ComputeError(ssum_dev.wolves, ssum_dev.sheep, thrd);
    fprintf('weighted sum fusion dev error: %2.2f \n',err);
    % Computeevaerror using dev threshold
    [err,~,~] = ComputeError(ssum_dev.wolves, ssum_eva.sheep, thrd);
    fprintf('weighted sum fusion eva error: %2.2f \n',err);

% %     2 TODO: Implement one classifier based fusion (SVM or logistic
% %     regression). Use the dev scores for training the classifier  and the
% %     eva scores for classification and computing the error. In the case of SVM
% %     either the output probabilities can be seen as the fusion scores for 
% %     computing the error or simply the error is (1-accuracy). For the case
% %     of logistic regression the  predicted values are the fusion scores.
% %     Classification labels should be zeroes for wolves and ones for sheep.
% %     Hint: use fitcsvm and predict Matlab functions for training an svm classifier and
% %     classifying eva scores. 

%% By using SVM %% 
     A=[dev.wolves;dev.sheep];
     B=[eva.wolves;eva.sheep];
     [L1]=[zeros((length(dev.wolves)),1);ones((length(dev.sheep)),1)];
     [L2]=[zeros((length(eva.wolves)),1);ones((length(eva.sheep)),1)];
     SVMModel_dev= fitcsvm(A,L1);
     SVMModel_eva= fitcsvm(B,L2);
     [ypredict_1,score_2] = predict(SVMModel_dev,A);
     [ypredict_2,score_2] = predict(SVMModel_eva,B);  

 %% For Classification %%
     accuracy_dev=length(find(ypredict_1 ~= L1))/length(L1);
     fprintf('svm fusion eva error: %2.2f \n',accuracy_dev*100);
     accuracy_eva=length(find(ypredict_2 ~= L2))/length(L2);
     fprintf('svm fusion eva error: %2.2f \n',accuracy_eva*100);     

%%    3 TODO: Implement one density based score fusion
%     Use the dev scores for estimating the authentic (f_sheep) and imposter
%     (f_wolves) densities and use the ratio f_sheep(eva)/f_wolves(eva) as
%     the fusion. Examples of density estimation approaches are Gaussian,
%     mexture of Gaussians, kernel density estimation, etc. Hint: use
%     fitgmdist and pdf Matlab functions for estimation of density function
%     and computation of likelihood.

%% Estimation of Authentic and Imposter %%  
F_sheep_A= fitgmdist(dev.sheep,1);
F_wolves_I= fitgmdist(dev.wolves,1);

%% Estimation of Desnsities%%
% ratio=f_sheep/f_wolves;
F_sheep_sheep1=pdf(F_sheep_A,eva.sheep);
F_sheep_wolves2=pdf(F_wolves_I,eva.sheep);
F_wolves_sheep3=pdf(F_sheep_A,eva.wolves);
F_wolves_wolves4=pdf(F_wolves_I,eva.wolves);
%% Ratio %%
F_sheep_fusion1=F_sheep_sheep1./F_sheep_wolves2;
F_sheep_fusion2=F_wolves_wolves4./F_wolves_sheep3;
% Compute simple sum fusion erros
thrd = ComputeDecisionThreshold(F_sheep_fusion2, F_sheep_fusion1);
% Compute dev error      
[err,~,~] = ComputeError(F_sheep_fusion2, F_sheep_fusion1,thrd);
fprintf('Density based score fusion eva error: %2.2f \n',err);
  
%%	REMARK: the implementation of two types of fusion is compulsory, the 
%	implementation of the third type is a plus.
                                                              %% MD ZIAUL HOQUE %%

