% % % % UBitName

UBitName = ['n','i','s','a','r','g','s','a'];

% % % % personNumber

personNumber = ['5','0','1','6','9','4','6','2'];

filename = 'university data.xlsx';

CS_Score = xlsread(filename,'C:C');
Research_Overhead = xlsread(filename,'D:D');
Admin_Base_Pay = xlsread(filename,'E:E');
Tuition = xlsread(filename,'F:F');

% % % % Sample Mean

% CS Score
mu1 = mean(CS_Score)
% Research Overhead
mu2 = mean(Research_Overhead)
% Admin Base Pay
mu3 = mean(Admin_Base_Pay)
% Tuition
mu4 = mean(Tuition)


% % % % Variance

% CS Score
var1 = var(CS_Score)
% Research Overhead
var2 = var(Research_Overhead)
% Admin Base Pay
var3 = var(Admin_Base_Pay)
% Tuition
var4 = var(Tuition)


% % % % Standard Deviation

% CS Score
sigma1 = std(CS_Score)
% Research Overhead
sigma2 = std(Research_Overhead)
% Admin Base Pay
sigma3 = std(Admin_Base_Pay)
% Tuition
sigma4 = std(Tuition)

% % % % Covariance and Correlation of Pair of Variables

Cov_12 = cov(CS_Score,Research_Overhead)
Cor_12 = corrcoef(CS_Score,Research_Overhead)

Cov_13 = cov(CS_Score,Admin_Base_Pay)
Cor_13 = corrcoef(CS_Score,Admin_Base_Pay)

Cov_14 = cov(CS_Score,Tuition)
Cor_14 = corrcoef(CS_Score,Tuition)

Cov_23 = cov(Research_Overhead,Admin_Base_Pay)
Cor_23 = corrcoef(Research_Overhead,Admin_Base_Pay)

Cov_24 = cov(Research_Overhead,Tuition)
Cor_24 = corrcoef(Research_Overhead,Tuition)

Cov_34 = cov(Admin_Base_Pay,Tuition)
Cor_34 = corrcoef(Admin_Base_Pay,Tuition)

% % % % Plot


Concatenation_1 = [CS_Score, Research_Overhead];
Concatenation_2 = [CS_Score, Admin_Base_Pay];
Concatenation_3 = [CS_Score, Tuition];
Concatenation_4 = [Research_Overhead, Admin_Base_Pay];
Concatenation_5 = [Research_Overhead,Tuition];
Concatenation_6 = [Admin_Base_Pay,Tuition];

corrplot(Concatenation,'varNames',{'CSscore','Research_Overhead','Admin_Base_Pay','Tuition'})

corrplot(Concatenation_1,'varNames',{'CSscore','Research_Overhead'})
corrplot(Concatenation_2,'varNames',{'CSscore','Admin_Base_Pay'})
corrplot(Concatenation_3,'varNames',{'CSscore','Tuition'})
corrplot(Concatenation_4,'varNames',{'Research_Overhead','Admin_Base_Pay'})
corrplot(Concatenation_5,'varNames',{'Research_Overhead','Tuition'})
corrplot(Concatenation_6,'varNames',{'Admin_Base_Pay','Tuition'})


Concatenation = [CS_Score, Research_Overhead,Admin_Base_Pay,Tuition];
BN_mean = mean(Concatenation);
BN_covariance = cov(Concatenation);

% % % % Covariance Matrix
covarianceMat = cov(Concatenation,'omitrows')

% % % % Correlation Coefficient matrix
correlationMat = corrcoef(Concatenation)

% % % % log likelihood
norm_one = normpdf(CS_Score,mu1,sigma1);
logLikeOne = sum(log(norm_one));
norm_two = normpdf(Research_Overhead,mu2,sigma2);
logLikeTwo = sum(log(norm_two));
norm_three = normpdf(Admin_Base_Pay,mu3,sigma3);
logLikeThree = sum(log(norm_three));
norm_four = normpdf(Tuition,mu4,sigma4);
logLikeFour = sum(log(norm_four));
logLikelihood = logLikeOne + logLikeTwo + logLikeThree + logLikeFour;

% % % % BNGraph

BNgraph = [0 1 0 1; 0 0 1 0; 0 0 0 0; 0 1 1 0];
% BNgraph = [0 0 0 0; 1 0 0 1; 0 1 0 1; 1 0 0 0]
% % % % BNlogLikelihood

% nodes = [3];
% BNlogLikeFirst = sum(log(mvnpdf(Concatenation(:, nodes), BN_mean(nodes), BN_covariance(nodes, nodes))))
nodes = [1 2 4];
BNlogLikeSecond_Numerator = sum(log(mvnpdf(Concatenation(:, nodes), BN_mean(nodes), BN_covariance(nodes, nodes))));
nodes = [2 4];
BNlogLikeSecond_Denominator = sum(log(mvnpdf(Concatenation(:, nodes), BN_mean(nodes), BN_covariance(nodes, nodes))));
% BNlogLikeSecond = BNlogLikeSecond_Numerator/BNlogLikeSecond_Denominator
% nodes = [2 3];
% BNlogLikeThird_Numerator = sum(log(mvnpdf(Concatenation(:, nodes), BN_mean(nodes), BN_covariance(nodes, nodes))));
% nodes = [3];
% BNlogLikeThird_Denominator = sum(log(mvnpdf(Concatenation(:, nodes), BN_mean(nodes), BN_covariance(nodes, nodes))));
% BNlogLikeThird = BNlogLikeThird_Numerator/BNlogLikeThird_Denominator
nodes = [4 2 3];
BNlogLikeFourth_Numerator = sum(log(mvnpdf(Concatenation(:, nodes), BN_mean(nodes), BN_covariance(nodes, nodes))));
% nodes = [2 3];
% BNlogLikeFourth_Denominator = sum(log(mvnpdf(Concatenation(:, nodes), BN_mean(nodes), BN_covariance(nodes, nodes))));
% BNlogLikeFourth = BNlogLikeFourth_Numerator/BNlogLikeFourth_Denominator
BNlogLikelihood = BNlogLikeSecond_Numerator - BNlogLikeSecond_Denominator + BNlogLikeFourth_Numerator;

% BNlogIdeal = sum(log(mvnpdf(Concatenation,BN_mean,BN_covariance)));

% BNlogLikelihood = logLikeThree + ((logLikeOne+logLikeTwo+logLikeFour)-(logLikeTwo+logLikeFour)) + ((logLikeTwo+logLikeThree)-logLikeThree) + ((logLikeFour+logLikeTwo+logLikeThree)-(logLikeTwo+logLikeThree))

% % % % Save Data For Project Submission

save('proj1.mat','UBitName', 'personNumber', 'mu1', 'mu2', 'mu3', 'mu4', 'var1', 'var2', 'var3', 'var4', 'sigma1', 'sigma2', 'sigma3', 'sigma4', 'covarianceMat', 'correlationMat', 'logLikelihood', 'BNgraph', 'BNlogLikelihood')