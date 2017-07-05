%% How to handle the different size of features
% Compute the histogram of each row independently.
% For example assume trnFeature_Set1{1,1} is a matrix of 128 by 3155.
% After compute the histogram using 52 bins (0:10:255), we obtain a 
% normalised histgram, a matrix of 128 by 52.
Histogram = hist(double(trnFeature_Set1{1,1})', 0:5:255)'./length(trnFeature_Set1{1,1});
% We concatenate individual row vector to form a 1280 dimension feature
% descriptor. The Feature is a 1 by 6656 row vector.
Feature = Histogram(:)'; 

%% How to apply KMean
F = rand(100, 10); % Generate 100 samples, each of which has 10 dimensions
[idx,C] = kmeans(F, 2); % Cluster F into two clusters, the centroid is C
doc kmeans

%% How to apply GMM
F = rand(100, 10); % Generate 100 samples, each of which has 10 dimensions
GMMModel = fitgmdist(F(1:50,:), 2);
idx = cluster(GMMModel, F(51:100,:));
doc fitgmdist

%% How to apply PCA
F = rand(100, 10); % Generate 100 samples, each of which has 10 dimensions
[coeff,score,latent] = pca(F);
% Each column of coeff contains coefficients for one principal component
% Principal component scores are the representations of X in the principal 
% component space. Rows of score correspond to observations, and columns 
% correspond to components. The principal component variances are the 
% eigenvalues of the covariance matrix of X.

%% How to apply LDA
F = rand(100, 10); % Generate 100 samples, each of which has 10 dimensions
C = unidrnd(3,[100,1]); % Generate class label for F
LDAModel = fitcdiscr(F(1:50,:),C(1:50)); % Fit a LDA model
idx = predict(LDAModel, F(51:100,:));

%% How to apply SVM
% Load the data
[X,Y] = wine_dataset;
X = X';
[~,Y] = find(Y');
% Divide the dataset into a training and testing class
idx = crossvalind('Kfold',Y,2);
X_trn = X(idx==1,:);
Y_trn = Y(idx==1,:);
X_tst = X(idx==2,:);
Y_tst = Y(idx==2,:);

% Training Procedure
% Construct a multi-class classifier by fitting a set of binary svm 
% classifiers using one v.s all scheme
% Create an SVM template, and specify the Gaussian kernel. 
% It is good practice to standardize the predictors.
t = templateSVM('Standardize',1,'KernelFunction','gaussian');
% Fit a multi-class SVM classifier
Mdl = fitcecoc(X_trn,Y_trn,'Learners',t,'FitPosterior',1,'Verbose',2);

% Testing Procedure
Y_tst_Predict = predict(Mdl,X_tst);
% Check accuracy
Diff = Y_tst-Y_tst_Predict;
ind = find(Diff==0);
Right = numel(ind);
Wrong = numel(Y_tst) - Right;
Accuracy = Right/(Right+Wrong);
disp(['Accuracy on testing set is: ' num2str(Accuracy*100) '%']);

%% How to apply Neural Networks
% Load the data
[X,Y] = wine_dataset;
X = X';
[~,Y] = find(Y');
FeatureSpace = 1:13; % Use all features
% Divide the dataset into a training and testing class
idx = crossvalind('Kfold',Y,2);
X_trn = X(idx==1,FeatureSpace);
Y_trn = Y(idx==1,:);
X_tst = X(idx==2,FeatureSpace);
Y_tst = Y(idx==2,:);

% Training Procedure
% Create a feedforward NNs using 1 layer with 7 hidden nodes
net = feedforwardnet(7);
% Train NNs on the training dataset
net = train(net,X_trn',Y_trn');
% Visualise the structure of NNs
view(net);

% Testing Procedure
Y_tst_Predict = net(X_tst');
% Why? Any idea?
Y_tst_Predict = round(Y_tst_Predict);
% Check accuracy
Diff = Y_tst-Y_tst_Predict';
ind = find(Diff==0);
Right = numel(ind);
Wrong = numel(Y_tst) - Right;
Accuracy = Right/(Right+Wrong);
disp(['Accuracy on testing set is: ' num2str(Accuracy*100) '%']);
