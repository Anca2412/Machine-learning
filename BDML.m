clear; close all; clc; 
%load wine_dataset
[X,Y] = wine_dataset;  %X is a 13X178 Matrix, Y 3X178
X=X'; %transpose X to a 178X13 Matrix
X=X(:,1:2); %We are interested in columns 1and 2 in X matrix (ie features 1 and 2)
[~,Y] = find(Y'); %Orient and turn Y to labelID matrix


idx = crossvalind('Kfold',Y,2);
X_trn = X(idx==1,:);
Y_trn = Y(idx==1,:);
X_tst = X(idx==2,:);
Y_tst = Y(idx==2,:);


%Training Set

h1=gscatter(X_trn(:,1),X_trn(:,2),Y_trn);
legend('Winery 1','Winery 2','Winery 3');
title('Training Set'); 
hold on;


%use built in function , build a linear discriminant using class labels
ldaModel = fitcdiscr(X_trn,Y_trn);

%Create a point to classify
%newPoint = mean(X);
%scatter(X_tst(:,1),X_tst(:,2),'kx')


% Coefficients for boundary between classes 2 and 3
K = ldaModel.Coeffs(2,3).Const;
L = ldaModel.Coeffs(2,3).Linear;
% Plot the curve of the boundary
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h2 = ezplot(f,[11 15 0 6]);

% Coefficients for boundary between classes 1 and 2
K = ldaModel.Coeffs(1,2).Const;
L = ldaModel.Coeffs(1,2).Linear;
% Plot the curve of the boundary
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h3 = ezplot(f,[11 15 0 6]);


nn = length(X_tst);
for i = 1:nn
    test_point = [X_tst(i,1), X_tst(i,2)]; 
    predictedClass = predict(ldaModel,test_point); 
    disp(test_point);disp(predictedClass);  
end

h4 = figure; 
h4 =gscatter(X_tst(:,1),X_tst(:,2),Y_tst);
title('Test Set'); 
 