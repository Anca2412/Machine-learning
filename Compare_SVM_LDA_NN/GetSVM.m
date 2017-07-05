    function [ resultSVM ] = GetSVM( X_trn, Y_trn, X_tst, Y_tst, index, resultSVM, maxCol )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    t = templateSVM('Standardize',1,'KernelFunction','polynomial');
% Fit a multi-class SVM classifier
    Mdl = fitcecoc(X_trn(:,1:maxCol),Y_trn,'Learners',t,'FitPosterior',1,'Verbose',2);

% Testing Procedure
    Y_tst_Predict = predict(Mdl,X_tst(:,1:maxCol));
    
% Check accuracy
    Diff = Y_tst-Y_tst_Predict;
    ind = find(Diff==0);
    Right = numel(ind);
    Wrong = numel(Y_tst) - Right;
    Accuracy = Right/(Right+Wrong);
%disp(['Accuracy on testing set is: ' num2str(Accuracy*100) '%']);
    resultSVM(index) = Accuracy*100;
end

